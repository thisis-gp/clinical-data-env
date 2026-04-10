"""
Baseline inference script for the Clinical Data Standardization Environment.

Runs an LLM agent against all 4 tasks via the OpenEnv WebSocket API and emits
structured logs in the exact format required by the OpenEnv hackathon:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
    API_BASE_URL  - LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME    - Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      - API key (checked first), fallback: API_KEY
    ENV_BASE_URL  - OpenEnv server URL (default: http://localhost:8000)

Usage:
    python inference.py
"""

import json
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from env_utils import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
ENV_NAME = "clinical_data_env"
BENCHMARK_SET = os.getenv("BENCHMARK_SET", "toy").strip().lower()
PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_ROOT.parent
LOGS_DIR = PROJECT_ROOT / "logs"
SUBMISSION_SCORE_FLOOR = 0.01
SUBMISSION_SCORE_CAP = 0.99

TASK_NAMES = [
    "task1_edc_to_sdtm",
    "task2_sdtm_validation",
    "task3_sdtm_to_adam",
    "task4_cross_domain_validation",
]

SYSTEM_PROMPT = """You are a clinical data standardization expert trained in CDISC standards.

You will receive a task description, study context (CDISC rules), and input data.
You must respond with ONLY a valid JSON object (no markdown, no explanation outside the JSON).

Your JSON must have this exact structure:
{
  "task_id": <1, 2, 3, or 4>,
  "output_data": <your answer - see task description for required format>,
  "reasoning": "<brief explanation of your choices>"
}

For Task 1 (EDC to SDTM): output_data is a JSON object with SDTM field to value pairs.
For Task 2 (Validation): output_data is {"violations": [{"field":"...","issue":"...","corrected_value":"..."},...]}
  IMPORTANT: If the record has NO violations, return {"violations": []} — an empty list.
  Not every record is wrong. Only flag fields that genuinely violate SDTM conventions.
For Task 3 (ADAM derivation): output_data is a JSON array of ADAM records, one per visit.
For Task 4 (Cross-domain validation): output_data is {"issues": [{"type":"...","domain":"...","field":"...","description":"..."},...]}
  IMPORTANT: If no cross-domain inconsistencies exist, return {"issues": []}.

If this is a RETRY (attempt > 1), you will receive feedback from your previous answer.
Read the feedback carefully and correct only the specific errors identified.

Return ONLY the JSON object. No markdown code blocks."""


def _submission_safe_score(score: float) -> float:
    """Keep logged task scores strictly inside (0, 1) for hackathon validation."""
    return min(max(float(score), SUBMISSION_SCORE_FLOOR), SUBMISSION_SCORE_CAP)


class TeeStream:
    """Write output to the original stream and a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_output(log_path: Path):
    """Mirror stdout and stderr to a timestamped log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _import_runtime_types():
    """Import client and action types in either package or script mode."""
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

    try:
        from clinical_data_env.client import ClinicalDataEnv
        from clinical_data_env.models import ClinicalAction
    except ImportError:
        from client import ClinicalDataEnv
        from models import ClinicalAction

    return ClinicalDataEnv, ClinicalAction


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def call_llm(obs_data: dict) -> dict:
    """
    Build a prompt from the observation and call the LLM.
    Returns the parsed action dict.
    """
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    task_id = obs_data.get("task_id", 1)
    task_description = obs_data.get("task_description", "")
    study_context = obs_data.get("study_context", "")
    input_data = obs_data.get("input_data", {})
    feedback = obs_data.get("feedback", "")
    case_number = obs_data.get("case_number", "?")
    total_cases = obs_data.get("total_cases", "?")
    case_attempt = obs_data.get("case_attempt", 1)
    pre_step_hints = obs_data.get("pre_step_hints", [])
    action_history = obs_data.get("action_history", [])
    field_scores = obs_data.get("field_scores", {})

    user_content = (
        f"TASK ID: {task_id}\n"
        f"CASE: {case_number} of {total_cases}\n\n"
        f"ATTEMPT: {case_attempt}\n\n"
        f"STUDY CONTEXT & RULES:\n{study_context}\n\n"
        f"TASK DESCRIPTION:\n{task_description}\n\n"
        f"INPUT DATA:\n{json.dumps(input_data, indent=2)}\n"
    )

    if pre_step_hints:
        hints_text = "\n".join(f"  - {h}" for h in pre_step_hints)
        user_content += f"\nHINTS (auto-detected from this case):\n{hints_text}\n"

    if feedback:
        user_content += f"\nFEEDBACK FROM PREVIOUS STEP:\n{feedback}\n"

    if field_scores:
        failed = [f for f, s in field_scores.items() if s == 0.0]
        passed = [f for f, s in field_scores.items() if s == 1.0]
        if failed:
            user_content += f"\nFIELDS YOU GOT WRONG: {', '.join(failed)}\n"
        if passed:
            user_content += f"FIELDS YOU GOT RIGHT (keep these): {', '.join(passed)}\n"

    if action_history and len(action_history) > 0:
        user_content += "\nYOUR PREVIOUS ATTEMPTS ON THIS CASE:\n"
        for attempt in action_history:
            attempt_num = attempt.get("attempt", "?")
            prev_reward = attempt.get("reward", 0.0)
            prev_output = attempt.get("output_data", {})
            user_content += (
                f"  Attempt {attempt_num} (score: {prev_reward:.2f}):\n"
                f"  {json.dumps(prev_output, separators=(',', ':'))}\n"
            )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        raw = "\n".join(inner).strip()

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, task_name: str, ClinicalAction) -> tuple[bool, int, float, list[float]]:
    """
    Run one full episode (one task, 5 cases) via an existing WebSocket session.

    Returns:
        (success, steps, final_score, per_step_rewards)
    """
    step_num = 0
    rewards: list[float] = []
    success = False

    print(
        f"[START] task={task_name} env={ENV_NAME} benchmark_set={BENCHMARK_SET} model={MODEL_NAME}",
        flush=True,
    )

    try:
        result = env.reset()
        obs = result.observation

        if obs.task_name != task_name:
            raise RuntimeError(
                f"Task mismatch after reset: expected {task_name}, got {obs.task_name}"
            )

        while True:
            if result.done:
                final_reward = result.reward or 0.0
                if not rewards:
                    rewards.append(final_reward)
                success = True
                break

            obs_dict = {
                "task_id": obs.task_id,
                "task_name": obs.task_name,
                "task_description": obs.task_description,
                "input_data": obs.input_data,
                "study_context": obs.study_context,
                "feedback": obs.feedback,
                "case_number": obs.case_number,
                "total_cases": obs.total_cases,
                "case_attempt": obs.case_attempt,
                "pre_step_hints": getattr(obs, "pre_step_hints", []),
                "action_history": getattr(obs, "action_history", []),
                "field_scores": getattr(obs, "field_scores", {}),
            }

            error_msg = "null"
            action_str = "null"

            try:
                action_dict = call_llm(obs_dict)
                action = ClinicalAction(
                    task_id=action_dict.get("task_id", obs.task_id),
                    output_data=action_dict.get("output_data", {}),
                    reasoning=action_dict.get("reasoning", ""),
                )

                result = env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                rewards.append(reward)
                step_num += 1

                action_str = json.dumps(action_dict, separators=(",", ":"))

                print(
                    f"[STEP]  step={step_num} "
                    f"action={action_str} "
                    f"reward={reward:.2f} "
                    f"done={'true' if done else 'false'} "
                    f"error=null",
                    flush=True,
                )

                if done:
                    success = True
                    break

            except json.JSONDecodeError as exc:
                error_msg = f"JSONDecodeError: {str(exc).replace(chr(10), ' ')}"
                step_num += 1
                rewards.append(0.0)
                print(
                    f"[STEP]  step={step_num} "
                    f"action={action_str} "
                    f"reward=0.00 "
                    f"done=false "
                    f"error={error_msg}",
                    flush=True,
                )
                try:
                    result = env.step(
                        ClinicalAction(
                            task_id=obs.task_id,
                            output_data={},
                            reasoning="parse error",
                        )
                    )
                    obs = result.observation
                    if result.done:
                        success = False
                        break
                except Exception:
                    break

            except Exception as exc:
                error_msg = str(exc).replace("\n", " ")
                step_num += 1
                rewards.append(0.0)
                print(
                    f"[STEP]  step={step_num} "
                    f"action={action_str} "
                    f"reward=0.00 "
                    f"done=false "
                    f"error={error_msg}",
                    flush=True,
                )
                break

    except Exception as exc:
        error_str = str(exc).replace("\n", " ")
        step_num += 1
        print(
            f"[STEP]  step={step_num} "
            f"action=null "
            f"reward=0.00 "
            f"done=false "
            f"error={error_str}",
            flush=True,
        )

    final_score = _submission_safe_score(sum(rewards) / len(rewards) if rewards else SUBMISSION_SCORE_FLOOR)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END]   success={'true' if success else 'false'} "
        f"steps={step_num} "
        f"score={final_score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )

    return success, step_num, final_score, rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print(
            "ERROR: No API key found. Set HF_TOKEN or API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOGS_DIR / f"inference-{BENCHMARK_SET}-{timestamp}.log"

    with tee_output(log_path):
        print(
            f"[RUN]   env={ENV_NAME} benchmark_set={BENCHMARK_SET} "
            f"env_base_url={ENV_BASE_URL} model={MODEL_NAME}",
            flush=True,
        )
        print(f"[LOG]   path={log_path}", flush=True)

        all_scores: list[float] = []

        ClinicalDataEnv, ClinicalAction = _import_runtime_types()

        with ClinicalDataEnv(base_url=ENV_BASE_URL).sync() as env:
            for task_name in TASK_NAMES:
                try:
                    _, _, score, _ = run_episode(env, task_name, ClinicalAction)
                    all_scores.append(score)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    all_scores.append(SUBMISSION_SCORE_FLOOR)

        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"\nOverall score: {overall:.4f}", flush=True)
        print(f"Per-task:      {', '.join(f'{s:.4f}' for s in all_scores)}", flush=True)


if __name__ == "__main__":
    main()
