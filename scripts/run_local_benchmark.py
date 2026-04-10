"""Run the local benchmark against a single fixed server port.

This helper keeps local evaluation predictable:
- it uses the configured SERVER_PORT / ENV_BASE_URL only
- it reuses an already-healthy local server on that port
- otherwise it starts the server on that same port, runs inference, then stops it

Examples:
    python scripts/run_local_benchmark.py
    python scripts/run_local_benchmark.py --tasks task4
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
SERVER_LOG = LOGS_DIR / "server-local-benchmark.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local inference against a single fixed environment port.")
    parser.add_argument(
        "--tasks",
        default=os.getenv("INFERENCE_TASKS", "all"),
        help="Comma-separated task ids or names to run, e.g. 'task4' or '1,2,4'.",
    )
    return parser.parse_args()


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def healthcheck(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as response:
            return response.status == 200
    except URLError:
        return False


def wait_for_health(url: str, timeout_seconds: int = 20) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if healthcheck(url):
            return True
        time.sleep(1)
    return False


def normalize_single_task(raw_tasks: str) -> str | None:
    tokens = [token.strip().lower() for token in raw_tasks.split(",") if token.strip()]
    if len(tokens) != 1:
        return None

    task_lookup = {
        "1": "1",
        "task1": "1",
        "task1_edc_to_sdtm": "1",
        "2": "2",
        "task2": "2",
        "task2_sdtm_validation": "2",
        "3": "3",
        "task3": "3",
        "task3_sdtm_to_adam": "3",
        "4": "4",
        "task4": "4",
        "task4_cross_domain_validation": "4",
    }
    return task_lookup.get(tokens[0])


def main() -> int:
    args = parse_args()
    server_port = int(os.getenv("SERVER_PORT", "8001"))
    env_base_url = os.getenv("ENV_BASE_URL", f"http://localhost:{server_port}")
    health_url = f"{env_base_url.rstrip('/')}/health"
    forced_task_id = normalize_single_task(args.tasks)
    server_started_here = False
    server_process: subprocess.Popen[str] | None = None
    log_handle = None
    server_env = os.environ.copy()
    if forced_task_id is not None:
        server_env["FORCE_TASK_ID"] = forced_task_id
    else:
        server_env.pop("FORCE_TASK_ID", None)

    if forced_task_id is not None and healthcheck(health_url):
        print(
            f"[SERVER] single-task run requested for task {forced_task_id}, but a server is already running on "
            f"port {server_port}. Stop that process first so the launcher can start a fresh single-task server.",
            file=sys.stderr,
        )
        return 1

    if healthcheck(health_url):
        print(f"[SERVER] reusing healthy server at {env_base_url}")
    else:
        if is_port_open("127.0.0.1", server_port):
            print(
                f"[SERVER] port {server_port} is already in use but {health_url} is not healthy. "
                "Stop the stale process on that port and rerun.",
                file=sys.stderr,
            )
            return 1

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_handle = SERVER_LOG.open("w", encoding="utf-8")
        server_process = subprocess.Popen(
            [sys.executable, "-m", "clinical_data_env.server.app"],
            cwd=str(PROJECT_ROOT.parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=server_env,
        )
        server_started_here = True

        if not wait_for_health(health_url):
            print(
                f"[SERVER] failed to become healthy at {health_url}. See {SERVER_LOG}.",
                file=sys.stderr,
            )
            if server_process.poll() is None:
                server_process.terminate()
                server_process.wait(timeout=10)
            log_handle.close()
            return 1

        print(f"[SERVER] started local server at {env_base_url}")

    try:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "inference.py"), "--tasks", args.tasks],
            cwd=str(PROJECT_ROOT),
            text=True,
            env=os.environ.copy(),
            check=False,
        )
        return result.returncode
    finally:
        if server_started_here and server_process is not None:
            if server_process.poll() is None:
                server_process.terminate()
                server_process.wait(timeout=10)
            print(f"[SERVER] stopped local server on port {server_port}")
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
