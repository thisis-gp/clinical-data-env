"""
FastAPI application for the Clinical Data Standardization Environment.

Endpoints:
    POST /reset   — Reset environment, returns first observation
    POST /step    — Submit an action, returns graded observation
    GET  /state   — Current episode state
    GET  /schema  — Action/observation JSON schemas
    WS   /ws      — WebSocket for persistent sessions
    GET  /health  — Health check
    GET  /web     — Interactive web UI
"""

import os

try:
    from ..env_utils import load_dotenv
except ImportError:
    from env_utils import load_dotenv

load_dotenv()
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import ClinicalAction, ClinicalObservation
    from .clinical_data_env_environment import ClinicalDataEnvironment
except ImportError:
    from models import ClinicalAction, ClinicalObservation
    from server.clinical_data_env_environment import ClinicalDataEnvironment


app = create_app(
    ClinicalDataEnvironment,
    ClinicalAction,
    ClinicalObservation,
    env_name="clinical_data_env",
    max_concurrent_envs=4,
)


def main(host: str | None = None, port: int | None = None):
    import uvicorn
    host = host or os.getenv("SERVER_HOST", "0.0.0.0")
    port = port or int(os.getenv("SERVER_PORT", "8000"))
    benchmark_set = os.getenv("BENCHMARK_SET", "toy").strip().lower()
    print(
        f"[SERVER] env=clinical_data_env benchmark_set={benchmark_set} "
        f"api=http://localhost:{port} ui=http://localhost:{port}/web/",
        flush=True,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
