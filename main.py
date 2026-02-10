#!/usr/bin/env python3
"""
Start fin-alpha backend and agent together.

Usage:
  python3 main.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _pick_python_executable() -> str:
    venv_python = ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _wait_for_backend(url: str, timeout_sec: int = 20) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(0.4)
    return False


def _terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _ensure_langgraph_compat(python_bin: str, env: dict) -> bool:
    """
    Ensure agent imports cleanly. Auto-repair known langgraph mismatch where
    langgraph-checkpoint/prebuilt from newer stacks conflicts with langgraph==0.0.26.
    """
    import_cmd = [python_bin, "-c", "import agents.agent"]
    first = subprocess.run(import_cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if first.returncode == 0:
        return True

    err_blob = f"{first.stdout}\n{first.stderr}"
    if "CheckpointAt" in err_blob and "langgraph.checkpoint.base" in err_blob:
        print("[FIX] Detected langgraph version conflict. Repairing environment...")
        uninstall_cmd = [
            python_bin,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "langgraph-checkpoint",
            "langgraph-prebuilt",
            "langgraph-sdk",
        ]
        subprocess.run(uninstall_cmd, cwd=ROOT, env=env, check=False)
        second = subprocess.run(import_cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        if second.returncode == 0:
            print("[FIX] Langgraph import issue repaired.")
            return True
        print("[ERROR] Langgraph repair attempted but import still fails.")
        print(second.stderr.strip() or second.stdout.strip())
        return False

    print("[ERROR] Agent import failed before startup.")
    print(first.stderr.strip() or first.stdout.strip())
    return False


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    python_bin = _pick_python_executable()
    host = env.get("FIN_ALPHA_HOST", "127.0.0.1")
    port = env.get("FIN_ALPHA_PORT", "8000")
    backend_base_url = f"http://{host}:{port}"
    backend_health_url = f"{backend_base_url}/api/health"
    env.setdefault("BACKEND_URL", backend_base_url)

    backend_cmd = [
        python_bin,
        "-m",
        "uvicorn",
        "backend.app:app",
        "--host",
        host,
        "--port",
        port,
    ]
    agent_cmd = [python_bin, "agents/run.py"]

    backend_proc: subprocess.Popen | None = None
    started_backend = False

    def _handle_signal(signum, frame):
        _terminate_process(backend_proc)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        print(f"[START] Using Python: {python_bin}")
        if not _ensure_langgraph_compat(python_bin, env):
            return 1
        if _wait_for_backend(backend_health_url, timeout_sec=2):
            print(f"[START] Reusing already-running backend on {backend_base_url}.")
        else:
            print(f"[START] Launching backend on {backend_base_url} ...")
            backend_proc = subprocess.Popen(backend_cmd, cwd=ROOT, env=env)
            started_backend = True

            if not _wait_for_backend(backend_health_url, timeout_sec=25):
                print("[ERROR] Backend failed to become healthy in time.")
                _terminate_process(backend_proc)
                return 1

        print("[START] Backend is healthy.")
        print("[START] Launching agent CLI...")
        agent_rc = subprocess.call(agent_cmd, cwd=ROOT, env=env)
        print(f"[EXIT] Agent exited with code {agent_rc}.")
        return agent_rc
    finally:
        if started_backend:
            _terminate_process(backend_proc)
            print("[STOP] Backend stopped.")
        else:
            print("[STOP] Backend left running.")


if __name__ == "__main__":
    raise SystemExit(main())
