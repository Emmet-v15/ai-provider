"""Find, start and stop the ai-provider server process.

The server is identified by whatever owns the listening socket on its port, so an
instance started by hand (`uv run server.py` in a console) is seen and can be
stopped just like one started here. Instances started here run the venv's python
with no console window, logging to `logs/server.log`.

State that has to outlive this process (the pid we launched, whether the server
*should* be running) lives in %LOCALAPPDATA%\\ai-provider. The tray's watchdog
reads `want` to decide whether a dead server is a crash or a deliberate stop.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import psutil


def _find_home() -> Path:
    env = os.getenv("AI_PROVIDER_HOME")
    if env:
        return Path(env)
    # Editable install: ctl/ai_provider_ctl/core.py -> repo root
    return Path(__file__).resolve().parents[2]


HOME = _find_home()
PORT = int(os.getenv("AI_PROVIDER_PORT", "8765"))
BASE_URL = f"http://127.0.0.1:{PORT}"
LOG_DIR = HOME / "logs"
LOG_FILE = LOG_DIR / "server.log"
LOG_ROTATE_BYTES = 50 * 1024 * 1024
STATE_DIR = Path(os.getenv("LOCALAPPDATA", Path.home())) / "ai-provider"
STATE_FILE = STATE_DIR / "state.json"

# Process names that make up one server: `uv run` -> venv python shim -> CPython.
_SERVER_CHAIN = {"python.exe", "pythonw.exe", "uv.exe"}

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


# ── persisted state ─────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(**updates) -> dict:
    state = load_state()
    state.update(updates)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))
    return state


# ── discovery ───────────────────────────────────────────────────────────────

def listener_pid() -> int | None:
    for conn in psutil.net_connections(kind="tcp"):
        if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == PORT:
            return conn.pid
    return None


def _alive(pid: int | None) -> psutil.Process | None:
    if not pid:
        return None
    try:
        p = psutil.Process(pid)
        return p if p.is_running() and p.status() != psutil.STATUS_ZOMBIE else None
    except psutil.Error:
        return None


def _launched_proc() -> psutil.Process | None:
    """The process we started, if it is still that process (pids get reused)."""
    state = load_state()
    p = _alive(state.get("pid"))
    if p is None:
        return None
    try:
        if abs(p.create_time() - state.get("create_time", 0)) > 1:
            return None
    except psutil.Error:
        return None
    return p


def _root_of(proc: psutil.Process) -> psutil.Process:
    """Walk up the uv/python launcher chain to the outermost server process."""
    def in_chain(p: psutil.Process) -> bool:
        # The name alone isn't enough: `ai-provider start` is itself a python.exe
        # and is the parent of the server it just launched.
        return p.name().lower() in _SERVER_CHAIN and "server.py" in " ".join(p.cmdline())

    root = proc
    try:
        parent = proc.parent()
        while parent is not None and in_chain(parent):
            root = parent
            parent = parent.parent()
    except psutil.Error:
        pass
    return root


def fetch_json(path: str, timeout: float = 3.0):
    with urllib.request.urlopen(BASE_URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def status(with_health: bool = True) -> dict:
    """state is one of: running, starting, unresponsive, stopped."""
    launched = _launched_proc()
    lpid = listener_pid()
    info: dict = {"port": PORT, "home": str(HOME), "want": load_state().get("want")}

    if lpid is None:
        if launched is not None:
            info.update(state="starting", pid=launched.pid, managed=True,
                        started_at=launched.create_time())
        else:
            info.update(state="stopped")
        return info

    proc = _alive(lpid)
    root = _root_of(proc) if proc else None
    managed = launched is not None and root is not None and root.pid == launched.pid
    info.update(pid=lpid, root_pid=root.pid if root else lpid, managed=managed,
                started_at=root.create_time() if root else None)
    if not managed and root is not None:
        try:
            info["cmdline"] = " ".join(root.cmdline())
        except psutil.Error:
            pass

    if not with_health:
        info["state"] = "running"
        return info
    try:
        info["health"] = fetch_json("/health")
        info["models"] = fetch_json("/models").get("models", [])
        info["state"] = "running"
    except Exception as e:
        info["state"] = "unresponsive"
        info["error"] = str(e)
    return info


# ── control ─────────────────────────────────────────────────────────────────

def _venv_python() -> Path:
    py = HOME / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        raise RuntimeError(f"no venv interpreter at {py}; run `uv sync` or create .venv first")
    return py


def _rotate_log() -> None:
    try:
        if LOG_FILE.stat().st_size > LOG_ROTATE_BYTES:
            old = LOG_FILE.with_suffix(".log.1")
            old.unlink(missing_ok=True)
            LOG_FILE.rename(old)
    except FileNotFoundError:
        pass


def start(wait: float = 120.0) -> dict:
    """Start the server in the background. No-op if one is already up."""
    st = status(with_health=False)
    if st["state"] != "stopped":
        save_state(want="running")
        return st

    LOG_DIR.mkdir(exist_ok=True)
    _rotate_log()
    log = open(LOG_FILE, "ab")
    log.write(f"\n===== ai-provider start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n".encode())
    log.flush()

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    cmd = [str(_venv_python()), "server.py"]
    flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    kwargs = dict(cwd=HOME, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        # Leave whatever job we're in (a terminal, the tray), so closing it doesn't
        # take the server down with it.
        proc = subprocess.Popen(cmd, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, **kwargs)
    except OSError:
        proc = subprocess.Popen(cmd, creationflags=flags, **kwargs)
    finally:
        log.close()

    save_state(want="running", pid=proc.pid,
               create_time=psutil.Process(proc.pid).create_time(),
               last_start=time.time())

    deadline = time.time() + wait
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see {LOG_FILE}")
        if listener_pid() is not None:
            try:
                fetch_json("/health", timeout=2)
                break
            except Exception:
                pass
        time.sleep(0.5)
    return status()


def stop(timeout: float = 15.0) -> dict | None:
    """Stop the server (managed or not) and everything it spawned.

    Returns the status it had before stopping, or None if nothing was running.
    Children (llama-server, TTS worker) are killed explicitly as well as by the
    server's own kill-on-close Job Object.
    """
    save_state(want="stopped")
    st = status(with_health=False)
    if st["state"] == "stopped":
        return None
    root = _alive(st.get("root_pid") or st.get("pid"))
    if root is None:
        return st

    procs = [root]
    try:
        procs += root.children(recursive=True)
    except psutil.Error:
        pass
    for p in reversed(procs):  # leaves first
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=timeout)

    deadline = time.time() + timeout
    while listener_pid() is not None and time.time() < deadline:
        time.sleep(0.25)
    return st


def restart() -> dict:
    stop()
    return start()


# ── formatting helpers shared by CLI and tray ───────────────────────────────

def loaded_models(st: dict) -> list[str]:
    return [m["name"] for m in st.get("models", []) if m.get("loaded")]


def uptime(st: dict) -> str:
    t = st.get("started_at")
    if not t:
        return "-"
    s = int(time.time() - t)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    return f"{m}m" if m else f"{s}s"
