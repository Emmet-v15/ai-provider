"""`ai-provider` — control the ai-provider server from any terminal."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import webbrowser

import psutil

from . import autostart, core


def _tailscale_ip() -> str | None:
    for name, addrs in psutil.net_if_addrs().items():
        if "tailscale" in name.lower():
            for a in addrs:
                if a.family == 2:  # AF_INET
                    return a.address
    return None


def cmd_status(args) -> int:
    st = core.status()
    if args.json:
        print(json.dumps(st, indent=2))
        return 0 if st["state"] == "running" else 3

    print(f"ai-provider: {st['state']}")
    if st["state"] == "stopped":
        print(f"  autostart: {'on' if autostart.is_enabled() else 'off'}")
        return 3
    how = "background (ai-provider start)" if st.get("managed") else "by hand: " + st.get("cmdline", "?")
    print(f"  pid:       {st.get('root_pid', st.get('pid'))}   up {core.uptime(st)}")
    print(f"  started:   {how}")
    urls = [core.BASE_URL]
    ts = _tailscale_ip()
    if ts:
        urls.append(f"http://{ts}:{core.PORT}")
    print(f"  url:       {'  '.join(urls)}")
    h = st.get("health")
    if h:
        v, g = h.get("vram", {}), h.get("gpu", {})
        print(f"  loaded:    {', '.join(core.loaded_models(st)) or 'nothing'}"
              f"   ({v.get('loaded_gb', 0)} / {v.get('max_vram_gb', '?')} GB budget)")
        if g:
            print(f"  gpu:       {g.get('name', '?')}  {g.get('mem_used_gb')}/{g.get('mem_total_gb')} GB"
                  f"  {g.get('gpu_util_pct')}%  {g.get('temp_c')}C")
        busy = {k: q for k, q in h.get("queues", {}).items() if q.get("running") or q.get("queued")}
        for k, q in busy.items():
            print(f"  queue:     {k} running {q['running']}, queued {q['queued']}")
    elif st.get("error"):
        print(f"  error:     {st['error']}")
    print(f"  autostart: {'on' if autostart.is_enabled() else 'off'}")
    print(f"  log:       {core.LOG_FILE}")
    return 0 if st["state"] == "running" else 1


def cmd_start(args) -> int:
    before = core.status(with_health=False)
    if before["state"] != "stopped":
        print(f"already {before['state']} (pid {before.get('root_pid', before.get('pid'))})")
        return 0
    print("starting ai-provider...", flush=True)
    try:
        st = core.start(wait=0 if args.no_wait else 120)
    except RuntimeError as e:
        print(f"failed: {e}", file=sys.stderr)
        return 1
    print(f"{st['state']} (pid {st.get('root_pid', st.get('pid'))}) on {core.BASE_URL}")
    return 0 if st["state"] in ("running", "starting") else 1


def cmd_stop(args) -> int:
    st = core.stop()
    if st is None:
        print("not running")
        return 0
    who = "" if st.get("managed") else " (it had been started by hand)"
    print(f"stopped pid {st.get('root_pid', st.get('pid'))}{who}")
    return 0


def cmd_restart(args) -> int:
    cmd_stop(args)
    args.no_wait = False
    return cmd_start(args)


def cmd_logs(args) -> int:
    path = core.LOG_FILE
    if not path.exists():
        print(f"no log yet at {path}")
        return 1
    with open(path, "rb") as f:
        lines = f.read().decode("utf-8", "replace").splitlines()
        for line in lines[-args.lines:]:
            print(line)
        if not args.follow:
            return 0
        try:
            while True:
                chunk = f.read()
                if chunk:
                    sys.stdout.write(chunk.decode("utf-8", "replace"))
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            return 0


def _launch_tray() -> None:
    cmd = autostart.tray_command()
    subprocess.Popen(cmd, creationflags=0x00000008 | 0x00000200, close_fds=True)  # DETACHED, NEW_GROUP


def cmd_tray(args) -> int:
    _launch_tray()
    print("tray icon launched (it exits immediately if one is already running)")
    return 0


def cmd_autostart(args) -> int:
    if args.action == "on":
        print(f"autostart on: tray runs at sign-in and starts the server\n  {autostart.enable()}")
        _launch_tray()
    elif args.action == "off":
        autostart.disable()
        print("autostart off (a running server and tray are left alone)")
    else:
        print("on" if autostart.is_enabled() else "off")
    return 0


def cmd_open(args) -> int:
    webbrowser.open(core.BASE_URL + "/documentation")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ai-provider", description=f"Control the ai-provider server ({core.HOME})")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("status", help="show whether it is running, what is loaded, GPU and queues")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("start", help="start in the background (no console window)")
    s.add_argument("--no-wait", action="store_true", help="don't wait for /health")
    s.set_defaults(fn=cmd_start)

    sub.add_parser("stop", help="stop the server and its model processes").set_defaults(fn=cmd_stop)
    sub.add_parser("restart", help="stop, then start").set_defaults(fn=cmd_restart)

    s = sub.add_parser("logs", help="print the server log")
    s.add_argument("-n", "--lines", type=int, default=50)
    s.add_argument("-f", "--follow", action="store_true")
    s.set_defaults(fn=cmd_logs)

    sub.add_parser("tray", help="launch the tray icon").set_defaults(fn=cmd_tray)

    s = sub.add_parser("autostart", help="run the tray (and server) at Windows sign-in")
    s.add_argument("action", nargs="?", choices=["on", "off", "status"], default="status")
    s.set_defaults(fn=cmd_autostart)

    sub.add_parser("open", help="open the docs in a browser").set_defaults(fn=cmd_open)

    args = p.parse_args(argv)
    if not args.cmd:
        args = p.parse_args(["status"])
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
