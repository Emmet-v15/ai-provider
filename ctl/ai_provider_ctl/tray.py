"""System-tray icon for ai-provider: status at a glance, start/stop, and a watchdog.

The watchdog restarts the server if it dies while it is meant to be running
(`want == "running"` in the shared state file). `ai-provider stop`, or Stop here,
sets `want = "stopped"`, so a deliberate stop is never undone. Launched with
`--login` (the autostart entry), it starts the server regardless.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
import webbrowser

import pystray
from PIL import Image, ImageDraw

from . import autostart, core

POLL_S = 5
RESTART_BACKOFF_S = 60

_COLORS = {
    "running": (46, 160, 67),
    "starting": (210, 153, 34),
    "unresponsive": (210, 153, 34),
    "stopped": (110, 118, 129),
    "error": (207, 34, 46),
}


def _icon_image(state: str) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((2, 2, 62, 62), radius=14, fill=(30, 34, 42, 255))
    d.text((10, 8), "AI", fill=(235, 238, 242, 255), font_size=30)
    d.ellipse((34, 34, 60, 60), fill=_COLORS.get(state, _COLORS["error"]) + (255,),
              outline=(30, 34, 42, 255), width=3)
    return img


class Tray:
    def __init__(self, login: bool):
        self.st: dict = {"state": "stopped"}
        self.busy = False  # a start/stop is in progress
        self.error: str | None = None
        self.login = login
        self.quitting = False
        self.icon = pystray.Icon("ai-provider", _icon_image("stopped"), "ai-provider", menu=self._menu())

    # ── menu ────────────────────────────────────────────────────────────────
    def _menu(self) -> pystray.Menu:
        item = pystray.MenuItem
        running = lambda _: self.st["state"] != "stopped"  # noqa: E731
        return pystray.Menu(
            item(lambda _: self._headline(), None, enabled=False),
            item(lambda _: self._detail(), None, enabled=False, visible=lambda _: bool(self._detail())),
            pystray.Menu.SEPARATOR,
            item("Start", self._do(core.start), enabled=lambda _: not running(_) and not self.busy),
            item("Stop", self._do(core.stop), enabled=lambda _: running(_) and not self.busy),
            item("Restart", self._do(core.restart), enabled=lambda _: running(_) and not self.busy),
            pystray.Menu.SEPARATOR,
            item("Open docs", lambda: webbrowser.open(core.BASE_URL + "/documentation"), default=True),
            item("Open log", lambda: os.startfile(core.LOG_FILE) if core.LOG_FILE.exists() else None),
            item("Open folder", lambda: os.startfile(core.HOME)),
            item("Start at sign-in", self._toggle_autostart, checked=lambda _: autostart.is_enabled()),
            pystray.Menu.SEPARATOR,
            item("Quit tray (server keeps running)", self._quit),
        )

    def _headline(self) -> str:
        if self.busy:
            return "ai-provider: working..."
        s = self.st
        line = f"ai-provider: {s['state']}"
        if s["state"] != "stopped":
            line += f" · up {core.uptime(s)}"
        return line

    def _detail(self) -> str:
        if self.error:
            return f"error: {self.error}"[:120]
        h = self.st.get("health")
        if not h:
            return ""
        v = h.get("vram", {})
        names = ", ".join(core.loaded_models(self.st)) or "nothing loaded"
        return f"{names} · {v.get('loaded_gb', 0)}/{v.get('max_vram_gb', '?')} GB"

    def _do(self, fn):
        def run():
            self.busy = True
            self.error = None
            self._refresh()
            try:
                fn()
            except Exception as e:
                self.error = str(e)
                self.icon.notify(str(e), "ai-provider")
            finally:
                self.busy = False
                self._refresh()
        return lambda: threading.Thread(target=run, daemon=True).start()

    def _toggle_autostart(self):
        autostart.disable() if autostart.is_enabled() else autostart.enable()

    def _quit(self):
        self.quitting = True
        self.icon.stop()

    # ── polling + watchdog ──────────────────────────────────────────────────
    def _refresh(self):
        try:
            self.st = core.status()
        except Exception as e:
            self.st = {"state": "error", "error": str(e)}
        state = "starting" if self.busy else self.st["state"]
        self.icon.icon = _icon_image(state)
        title = f"ai-provider: {state}"
        if self._detail():
            title += f"\n{self._detail()}"
        self.icon.title = title[:127]
        self.icon.update_menu()

    def _watchdog(self):
        st = self.st
        if self.busy or st["state"] != "stopped":
            return
        saved = core.load_state()
        if saved.get("want") != "running":
            return
        if time.time() - saved.get("last_start", 0) < RESTART_BACKOFF_S:
            return
        self.icon.notify("ai-provider stopped unexpectedly; restarting it.", "ai-provider")
        self._do(core.start)()

    def _loop(self, icon):
        icon.visible = True
        if self.login:
            core.save_state(want="running")
        self._refresh()
        if self.st["state"] == "stopped" and core.load_state().get("want") == "running":
            self._do(core.start)()
        while not self.quitting:
            time.sleep(POLL_S)
            if not self.busy:
                self._refresh()
                self._watchdog()

    def run(self):
        self.icon.run(setup=self._loop)


def _single_instance() -> bool:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _single_instance.handle = k32.CreateMutexW(None, False, "Local\\ai-provider-tray")
    return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS


def main() -> None:
    if not _single_instance():
        return
    Tray(login="--login" in sys.argv).run()


if __name__ == "__main__":
    main()
