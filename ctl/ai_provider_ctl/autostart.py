"""Launch the tray at Windows sign-in via HKCU\\...\\Run (per-user, no admin)."""

from __future__ import annotations

import shutil
import sys
import winreg
from pathlib import Path

_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_NAME = "ai-provider"


def tray_command(*args: str) -> str:
    exe = shutil.which("ai-provider-tray")
    if exe:
        return f'"{exe}"' + "".join(f" {a}" for a in args)
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    return f'"{pythonw}" -m ai_provider_ctl.tray' + "".join(f" {a}" for a in args)


def is_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _KEY) as k:
            winreg.QueryValueEx(k, _NAME)
            return True
    except FileNotFoundError:
        return False


def enable() -> str:
    cmd = tray_command("--login")
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _KEY, 0, winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, _NAME, 0, winreg.REG_SZ, cmd)
    return cmd


def disable() -> None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _KEY, 0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, _NAME)
    except FileNotFoundError:
        pass
