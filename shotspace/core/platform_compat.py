"""What differs between Linux and Windows, in one place (standard library only: the GPU workers import it too).

* `link_file`: Windows does not let an ordinary user make symbolic links, so a frame is linked hard (same disk) or
  copied instead.  On Linux it stays a symbolic link, exactly as before.
* `pid_alive`: on Windows `os.kill(pid, 0)` does not ask whether a process lives - it ENDS it.  Never use it there.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def link_file(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    """`dst` shows the content of `src`: a symbolic link on Linux; on Windows a hard link, or a copy across disks."""
    if os.name != "nt":
        os.symlink(src, dst)
        return
    src_p = Path(src).resolve()
    try:
        os.link(src_p, dst)
    except OSError:
        shutil.copyfile(src_p, dst)


def pid_alive(pid: int) -> bool:
    """Is the process `pid` still running?"""
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE = 0x1000, 259
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return ctypes.get_last_error() == 5  # access denied: it exists
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def replace_file(src: str | os.PathLike, dst: str | os.PathLike, wait_s: float = 3.0) -> None:
    """`os.replace(src, dst)`.  On Windows a file that another request is reading at that moment cannot be replaced
    (sharing violation); those readers finish within moments, so it is tried again for up to `wait_s`."""
    if os.name != "nt":
        os.replace(src, dst)
        return
    import time

    deadline = time.monotonic() + wait_s
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
