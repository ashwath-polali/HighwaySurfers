"""Screen capture of the game client area.

Primary backend: dxcam (DXGI desktop duplication, fast, Windows only).
Fallback: mss (slower but always works).

The game window is located by title via pygetwindow, and the *client area*
(no title bar or borders) is computed with Win32 calls so HUD fractions in
config line up with what the game actually renders.
"""
import ctypes
import ctypes.wintypes as wintypes
import time

import numpy as np


def _make_dpi_aware() -> None:
    """Without this, coordinates are lied to on scaled displays (common on laptops)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def _find_window(window_title: str):
    """Pick the best matching window. getWindowsWithTitle is a case-insensitive
    substring match, so several windows (a browser tab, a chat) can match. Take
    the largest visible one, which is the actual game client."""
    import pygetwindow as gw

    wins = [w for w in gw.getWindowsWithTitle(window_title) if w.title.strip()]
    wins = [w for w in wins if w.visible and w.width > 200 and w.height > 200]
    if not wins:
        raise RuntimeError(
            f"No usable window with title containing '{window_title}' found. "
            "Is the game running and un-minimized?"
        )
    return max(wins, key=lambda w: w.width * w.height)


def find_game_client_region(window_title: str) -> tuple:
    """Returns (left, top, right, bottom) of the window's client area in screen px."""
    win = _find_window(window_title)
    hwnd = win._hWnd

    rect = wintypes.RECT()
    if not ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("GetClientRect failed")
    pt = wintypes.POINT(0, 0)
    if not ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        raise RuntimeError("ClientToScreen failed")

    left, top = pt.x, pt.y
    right, bottom = left + rect.right, top + rect.bottom
    if rect.right < 200 or rect.bottom < 200:
        raise RuntimeError(
            f"game client area looks too small ({rect.right}x{rect.bottom}). "
            "Is the window minimized?"
        )
    return (left, top, right, bottom)


def get_game_hwnd(window_title: str):
    """Win32 handle of the game window, for focus checks and key routing."""
    return _find_window(window_title)._hWnd


def is_foreground(hwnd) -> bool:
    """True when the game window currently has keyboard focus.

    This is read-only: it never touches window state. We deliberately do NOT
    force the window to the foreground. A fullscreen game minimizes itself when
    another process fights it for focus, so the bot stays passive: it drives
    only while the player already has the game focused, and otherwise waits.
    """
    u = ctypes.windll.user32
    u.GetForegroundWindow.restype = wintypes.HWND  # full handle, not a truncated int
    fg = u.GetForegroundWindow()
    return fg is not None and int(fg) == int(hwnd)


def activate_game(window_title: str) -> None:
    """No-op kept for API compatibility. The bot never forces window focus (that
    minimizes a fullscreen game); the player keeps the game focused instead."""
    return


class Capture:
    """Unified capture interface: .read() -> (bgr_frame, timestamp) or (None, t)."""

    def __init__(self, region: tuple, target_fps: int):
        self.region = region
        self.target_fps = target_fps
        self.backend = None
        self._camera = None
        self._sct = None
        self._mss_monitor = None
        self._last_mss_t = 0.0
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import dxcam

            self._camera = dxcam.create(output_color="BGR")
            if self._camera is None:
                raise RuntimeError("dxcam.create returned None")
            self._camera.start(region=self.region, target_fps=self.target_fps)
            # Fail fast if the region is invalid.
            frame = self._camera.get_latest_frame()
            if frame is None:
                raise RuntimeError("dxcam produced no frame")
            self.backend = "dxcam"
            print(f"[capture] dxcam @ {self.target_fps}fps region={self.region}")
            return
        except Exception as e:  # noqa: BLE001 - any dxcam failure falls back to mss
            print(f"[capture] dxcam unavailable ({e}); falling back to mss. "
                  "(dxcam only captures the primary monitor. If the game is on "
                  "a second screen, move it to the primary one for the fast path.)")
            self._camera = None

        import mss

        self._sct = mss.mss()
        l, t, r, b = self.region
        self._mss_monitor = {"left": l, "top": t, "width": r - l, "height": b - t}
        self.backend = "mss"
        print(f"[capture] mss region={self.region}")

    def read(self):
        """Blocks until a new frame (dxcam) or paces itself (mss)."""
        if self.backend == "dxcam":
            frame = self._camera.get_latest_frame()  # blocks on new frame
            return frame, time.perf_counter()
        # mss path: throttle to target fps
        min_dt = 1.0 / self.target_fps
        now = time.perf_counter()
        wait = self._last_mss_t + min_dt - now
        if wait > 0:
            time.sleep(wait)
        shot = self._sct.grab(self._mss_monitor)
        self._last_mss_t = time.perf_counter()
        frame = np.asarray(shot)[:, :, :3]  # BGRA -> BGR
        return np.ascontiguousarray(frame), self._last_mss_t

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
        if self._sct is not None:
            self._sct.close()


def open_capture(cfg) -> Capture:
    _make_dpi_aware()
    region = find_game_client_region(cfg.window_title)
    return Capture(region, cfg.target_fps)
