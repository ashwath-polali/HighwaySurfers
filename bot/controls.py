"""Key injection + global hotkeys.

Keys go out via pydirectinput (scancode SendInput, which the game accepts).
Everything is non-blocking: the planner asks for key states / timed taps and
`Controls.update()` is pumped every frame to release expired taps.

Global hotkeys (work even while the game has focus, via low-level hook):
  F8  toggle autopilot on/off
  F9  PANIC: release all keys and quit
"""
import time
import threading

import pydirectinput

pydirectinput.PAUSE = 0.0
pydirectinput.FAILSAFE = False

GAS = "w"
BRAKE = "s"
LEFT = "a"
RIGHT = "d"
ALL_KEYS = (GAS, BRAKE, LEFT, RIGHT)


class Controls:
    def __init__(self):
        self._down = set()
        self._tap_until = {}  # key -> perf_counter deadline
        self._lock = threading.Lock()

    # -- low level ---------------------------------------------------------
    def _key_down(self, key: str) -> None:
        if key not in self._down:
            pydirectinput.keyDown(key)
            self._down.add(key)

    def _key_up(self, key: str) -> None:
        if key in self._down:
            pydirectinput.keyUp(key)
            self._down.discard(key)
        self._tap_until.pop(key, None)

    # -- public API --------------------------------------------------------
    def set_key(self, key: str, held: bool) -> None:
        """Hold or release a key persistently (cancels any pending tap timer)."""
        with self._lock:
            self._tap_until.pop(key, None)
            if held:
                self._key_down(key)
            else:
                self._key_up(key)

    def tap(self, key: str, ms: float) -> None:
        """Press key now, auto-release after ms (extended if called again)."""
        with self._lock:
            self._key_down(key)
            self._tap_until[key] = time.perf_counter() + ms / 1000.0

    def update(self) -> None:
        """Pump once per frame: release expired taps."""
        now = time.perf_counter()
        with self._lock:
            expired = [k for k, t in self._tap_until.items() if now >= t]
            for k in expired:
                self._key_up(k)

    def release_all(self) -> None:
        with self._lock:
            for k in list(self._down):
                self._key_up(k)

    def held(self, key: str) -> bool:
        return key in self._down

    def steer_state(self) -> str:
        if LEFT in self._down:
            return "L"
        if RIGHT in self._down:
            return "R"
        return "-"


class Hotkeys:
    """Global F8 (toggle) / F9 (panic quit) listener via pynput low-level hook."""

    def __init__(self):
        self.autopilot = False
        self.quit = False
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.f8:
                self.autopilot = not self.autopilot
                print(f"[hotkeys] autopilot {'ON' if self.autopilot else 'OFF'}")
            elif key == keyboard.Key.f9:
                print("[hotkeys] PANIC quit")
                self.quit = True

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
