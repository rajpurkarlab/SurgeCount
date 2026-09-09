"""Minimal pynput.keyboard-compatible listener backed by evdev.

pynput's Linux backend is X11-only and silently produces zero events on Wayland
sessions. This module mimics the small subset of pynput.keyboard that lerobot's
recording loop uses (`Listener` + `Key.enter/.esc/.backspace/.right/.left`),
reading from /dev/input/event* directly so it works regardless of the display
server.

Requires the user to be in the `input` group (so /dev/input/event* is readable
without root).
"""

from __future__ import annotations

import logging
import select
import threading
from typing import Callable

import evdev


class _SpecialKey:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Key.{self.name}"


class Key:
    enter = _SpecialKey("enter")
    esc = _SpecialKey("esc")
    backspace = _SpecialKey("backspace")
    right = _SpecialKey("right")
    left = _SpecialKey("left")
    space = _SpecialKey("space")


class _CharKey:
    __slots__ = ("char",)

    def __init__(self, char: str) -> None:
        self.char = char

    def __repr__(self) -> str:
        return f"'{self.char}'"


_SPECIAL_KEYS = {
    evdev.ecodes.KEY_ENTER: Key.enter,
    evdev.ecodes.KEY_KPENTER: Key.enter,
    evdev.ecodes.KEY_ESC: Key.esc,
    evdev.ecodes.KEY_BACKSPACE: Key.backspace,
    evdev.ecodes.KEY_LEFT: Key.left,
    evdev.ecodes.KEY_RIGHT: Key.right,
    evdev.ecodes.KEY_SPACE: Key.space,
}


def _build_char_map() -> dict[int, str]:
    m: dict[int, str] = {}
    for c in "abcdefghijklmnopqrstuvwxyz":
        code = getattr(evdev.ecodes, f"KEY_{c.upper()}", None)
        if code is not None:
            m[code] = c
    for d in "0123456789":
        code = getattr(evdev.ecodes, f"KEY_{d}", None)
        if code is not None:
            m[code] = d
        kp_code = getattr(evdev.ecodes, f"KEY_KP{d}", None)
        if kp_code is not None:
            m[kp_code] = d
    return m


_CHAR_KEYS = _build_char_map()


def _find_keyboards() -> list[evdev.InputDevice]:
    devices: list[evdev.InputDevice] = []
    for path in evdev.list_devices():
        try:
            d = evdev.InputDevice(path)
        except (PermissionError, OSError):
            continue
        caps = d.capabilities().get(evdev.ecodes.EV_KEY, [])
        if evdev.ecodes.KEY_A in caps and evdev.ecodes.KEY_ENTER in caps:
            devices.append(d)
        else:
            d.close()
    return devices


class Listener:
    """Drop-in replacement for pynput.keyboard.Listener using evdev."""

    def __init__(
        self,
        on_press: Callable | None = None,
        on_release: Callable | None = None,
    ) -> None:
        self._on_press = on_press
        self._on_release = on_release
        self._devices: list[evdev.InputDevice] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self._devices = _find_keyboards()
        except Exception as e:
            logging.warning("evdev: failed to enumerate input devices: %s", e)
            self._devices = []

        if not self._devices:
            logging.warning(
                "evdev keyboard listener found no readable devices. "
                "Add your user to the 'input' group "
                "(sudo usermod -aG input $USER) and re-login."
            )
            return

        logging.info(
            "evdev keyboard listener attached to: %s",
            ", ".join(f"{d.name} ({d.path})" for d in self._devices),
        )
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="evdev-keyboard"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for d in self._devices:
            try:
                d.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        fd_to_dev = {d.fd: d for d in self._devices}
        def _trace(msg: str) -> None:
            try:
                import time as _t
                with open("/tmp/lerobot_keylog", "a") as _kf:
                    _kf.write(f"{_t.strftime('%H:%M:%S')} [_run] {msg}\n")
            except Exception:
                pass
        _trace(f"thread start; watching {len(fd_to_dev)} device(s): "
               + ", ".join(f"{d.path}={d.name}" for d in fd_to_dev.values()))
        while not self._stop_event.is_set() and fd_to_dev:
            try:
                r, _, _ = select.select(list(fd_to_dev.keys()), [], [], 0.5)
            except (OSError, ValueError) as e:
                _trace(f"select failed: {e!r}; thread exiting")
                return
            for fd in r:
                dev = fd_to_dev.get(fd)
                if dev is None:
                    continue
                try:
                    for event in dev.read():
                        if event.type != evdev.ecodes.EV_KEY:
                            continue
                        if event.value != 1:
                            continue
                        _trace(f"key down on {dev.path}: code={event.code}")
                        self._dispatch_press(event.code)
                except OSError as e:
                    _trace(f"OSError on {dev.path}: {e!r}; dropping device")
                    fd_to_dev.pop(fd, None)
        _trace(f"thread exit; devices left: {len(fd_to_dev)}")

    def _dispatch_press(self, code: int) -> None:
        if self._on_press is None:
            return
        special = _SPECIAL_KEYS.get(code)
        if special is not None:
            try:
                self._on_press(special)
            except Exception as e:
                logging.exception("on_press handler raised: %s", e)
            return
        char = _CHAR_KEYS.get(code)
        if char is not None:
            try:
                self._on_press(_CharKey(char))
            except Exception as e:
                logging.exception("on_press handler raised: %s", e)


class _KeyboardNamespace:
    Listener = Listener
    Key = Key


keyboard = _KeyboardNamespace()
