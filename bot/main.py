"""Entry point.

    python run.py view        live perception overlay, no keys sent (tune here first)
    python run.py shot        save one annotated frame pair to disk and exit
    python run.py probe       measure input dead time  -> calibration.json
    python run.py calibrate   measure steering response -> calibration.json
    python run.py drive       autopilot (F8 toggles, F9 panic-quits)
"""
import argparse
import os
import time
import traceback

import cv2

from .config import Config, load_calibration
from .capture import (open_capture, activate_game, get_game_hwnd,
                      is_foreground, focus_hwnd)
from .vision import Vision
from .tracker import Tracker
from .planner import Planner
from .controls import Controls, Hotkeys, GAS, BRAKE, LEFT, RIGHT
from .telemetry import Telemetry


def _first_frame(capture):
    for _ in range(120):
        frame, _t = capture.read()
        if frame is not None:
            return frame
    raise RuntimeError("no frames from capture")


def _log_crash(cfg, where: str, extra: str = "") -> None:
    """Append a full traceback to crash.log so an unattended run is diagnosable."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "crash.log")
    tb = traceback.format_exc()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"=== crash in {where} @ frame ===\n{extra}\n{tb}\n")
    except OSError:
        pass
    print(f"[crash] {where}: {tb.strip().splitlines()[-1] if tb.strip() else 'unknown'}")


def _click_canvas(region) -> None:
    """Click the middle of the road to give the 3D view keyboard focus, so WASD
    drives the car instead of typing into the chat box."""
    import pydirectinput

    l, t, r, b = region
    x = int(l + (r - l) * 0.5)
    y = int(t + (b - t) * 0.45)
    pydirectinput.moveTo(x, y)
    pydirectinput.click()


def _place_debug_windows(names, region) -> None:
    """Park debug windows OUTSIDE the capture region. Both capture backends grab
    a rectangle of the desktop, so a window sitting over the game would be warped
    back into perception, a silent feedback loop. Move them to the right of the
    region (or below if there's no room) and warn if they might still overlap."""
    l, t, r, b = region
    x = r + 8          # just right of the captured area
    y = t
    for i, name in enumerate(names):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(name, x, y + i * 300)
    print(f"[debug] windows parked at x>={x}. Keep them off the game window. "
          "If one overlaps the captured area it will corrupt perception.")


def mode_view(cfg, save_one: bool = False) -> None:
    capture = open_capture(cfg)
    frame = _first_frame(capture)
    vision = Vision(cfg, frame.shape)
    tracker = Tracker(cfg)
    planner = Planner(cfg, load_calibration(cfg))
    print("[view] q in the overlay window quits. No keys are sent to the game.")
    if not save_one:
        _place_debug_windows(["frame", "bev"], capture.region)
    fps, prev_t = 0.0, None
    try:
        while True:
            frame, t = capture.read()
            if frame is None:
                continue
            if prev_t is not None:
                inst = 1.0 / max(t - prev_t, 1e-6)
                fps = 0.9 * fps + 0.1 * inst if fps else inst
            prev_t = t
            per = vision.process(frame, want_masks=True)
            tracks = tracker.update(per.blobs)
            plan = planner.plan(per, tracks, fps or cfg.target_fps)
            vis, bev_vis = vision.draw_debug(frame, per, plan)
            cv2.putText(vis, f"{fps:5.1f} fps  blobs={len(per.blobs)} tracks={len(tracks)}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if save_one:
                cv2.imwrite("debug_frame.jpg", vis)
                cv2.imwrite("debug_bev.jpg", bev_vis)
                cv2.imwrite("debug_obstacle_mask.jpg", per.masks["obstacle"])
                print("[shot] wrote debug_frame.jpg / debug_bev.jpg / debug_obstacle_mask.jpg")
                return
            cv2.imshow("frame", vis)
            cv2.imshow("bev", bev_vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
    finally:
        capture.close()
        cv2.destroyAllWindows()


def mode_probe(cfg) -> None:
    from .calibration import run_probe
    capture = open_capture(cfg)
    frame = _first_frame(capture)
    vision = Vision(cfg, frame.shape)
    activate_game(cfg.window_title)
    try:
        run_probe(cfg, capture, vision)
    finally:
        capture.close()


def mode_calibrate(cfg) -> None:
    from .calibration import run_steer_calibration
    capture = open_capture(cfg)
    frame = _first_frame(capture)
    vision = Vision(cfg, frame.shape)
    activate_game(cfg.window_title)
    try:
        run_steer_calibration(cfg, capture, vision, load_calibration(cfg))
    finally:
        capture.close()


def mode_drive(cfg, overlay: bool, autostart: bool = True) -> None:
    from .autoplay import UINavigator

    capture = open_capture(cfg)
    frame = _first_frame(capture)
    vision = Vision(cfg, frame.shape)
    tracker = Tracker(cfg)
    cal = load_calibration(cfg)
    planner = Planner(cfg, cal)
    controls = Controls()
    hotkeys = Hotkeys()
    navigator = UINavigator(capture.region)
    telemetry = Telemetry(cfg)
    hotkeys.start()
    hotkeys.autopilot = autostart
    try:
        hwnd = get_game_hwnd(cfg.window_title)
    except Exception:
        hwnd = None

    print(f"[drive] latency={cal['latency_ms']:.0f}ms "
          f"vmax={cal['steer_vmax_px_s']:.0f}px/s coast={cal['steer_coast_s']:.2f}s")
    if "latency_samples_ms" not in cal:
        print("[drive] WARNING: no probe data found, using default latency. "
              "Run `python run.py probe` for real numbers.")
    print(f"[drive] autopilot starts {'ON' if autostart else 'OFF'}. "
          "F8 = autopilot on/off, F9 = panic quit.")
    activate_game(cfg.window_title)
    if overlay:
        _place_debug_windows(["bot"], capture.region)
    # Grace period so the game settles into the foreground before keys start.
    for i in (3, 2, 1):
        print(f"[drive] taking control in {i}...")
        time.sleep(1.0)

    fps, prev_t = 0.0, None
    last_brake_t = 0.0
    last_focus_t = 0.0
    crash_streak = 0
    was_ui = True   # force a canvas-focus click when the first run begins
    try:
        while not hotkeys.quit:
            frame, t = capture.read()
            if frame is None:
                controls.update()
                continue
            if prev_t is not None:
                inst = 1.0 / max(t - prev_t, 1e-6)
                fps = 0.9 * fps + 0.1 * inst if fps else inst
            prev_t = t

            # Death screen / main menu? Click back into the run, skip driving.
            if hotkeys.autopilot and navigator.step(frame, t, controls):
                was_ui = True
                continue
            # Just came out of a menu/death screen into a live run: focus the
            # 3D view so keys drive the car instead of landing in chat.
            if hotkeys.autopilot and was_ui:
                was_ui = False
                if hwnd:
                    focus_hwnd(hwnd)
                _click_canvas(capture.region)
                last_focus_t = t

            try:
                per = vision.process(frame, want_masks=overlay)
                tracks = tracker.update(per.blobs)
                plan = planner.plan(per, tracks, fps or cfg.target_fps)

                game_focused = hwnd is None or is_foreground(hwnd)
                if hotkeys.autopilot and not game_focused:
                    # Game lost focus: never send keys (they would leak into
                    # chat or another app). Reclaim focus at most once a second.
                    controls.release_all()
                    if t - last_focus_t > 1.0:
                        focus_hwnd(hwnd)
                        last_focus_t = t
                elif hotkeys.autopilot:
                    # throttle
                    if plan.brake_tap_ms > 0 and t - last_brake_t > 0.35:
                        controls.set_key(GAS, False)
                        controls.tap(BRAKE, plan.brake_tap_ms)
                        last_brake_t = t
                    else:
                        controls.set_key(GAS, plan.gas)
                    # steering: exactly one of L/R/none
                    if plan.steer_key == LEFT:
                        controls.set_key(RIGHT, False)
                        controls.set_key(LEFT, True)
                    elif plan.steer_key == RIGHT:
                        controls.set_key(LEFT, False)
                        controls.set_key(RIGHT, True)
                    else:
                        controls.set_key(LEFT, False)
                        controls.set_key(RIGHT, False)
                else:
                    controls.release_all()
                controls.update()

                keys = ("W" if controls.held(GAS) else "-") + \
                       ("S" if controls.held(BRAKE) else "-") + controls.steer_state()
                telemetry.log(per, plan, fps, hotkeys.autopilot, keys)

                if overlay:
                    vis, bev_vis = vision.draw_debug(frame, per, plan)
                    telemetry.maybe_dump_frame(vis, bev_vis)
                    cv2.imshow("bot", bev_vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                crash_streak = 0
            except Exception:
                # A single bad frame shouldn't kill an unattended run: log the
                # full traceback, drop the keys, and keep going. Bail only if
                # every frame is failing (a real, non-transient bug).
                crash_streak += 1
                _log_crash(cfg, "drive-loop", f"fps={fps:.1f} streak={crash_streak}")
                controls.release_all()
                if crash_streak >= 30:
                    print("[drive] too many consecutive errors; stopping. See crash.log.")
                    break
    finally:
        controls.release_all()
        hotkeys.stop()
        telemetry.close()
        capture.close()
        cv2.destroyAllWindows()
        print("[drive] stopped, all keys released.")


def main() -> None:
    parser = argparse.ArgumentParser(description="highway driving autopilot")
    parser.add_argument("mode", choices=["view", "shot", "probe", "calibrate", "drive"])
    parser.add_argument("--overlay", action="store_true",
                        help="drive mode: show live BEV window + dump debug frames")
    parser.add_argument("--debug-frames", type=int, default=0,
                        help="dump annotated frames every N frames (needs --overlay)")
    parser.add_argument("--no-autostart", action="store_true",
                        help="drive mode: start with autopilot OFF (press F8 to engage)")
    args = parser.parse_args()

    cfg = Config()
    if args.debug_frames:
        cfg.debug_frame_every = args.debug_frames

    if args.mode == "view":
        mode_view(cfg)
    elif args.mode == "shot":
        mode_view(cfg, save_one=True)
    elif args.mode == "probe":
        mode_probe(cfg)
    elif args.mode == "calibrate":
        mode_calibrate(cfg)
    elif args.mode == "drive":
        mode_drive(cfg, overlay=args.overlay, autostart=not args.no_autostart)


if __name__ == "__main__":
    main()
