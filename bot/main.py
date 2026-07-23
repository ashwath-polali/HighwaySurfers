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
from .capture import open_capture, get_game_hwnd, is_foreground
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
    try:
        run_probe(cfg, capture, vision)
    finally:
        capture.close()


def mode_calibrate(cfg) -> None:
    from .calibration import run_steer_calibration
    capture = open_capture(cfg)
    frame = _first_frame(capture)
    vision = Vision(cfg, frame.shape)
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
    hotkeys = Hotkeys()
    navigator = UINavigator(capture.region)
    telemetry = Telemetry(cfg)
    try:
        hwnd = get_game_hwnd(cfg.window_title)
    except Exception:
        hwnd = None

    # The single source of truth for "may we send input right now": only when
    # the game window is the foreground window. Controls enforces this on every
    # key/click, and the loop below also skips whole frames when it's false, so
    # nothing can leak into another app.
    def focused() -> bool:
        return hwnd is not None and is_foreground(hwnd)
    controls = Controls(gate=focused)

    hotkeys.start()
    hotkeys.autopilot = autostart

    print(f"[drive] latency={cal['latency_ms']:.0f}ms "
          f"vmax={cal['steer_vmax_px_s']:.0f}px/s coast={cal['steer_coast_s']:.2f}s")
    if hwnd is None:
        print("[drive] WARNING: could not find the game window; input stays "
              "disabled until it is open. Nothing will be typed anywhere.")
    if "latency_samples_ms" not in cal:
        print("[drive] no probe data found, using default latency.")
    print(f"[drive] autopilot starts {'ON' if autostart else 'OFF'}. "
          "F8 = autopilot on/off, F9 = panic quit.")
    print("[drive] KEEP THE GAME WINDOW FOCUSED. The bot only acts while the "
          "game is the active window; click it now.")
    if overlay:
        _place_debug_windows(["bot"], capture.region)
    for i in (3, 2, 1):
        print(f"[drive] taking control in {i}...")
        time.sleep(1.0)

    l, tp, r, b = capture.region
    canvas_x, canvas_y = int(l + (r - l) * 0.5), int(tp + (b - tp) * 0.45)

    fps, prev_t = 0.0, None
    last_brake_t = last_steer_t = last_hint_t = 0.0
    crash_streak = 0
    was_ui = True   # click the road to grab canvas focus when a run begins
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

            # HARD GATE (passive): if autopilot is on but the game is not the
            # foreground window, do nothing at all — no keys, no clicks, and no
            # window manipulation (forcing focus minimizes a fullscreen game).
            # Just wait for the player to focus the game.
            if hotkeys.autopilot and not focused():
                controls.release_all()
                controls.update()
                if t - last_hint_t > 4.0:
                    print("[drive] waiting — click the game window to let the "
                          "bot drive (it will not touch anything else).")
                    last_hint_t = t
                was_ui = True
                continue

            # Death screen / main menu? Click back into the run (gated), skip driving.
            if hotkeys.autopilot and navigator.step(frame, t, controls):
                was_ui = True
                controls.update()
                continue
            # First live frame after a menu: click the road so keys drive the
            # car instead of landing in the chat box.
            if hotkeys.autopilot and was_ui:
                was_ui = False
                controls.click(canvas_x, canvas_y)

            try:
                per = vision.process(frame, want_masks=overlay)
                tracks = tracker.update(per.blobs)
                plan = planner.plan(per, tracks, fps or cfg.target_fps)

                if hotkeys.autopilot:
                    # throttle
                    if plan.brake_tap_ms > 0 and t - last_brake_t > 0.35:
                        controls.set_key(GAS, False)
                        controls.tap(BRAKE, plan.brake_tap_ms)
                        last_brake_t = t
                    else:
                        controls.set_key(GAS, plan.gas)
                    # steering: a short tap, then let the car settle before the
                    # next one (holding the key oversteers at high responsiveness)
                    if plan.steer_key and t - last_steer_t > cfg.steer_cooldown_s:
                        controls.tap(plan.steer_key, plan.steer_tap_ms)
                        last_steer_t = t
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
