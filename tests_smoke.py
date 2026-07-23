"""Headless smoke test: no game needed.

Renders synthetic frames (gray road, white dashed lanes, one red car ahead),
runs Vision -> Tracker -> Planner end to end, and asserts the pipeline sees
the road, the lanes, the car, and produces a sane plan.

    .venv\\Scripts\\python.exe tests_smoke.py
"""
import numpy as np
import cv2

from bot.config import Config, load_calibration
from bot.vision import Vision
from bot.tracker import Tracker
from bot.planner import Planner
from bot.autoplay import UINavigator

W, H = 960, 540
ROAD_GRAY = (120, 120, 120)
GRASS = (70, 170, 80)


def make_frame(dash_phase: float, car_y_frac: float) -> np.ndarray:
    """Perspective road like the game: converging edges, dashed lines, one car."""
    img = np.full((H, W, 3), GRASS, np.uint8)
    horizon_y = int(0.28 * H)
    # road polygon: wide at bottom, narrow at horizon
    bl, br = 0.14 * W, 0.86 * W
    tl, tr = 0.40 * W, 0.60 * W
    road = np.array([[tl, horizon_y], [tr, horizon_y], [br, H], [bl, H]], np.int32)
    cv2.fillPoly(img, [road], ROAD_GRAY)
    # 5 lanes -> 4 boundary lines, dashed
    for k in range(1, 5):
        f = k / 5.0
        x_top = tl + f * (tr - tl)
        x_bot = bl + f * (br - bl)
        for seg in range(24):
            t0 = (seg * 2 + (dash_phase % 2.0)) / 48.0
            t1 = t0 + 1.0 / 48.0
            if t1 > 1:
                continue
            # non-linear spacing to mimic perspective compression
            g0, g1 = t0 ** 1.6, t1 ** 1.6
            p0 = (int(x_top + g0 * (x_bot - x_top)), int(horizon_y + g0 * (H - horizon_y)))
            p1 = (int(x_top + g1 * (x_bot - x_top)), int(horizon_y + g1 * (H - horizon_y)))
            cv2.line(img, p0, p1, (250, 250, 250), max(1, int(1 + 3 * g0)))
    # A red car in the center lane, ahead of us. Kept in the upper/mid road so
    # it never falls into the own-car ignore box at the bottom of the frame;
    # car_y_frac grows so it closes on us (tracker should read vy > 0).
    cy = int(horizon_y + car_y_frac * (H - horizon_y))
    scale = 0.5 + 1.4 * car_y_frac
    cw, ch = int(46 * scale), int(58 * scale)
    # center of the road at this depth (road narrows toward the horizon)
    road_cx = (tl + tr) / 2 + (car_y_frac ** 1.6) * ((bl + br) / 2 - (tl + tr) / 2)
    cx = int(road_cx)
    cv2.rectangle(img, (cx - cw // 2, cy - ch), (cx + cw // 2, cy),
                  (40, 40, 200), -1)
    return img


def main() -> None:
    cfg = Config()
    frame0 = make_frame(0.0, 0.45)
    vision = Vision(cfg, frame0.shape)
    tracker = Tracker(cfg)
    planner = Planner(cfg, load_calibration(cfg))

    per = None
    tracks = []
    plan = None
    for i in range(12):
        # a car dead ahead in the center lane, closing on us
        frame = make_frame(dash_phase=i * 0.35, car_y_frac=0.30 + i * 0.03)
        per = vision.process(frame, want_masks=True)
        tracks = tracker.update(per.obstacles)
        plan = planner.plan(per, tracks, fps=30.0)

    assert per.road_quality > 0.5, f"road poorly detected: q={per.road_quality}"
    assert per.road_right - per.road_left > 0.5 * cfg.bev_w, \
        f"road span too narrow: {per.road_left:.0f}..{per.road_right:.0f}"
    assert abs(per.car_x - cfg.bev_w / 2) < 1, "car_x should be fixed at center"
    assert len(per.obstacles) >= 1, "car ahead not detected as obstacle"
    assert len(tracks) >= 1, "tracker lost the car"
    # a car dead ahead: the planner should route around it (a steered path)
    assert plan.state in ("CRUISE", "CHANGE", "SLOW", "BRAKE_WAIT")
    assert plan.n_threats >= 1, "car ahead should register as a threat"
    assert len(plan.path_x) >= 2, "planner produced no route"
    assert plan.steer_key is not None, \
        f"should steer around a car dead ahead, got {plan.state}/{plan.steer_key}"

    # UI navigator: menu on a gray-road background (like the real game), with a
    # top grass strip that the cy filter must reject, and a centered green Play.
    menu = np.full((H, W, 3), (90, 90, 90), np.uint8)
    cv2.rectangle(menu, (0, 0), (W, int(0.20 * H)), GRASS, -1)  # grass, up top
    cv2.rectangle(menu, (int(0.44 * W), int(0.72 * H)), (int(0.56 * W), int(0.79 * H)),
                  (55, 210, 60), -1)
    cv2.putText(menu, "Play", (int(0.465 * W), int(0.775 * H)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    nav = UINavigator((0, 0, W, H))
    state, xy = nav.detect(menu)
    assert state == "menu" and xy is not None, f"menu not detected: {state}"
    # death screen: green Revive left + orange Menu right -> must pick orange
    death = np.full((H, W, 3), (60, 60, 60), np.uint8)
    cv2.rectangle(death, (int(0.28 * W), int(0.84 * H)), (int(0.47 * W), int(0.92 * H)),
                  (60, 220, 65), -1)
    cv2.putText(death, "Revive", (int(0.30 * W), int(0.90 * H)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.rectangle(death, (int(0.53 * W), int(0.84 * H)), (int(0.72 * W), int(0.92 * H)),
                  (30, 165, 250), -1)
    cv2.putText(death, "Menu", (int(0.57 * W), int(0.90 * H)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    state, xy = nav.detect(death)
    assert state == "death", f"death screen not detected: {state}"
    assert xy[0] > 0.5 * W, "must click the ORANGE Menu button, not green Revive"

    print("SMOKE TEST PASSED")
    print(f"  road_q: {per.road_quality:.2f}  road: {per.road_left:.0f}..{per.road_right:.0f}"
          f"  obstacles: {len(per.obstacles)}  tracks: {len(tracks)}")
    print(f"  plan: {plan.state}  steer={plan.steer_key}  threats={plan.n_threats}"
          f"  route_depth={plan.depth}  route_pts={len(plan.path_x)}")


if __name__ == "__main__":
    main()
