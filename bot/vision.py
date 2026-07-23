"""Perception: frame -> fixed bird's-eye view -> road region -> obstacles.

Rebuilt to drop the road-edge fit, which drifted (locking onto grass, skewing
the whole view) and corrupted everything downstream. The new method has nothing
to mislock:

  1. warpPerspective the road trapezoid to a fixed top-down raster (STABLE: a
     constant transform, never re-fit per frame).
  2. Road = the largest connected gray-ish region (road + lane lines). Grass and
     buildings are a different color, so they fall outside it.
  3. Fill each road row left-edge..right-edge so car-shaped holes are inside the
     road region; obstacles are the colored (non-gray, non-white) blobs in there.

The player's car sits at a FIXED spot: bottom-center, (bev_w/2, bev_h). Obstacles
are reported in this same fixed space, so the planner reasons purely about where
cars are relative to us, with no lateral-position estimate to go wrong.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Perception:
    obstacles: list        # (cx, cy_bottom, w, h, area) in fixed BEV
    road_left: float       # road bounds near the car (bottom rows), fixed BEV
    road_right: float
    car_x: float           # bev_w / 2 (constant)
    road_quality: float    # fraction of rows that found road (health metric)
    masks: Optional[dict] = None


class Vision:
    def __init__(self, cfg, frame_shape):
        self.cfg = cfg
        h, w = frame_shape[:2]
        self.frame_w, self.frame_h = w, h

        src = np.float32([
            [cfg.trap_top_left_x * w, cfg.trap_top_y * h],
            [cfg.trap_top_right_x * w, cfg.trap_top_y * h],
            [cfg.trap_bottom_right_x * w, cfg.trap_bottom_y * h],
            [cfg.trap_bottom_left_x * w, cfg.trap_bottom_y * h],
        ])
        dst = np.float32([
            [0, 0], [cfg.bev_w, 0], [cfg.bev_w, cfg.bev_h], [0, cfg.bev_h],
        ])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.src_trapezoid = src

        ignore = np.zeros((h, w), dtype=np.uint8)
        for box in (cfg.own_car_box, cfg.top_hud_box):
            l, t, r, b = box
            ignore[int(t * h):int(b * h), int(l * w):int(r * w)] = 255
        self.bev_ignore = cv2.warpPerspective(
            ignore, self.M, (cfg.bev_w, cfg.bev_h)) > 0
        self._cols = np.arange(cfg.bev_w)

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, want_masks: bool = False) -> Perception:
        cfg = self.cfg
        W, H = cfg.bev_w, cfg.bev_h
        bev = cv2.warpPerspective(frame, self.M, (W, H))
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        s_, v_ = hsv[:, :, 1], hsv[:, :, 2]
        gray = ((s_ <= cfg.road_sat_max)
                & (v_ >= cfg.road_val_min) & (v_ <= cfg.road_val_max))
        white = (s_ <= cfg.white_sat_max) & (v_ >= cfg.white_val_min)

        # --- road = largest connected gray/white region (car holes included) ---
        roadish = (gray | white | self.bev_ignore).astype(np.uint8)
        roadish = cv2.morphologyEx(roadish, cv2.MORPH_CLOSE,
                                   np.ones((7, 7), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(roadish)
        road_region = np.zeros((H, W), dtype=bool)
        quality = 0.0
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            road_cc = labels == biggest
            any_row = road_cc.any(axis=1)
            if any_row.any():
                left = road_cc.argmax(axis=1)
                right = W - 1 - road_cc[:, ::-1].argmax(axis=1)
                road_region = ((self._cols[None, :] >= left[:, None])
                               & (self._cols[None, :] <= right[:, None])
                               & any_row[:, None])
                quality = float(any_row.mean())

        # --- obstacles = colored blobs sitting inside the road ---
        obstacle = road_region & ~gray & ~white & ~self.bev_ignore
        ob = obstacle.astype(np.uint8) * 255
        ob = cv2.morphologyEx(ob, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ob = cv2.morphologyEx(ob, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        obstacles = []
        n2, _l, stats2, cents = cv2.connectedComponentsWithStats(ob)
        for i in range(1, n2):
            x, y, bw, bh, area = stats2[i]
            if area < cfg.min_blob_area or bw > 0.6 * W:
                continue
            obstacles.append((float(cents[i][0]), float(y + bh),
                              float(bw), float(bh), int(area)))

        # --- road bounds near the car (bottom rows) ---
        near = road_region[int(H * 0.75):, :]
        cols_any = near.any(axis=0)
        if cols_any.any():
            road_left = float(np.argmax(cols_any))
            road_right = float(W - 1 - np.argmax(cols_any[::-1]))
        else:
            road_left, road_right = 0.0, float(W)

        masks = None
        if want_masks:
            masks = {"bev": bev, "obstacle": ob,
                     "road": road_region.astype(np.uint8) * 255}
        return Perception(obstacles=obstacles, road_left=road_left,
                          road_right=road_right, car_x=W / 2.0,
                          road_quality=quality, masks=masks)

    # ------------------------------------------------------------------
    def draw_debug(self, frame: np.ndarray, per: Perception, plan=None) -> tuple:
        vis = frame.copy()
        cv2.polylines(vis, [self.src_trapezoid.astype(np.int32)], True,
                      (0, 255, 255), 2)
        h, w = vis.shape[:2]
        for box, color in ((self.cfg.own_car_box, (255, 0, 255)),
                           (self.cfg.top_hud_box, (255, 128, 0))):
            l, t, r, b = box
            cv2.rectangle(vis, (int(l * w), int(t * h)), (int(r * w), int(b * h)),
                          color, 1)

        bev = per.masks["bev"].copy() if per.masks else np.zeros(
            (self.cfg.bev_h, self.cfg.bev_w, 3), np.uint8)
        cv2.line(bev, (int(per.road_left), 0), (int(per.road_left), self.cfg.bev_h),
                 (0, 128, 255), 1)
        cv2.line(bev, (int(per.road_right), 0), (int(per.road_right), self.cfg.bev_h),
                 (0, 128, 255), 1)
        for (cx, cyb, bw, bh, _a) in per.obstacles:
            cv2.rectangle(bev, (int(cx - bw / 2), int(cyb - bh)),
                          (int(cx + bw / 2), int(cyb)), (0, 0, 255), 2)
        cv2.circle(bev, (int(per.car_x), self.cfg.bev_h - 4), 5, (0, 255, 0), -1)
        if plan is not None:
            cv2.putText(bev, f"{plan.state} d={getattr(plan, 'depth', 0)} "
                        f"steer={plan.steer_key or '-'}",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            # the planned route through the walls
            path = getattr(plan, "path_x", None) or []
            nr = self.cfg.grid_rows
            pts = [(int(per.car_x), self.cfg.bev_h)]
            for i, px in enumerate(path):
                y = int(self.cfg.bev_h * (1.0 - (i + 0.5) / nr))
                pts.append((int(px), y))
            for a, b in zip(pts, pts[1:]):
                cv2.line(bev, a, b, (0, 255, 0), 2)
            for p in pts[1:]:
                cv2.circle(bev, p, 2, (0, 255, 255), -1)
        return vis, bev
