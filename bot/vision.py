"""Perception: frame -> BEV warp -> road-edge fit -> RECTIFIED road view ->
lane model, obstacles, flow.

Core trick: the game world is flat-shaded. The road is uniform gray, lane
markings are white. Inside the road, ANYTHING that is neither gray nor white
is an obstacle. No ML, nothing to break when new car models appear.

Rectification: the capture trapezoid is deliberately generous (the camera pans
as the car changes lanes), so in raw BEV the road edges are slanted lines and
grass/buildings intrude near the horizon. Each frame we fit left(y)/right(y)
edge lines and horizontally remap every row so the road spans the full raster
width. In rectified space:
  - lane lines are vertical, lane width constant,
  - off-road content is geometrically gone,
  - the car's own position IS its x within the road. own_x moves as the car
    moves, and own_vx (+ = rightward, px/frame) is the steering feedback signal.

Coordinates: x 0..bev_w = road left..right edge; y 0..bev_h far..near.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Perception:
    blobs: list                    # [(cx, cy_bottom, w, h, area), ...] rectified px
    road_left: float               # rectified: always 0 (kept for telemetry)
    road_right: float              # rectified: always bev_w
    lane_phase: float              # boundary model: boundary_i = phase + i*width
    lane_width: float
    lane_centers: list             # rectified x of each lane center
    own_x: float                   # rectified x of own car, smoothed (moves with the car!)
    own_x_raw: float               # unsmoothed per-frame measurement (calibration)
    own_vx: float                  # px/frame, + = car drifting right
    own_lane: int
    dy: float                      # forward flow px/frame (+ = moving forward)
    edge_quality: float            # 0..1, how confidently the road edges fit
    masks: Optional[dict] = None


def _subpixel_peak(scores: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(scores) - 1:
        return float(idx)
    a, b, c = scores[idx - 1], scores[idx], scores[idx + 1]
    denom = a - 2 * b + c
    if abs(denom) < 1e-9:
        return float(idx)
    return idx + 0.5 * (a - c) / denom


def profile_shift(prev: np.ndarray, cur: np.ndarray, max_shift: int) -> Optional[float]:
    """How far did `prev`'s content move to become `cur`? + = right/down.
    Normalized cross-correlation, subpixel refined. Returns None when there
    isn't enough signal."""
    if prev is None or cur is None or len(prev) != len(cur):
        return None
    p = prev.astype(np.float32)
    c = cur.astype(np.float32)
    if p.sum() < 20 or c.sum() < 20:
        return None
    p = p - p.mean()
    c = c - c.mean()
    n = len(p)
    scores = np.empty(2 * max_shift + 1, dtype=np.float32)
    for i, s in enumerate(range(-max_shift, max_shift + 1)):
        if s >= 0:
            a, b = c[s:], p[: n - s]
        else:
            a, b = c[:s], p[-s:]
        denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6
        scores[i] = (a * b).sum() / denom
    best = int(np.argmax(scores))
    if scores[best] < 0.25:
        return None
    # plain float, not np.float32: this value flows into telemetry (json) and
    # the planner's safe-distance calc downstream.
    return float(_subpixel_peak(scores, best) - max_shift)


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
        self.bev_ignore_u8 = cv2.warpPerspective(
            ignore, self.M, (cfg.bev_w, cfg.bev_h))

        self._ys = np.arange(cfg.bev_h, dtype=np.float32)
        self._map_y = np.repeat(self._ys[:, None], cfg.bev_w, axis=1)

        # Temporal state
        self.edge_left_ab = None    # (slope, intercept) of left(y)
        self.edge_right_ab = None
        self.prev_row_profile = None
        self.lane_phase = None
        self.lane_width = cfg.lane_width_guess_px
        self.own_x_ema = None
        self.prev_own_x_raw = None
        self.own_vx_ema = 0.0
        self.dy_ema = 0.0

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, want_masks: bool = False) -> Perception:
        cfg = self.cfg
        bev = cv2.warpPerspective(frame, self.M, (cfg.bev_w, cfg.bev_h))
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        s_, v_ = hsv[:, :, 1], hsv[:, :, 2]
        gray_mask = ((s_ <= cfg.road_sat_max)
                     & (v_ >= cfg.road_val_min) & (v_ <= cfg.road_val_max))
        white_mask = (s_ <= cfg.white_sat_max) & (v_ >= cfg.white_val_min)
        # Ignored regions (own car / HUD) count as road for edge fitting: the
        # car sits dead-center and would otherwise punch a hole in the road.
        roadish = gray_mask | white_mask | (self.bev_ignore_u8 > 0)

        left_arr, right_arr, quality = self._fit_road_edges(roadish)

        # --- rectify: road left..right -> full raster width, every row ---
        span = (right_arr - left_arr)[:, None]
        xs = (np.arange(cfg.bev_w, dtype=np.float32) + 0.5) / cfg.bev_w
        map_x = left_arr[:, None] + xs[None, :] * span
        rect = cv2.remap(bev, map_x.astype(np.float32), self._map_y,
                         cv2.INTER_LINEAR)
        rect_ignore = cv2.remap(self.bev_ignore_u8, map_x.astype(np.float32),
                                self._map_y, cv2.INTER_NEAREST) > 0

        rhsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
        rs, rv = rhsv[:, :, 1], rhsv[:, :, 2]
        rgray = ((rs <= cfg.road_sat_max)
                 & (rv >= cfg.road_val_min) & (rv <= cfg.road_val_max))
        rwhite = (rs <= cfg.white_sat_max) & (rv >= cfg.white_val_min)

        # --- obstacles ---
        obstacle = ~rgray & ~rwhite & ~rect_ignore
        margin = int(cfg.bev_w * 0.03)  # edge-fit slop
        obstacle[:, :margin] = False
        obstacle[:, cfg.bev_w - margin:] = False
        obstacle_u8 = obstacle.astype(np.uint8) * 255
        obstacle_u8 = cv2.morphologyEx(
            obstacle_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        obstacle_u8 = cv2.morphologyEx(
            obstacle_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        blobs = []
        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(obstacle_u8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < cfg.min_blob_area:
                continue
            blobs.append((float(centroids[i][0]), float(y + bh),
                          float(bw), float(bh), int(area)))

        # --- lane model (vertical lines in rectified space) ---
        col_profile = rwhite.sum(axis=0).astype(np.float32)
        self._update_lane_model(col_profile)
        lane_centers = self._lane_centers()

        # --- own position within the road + lateral velocity ---
        # own_x_raw is where screen-center falls between the road edges at the
        # bottom row, remapped onto 0..bev_w. It moves as the car changes lanes.
        bottom = cfg.bev_h - 1
        l_b, r_b = left_arr[bottom], right_arr[bottom]
        own_x_raw = (cfg.bev_w / 2.0 - l_b) / max(r_b - l_b, 1e-6) * cfg.bev_w
        own_x_raw = float(np.clip(own_x_raw, 0.0, cfg.bev_w))
        if self.own_x_ema is None:
            self.own_x_ema = own_x_raw
            self.prev_own_x_raw = own_x_raw
        vx_inst = own_x_raw - self.prev_own_x_raw
        self.prev_own_x_raw = own_x_raw
        self.own_x_ema = 0.55 * self.own_x_ema + 0.45 * own_x_raw
        self.own_vx_ema = 0.6 * self.own_vx_ema + 0.4 * vx_inst

        # --- forward flow from dash phase ---
        row_profile = rwhite.sum(axis=1).astype(np.float32)
        dy = profile_shift(self.prev_row_profile, row_profile, cfg.max_forward_shift)
        self.prev_row_profile = row_profile
        if dy is not None and dy >= 0:
            self.dy_ema = 0.7 * self.dy_ema + 0.3 * dy
        else:
            # Decay toward zero when the flow is unreadable so a stale reading
            # can't keep feeding the safe-distance calc a phantom high speed.
            self.dy_ema *= 0.9

        own_lane = self._lane_index(self.own_x_ema, lane_centers)
        masks = None
        if want_masks:
            masks = {
                "bev": rect,
                "bev_raw": bev,
                "gray": rgray.astype(np.uint8) * 255,
                "white": rwhite.astype(np.uint8) * 255,
                "obstacle": obstacle_u8,
                "edges": (left_arr, right_arr),
            }

        return Perception(
            blobs=blobs, road_left=0.0, road_right=float(cfg.bev_w),
            lane_phase=self.lane_phase if self.lane_phase is not None else 0.0,
            lane_width=self.lane_width, lane_centers=lane_centers,
            own_x=self.own_x_ema, own_x_raw=own_x_raw,
            own_vx=self.own_vx_ema, own_lane=own_lane,
            dy=self.dy_ema, edge_quality=quality, masks=masks,
        )

    # ------------------------------------------------------------------
    def _fit_road_edges(self, roadish: np.ndarray) -> tuple:
        """Fit left(y), right(y) lines to the road edges, bottom-up per band."""
        cfg = self.cfg
        H, W = cfg.bev_h, cfg.bev_w
        n_bands = 12
        band_h = H // n_bands
        pts_y, pts_l, pts_r, wts = [], [], [], []
        center = W // 2
        for b in range(n_bands - 1, -1, -1):  # bottom band first
            y0, y1 = b * band_h, min((b + 1) * band_h, H)
            frac = roadish[y0:y1, :].mean(axis=0)
            is_r = frac >= 0.5
            c = center
            if not is_r[c]:
                near = np.where(is_r)[0]
                if len(near) == 0:
                    continue
                cand = near[np.argmin(np.abs(near - c))]
                if abs(cand - c) > W * 0.25:
                    continue
                c = int(cand)
            # walk out with gap tolerance (cars punch holes in the gray)
            gap_tol = 12
            left = c
            i, gap = c, 0
            while i > 0:
                i -= 1
                if is_r[i]:
                    left, gap = i, 0
                else:
                    gap += 1
                    if gap > gap_tol:
                        break
            right = c
            i, gap = c, 0
            while i < W - 1:
                i += 1
                if is_r[i]:
                    right, gap = i, 0
                else:
                    gap += 1
                    if gap > gap_tol:
                        break
            if right - left < W * 0.25:
                continue
            yc = (y0 + y1) / 2.0
            pts_y.append(yc)
            pts_l.append(float(left))
            pts_r.append(float(right + 1))
            wts.append(float(is_r[left:right].mean()))
            center = (left + right) // 2

        quality = len(pts_y) / n_bands
        if len(pts_y) >= 3:
            yv = np.array(pts_y)
            wv = np.array(wts) + 1e-3
            la = np.polyfit(yv, np.array(pts_l), 1, w=wv)
            ra = np.polyfit(yv, np.array(pts_r), 1, w=wv)
            a = 0.35  # temporal smoothing
            if self.edge_left_ab is None:
                self.edge_left_ab, self.edge_right_ab = la, ra
            else:
                self.edge_left_ab = (1 - a) * self.edge_left_ab + a * la
                self.edge_right_ab = (1 - a) * self.edge_right_ab + a * ra

        if self.edge_left_ab is None:
            left_arr = np.zeros(H, np.float32)
            right_arr = np.full(H, W, np.float32)
        else:
            left_arr = np.clip(np.polyval(self.edge_left_ab, self._ys),
                               0, W * 0.45).astype(np.float32)
            right_arr = np.clip(np.polyval(self.edge_right_ab, self._ys),
                                W * 0.55, W).astype(np.float32)
        return left_arr, right_arr, quality

    # ------------------------------------------------------------------
    def _update_lane_model(self, col_profile: np.ndarray) -> None:
        peaks = self._find_peaks(col_profile)
        if len(peaks) >= 2:
            gaps = np.diff(peaks)
            good = gaps[(gaps > self.lane_width * 0.6) & (gaps < self.lane_width * 1.6)]
            if len(good) > 0:
                self.lane_width = 0.85 * self.lane_width + 0.15 * float(np.median(good))
        if len(peaks) >= 1:
            ref = float(peaks[np.argmin(np.abs(peaks - self.cfg.bev_w / 2))])
            new_phase = ref % self.lane_width
            if self.lane_phase is None:
                self.lane_phase = new_phase
            else:
                diff = (new_phase - self.lane_phase + self.lane_width / 2) \
                    % self.lane_width - self.lane_width / 2
                self.lane_phase = (self.lane_phase + 0.25 * diff) % self.lane_width
        elif self.lane_phase is None:
            self.lane_phase = 0.0

    @staticmethod
    def _find_peaks(prof: np.ndarray) -> np.ndarray:
        if prof.max() <= 0:
            return np.array([])
        thresh = max(prof.max() * 0.35, 3.0)
        above = prof >= thresh
        peaks = []
        i, n = 0, len(prof)
        while i < n:
            if above[i]:
                j = i
                while j < n and above[j]:
                    j += 1
                peaks.append(i + int(np.argmax(prof[i:j])))
                i = j
            else:
                i += 1
        return np.array(peaks)

    def _lane_centers(self) -> list:
        """Lane centers across the full rectified width via the boundary model."""
        W = self.cfg.bev_w
        w = self.lane_width
        phase = self.lane_phase if self.lane_phase is not None else 0.0
        k0 = int(np.floor((0 - phase) / w))
        centers = []
        b = phase + k0 * w
        while b < W + w:
            c = b + w / 2
            if -0.25 * w <= c - w / 2 and c + w / 2 <= W + 0.25 * w:
                centers.append(float(c))
            b += w
        if not centers:
            n_lanes = max(1, round(W / w))
            lane_w = W / n_lanes
            centers = [(i + 0.5) * lane_w for i in range(int(n_lanes))]
        return centers

    @staticmethod
    def _lane_index(x: float, lane_centers: list) -> int:
        if not lane_centers:
            return 0
        return int(np.argmin([abs(x - c) for c in lane_centers]))

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

        bev_vis = per.masks["bev"].copy() if per.masks else np.zeros(
            (self.cfg.bev_h, self.cfg.bev_w, 3), np.uint8)
        for c in per.lane_centers:
            cv2.line(bev_vis, (int(c), 0), (int(c), self.cfg.bev_h), (80, 80, 80), 1)
        for (cx, cyb, bw, bh, _area) in per.blobs:
            cv2.rectangle(bev_vis, (int(cx - bw / 2), int(cyb - bh)),
                          (int(cx + bw / 2), int(cyb)), (0, 0, 255), 1)
        cv2.circle(bev_vis, (int(per.own_x), self.cfg.bev_h - 4), 5, (0, 255, 0), -1)
        cv2.putText(bev_vis, f"vx={per.own_vx:+.1f} dy={per.dy:.1f} "
                    f"q={per.edge_quality:.1f}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if plan is not None:
            cv2.putText(bev_vis, f"{plan.state} ln{per.own_lane}->{plan.target_lane}",
                        (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            if plan.target_x is not None:
                cv2.line(bev_vis, (int(plan.target_x), self.cfg.bev_h - 20),
                         (int(plan.target_x), self.cfg.bev_h), (0, 255, 255), 2)
        if per.masks and "edges" in per.masks:
            la, ra = per.masks["edges"]
            raw = per.masks["bev_raw"].copy()
            for y in range(0, self.cfg.bev_h, 4):
                cv2.circle(raw, (int(la[y]), y), 1, (0, 255, 255), -1)
                cv2.circle(raw, (int(ra[y]), y), 1, (0, 255, 255), -1)
            per.masks["bev_raw_edges"] = raw
        return vis, bev_vis
