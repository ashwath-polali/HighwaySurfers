"""Perception: frame -> bird's-eye view -> road span, lane model, obstacles, flow.

Core trick: the game world is flat-shaded. The road is uniform gray, lane
markings are white. Inside the road span, ANYTHING that is neither gray nor
white is an obstacle. No ML, nothing to break when new car models appear.

All outputs live in BEV (bird's-eye view) pixel space:
  x: 0..bev_w (left..right), y: 0..bev_h (far..near). The player car sits at
  x = bev_w/2, y = bev_h (bottom edge) because the camera is locked to it.
"""
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Perception:
    blobs: list                    # [(cx, cy_bottom, w, h, area), ...] BEV px
    road_left: float               # BEV x of left road edge
    road_right: float
    lane_phase: float              # boundary model: boundary_i = phase + i*width
    lane_width: float
    lane_centers: list             # BEV x of each lane center within road span
    own_x: float                   # BEV x of own car (constant: bev_w/2)
    own_lane: int                  # index into lane_centers (clamped)
    dx: float                      # lateral world shift px/frame (+ = content moved right)
    dy: float                      # forward flow px/frame (+ = moving forward)
    masks: Optional[dict] = None   # debug masks when requested


def _subpixel_peak(scores: np.ndarray, idx: int) -> float:
    """Parabolic refinement of an argmax index."""
    if idx <= 0 or idx >= len(scores) - 1:
        return float(idx)
    a, b, c = scores[idx - 1], scores[idx], scores[idx + 1]
    denom = a - 2 * b + c
    if abs(denom) < 1e-9:
        return float(idx)
    return idx + 0.5 * (a - c) / denom


def profile_shift(prev: np.ndarray, cur: np.ndarray, max_shift: int) -> Optional[float]:
    """How far did `prev`'s content move to become `cur`? + = moved right/down.

    Normalized cross-correlation over integer shifts with subpixel refinement.
    Returns None when there is not enough signal to trust.
    """
    if prev is None or cur is None or len(prev) != len(cur):
        return None
    p = prev.astype(np.float32)
    c = cur.astype(np.float32)
    if p.sum() < 20 or c.sum() < 20:  # not enough markings visible
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
    if scores[best] < 0.25:  # weak correlation -> unreliable
        return None
    return _subpixel_peak(scores, best) - max_shift


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

        # Ignore mask (own car + HUD), warped into BEV once.
        ignore = np.zeros((h, w), dtype=np.uint8)
        for box in (cfg.own_car_box, cfg.top_hud_box):
            l, t, r, b = box
            ignore[int(t * h):int(b * h), int(l * w):int(r * w)] = 255
        self.bev_ignore = cv2.warpPerspective(
            ignore, self.M, (cfg.bev_w, cfg.bev_h)) > 0

        # Temporal state
        self.prev_col_profile = None   # white-mask column sums (lateral flow)
        self.prev_row_profile = None   # white-mask row sums (forward flow)
        self.lane_phase = None
        self.lane_width = cfg.lane_width_guess_px
        self.dx_ema = 0.0
        self.dy_ema = 0.0

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, want_masks: bool = False) -> Perception:
        cfg = self.cfg
        bev = cv2.warpPerspective(frame, self.M, (cfg.bev_w, cfg.bev_h))
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        h_, s_, v_ = cv2.split(hsv)

        gray_mask = ((s_ <= cfg.road_sat_max)
                     & (v_ >= cfg.road_val_min) & (v_ <= cfg.road_val_max))
        white_mask = (s_ <= cfg.white_sat_max) & (v_ >= cfg.white_val_min)

        # Ignored regions (own car / HUD) count as road for span-finding: the
        # black car sits dead-center and would otherwise split the road in two.
        road_left, road_right = self._road_span(
            gray_mask | white_mask | self.bev_ignore)

        # --- lane boundary model from white-line column histogram ---
        col_profile = self._column_profile(white_mask, road_left, road_right)
        self._update_lane_model(col_profile, road_left, road_right)

        # --- flow ---
        dx = profile_shift(self.prev_col_profile, col_profile, cfg.max_lateral_shift)
        row_profile = self._row_profile(white_mask, road_left, road_right)
        dy = profile_shift(self.prev_row_profile, row_profile, cfg.max_forward_shift)
        self.prev_col_profile = col_profile
        self.prev_row_profile = row_profile
        if dx is not None:
            self.dx_ema = 0.5 * self.dx_ema + 0.5 * dx
        if dy is not None and dy >= 0:
            self.dy_ema = 0.7 * self.dy_ema + 0.3 * dy

        # --- obstacles: on-road, not gray, not white, not ignored ---
        on_road = np.zeros_like(gray_mask)
        li, ri = int(max(0, road_left)), int(min(cfg.bev_w, road_right))
        on_road[:, li:ri] = True
        obstacle = on_road & ~gray_mask & ~white_mask & ~self.bev_ignore
        obstacle_u8 = obstacle.astype(np.uint8) * 255
        obstacle_u8 = cv2.morphologyEx(
            obstacle_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        obstacle_u8 = cv2.morphologyEx(
            obstacle_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        blobs = []
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(obstacle_u8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < cfg.min_blob_area:
                continue
            cx = centroids[i][0]
            cy_bottom = y + bh  # nearest point of the obstacle to us
            blobs.append((float(cx), float(cy_bottom), float(bw), float(bh), int(area)))

        lane_centers = self._lane_centers(road_left, road_right)
        own_x = cfg.bev_w / 2.0
        own_lane = self._lane_index(own_x, lane_centers)

        masks = None
        if want_masks:
            masks = {
                "bev": bev,
                "gray": gray_mask.astype(np.uint8) * 255,
                "white": white_mask.astype(np.uint8) * 255,
                "obstacle": obstacle_u8,
            }

        return Perception(
            blobs=blobs, road_left=road_left, road_right=road_right,
            lane_phase=self.lane_phase if self.lane_phase is not None else 0.0,
            lane_width=self.lane_width, lane_centers=lane_centers,
            own_x=own_x, own_lane=own_lane,
            dx=self.dx_ema, dy=self.dy_ema, masks=masks,
        )

    # ------------------------------------------------------------------
    def _road_span(self, roadish: np.ndarray) -> tuple:
        """Contiguous run of road columns around the car, from bottom rows."""
        cfg = self.cfg
        rows = int(cfg.bev_h * (1 - cfg.road_span_row_frac))
        sample = roadish[rows:, :]
        col_frac = sample.mean(axis=0)
        is_road_col = col_frac >= cfg.road_span_gray_frac
        center = cfg.bev_w // 2
        if not is_road_col[center]:
            # A car directly ahead can shadow the center column; widen search.
            near = np.where(is_road_col)[0]
            if len(near) == 0:
                return 0.0, float(cfg.bev_w)
            center = int(near[np.argmin(np.abs(near - center))])
        left = center
        while left > 0 and is_road_col[left - 1]:
            left -= 1
        right = center
        while right < cfg.bev_w - 1 and is_road_col[right + 1]:
            right += 1
        return float(left), float(right + 1)

    def _column_profile(self, white_mask: np.ndarray, road_left: float,
                        road_right: float) -> np.ndarray:
        """White pixels per column, zeroed outside the road span."""
        prof = white_mask.sum(axis=0).astype(np.float32)
        li, ri = int(road_left), int(road_right)
        prof[:li] = 0
        prof[ri:] = 0
        return prof

    def _row_profile(self, white_mask: np.ndarray, road_left: float,
                     road_right: float) -> np.ndarray:
        li, ri = int(road_left), int(road_right)
        if ri - li < 4:
            return white_mask.sum(axis=1).astype(np.float32)
        return white_mask[:, li:ri].sum(axis=1).astype(np.float32)

    def _update_lane_model(self, col_profile: np.ndarray, road_left: float,
                           road_right: float) -> None:
        """Fit boundary_i = phase + i*width to white-line histogram peaks."""
        peaks = self._find_peaks(col_profile)
        if len(peaks) >= 2:
            gaps = np.diff(peaks)
            # ignore double-detections and merged peaks
            good = gaps[(gaps > self.lane_width * 0.6) & (gaps < self.lane_width * 1.6)]
            if len(good) > 0:
                self.lane_width = 0.85 * self.lane_width + 0.15 * float(np.median(good))
        if len(peaks) >= 1:
            # phase = representative peak folded into [0, width)
            ref = float(peaks[np.argmin(np.abs(peaks - self.cfg.bev_w / 2))])
            new_phase = ref % self.lane_width
            if self.lane_phase is None:
                self.lane_phase = new_phase
            else:
                # move along shortest wrap-around path
                diff = (new_phase - self.lane_phase + self.lane_width / 2) % self.lane_width \
                    - self.lane_width / 2
                self.lane_phase = (self.lane_phase + 0.25 * diff) % self.lane_width
        elif self.lane_phase is None:
            self.lane_phase = (road_left % self.lane_width)

    @staticmethod
    def _find_peaks(prof: np.ndarray) -> np.ndarray:
        if prof.max() <= 0:
            return np.array([])
        thresh = max(prof.max() * 0.35, 3.0)
        above = prof >= thresh
        peaks = []
        i = 0
        n = len(prof)
        while i < n:
            if above[i]:
                j = i
                while j < n and above[j]:
                    j += 1
                seg = prof[i:j]
                peaks.append(i + int(np.argmax(seg)))
                i = j
            else:
                i += 1
        return np.array(peaks)

    def _lane_centers(self, road_left: float, road_right: float) -> list:
        """Centers of full lanes inside the road span, using the boundary model."""
        w = self.lane_width
        phase = self.lane_phase if self.lane_phase is not None else 0.0
        # first boundary at or left of road_left
        k0 = int(np.floor((road_left - phase) / w))
        centers = []
        b = phase + k0 * w
        while b < road_right + w:
            c = b + w / 2
            if road_left - 0.25 * w <= c - w / 2 and c + w / 2 <= road_right + 0.25 * w:
                centers.append(float(c))
            b += w
        if not centers:  # degenerate fallback: split span evenly by guess width
            n_lanes = max(1, round((road_right - road_left) / w))
            lane_w = (road_right - road_left) / n_lanes
            centers = [road_left + (i + 0.5) * lane_w for i in range(int(n_lanes))]
        return centers

    @staticmethod
    def _lane_index(x: float, lane_centers: list) -> int:
        if not lane_centers:
            return 0
        return int(np.argmin([abs(x - c) for c in lane_centers]))

    # ------------------------------------------------------------------
    def draw_debug(self, frame: np.ndarray, per: Perception,
                   plan=None) -> tuple:
        """Returns (annotated_frame, annotated_bev)."""
        vis = frame.copy()
        cv2.polylines(vis, [self.src_trapezoid.astype(np.int32)], True, (0, 255, 255), 2)
        h, w = vis.shape[:2]
        for box, color in ((self.cfg.own_car_box, (255, 0, 255)),
                           (self.cfg.top_hud_box, (255, 128, 0))):
            l, t, r, b = box
            cv2.rectangle(vis, (int(l * w), int(t * h)), (int(r * w), int(b * h)),
                          color, 1)

        bev_vis = per.masks["bev"].copy() if per.masks else np.zeros(
            (self.cfg.bev_h, self.cfg.bev_w, 3), np.uint8)
        cv2.line(bev_vis, (int(per.road_left), 0), (int(per.road_left), self.cfg.bev_h),
                 (0, 128, 255), 1)
        cv2.line(bev_vis, (int(per.road_right), 0), (int(per.road_right), self.cfg.bev_h),
                 (0, 128, 255), 1)
        for c in per.lane_centers:
            cv2.line(bev_vis, (int(c), 0), (int(c), self.cfg.bev_h), (80, 80, 80), 1)
        for (cx, cyb, bw, bh, area) in per.blobs:
            cv2.rectangle(bev_vis, (int(cx - bw / 2), int(cyb - bh)),
                          (int(cx + bw / 2), int(cyb)), (0, 0, 255), 1)
        cv2.circle(bev_vis, (int(per.own_x), self.cfg.bev_h - 4), 5, (0, 255, 0), -1)
        cv2.putText(bev_vis, f"dx={per.dx:+.1f} dy={per.dy:.1f}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if plan is not None:
            cv2.putText(bev_vis, f"{plan.state} ln{per.own_lane}->{plan.target_lane}",
                        (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            if plan.target_x is not None:
                cv2.line(bev_vis, (int(plan.target_x), self.cfg.bev_h - 20),
                         (int(plan.target_x), self.cfg.bev_h), (0, 255, 255), 2)
        return vis, bev_vis
