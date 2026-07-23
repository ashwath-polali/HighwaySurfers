"""Planning + control.

Latency compensation: every obstacle is projected forward by
(measured input dead time + a maneuver margin) before any decision is made,
so we always plan against the world as it will be when our keys land.

States:
  CRUISE      current lane clear -> full gas, stay centered
  CHANGE      committed lane change toward target_x
  BRAKE_WAIT  nothing reachable is clear -> shed speed until a gap opens
"""
from dataclasses import dataclass
from typing import Optional

from .controls import GAS, BRAKE, LEFT, RIGHT


@dataclass
class Plan:
    state: str
    target_lane: int
    target_x: Optional[float]
    gas: bool
    brake_tap_ms: float          # 0 = no brake
    steer_key: Optional[str]     # LEFT / RIGHT / None
    clear_dists: list            # per-lane clear distance (debug/telemetry)
    safe_dist: float


class Planner:
    def __init__(self, cfg, calibration: dict):
        self.cfg = cfg
        self.cal = calibration
        self.state = "CRUISE"
        self.target_lane: Optional[int] = None
        self.frames_in_state = 0

    # ------------------------------------------------------------------
    def plan(self, per, tracks: list, fps: float) -> Plan:
        cfg = self.cfg
        fps = max(fps, 10.0)
        lanes = per.lane_centers
        n_lanes = len(lanes)
        if n_lanes == 0:
            return Plan("CRUISE", 0, None, True, 0.0, None, [], 0.0)

        latency_frames = (self.cal["latency_ms"] / 1000.0) * fps
        lookahead = latency_frames + cfg.lookahead_extra_s * fps

        # --- per-lane clear distance against PREDICTED obstacle positions ---
        own_y = cfg.bev_h
        clear = [float("inf")] * n_lanes
        side_block = [False] * n_lanes
        for tr in tracks:
            px, py = tr.predict(lookahead)
            occupied = self._lanes_of(px, tr.w, lanes)
            for li in occupied:
                dist = own_y - py
                if dist > 0:
                    clear[li] = min(clear[li], dist)
                # beside-or-slightly-behind blocks a change into that lane
                if -cfg.side_margin_behind <= dist <= cfg.side_margin_ahead:
                    side_block[li] = True

        # Speed-scaled safety distance (dy is px/frame of forward flow).
        safe = min(max(cfg.safe_dist_speed_k * per.dy, cfg.safe_dist_min),
                   cfg.safe_dist_max)
        brake_dist = cfg.brake_dist_frac * safe

        cur = min(max(per.own_lane, 0), n_lanes - 1)

        # --- state machine ---
        self.frames_in_state += 1
        if self.state == "CHANGE":
            done = (self.target_lane is not None
                    and abs(per.own_x - lanes[self.target_lane]) < cfg.steer_tol_px * 1.5)
            aborted = self.target_lane is None or self.target_lane >= n_lanes
            if done or aborted or self.frames_in_state > int(3.0 * fps):
                self._to("CRUISE")
                self.target_lane = None

        if self.state == "CRUISE":
            if clear[cur] < safe:
                best = self._best_reachable(cur, clear, side_block, n_lanes)
                if best is not None and clear[best] > clear[cur] * cfg.change_gain_req:
                    self.target_lane = best
                    self._to("CHANGE")
                elif clear[cur] < brake_dist:
                    self._to("BRAKE_WAIT")

        elif self.state == "CHANGE":
            tl = self.target_lane
            # target lane got dangerous mid-change -> re-evaluate
            if tl is not None and tl < n_lanes and clear[tl] < brake_dist:
                best = self._best_reachable(cur, clear, side_block, n_lanes)
                if best is not None and best != tl and clear[best] > clear[tl]:
                    self.target_lane = best
                elif clear[cur] >= clear[tl]:
                    self.target_lane = None
                    self._to("BRAKE_WAIT" if clear[cur] < brake_dist else "CRUISE")

        elif self.state == "BRAKE_WAIT":
            best = self._best_reachable(cur, clear, side_block, n_lanes)
            if clear[cur] >= safe:
                self._to("CRUISE")
            elif best is not None and clear[best] > max(safe * 0.8, clear[cur] * 1.3):
                self.target_lane = best
                self._to("CHANGE")

        # --- outputs ---
        target_lane = self.target_lane if (
            self.state == "CHANGE" and self.target_lane is not None) else cur
        target_lane = min(max(target_lane, 0), n_lanes - 1)
        target_x = lanes[target_lane]

        effective_clear = min(clear[cur], clear[target_lane]) \
            if self.state == "CHANGE" else clear[cur]
        gas = effective_clear > safe * 0.75 and self.state != "BRAKE_WAIT"
        brake_ms = 0.0
        if self.state == "BRAKE_WAIT" or effective_clear < brake_dist:
            # deeper deficit -> longer tap
            deficit = 1.0 - min(effective_clear / max(brake_dist, 1e-6), 1.0)
            brake_ms = self.cfg.brake_tap_ms * (0.5 + 0.5 * deficit)

        steer_key = self._steer(per, target_x, fps)

        return Plan(self.state, target_lane, target_x, gas, brake_ms, steer_key,
                    clear, safe)

    # ------------------------------------------------------------------
    def _to(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.frames_in_state = 0

    def _lanes_of(self, x: float, w: float, lanes: list) -> list:
        """Lane indices a blob at center x with width w occupies."""
        cfg = self.cfg
        half = max(w / 2, cfg.straddle_frac * cfg.lane_width_guess_px)
        out = []
        for i, c in enumerate(lanes):
            lane_half = (lanes[1] - lanes[0]) / 2 if len(lanes) > 1 else cfg.lane_width_guess_px / 2
            if abs(x - c) < lane_half + half * 0.6:
                out.append(i)
        return out or [min(range(len(lanes)), key=lambda i: abs(x - lanes[i]))]

    def _best_reachable(self, cur: int, clear: list, side_block: list,
                        n_lanes: int) -> Optional[int]:
        """Adjacent lanes first; a 2-away lane only if the intermediate is safe."""
        cands = []
        for d in (-1, 1):
            li = cur + d
            if 0 <= li < n_lanes and not side_block[li]:
                cands.append((clear[li], li))
        for d in (-2, 2):
            li = cur + d
            mid = cur + d // 2
            if (0 <= li < n_lanes and not side_block[li]
                    and not side_block[mid] and clear[mid] > self.cfg.safe_dist_min):
                cands.append((clear[li] * 0.9, li))  # slight penalty: longer maneuver
        if not cands:
            return None
        best_clear, best = max(cands)
        return best if best_clear > 0 else None

    # ------------------------------------------------------------------
    def _steer(self, per, target_x: float, fps: float) -> Optional[str]:
        """Predictive bang-bang: hold toward target, release early by coast distance.

        own lateral velocity = -dx (world shifts opposite to the car).
        """
        cfg = self.cfg
        err = target_x - per.own_x           # + = need to move right
        v_own = -per.dx * fps                # px/s, + = moving right
        coast = v_own * self.cal["steer_coast_s"]  # px we'll drift after release
        # where we'd end up if we released right now
        settle = coast
        remaining = err - settle
        if abs(err) < cfg.steer_tol_px and abs(v_own) < 40:
            return None
        if remaining > cfg.steer_tol_px * 0.6:
            return RIGHT
        if remaining < -cfg.steer_tol_px * 0.6:
            return LEFT
        return None
