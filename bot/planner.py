"""Planning + control, built around how the game actually drives.

Controls are HELD, not tapped:
  - Gas (W): held to build and hold top speed. Released only to brake.
  - Brake (S): held only when no reachable lane is safe (a dodge is impossible
    at this speed), to buy time. Braking costs distance, so it is a last resort.
  - Steer (A/D): held toward the target lane. Hold length sets the swing size,
    so we keep holding while far and release early by the coast distance, letting
    momentum finish the move.

Planning is time-to-collision (TTC) based: we react to how SOON a car reaches us,
not just how far, so it scales with speed. The bot commits to a lane and only
re-plans when the current lane is actually threatened, which stops the constant
lane-thrashing that wrecked earlier runs.

Latency compensation: every obstacle is projected forward by
(measured input dead time + a margin) before any decision, so we plan against the
world as it will be when our keys land.
"""
from dataclasses import dataclass
from typing import Optional

from .controls import LEFT, RIGHT

INF = float("inf")


@dataclass
class Plan:
    state: str
    target_lane: int
    target_x: Optional[float]
    gas: bool                    # hold W
    brake: bool                  # hold S
    steer_key: Optional[str]     # hold LEFT / RIGHT / None
    clear_dists: list            # per-lane nearest obstacle distance (px), debug
    ttc: list                    # per-lane time-to-collision (s), debug
    danger_dist: float           # distance that counts as a threat now (telemetry)


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
        fps = max(fps, 6.0)
        lanes = per.lane_centers
        n = len(lanes)
        if n == 0:
            return Plan("CRUISE", 0, None, True, False, None, [], [], 0.0)

        lookahead = (self.cal["latency_ms"] / 1000.0 + cfg.lookahead_extra_s) * fps
        dy = max(per.dy, cfg.dy_floor)          # forward flow, px/frame
        speed_px_s = dy * fps

        # --- per-lane nearest obstacle (against PREDICTED positions) ---
        own_y = cfg.bev_h
        clear = [INF] * n
        side_block = [False] * n
        for tr in tracks:
            px, py = tr.predict(lookahead)
            for li in self._lanes_of(px, tr.w, lanes):
                dist = own_y - py
                if dist > 0:
                    clear[li] = min(clear[li], dist)
                if -cfg.side_margin_behind <= dist <= cfg.side_margin_ahead:
                    side_block[li] = True

        # TTC per lane (seconds until we reach that obstacle at current speed)
        ttc = [c / speed_px_s if c != INF else INF for c in clear]
        danger_dist = cfg.ttc_danger_s * speed_px_s

        cur = min(max(per.own_lane, 0), n - 1)
        self.frames_in_state += 1

        # --- state machine (commit to lanes; only re-plan on real threats) ---
        if self.state == "CHANGE":
            tl = self.target_lane
            valid = tl is not None and 0 <= tl < n
            reached = valid and abs(per.own_x - lanes[tl]) < cfg.lane_reached_px
            if not valid or reached:
                self._to("CRUISE")
                self.target_lane = None
            elif self.frames_in_state >= cfg.change_commit_min_frames:
                # target went dangerous mid-move -> re-pick or brake
                if ttc[tl] < cfg.ttc_brake_s:
                    best = self._best_lane(cur, ttc, side_block, n)
                    if best is not None and ttc[best] > ttc[tl] + cfg.ttc_change_margin_s:
                        self.target_lane = best
                        self.frames_in_state = 0
                    elif ttc[cur] < cfg.ttc_brake_s:
                        self._to("BRAKE_WAIT")
                        self.target_lane = None

        elif self.state == "BRAKE_WAIT":
            best = self._best_lane(cur, ttc, side_block, n)
            if ttc[cur] >= cfg.ttc_danger_s:
                self._to("CRUISE")
            elif best is not None and ttc[best] >= cfg.ttc_danger_s:
                self.target_lane = best
                self._to("CHANGE")

        else:  # CRUISE
            if ttc[cur] < cfg.ttc_danger_s:
                best = self._best_lane(cur, ttc, side_block, n)
                if best is not None and ttc[best] > ttc[cur] + cfg.ttc_change_margin_s:
                    self.target_lane = best
                    self._to("CHANGE")
                elif ttc[cur] < cfg.ttc_brake_s:
                    self._to("BRAKE_WAIT")

        # --- outputs ---
        if self.state == "CHANGE" and self.target_lane is not None \
                and 0 <= self.target_lane < n:
            target_lane = self.target_lane
        else:
            target_lane = cur
        target_x = lanes[target_lane]

        brake = self.state == "BRAKE_WAIT"
        gas = not brake                          # hold W to keep speed unless braking
        steer_key = self._steer(per, target_x, fps)

        return Plan(self.state, target_lane, target_x, gas, brake, steer_key,
                    clear, ttc, danger_dist)

    # ------------------------------------------------------------------
    def _to(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.frames_in_state = 0

    def _lanes_of(self, x: float, w: float, lanes: list) -> list:
        """Lane indices a blob at center x with width w occupies."""
        cfg = self.cfg
        lane_w = (lanes[1] - lanes[0]) if len(lanes) > 1 else cfg.lane_width_guess_px
        half = max(w / 2, cfg.straddle_frac * lane_w)
        out = [i for i, c in enumerate(lanes) if abs(x - c) < lane_w / 2 + half * 0.6]
        return out or [min(range(len(lanes)), key=lambda i: abs(x - lanes[i]))]

    def _best_lane(self, cur: int, ttc: list, side_block: list, n: int) -> Optional[int]:
        """Safest reachable lane (highest TTC). Adjacent first; a 2-away lane
        only if the lane between it and us is itself safe to cross."""
        cands = []
        for d in (-1, 1):
            li = cur + d
            if 0 <= li < n and not side_block[li]:
                cands.append((ttc[li], li))
        for d in (-2, 2):
            li, mid = cur + d, cur + d // 2
            if (0 <= li < n and not side_block[li] and not side_block[mid]
                    and ttc[mid] >= self.cfg.ttc_danger_s):
                cands.append((ttc[li] - 0.2, li))  # small penalty: longer move
        if not cands:
            return None
        best_ttc, best = max(cands)
        return best if best_ttc > 0 else None

    # ------------------------------------------------------------------
    def _steer(self, per, target_x: float, fps: float) -> Optional[str]:
        """Hold A/D toward the target, release early by the coast distance.

        own_x/own_vx are in rectified road space: press RIGHT to raise own_x,
        LEFT to lower it. Releasing while still `lead` px short lets the car's
        lateral momentum carry it the rest of the way (that is how a real swing
        finishes), which prevents the overshoot-and-oscillate failure.
        """
        cfg = self.cfg
        if per.edge_quality < cfg.steer_min_edge_q:
            return None
        lead = per.own_vx * (cfg.steer_release_lead_s * fps)  # px we will coast
        predicted = per.own_x + lead
        remaining = target_x - predicted
        if abs(remaining) < cfg.steer_deadband_px:
            return None
        return RIGHT if remaining > 0 else LEFT
