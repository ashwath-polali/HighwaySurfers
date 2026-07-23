"""Planning + control, built around how the game actually drives.

Controls are HELD, not tapped:
  - Gas (W): held to build and hold top speed. Released only to brake.
  - Brake (S): held only when no reachable lane is safe (a dodge is impossible
    at this speed), to buy time. Braking costs distance, so it is a last resort.
  - Steer (A/D): held toward the target lane. Hold length sets the swing size,
    so we keep holding while far and release early by the coast distance, letting
    momentum finish the move.

Planning is time-to-collision (TTC) based: we react to how SOON a car reaches us
rather than how far, so it scales with speed. The bot commits to a lane and only
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
        # Commit a lane change to a target X POSITION, not a lane index: lane
        # indices are re-derived each frame from a drifting lane model, so a
        # stored index can silently point at a different lane next frame.
        self.target_x: Optional[float] = None
        self.frames_in_state = 0
        self.frame_i = 0
        self._steer_dir: Optional[str] = None
        self._steer_frame = 0

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

        def nearest(px):
            return min(range(n), key=lambda i: abs(lanes[i] - px))

        # --- state machine (commit to a lane; only re-plan on real threats) ---
        if self.state == "CHANGE":
            tx = self.target_x
            if tx is None:
                self._to("CRUISE")
            else:
                tl = nearest(tx)
                if abs(per.own_x - tx) < cfg.lane_reached_px:
                    self._to("CRUISE")
                    self.target_x = None
                elif self.frames_in_state >= cfg.change_commit_min_frames \
                        and ttc[tl] < cfg.ttc_brake_s:
                    # target went dangerous mid-move: re-pick, else brake
                    best = self._best_lane(cur, ttc, side_block, n)
                    if best is not None and ttc[best] > ttc[tl] + cfg.ttc_change_margin_s:
                        self.target_x = lanes[best]
                        self.frames_in_state = 0
                    else:
                        self._to("BRAKE_WAIT")
                        self.target_x = None

        elif self.state == "BRAKE_WAIT":
            best = self._best_lane(cur, ttc, side_block, n)
            if ttc[cur] >= cfg.ttc_danger_s:
                self._to("CRUISE")
            elif best is not None and ttc[best] >= cfg.ttc_danger_s:
                self.target_x = lanes[best]
                self._to("CHANGE")

        else:  # CRUISE
            if ttc[cur] < cfg.ttc_danger_s:
                best = self._best_lane(cur, ttc, side_block, n)
                if best is not None and ttc[best] > ttc[cur] + cfg.ttc_change_margin_s:
                    self.target_x = lanes[best]
                    self._to("CHANGE")
                else:
                    # Threatened with nowhere safer to go: brake. (Previously it
                    # only braked below ttc_brake_s, so a car closing in the gap
                    # between the two thresholds got no reaction at all.)
                    self._to("BRAKE_WAIT")

        # --- outputs ---
        if self.state == "CHANGE" and self.target_x is not None:
            target_x = self.target_x
            target_lane = nearest(target_x)
        else:
            target_lane = cur
            target_x = lanes[cur]

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
        """Safest reachable lane. Score by TTC (capped so 'plenty of room' lanes
        tie and the tie breaks toward the NEARER lane, not a fixed side). A
        2-away lane pays a penalty and needs the lane between to be safe."""
        cap = self.cfg.ttc_danger_s * 2.0   # beyond this, more room does not matter
        cands = []
        for d in (-1, 1):
            li = cur + d
            if 0 <= li < n and not side_block[li]:
                cands.append((min(ttc[li], cap), -abs(d), li))
        for d in (-2, 2):
            li, mid = cur + d, cur + d // 2
            if (0 <= li < n and not side_block[li] and not side_block[mid]
                    and ttc[mid] >= self.cfg.ttc_danger_s):
                cands.append((min(ttc[li], cap) - 0.3, -abs(d), li))  # longer move
        if not cands:
            return None
        cands.sort(reverse=True)                # highest score, then nearest lane
        best_score, _, best = cands[0]
        return best if best_score > 0 else None

    # ------------------------------------------------------------------
    def _steer(self, per, target_x: float, fps: float) -> Optional[str]:
        """Hold A/D toward the target, release early by the coast distance.

        Sign measured from real play (records): pressing D/RIGHT LOWERS own_x and
        A/LEFT RAISES it (the chase cam makes the rectified road move opposite to
        the car). So to raise own_x we press LEFT, to lower it we press RIGHT.
        Releasing while still `lead` px short lets momentum finish the swing.
        """
        cfg = self.cfg
        self.frame_i += 1
        if per.edge_quality < cfg.steer_min_edge_q:
            self._steer_dir = None
            return None
        # Release lead must cover the WHOLE delay before the car stops turning:
        # the key-up is itself delayed by the input dead time, and only then does
        # the car coast to a stop. Mirror the latency the obstacle lookahead uses.
        lead_s = self.cal["latency_ms"] / 1000.0 + self.cal["steer_coast_s"]
        predicted = per.own_x + per.own_vx * (lead_s * fps)
        remaining = target_x - predicted
        steering = self._steer_dir is not None
        # Hysteresis: release inside the deadband; when idle, only start once the
        # drift exceeds the larger engage band (stops chatter around the center).
        thresh = cfg.steer_deadband_px if steering else cfg.steer_engage_px
        if abs(remaining) < thresh:
            self._steer_dir = None
            return None
        # remaining > 0 means own_x must RISE, and raising own_x needs LEFT (a).
        want = LEFT if remaining > 0 else RIGHT
        # Don't snap straight into the opposite direction: coast a few frames
        # first. Under input lag, instant reversals turn into flip-flop chatter.
        if (self._steer_dir is not None and want != self._steer_dir
                and self.frame_i - self._steer_frame < cfg.steer_reversal_frames):
            return None
        self._steer_dir = want
        self._steer_frame = self.frame_i
        return want
