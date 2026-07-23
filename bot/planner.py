"""Planning + control: plan a route through the walls, then follow it.

This game is "thread the gap in a wall," where each wall is a row of cars with
holes between them. A reactive dodge-the-nearest-car controller jitters and gets
trapped, because it only ever looks one car ahead. Instead we PLAN:

  1. Lay a grid over the road ahead (columns across, rows into the distance).
  2. Mark cells blocked by cars, inflated by our own width so the plan can treat
     the car as a point.
  3. Shortest-path search (dynamic programming) from our position forward, where
     the route may only shift a few columns per row (our real steering rate) and
     pays for lateral movement. The result is the smoothest collision-free route
     through ALL the visible walls at once, not just the next car.
  4. Steer smoothly toward the route a step ahead. A stable global route means a
     stable target, which means fine corrections instead of jitter.

Speed: full gas while a route runs far enough ahead; coast only when the road
tightens to nearly blocked; brake only when boxed in. Held controls; D/RIGHT
moves the car right, A/LEFT left. Obstacles are projected forward by the measured
input latency before planning.
"""
from dataclasses import dataclass, field
from typing import Optional

from .controls import LEFT, RIGHT

INF = float("inf")


@dataclass
class Plan:
    state: str
    target_x: Optional[float]
    gas: bool
    brake: bool
    steer_key: Optional[str]
    n_threats: int
    path_x: list = field(default_factory=list)  # route x per row, for debug/telemetry
    depth: int = 0                                # how many rows the route reaches


class Planner:
    def __init__(self, cfg, calibration: dict):
        self.cfg = cfg
        self.cal = calibration
        self.state = "CRUISE"
        self.prev_target_col: Optional[int] = None

    def plan(self, per, tracks: list, fps: float) -> Plan:
        cfg = self.cfg
        fps = max(fps, 6.0)
        car_x, car_y = per.car_x, cfg.bev_h
        NC, NR = cfg.grid_cols, cfg.grid_rows
        rl = per.road_left + cfg.road_edge_margin_px
        rr = per.road_right - cfg.road_edge_margin_px
        if rr - rl < 20:                       # no usable road this frame
            return Plan("CRUISE", car_x, True, False, None, 0)

        col_x = [rl + (c + 0.5) / NC * (rr - rl) for c in range(NC)]
        row_y = [car_y * (1.0 - (r + 0.5) / NR) for r in range(NR)]  # r0 near..far
        lookahead = (self.cal["latency_ms"] / 1000.0 + cfg.lookahead_extra_s) * fps

        # --- occupancy grid (obstacles inflated by our own size) ---
        blocked = [[False] * NC for _ in range(NR)]
        infl = cfg.player_half_px + cfg.safety_margin_px
        n_threats = 0
        for tr in tracks:
            px, py = tr.predict(lookahead)      # project BOTH axes by input latency
            if car_y - tr.y > 0:
                n_threats += 1
            x0, x1 = px - tr.w / 2 - infl, px + tr.w / 2 + infl
            y0, y1 = py - tr.h - cfg.pad_y_px, py + cfg.pad_y_px
            for r in range(NR):
                if y0 <= row_y[r] <= y1:
                    row = blocked[r]
                    for c in range(NC):
                        if x0 <= col_x[c] <= x1:
                            row[c] = True

        # --- shortest-path (DP) from our column forward ---
        start_c = min(range(NC), key=lambda c: abs(col_x[c] - car_x))
        S = cfg.max_col_step

        def span_clear(row: int, a: int, b: int) -> bool:
            """No blocked cell between columns a..b on `row` (the cells a diagonal
            step sweeps across). Without this the route clips a car's corner."""
            lo, hi = (a, b) if a < b else (b, a)
            return not any(blocked[row][k] for k in range(lo, hi + 1))

        cost = [[INF] * NC for _ in range(NR)]
        par = [[-1] * NC for _ in range(NR)]
        for c in range(NC):
            if not blocked[0][c] and abs(c - start_c) <= S and span_clear(0, start_c, c):
                cost[0][c] = abs(c - start_c)
                par[0][c] = start_c
        for r in range(1, NR):
            prev = cost[r - 1]
            for c in range(NC):
                if blocked[r][c]:
                    continue
                best, bp = INF, -1
                for c2 in range(max(0, c - S), min(NC, c + S + 1)):
                    if prev[c2] == INF:
                        continue
                    # the step must not cross a car on either row it touches
                    if not span_clear(r, c2, c) or not span_clear(r - 1, c2, c):
                        continue
                    # lateral moves cost MORE the farther out they happen, so the
                    # route front-loads its shift: it gets into the gap lane early
                    # instead of driving straight and swerving at the last row.
                    cand = prev[c2] + abs(c - c2) * (1.0 + cfg.early_bias * r / NR)
                    if cand < best:
                        best, bp = cand, c2
                if bp >= 0:
                    stick = 0.0
                    if self.prev_target_col is not None:
                        stick = cfg.path_stick_bias * abs(c - self.prev_target_col) / NC
                    cost[r][c] = best + 0.04 * abs(c - start_c) + stick
                    par[r][c] = bp

        # deepest row the route reaches
        depth = 0
        for r in range(NR):
            if any(cost[r][c] < INF for c in range(NC)):
                depth = r
        end_row = depth
        free_end = [c for c in range(NC) if cost[end_row][c] < INF]
        if not free_end:
            # even the first step is blocked: hold position, brake
            self.state = "BRAKE_WAIT"
            self.prev_target_col = start_c
            return Plan("BRAKE_WAIT", car_x, False, True, None, n_threats, [], 0)
        # tie-break toward last frame's route so symmetric gaps don't flip-flop
        ref = self.prev_target_col if self.prev_target_col is not None else start_c
        end_c = min(free_end, key=lambda c: (round(cost[end_row][c], 3), abs(c - ref)))

        # backtrack the route to a list of columns per row
        cols = []
        r, c = end_row, end_c
        while r >= 0 and c >= 0:
            cols.append(c)
            c = par[r][c]
            r -= 1
        cols.reverse()
        path_x = [col_x[c] for c in cols]

        # --- steering: aim at the route a couple rows ahead ---
        ti = min(cfg.steer_target_row, len(cols) - 1)
        target_col = cols[ti]
        target_x = col_x[target_col]
        self.prev_target_col = cols[min(1, len(cols) - 1)]

        err = target_x - car_x
        if abs(err) < cfg.steer_deadband_px:
            steer = None
        else:
            steer = RIGHT if err > 0 else LEFT   # target to the right -> press D

        # --- speed: only ease off when the road ahead genuinely tightens ---
        if depth <= cfg.brake_depth:
            brake, gas = True, False
            self.state = "BRAKE_WAIT"
        elif depth < cfg.slow_depth:
            brake, gas = False, False            # coast, it is getting tight
            self.state = "SLOW"
        else:
            brake, gas = False, True
            self.state = "CHANGE" if steer else "CRUISE"

        return Plan(self.state, target_x, gas, brake, steer, n_threats, path_x, depth)
