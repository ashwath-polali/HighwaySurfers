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
        self.prev_target_x: Optional[float] = None

    def plan(self, per, tracks: list, fps: float) -> Plan:
        cfg = self.cfg
        fps = max(fps, 6.0)
        car_x, car_y = per.car_x, cfg.bev_h
        lookahead = (self.cal["latency_ms"] / 1000.0 + cfg.lookahead_extra_s) * fps
        return self._plan_gaps(per, tracks, fps, car_x, car_y, lookahead)

    # ------------------------------------------------------------------
    def _plan_gaps(self, per, tracks, fps, car_x, car_y, lookahead) -> Plan:
        """Thread the gap: find the free lateral intervals in the wall ahead and
        aim at the nearest safe point inside one. Robust where the grid DP gave
        up (it collapsed to no-route when a wall got close)."""
        cfg = self.cfg
        infl = cfg.player_half_px + cfg.safety_margin_px
        rl, rr = per.road_left, per.road_right

        near = []
        for tr in tracks:
            px, _py = tr.predict(lookahead)
            d = car_y - tr.y
            if d > 0:
                near.append((px, d, tr.w))
        n_threats = len(near)
        if not near:
            return Plan("CRUISE", car_x, True, False, None, 0, [car_x, car_x], 6)

        # --- group cars into walls (layers by depth) and find each wall's gaps ---
        layers = []
        for px, d, w in sorted(near, key=lambda o: o[1]):
            if layers and d - layers[-1]["d0"] <= cfg.wall_band_px:
                layers[-1]["obs"].append((px, w))
                layers[-1]["d"] = (layers[-1]["d"] + d) / 2
            else:
                layers.append({"d0": d, "d": d, "obs": [(px, w)]})
        for L in layers:
            L["gaps"] = self._gaps(L["obs"], rl, rr, infl, cfg.min_center_gap)
        d_min = layers[0]["d"]
        center_clear = min((d for px, d, w in near
                            if abs(px - car_x) < cfg.path_half_px + w / 2), default=INF)

        # --- pick the nearest-wall gap that leads deepest through later walls ---
        ref = self.prev_target_x if self.prev_target_x is not None else car_x
        best = self._choose_gap(layers, car_x, ref, cfg)
        if best is None:                          # no fitting gap: aim at widest
            allg = layers[0]["gaps"] or [(rl, rr)]
            widest = max(allg, key=lambda ab: ab[1] - ab[0])
            target_x = (widest[0] + widest[1]) / 2
        else:
            a, b = best
            # Aim so the car ends up safely INSIDE the gap: stay put if it already
            # is, else move to a buffered point (room to absorb overshoot). Buffer
            # shrinks for tight gaps (then we aim at the center).
            buf = min(cfg.overshoot_buf, (b - a) / 2)
            target_x = max(a + buf, min(car_x, b - buf))
        self.prev_target_x = target_x

        err = target_x - car_x
        steer = (RIGHT if err > 0 else LEFT) if abs(err) >= cfg.steer_deadband_px else None
        # Full speed when threading straight; ease off (coast) while making a big
        # swing so there is time to complete it. This is why a human runs ~300 on
        # hard and ~480 on easy without ever braking.
        gas = not (steer is not None and abs(err) > cfg.slow_err_px)
        return Plan("CHANGE" if steer else "CRUISE", target_x, gas, False, steer,
                    n_threats, [car_x, target_x], len(layers))

    # ------------------------------------------------------------------
    @staticmethod
    def _gaps(obs, rl, rr, infl, min_gap):
        """Intervals where the CAR CENTER can safely sit. Cars are inflated by our
        own half-width + safety, so any positive interval already fits the car;
        we only drop slivers narrower than min_gap. (Do not also subtract the car
        width when aiming, or a real gap vanishes.)"""
        blocked = sorted((max(rl, px - w / 2 - infl), min(rr, px + w / 2 + infl))
                         for px, w in obs)
        merged = []
        for a, b in blocked:
            if merged and a <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        free, cur = [], rl
        for a, b in merged:
            if a - cur >= 0:
                free.append((cur, a))
            cur = max(cur, b)
        free.append((cur, rr))
        return [(a, b) for a, b in free if b - a >= min_gap]

    def _choose_gap(self, layers, car_x, ref_x, cfg):
        """Nearest-wall gap that reaches through the most subsequent walls; ties
        broken toward last frame's aim (stability). Reachability: between walls we
        can move ~reach_ratio px laterally per px of forward distance."""
        from functools import lru_cache

        def overlap_reach(g1, d1, g2, d2):
            reach = cfg.reach_ratio * abs(d2 - d1) + 1.0
            return g1[0] - reach <= g2[1] and g2[0] <= g1[1] + reach

        @lru_cache(maxsize=None)
        def depth_from(k, gi):
            if k + 1 >= len(layers):
                return k
            g = layers[k]["gaps"][gi]
            best = k
            for gj, g2 in enumerate(layers[k + 1]["gaps"]):
                if overlap_reach(g, layers[k]["d"], g2, layers[k + 1]["d"]):
                    best = max(best, depth_from(k + 1, gj))
            return best

        gaps0 = layers[0]["gaps"]
        if not gaps0:
            return None
        scored = []
        for gi, g in enumerate(gaps0):
            reach_depth = depth_from(0, gi)
            aim = max(g[0], min(ref_x, g[1]))
            # Strongly COMMIT to the gap we were already threading (the one holding
            # last frame's aim): give it a big reach bonus so we only switch gaps
            # when another reaches several walls deeper. Otherwise the choice
            # flip-flops between gaps and the target swings into a car.
            holding = g[0] - cfg.gap_hold_tol <= ref_x <= g[1] + cfg.gap_hold_tol
            score = reach_depth + (cfg.gap_hold_bonus if holding else 0)
            scored.append((-score, abs(aim - ref_x), abs(aim - car_x), g))
        scored.sort()
        return scored[0][3]

    # ------------------------------------------------------------------
    def _plan_dp(self, per, tracks: list, fps: float) -> Plan:
        cfg = self.cfg
        fps = max(fps, 6.0)
        car_x, car_y = per.car_x, cfg.bev_h
        NC, NR = cfg.grid_cols, cfg.grid_rows
        rl = per.road_left + cfg.road_edge_margin_px
        rr = per.road_right - cfg.road_edge_margin_px
        if rr - rl < 20:                       # no usable road this frame
            return Plan("CRUISE", car_x, True, False, None, 0)
        lookahead = (self.cal["latency_ms"] / 1000.0 + cfg.lookahead_extra_s) * fps

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

        # How close is the nearest car actually in our column right now?
        center_clear = INF
        blocker_x = None
        for tr in tracks:
            px, _py = tr.predict(lookahead)
            d = car_y - tr.y
            if d > 0 and abs(px - car_x) < cfg.path_half_px + tr.w / 2 and d < center_clear:
                center_clear, blocker_x = d, px

        err = target_x - car_x
        if center_clear > cfg.go_straight_px:
            steer = None                         # lane ahead is open: hold straight
        elif abs(err) >= cfg.steer_deadband_px:
            steer = RIGHT if err > 0 else LEFT   # follow the route around the car
        else:
            steer = None
        # Hard safety: a car is close in our lane but the route didn't pick a side.
        # Never coast into it; dodge away from it (toward the emptier side).
        if steer is None and center_clear <= cfg.go_straight_px and blocker_x is not None:
            steer = LEFT if blocker_x >= car_x else RIGHT

        # --- speed: a human NEVER brakes here; they feather the gas by density.
        # Hold gas (accelerate toward top speed) when the route runs deep = open
        # road; ease off (coast) when it is dense so we hold a controllable speed.
        # We do not brake: braking bogs the car down, which is what wrecked runs.
        brake = False
        gas = depth >= cfg.slow_depth
        self.state = "CHANGE" if steer else ("CRUISE" if gas else "SLOW")

        return Plan(self.state, target_x, gas, brake, steer, n_threats, path_x, depth)
