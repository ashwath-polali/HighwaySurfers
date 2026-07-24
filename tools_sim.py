"""Closed-loop simulator: run the real Tracker+Planner against a synthetic
highway (walls of cars with gaps) using the car's measured steering physics, so
driving quality (survival, jitter, collisions) can be tuned with no live runs.

    .venv\\Scripts\\python.exe tools_sim.py

The car is fixed at BEV center; the world scrolls toward it. Pressing D moves the
car right (world x up), A moves it left, with input latency + a velocity ramp.
"""
import types
import random
import statistics

from bot.config import Config
from bot.tracker import Tracker
from bot.planner import Planner
from bot.controls import LEFT, RIGHT

BEVW, BEVH = 240, 360
LOOKAHEAD = 380.0            # world forward distance the BEV covers
FPS = 30.0
DT = 1.0 / FPS
VMAX = 6.5                   # px/frame lateral (~195 px/s, from recordings)
TAU_ON, TAU_OFF = 0.14, 0.10
LAT_FRAMES = round(0.13 * FPS)
FWD_FULL = 6.5              # world/frame at full gas
CAR_HALF, CAR_LEN = 16.0, 42.0
CAL = {"latency_ms": 130.0, "steer_coast_s": 0.04, "steer_vmax_px_s": 195.0}


def make_walls(density, n=80):
    walls = []
    y = 260.0
    for _ in range(n):
        # gapw is car-center spacing across the gap; free space = gapw - 34 (car
        # body). Keep gaps passable (free space >= ~car width) like the real game.
        if density == "hard":
            gapw = random.uniform(76, 98); step = random.uniform(150, 240)
        else:
            gapw = random.uniform(110, 165); step = random.uniform(240, 380)
        gapx = random.uniform(45, 195)
        x = 8.0
        while x < 235:
            if abs(x - gapx) < gapw / 2:
                x = gapx + gapw / 2 + 2
                continue
            walls.append([x + 17, y, 34.0])
            x += 40
        y += step
    return walls


def sim(cfg, density, seed, max_frames=4000):
    random.seed(seed)
    walls = make_walls(density)
    tracker, planner = Tracker(cfg), Planner(cfg, CAL)
    cx, vx, CY = 120.0, 0.0, 0.0
    queue, flips, last = [], 0, None
    for frame in range(max_frames):
        bev = []
        for ox, oy, ow in walls:
            d = oy - CY
            if 0 < d < LOOKAHEAD:
                bx = 120 + (ox - cx)
                by = BEVH * (1 - d / LOOKAHEAD)
                if -ow < bx < BEVW + ow:
                    bev.append((bx, by, ow, 28.0, ow * 28))
        rl, rr = max(0.0, 120 - cx), min(240.0, 120 + (240 - cx))
        tracks = tracker.update(bev)
        per = types.SimpleNamespace(car_x=120.0, road_left=rl, road_right=rr,
                                    obstacles=bev, road_quality=1.0, masks=None)
        plan = planner.plan(per, tracks, FPS)
        s = "a" if plan.steer_key == LEFT else ("d" if plan.steer_key == RIGHT else "-")
        if s in "ad" and last is not None and s != last:
            flips += 1
        if s in "ad":
            last = s

        queue.append(s)
        eff = queue.pop(0) if len(queue) > LAT_FRAMES else "-"
        des = VMAX if eff == "d" else (-VMAX if eff == "a" else 0.0)
        tau = TAU_ON if eff in "ad" else TAU_OFF
        vx += (des - vx) * (DT / tau)
        cx = max(CAR_HALF, min(240 - CAR_HALF, cx + vx))
        CY += FWD_FULL * (1.0 if plan.gas else 0.72)

        for ox, oy, ow in walls:
            if abs(oy - CY) < CAR_LEN / 2 + 12 and abs(ox - cx) < CAR_HALF + ow / 2:
                return CY, flips, frame, "crash"
    return CY, flips, frame, "survived"


def bench(cfg, label):
    for density in ("easy", "hard"):
        dist, fl, crashes = [], [], 0
        for seed in range(12):
            d, f, fr, how = sim(cfg, density, seed)
            dist.append(d); fl.append(f)
            if how == "crash":
                crashes += 1
        print(f"  {label:16} {density:5}: median dist {statistics.median(dist):6.0f}  "
              f"crashed {crashes}/12  jitter(flips) med {statistics.median(fl):.0f}")


def bench_hard(cfg, seeds=16):
    dist, crashes = [], 0
    for seed in range(seeds):
        d, f, fr, how = sim(cfg, "hard", seed)
        dist.append(d)
        if how == "crash":
            crashes += 1
    return statistics.median(dist), crashes


if __name__ == "__main__":
    print("closed-loop sim:")
    bench(Config(), "current")
    print("\nsweep (hard, 16 seeds) -- inflation & max_col_step:")
    print(f"{'p_half':>6} {'safety':>6} {'colStep':>7} {'go_str':>6} | {'medDist':>7} {'crash/16':>8}")
    for ph, sm in ((20, 7), (14, 4), (10, 2)):
        for cs in (3, 4, 6):
            for gs in (190.0, 260.0):
                cfg = Config()
                cfg.player_half_px = ph
                cfg.safety_margin_px = sm
                cfg.max_col_step = cs
                cfg.go_straight_px = gs
                md, cr = bench_hard(cfg)
                print(f"{ph:>6} {sm:>6} {cs:>7} {gs:>6.0f} | {md:>7.0f} {cr:>8}")
