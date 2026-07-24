"""Offline replay: run the planner against a recorded HUMAN session and compare
its decisions to what the human actually did, frame by frame. Tunes the planner
against real play with no live runs.

    .venv\\Scripts\\python.exe tools_replay.py records\\<session>
"""
import json
import sys
import types
import collections

from bot.config import Config
from bot.tracker import Tracker
from bot.planner import Planner
from bot.controls import LEFT, RIGHT


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def evaluate(rows, cfg):
    tracker = Tracker(cfg)
    planner = Planner(cfg, {"latency_ms": 130.0, "steer_coast_s": 0.04,
                            "steer_vmax_px_s": 200.0})
    agree = collections.Counter()
    total = collections.Counter()
    bot_steer = danger = close = 0
    car_y = cfg.bev_h
    for r in rows:
        hk = r["keys"]
        human = "a" if hk["a"] and not hk["d"] else ("d" if hk["d"] and not hk["a"] else "-")
        obs = [(o[0], o[1], o[2], 28.0, o[2] * 28.0) for o in r["obstacles"]]
        tracks = tracker.update(obs)
        per = types.SimpleNamespace(car_x=r["car_x"], road_left=r["road"][0],
                                    road_right=r["road"][1], obstacles=obs,
                                    road_quality=1.0, masks=None)
        plan = planner.plan(per, tracks, r.get("fps", 30.0) or 30.0)
        bot = "a" if plan.steer_key == LEFT else ("d" if plan.steer_key == RIGHT else "-")
        total[human] += 1
        if bot == human:
            agree[human] += 1
        if bot in "ad":
            bot_steer += 1
        # a car actually close in the car's own column (real threat this frame)
        nic = min((car_y - o[1] for o in obs
                   if abs(o[0] - r["car_x"]) < cfg.path_half_px + o[2] / 2
                   and car_y - o[1] > 0), default=1e9)
        if nic < 110:
            close += 1
            if bot == "-":
                danger += 1
    n = sum(total.values())
    sf = total["a"] + total["d"]
    return {
        "straight": 100 * agree["-"] // max(total["-"], 1),
        "match": 100 * (agree["a"] + agree["d"]) // max(sf, 1),
        "react": 100 * (close - danger) // max(close, 1),
        "danger": danger,
        "bot_steer": 100 * bot_steer // n,
        "human_steer": 100 * sf // n,
    }


def run(session_dir):
    rows = load(session_dir.rstrip("/\\") + "/play.jsonl")
    print(f"replayed {len(rows)} frames from {session_dir}")
    print(f"human steers {evaluate(rows, Config())['human_steer']}% of frames\n")
    print(f"{'minAge':>6} {'go_str':>6} {'earlyB':>6} | {'straight%':>9} "
          f"{'match%':>6} {'react-to-close%':>15} {'botSteer%':>9} {'danger':>6}")
    for ma in (1, 2):
        for gs in (150.0, 210.0, 270.0):
            for eb in (0.5, 1.5):
                cfg = Config()
                cfg.track_min_age = ma
                cfg.go_straight_px = gs
                cfg.early_bias = eb
                m = evaluate(rows, cfg)
                print(f"{ma:>6} {gs:>6.0f} {eb:>6} | {m['straight']:>9} {m['match']:>6} "
                      f"{m['react']:>15} {m['bot_steer']:>9} {m['danger']:>6}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "records")
