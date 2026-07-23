"""Per-frame JSONL telemetry + optional annotated frame dumps.

Everything lands in runs/<timestamp>/ so a bad run can be replayed and
diagnosed offline.
"""
import json
import os
import time
from datetime import datetime

import cv2


class Telemetry:
    def __init__(self, cfg):
        self.cfg = cfg
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(cfg.runs_dir, stamp)
        os.makedirs(self.dir, exist_ok=True)
        self._f = open(os.path.join(self.dir, "telemetry.jsonl"), "w",
                       encoding="utf-8", buffering=1)
        self._frame_i = 0
        print(f"[telemetry] logging to {self.dir}")

    def log(self, per, plan, fps: float, autopilot: bool, keys: str,
            proc_ms: float = 0.0) -> None:
        # Everything is coerced to plain float/int: numpy scalars (float32) reach
        # here from the vision math and json.dumps cannot serialize them.
        rec = {
            "t": round(time.perf_counter(), 3),
            "fps": round(float(fps), 1),
            "proc_ms": round(float(proc_ms), 1),
            "auto": bool(autopilot),
            "state": plan.state if plan else None,
            "keys": keys,
            "steer": plan.steer_key if plan else None,
            "gas": bool(plan.gas) if plan else None,
            "brake": bool(plan.brake) if plan else None,
            "n_threats": int(plan.n_threats) if plan else 0,
            "car_x": round(float(per.car_x), 1),
            "road": [round(float(per.road_left), 1), round(float(per.road_right), 1)],
            "road_q": round(float(per.road_quality), 2),
            "obstacles": [[round(float(o[0]), 1), round(float(o[1]), 1),
                           round(float(o[2]), 1)] for o in per.obstacles],
        }
        self._f.write(json.dumps(rec) + "\n")

    def maybe_dump_frame(self, annotated, bev) -> None:
        n = self.cfg.debug_frame_every
        self._frame_i += 1
        if n and self._frame_i % n == 0:
            cv2.imwrite(os.path.join(self.dir, f"f{self._frame_i:06d}.jpg"), annotated)
            cv2.imwrite(os.path.join(self.dir, f"f{self._frame_i:06d}_bev.jpg"), bev)

    def close(self) -> None:
        self._f.close()


class Recorder:
    """Logs a human play session: the perceived state plus the real keys the
    player held that frame. This is the ground truth for how a person actually
    drives (fine A/D corrections, gas/brake rhythm) and for the input->motion
    response (keys vs own_x in the same row)."""

    def __init__(self, cfg):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(cfg.runs_dir, "..", "records", stamp)
        self.dir = os.path.normpath(self.dir)
        os.makedirs(self.dir, exist_ok=True)
        self._f = open(os.path.join(self.dir, "play.jsonl"), "w",
                       encoding="utf-8", buffering=1)
        self.n = 0
        print(f"[record] logging to {self.dir}")

    def log(self, per, keys: dict, fps: float) -> None:
        rec = {
            "t": round(time.perf_counter(), 3),
            "fps": round(float(fps), 1),
            "keys": {k: bool(v) for k, v in keys.items()},
            "car_x": round(float(per.car_x), 1),
            "road": [round(float(per.road_left), 1), round(float(per.road_right), 1)],
            # obstacles in fixed BEV: center x, nearest edge y, width
            "obstacles": [[round(float(o[0]), 1), round(float(o[1]), 1),
                           round(float(o[2]), 1)] for o in per.obstacles],
        }
        self._f.write(json.dumps(rec) + "\n")
        self.n += 1

    def close(self) -> None:
        self._f.close()
        print(f"[record] saved {self.n} frames to {self.dir}")
