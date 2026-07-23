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

    def log(self, per, plan, fps: float, autopilot: bool, keys: str) -> None:
        # Everything is coerced to plain float/int: numpy scalars (float32) reach
        # here from the vision math and json.dumps cannot serialize them.
        rec = {
            "t": round(time.perf_counter(), 3),
            "fps": round(float(fps), 1),
            "auto": bool(autopilot),
            "state": plan.state if plan else None,
            "keys": keys,
            "own_lane": int(per.own_lane),
            "own_x": round(float(per.own_x), 1),
            "own_vx": round(float(per.own_vx), 2),
            "dy": round(float(per.dy), 2),
            "edge_q": round(float(per.edge_quality), 2),
            "lanes": [round(float(c), 1) for c in per.lane_centers],
            "clear": [None if c == float("inf") else round(float(c), 1)
                      for c in (plan.clear_dists if plan else [])],
            "safe": round(float(plan.safe_dist), 1) if plan else None,
            "target_lane": int(plan.target_lane) if plan else None,
            "steer_tap": round(float(plan.steer_tap_ms), 0) if plan else 0,
            "n_blobs": int(len(per.blobs)),
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
