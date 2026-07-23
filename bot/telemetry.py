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
        rec = {
            "t": round(time.perf_counter(), 3),
            "fps": round(fps, 1),
            "auto": autopilot,
            "state": plan.state if plan else None,
            "keys": keys,
            "own_lane": per.own_lane,
            "own_x": round(per.own_x, 1),
            "own_vx": round(per.own_vx, 2),
            "dy": round(per.dy, 2),
            "edge_q": round(per.edge_quality, 2),
            "lanes": [round(c, 1) for c in per.lane_centers],
            "clear": [None if c == float("inf") else round(c, 1)
                      for c in (plan.clear_dists if plan else [])],
            "safe": round(plan.safe_dist, 1) if plan else None,
            "target_lane": plan.target_lane if plan else None,
            "n_blobs": len(per.blobs),
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
