"""Constant-velocity blob tracker in BEV space.

Traffic in this game is lane-locked and near-constant speed, so a greedy
nearest-neighbor association with EMA velocities is enough. No Kalman needed.
Velocities here are RELATIVE closing speeds already (BEV moves with our car),
which is exactly what the planner wants.
"""
from dataclasses import dataclass, field


@dataclass
class Track:
    tid: int
    x: float          # BEV cx
    y: float          # BEV bottom edge (nearest point to us)
    w: float
    h: float
    vx: float = 0.0   # px/frame
    vy: float = 0.0   # px/frame; + = coming toward us (down-screen)
    age: int = 1
    missed: int = 0

    def predict(self, frames: float) -> tuple:
        return self.x + self.vx * frames, self.y + self.vy * frames


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks: list = []
        self._next_id = 1

    def update(self, blobs: list) -> list:
        """blobs: [(cx, cy_bottom, w, h, area), ...] -> live tracks (age-gated)."""
        cfg = self.cfg
        unmatched = list(range(len(blobs)))
        # Greedy: closest (track, blob) pairs first, under an anisotropic gate
        # (tight across lanes, wide along the travel direction).
        pairs = []
        for ti, tr in enumerate(self.tracks):
            px, py = tr.predict(1.0)
            for bi in range(len(blobs)):
                bx, by = blobs[bi][0], blobs[bi][1]
                d = ((px - bx) / cfg.track_match_x) ** 2 \
                    + ((py - by) / cfg.track_match_y) ** 2
                if d <= 1.0:
                    pairs.append((d, ti, bi))
        pairs.sort()
        used_t, used_b = set(), set()
        for d, ti, bi in pairs:
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            tr = self.tracks[ti]
            bx, by, bw, bh, _ = blobs[bi]
            a = cfg.vel_ema_alpha
            tr.vx = (1 - a) * tr.vx + a * (bx - tr.x)
            tr.vy = (1 - a) * tr.vy + a * (by - tr.y)
            tr.x, tr.y, tr.w, tr.h = bx, by, bw, bh
            tr.age += 1
            tr.missed = 0
            if bi in unmatched:
                unmatched.remove(bi)

        # Age out lost tracks.
        survivors = []
        for ti, tr in enumerate(self.tracks):
            if ti in used_t:
                survivors.append(tr)
                continue
            tr.missed += 1
            if tr.missed <= cfg.track_max_missed:
                # coast on prediction so a briefly-occluded car isn't forgotten
                tr.x, tr.y = tr.predict(1.0)
                survivors.append(tr)
        self.tracks = survivors

        # New tracks for unmatched blobs.
        for bi in unmatched:
            bx, by, bw, bh, _ = blobs[bi]
            self.tracks.append(Track(self._next_id, bx, by, bw, bh))
            self._next_id += 1

        # Only surface tracks confirmed across a few frames. The previous
        # `or t.missed == 0` clause let brand-new (age 1) tracks through and
        # made the age gate a no-op, so single-frame color noise could spawn a
        # phantom obstacle and trigger a swerve.
        return [t for t in self.tracks if t.age >= cfg.track_min_age]
