"""Planning + control: dodge by gaps, in the fixed bird's-eye view.

The car sits at a fixed spot (bottom-center). For any lateral column we can ask
"how far ahead is the nearest car in it?" and drive toward the emptiest reachable
column, committing until our own column is clear again. Looking at the WHOLE
column ahead (not just the point beside us) is what stops it from diving into a
lane that has another car further up.

  - Column ahead clear past `safe_ahead_px`: hold gas, don't steer.
  - Otherwise aim for the reachable column (left / straight / right) with the most
    room, and hold that steer key. To switch sides it must be clearly better, so
    it commits instead of dithering.
  - Nothing reachable has room (`brake_ahead_px`): brake to buy time.

Held controls; D/RIGHT moves the car right, A/LEFT left. Obstacles are projected
forward by the measured input latency before deciding.
"""
from dataclasses import dataclass
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


class Planner:
    def __init__(self, cfg, calibration: dict):
        self.cfg = cfg
        self.cal = calibration
        self.state = "CRUISE"
        self.dodge_dir: Optional[str] = None

    def plan(self, per, tracks: list, fps: float) -> Plan:
        cfg = self.cfg
        fps = max(fps, 6.0)
        car_x = per.car_x
        car_y = cfg.bev_h
        lookahead = (self.cal["latency_ms"] / 1000.0 + cfg.lookahead_extra_s) * fps

        # Predicted obstacle positions (x, nearest-edge y, half width).
        obs = []
        for tr in tracks:
            px, _py = tr.predict(lookahead)
            dist = car_y - tr.y
            if dist > 0:
                obs.append((px, dist, tr.w / 2.0))

        def clear_ahead(center: float) -> float:
            """Distance to the nearest car whose body overlaps this column."""
            best = INF
            for (x, dist, hw) in obs:
                if abs(x - center) < cfg.path_half_px + hw:
                    best = min(best, dist)
            return best

        def on_road(x: float) -> bool:
            return (per.road_left + cfg.road_edge_margin_px
                    < x < per.road_right - cfg.road_edge_margin_px)

        center_clear = clear_ahead(car_x)
        n_threats = sum(1 for (_x, d, _h) in obs if d < cfg.safe_ahead_px)

        steer = None
        brake = False
        if center_clear >= cfg.safe_ahead_px:
            self.dodge_dir = None
            self.state = "CRUISE"
        else:
            left_x, right_x = car_x - cfg.dodge_shift_px, car_x + cfg.dodge_shift_px
            left = clear_ahead(left_x) if on_road(left_x) else -1.0
            right = clear_ahead(right_x) if on_road(right_x) else -1.0

            # Commit: keep the current dodge side unless the other is clearly better.
            if self.dodge_dir == "L" and left > cfg.brake_ahead_px:
                chosen = "L"
            elif self.dodge_dir == "R" and right > cfg.brake_ahead_px:
                chosen = "R"
            else:
                chosen = None
                best_side = max(left, right)
                if best_side > center_clear * cfg.dodge_gain and best_side > cfg.brake_ahead_px:
                    chosen = "L" if left >= right else "R"

            if chosen is None:
                if center_clear < cfg.brake_ahead_px and max(left, right) < cfg.brake_ahead_px:
                    self.state = "BRAKE_WAIT"
                    brake = True
                else:
                    self.state = "CRUISE"   # ride it out; nowhere better to go yet
                self.dodge_dir = None
            else:
                self.state = "CHANGE"
                self.dodge_dir = chosen
                steer = RIGHT if chosen == "R" else LEFT

        # Speed control: only floor it when the road directly ahead is open.
        # In traffic we coast (no gas, no brake) so the car sheds a little speed
        # and there is time to actually complete a dodge. This is the difference
        # between surviving and rocketing into the first gap that closes.
        gas = (not brake) and center_clear >= cfg.safe_ahead_px
        if steer == RIGHT:
            target_x = car_x + cfg.dodge_shift_px
        elif steer == LEFT:
            target_x = car_x - cfg.dodge_shift_px
        else:
            target_x = car_x
        return Plan(self.state, target_x, gas, brake, steer, n_threats)
