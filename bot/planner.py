"""Planning + control: dodge by gaps, in the fixed bird's-eye view.

The car is at a fixed spot (bottom-center). Each obstacle has a position and a
tracked closing speed. The logic is deliberately simple and hard to break:

  - If nothing is in our column ahead: hold gas, don't steer.
  - If a car is in our column: pick the nearer clear side (left/right), commit to
    it, and hold that steer key until our column is clear again. Holding until
    clear means the swing lasts exactly as long as the danger does.
  - If both sides are blocked: brake to buy time.

Controls are HELD (gas to keep speed, brake only in the pinch, steer toward the
gap). Physical key directions: D/RIGHT moves the car right, A/LEFT moves it left.
Obstacles are projected forward by the measured input latency before deciding, so
we plan against where the world will be when the keypress lands.
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

        # Threats: obstacles ahead that are near, or closing fast enough to matter.
        threats = []   # (x_pred, dist, half_width)
        for tr in tracks:
            px, _py = tr.predict(lookahead)
            dist = car_y - tr.y
            if dist <= 0:
                continue
            closing = tr.vy * fps
            ttc = dist / closing if closing > cfg.min_closing_px_s else INF
            if dist < cfg.react_dist_px or ttc < cfg.ttc_danger_s:
                threats.append((px, dist, tr.w / 2.0))

        def blocked(center: float) -> bool:
            return any(abs(x - center) < cfg.path_half_px + hw
                       for (x, _d, hw) in threats)

        steer = None
        brake = False
        if not blocked(car_x):
            self.dodge_dir = None
            self.state = "CRUISE"
        else:
            left_x = car_x - cfg.dodge_shift_px
            right_x = car_x + cfg.dodge_shift_px
            left_ok = (not blocked(left_x)
                       and left_x > per.road_left + cfg.road_edge_margin_px)
            right_ok = (not blocked(right_x)
                        and right_x < per.road_right - cfg.road_edge_margin_px)

            # keep a committed dodge if that side is still clear (avoid flip-flop)
            if self.dodge_dir == "L" and left_ok:
                chosen = "L"
            elif self.dodge_dir == "R" and right_ok:
                chosen = "R"
            elif left_ok and right_ok:
                nearest = min(threats, key=lambda t: t[1])
                chosen = "R" if nearest[0] <= car_x else "L"  # away from the blocker
            elif left_ok:
                chosen = "L"
            elif right_ok:
                chosen = "R"
            else:
                chosen = None

            self.dodge_dir = chosen
            if chosen is None:
                self.state = "BRAKE_WAIT"
                brake = True
            else:
                self.state = "CHANGE"
                steer = RIGHT if chosen == "R" else LEFT

        gas = not brake
        if steer == RIGHT:
            target_x = car_x + cfg.dodge_shift_px
        elif steer == LEFT:
            target_x = car_x - cfg.dodge_shift_px
        else:
            target_x = car_x
        return Plan(self.state, target_x, gas, brake, steer, len(threats))
