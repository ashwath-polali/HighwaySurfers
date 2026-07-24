"""All tunables in one place.

Coordinates that describe screen layout are expressed as FRACTIONS of the
captured client area (0..1) so they survive window resizes. Tune them with
`python run.py view` which draws every region on screen.
"""
from dataclasses import dataclass, field
import json
import os

# Project root = the folder holding this package, so calibration + telemetry
# land in the same place no matter which directory the bot is launched from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # ---- capture ----
    # OS window title to capture; must match the player window's title exactly.
    window_title: str = "Roblox"
    target_fps: int = 60

    # ---- road trapezoid (fractions of client area) ----
    # Generous on purpose: the camera pans as the car changes lanes, so the
    # trapezoid must contain the road at every lateral offset. Grass that
    # sneaks in is rejected later by the road-span detector.
    trap_bottom_y: float = 0.97
    # Reach further toward the horizon so traffic is seen well before impact.
    # Grass pulled in at the corners is rejected by the road-region detector.
    trap_top_y: float = 0.18
    trap_bottom_left_x: float = 0.05
    trap_bottom_right_x: float = 0.95
    trap_top_left_x: float = 0.30
    trap_top_right_x: float = 0.70

    # ---- bird's-eye view (BEV) raster ----
    bev_w: int = 240
    bev_h: int = 360

    # ---- masks (fractions of client area) ----
    # Own car + speed/distance HUD live here; never treat as obstacle.
    own_car_box: tuple = (0.40, 0.66, 0.60, 1.00)  # l, t, r, b
    # Top HUD strip (coins / timer / gems).
    top_hud_box: tuple = (0.00, 0.00, 1.00, 0.10)

    # ---- color thresholds (HSV, uint8 ranges) ----
    road_sat_max: int = 70
    road_val_min: int = 35
    road_val_max: int = 175
    white_sat_max: int = 70
    white_val_min: int = 190

    # ---- perception ----
    min_blob_area: int = 22          # BEV pixels
    road_span_row_frac: float = 0.30  # bottom fraction of BEV rows used to find road span
    road_span_gray_frac: float = 0.55  # column is "road" if >= this frac of sampled rows is gray
    lane_width_guess_px: float = 44.0  # BEV px; refined online
    # Hard bounds on lane width so the estimate can't run away. Without them the
    # width EMA collapsed toward tiny gaps (thick/aliased lines read as many
    # peaks), yielding 20-32 phantom lanes and a garbage lane model.
    min_lane_width_px: float = 34.0
    max_lane_width_px: float = 64.0
    max_lateral_shift: int = 24      # px/frame search window for lateral flow
    max_forward_shift: int = 80      # px/frame search window for forward flow

    # ---- tracking ----
    # Anisotropic gate: cars barely move across lanes frame-to-frame but close
    # FAST along y (BEV perspective magnifies distant motion). A circular gate
    # drops fast closers and their velocity never gets estimated.
    track_match_x: float = 26.0      # px gate across lanes
    track_match_y: float = 110.0     # px gate along travel direction
    track_max_missed: int = 5
    track_min_age: int = 1
    vel_ema_alpha: float = 0.45

    # ---- planning: time-to-collision based (distances in BEV px) ----
    # We react to how SOON a car arrives rather than how far it is, so behavior
    # scales with speed. The bot commits to a lane and only re-plans when the
    # current lane is actually threatened, instead of chasing the best lane.
    lookahead_extra_s: float = 0.20   # prediction margin on top of measured latency
    # Collision timing comes from each obstacle's tracked closing speed, not the
    # dashed-line forward flow (that aliases and reads ~0). A very close obstacle
    # is a threat regardless of measured closing speed.
    # ---- path planner (grid search over the road ahead; fixed BEV px) ----
    road_edge_margin_px: float = 16.0  # keep this far inside the road edge
    grid_cols: int = 23               # lateral resolution of the plan
    grid_rows: int = 12               # depth (lookahead) resolution
    player_half_px: float = 16.0      # our car half-width, for obstacle inflation
    safety_margin_px: float = 4.0     # extra clearance around each car
    pad_y_px: float = 16.0            # obstacle length padding along travel
    max_col_step: int = 3             # most columns the route may shift per row
    early_bias: float = 0.5           # front-load moves: pre-position into the gap
                                      # early instead of swerving at the last row
    steer_target_row: int = 3         # steer toward the route point this far ahead
    steer_deadband_px: float = 12.0   # within this of the route -> don't steer
    path_half_px: float = 26.0        # half-width of the column that must be clear
    go_straight_px: float = 190.0     # nearest car in our column farther than this
                                      # -> just hold straight (like a human)
    wall_band_px: float = 55.0        # cars within this depth of the closest = one wall
    min_center_gap: float = 5.0       # min car-center room in an (inflated) gap
    overshoot_buf: float = 9.0        # aim this far inside a gap edge (overshoot room)
    reach_ratio: float = 1.0          # px lateral reachable per px forward (between walls)
    gap_hold_bonus: float = 3.0       # commit to the gap we are already threading
    gap_hold_tol: float = 8.0         # ref counts as inside a gap within this
    close_px: float = 90.0            # wall this close + still needing to move -> ease gas
    slow_err_px: float = 60.0         # ease gas while target is farther than this (big swing)
    # Route reaches at least slow_depth rows -> full gas; between brake and slow
    # -> coast; brake_depth or fewer -> brake. (grid has grid_rows rows.)
    brake_depth: int = 0              # only a fully blocked road -> brake
    slow_depth: int = 3               # route shallower than this -> coast (ease off)
    path_stick_bias: float = 0.9      # prefer last frame's route (kills flip-flop)

    # ---- default calibration (overridden by calibration.json) ----
    latency_ms_default: float = 180.0
    steer_vmax_px_s_default: float = 130.0   # lateral speed while key held (rectified px/s)
    steer_coast_s_default: float = 0.12      # coast distance = v * this after release

    # ---- telemetry ----
    runs_dir: str = os.path.join(_ROOT, "runs")
    debug_frame_every: int = 0       # 0 = off; N = dump annotated frame every N frames

    calibration_path: str = os.path.join(_ROOT, "calibration.json")


def load_calibration(cfg: Config) -> dict:
    """Returns calibration dict with defaults filled in."""
    cal = {
        "latency_ms": cfg.latency_ms_default,
        "steer_vmax_px_s": cfg.steer_vmax_px_s_default,
        "steer_coast_s": cfg.steer_coast_s_default,
    }
    if os.path.exists(cfg.calibration_path):
        try:
            with open(cfg.calibration_path, "r", encoding="utf-8") as f:
                cal.update(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[config] failed to read {cfg.calibration_path}: {e}; using defaults")
    return cal


def save_calibration(cfg: Config, updates: dict) -> None:
    cal = {}
    if os.path.exists(cfg.calibration_path):
        try:
            with open(cfg.calibration_path, "r", encoding="utf-8") as f:
                cal = json.load(f)
        except (json.JSONDecodeError, OSError):
            cal = {}
    cal.update(updates)
    with open(cfg.calibration_path, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print(f"[config] saved calibration -> {cfg.calibration_path}")
