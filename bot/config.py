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
    trap_top_y: float = 0.30
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
    max_lateral_shift: int = 24      # px/frame search window for lateral flow
    max_forward_shift: int = 80      # px/frame search window for forward flow

    # ---- tracking ----
    # Anisotropic gate: cars barely move across lanes frame-to-frame but close
    # FAST along y (BEV perspective magnifies distant motion). A circular gate
    # drops fast closers and their velocity never gets estimated.
    track_match_x: float = 26.0      # px gate across lanes
    track_match_y: float = 110.0     # px gate along travel direction
    track_max_missed: int = 5
    track_min_age: int = 2
    vel_ema_alpha: float = 0.45

    # ---- planning (distances in BEV px; BEV bottom is the car) ----
    lookahead_extra_s: float = 0.30  # margin on top of measured latency
    safe_dist_speed_k: float = 9.0   # safe_dist = k * forward_flow(px/frame)
    safe_dist_min: float = 70.0
    safe_dist_max: float = 320.0
    brake_dist_frac: float = 0.45    # brake when clear < brake_dist_frac * safe_dist
    change_gain_req: float = 1.15    # target lane must beat current by this factor
    side_margin_ahead: float = 55.0  # blob this close beside us blocks a lane change
    side_margin_behind: float = 25.0
    straddle_frac: float = 0.28      # blob within this frac of lane width of a boundary occupies both lanes

    # ---- steering controller ----
    # The game runs at very high responsiveness, so we nudge with short taps and
    # let the car settle between them instead of holding a key (which oversteers).
    steer_tol_px: float = 6.0        # lane-change completion tolerance
    steer_deadband_px: float = 12.0  # don't steer within this of the target
    steer_tap_min_ms: int = 22       # gentlest nudge
    steer_tap_max_ms: int = 80       # biggest single nudge
    steer_cooldown_s: float = 0.10   # minimum gap between steer taps (let it settle)
    steer_min_edge_q: float = 0.7    # below this the road fit is too noisy to steer on
    brake_tap_ms: int = 140

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
