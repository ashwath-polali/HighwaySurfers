"""Measurement routines. Run these BEFORE trusting the bot at speed.

probe     input dead time: inject a steer tap, watch the car's position
          within the road, measure ms until it visibly moves. This number
          feeds every prediction the planner makes.
calibrate steering response: hold left/right for several durations, track
          how far the car slides, extract max lateral speed and post-release
          coast.

Both assume: the game window focused, car already at cruising speed on open
road (pick easy mode / low traffic), bot holds the gas for you during the run.

Everything is measured in rectified road space (own_x_raw, 0..bev_w spanning
the road's full width), the same space the steering controller plans in.
"""
import statistics
import time

import numpy as np

from .controls import Controls, GAS, LEFT, RIGHT
from .config import save_calibration


def _countdown(msg: str, secs: int = 5) -> None:
    print(f"\n=== {msg} ===")
    print("Click into the game window NOW.")
    for i in range(secs, 0, -1):
        print(f"  starting in {i}...")
        time.sleep(1)


def _settle(capture, vision, seconds: float) -> float:
    """Process frames for a while; return the last own_x_raw seen."""
    last = 0.0
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        frame, _ = capture.read()
        if frame is not None:
            last = vision.process(frame).own_x_raw
    return last


def run_probe(cfg, capture, vision, trials: int = 10) -> dict:
    """Measure input->screen dead time via steer taps."""
    controls = Controls()
    _countdown("LATENCY PROBE: bot will hold gas and tap left/right. Hands off!", 5)
    controls.set_key(GAS, True)
    results_ms = []
    fps_samples = []
    try:
        # settle + warm the estimators, sampling fps as we go
        t_end = time.perf_counter() + 2.0
        prev_t = None
        while time.perf_counter() < t_end:
            frame, t = capture.read()
            if frame is None:
                continue
            vision.process(frame)
            if prev_t is not None:
                fps_samples.append(1.0 / max(t - prev_t, 1e-6))
            prev_t = t

        direction = LEFT
        for trial in range(trials):
            baseline = _settle(capture, vision, 0.8)  # quiet, car centered-ish
            sign = -1.0 if direction == LEFT else 1.0  # LEFT lowers own_x

            tap_ms = 130.0
            t0 = time.perf_counter()
            controls.tap(direction, tap_ms)
            detected = None
            deadline = t0 + 1.2
            while time.perf_counter() < deadline:
                controls.update()
                frame, t_frame = capture.read()
                if frame is None:
                    continue
                per = vision.process(frame)
                disp = (per.own_x_raw - baseline) * sign
                if disp > 3.5:  # moved clearly in the commanded direction
                    detected = (t_frame - t0) * 1000.0
                    break
            controls.update()
            if detected is not None:
                results_ms.append(detected)
                print(f"  trial {trial + 1}: {detected:.0f} ms")
            else:
                print(f"  trial {trial + 1}: no response detected (ignored)")

            # recenter with an opposite tap, then alternate to stay near center
            opposite = RIGHT if direction == LEFT else LEFT
            controls.tap(opposite, tap_ms)
            _settle(capture, vision, 0.6)
            direction = opposite
    finally:
        controls.release_all()

    if not results_ms:
        print("PROBE FAILED: no responses detected. Check `view` mode first.")
        return {}
    med = statistics.median(results_ms)
    p90 = sorted(results_ms)[max(0, int(len(results_ms) * 0.9) - 1)]
    fps = statistics.median(fps_samples) if fps_samples else 0.0
    out = {
        "latency_ms": round(med, 1),
        "latency_p90_ms": round(p90, 1),
        "latency_samples_ms": [round(x, 1) for x in results_ms],
        "capture_fps": round(fps, 1),
    }
    print(f"\nDead time: median {med:.0f} ms, p90 {p90:.0f} ms, capture ~{fps:.0f} fps")
    save_calibration(cfg, out)
    return out


def run_steer_calibration(cfg, capture, vision, cal: dict) -> dict:
    """Measure lateral speed while a key is held + coast after release."""
    controls = Controls()
    _countdown("STEER CALIBRATION: bot will hold gas and weave. Hands off!", 5)
    controls.set_key(GAS, True)
    hold_set = [80, 140, 220, 320]
    curves = []
    try:
        _settle(capture, vision, 2.0)
        direction = LEFT
        for hold_ms in hold_set:
            for _rep in range(2):
                baseline = _settle(capture, vision, 0.8)
                sign = -1.0 if direction == LEFT else 1.0

                t0 = time.perf_counter()
                controls.tap(direction, hold_ms)
                samples = []
                while time.perf_counter() < t0 + hold_ms / 1000.0 + 0.9:
                    controls.update()
                    frame, t_frame = capture.read()
                    if frame is None:
                        continue
                    per = vision.process(frame)
                    disp = (per.own_x_raw - baseline) * sign  # + = commanded dir
                    samples.append(((t_frame - t0) * 1000.0, disp))
                controls.update()
                curves.append({"hold_ms": hold_ms,
                               "dir": "L" if direction == LEFT else "R",
                               "samples": [(round(a, 1), round(b, 2)) for a, b in samples]})
                total = samples[-1][1] if samples else 0.0
                print(f"  hold {hold_ms}ms {('L' if direction == LEFT else 'R')}: "
                      f"slid {total:+.1f} px")
                direction = RIGHT if direction == LEFT else LEFT
    finally:
        controls.release_all()

    # ---- extract model ----
    dead_s = cal.get("latency_ms", cfg.latency_ms_default) / 1000.0
    vmaxes, coasts = [], []
    for c in curves:
        s = c["samples"]
        if len(s) < 5:
            continue
        t_arr = np.array([p[0] for p in s]) / 1000.0
        d_arr = np.array([p[1] for p in s])
        hold_s = c["hold_ms"] / 1000.0
        in_hold = (t_arr > dead_s + 0.02) & (t_arr < dead_s + hold_s)
        v_hold = None
        if in_hold.sum() >= 2:
            v_hold = np.polyfit(t_arr[in_hold], d_arr[in_hold], 1)[0]  # px/s
            if v_hold > 10:
                vmaxes.append(v_hold)
        after = t_arr > dead_s + hold_s + 0.02
        if after.sum() >= 3 and in_hold.sum() >= 2 and v_hold and v_hold > 10:
            d_release = d_arr[in_hold][-1]
            d_final = d_arr[-1]
            coasts.append(max(d_final - d_release, 0.0) / v_hold)

    out = {"steer_curves": curves}
    if vmaxes:
        out["steer_vmax_px_s"] = round(float(np.median(vmaxes)), 1)
    if coasts:
        out["steer_coast_s"] = round(float(np.median(coasts)), 3)
    print(f"\nSteer model: vmax={out.get('steer_vmax_px_s', 'n/a')} px/s, "
          f"coast={out.get('steer_coast_s', 'n/a')} s")
    save_calibration(cfg, out)
    return out
