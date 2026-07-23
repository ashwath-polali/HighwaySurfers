# HighwaySurfers — Roblox highway autopilot

Watches the Roblox window (screen capture only — nothing touches the game
process) and drives with synthetic W/A/S/D. Perception is classic OpenCV:
the road is gray, lines are white, so anything else on the road is a car.

## Setup (once)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Game setup, every session

- Roblox **windowed** (don't move/resize the window while the bot runs)
- Graphics quality **level 1** (flat game — looks the same, halves your latency)
- Close the leaderboard/people panel (it covers the road)
- Default chase camera

## Workflow

```powershell
# 1. Sanity: does perception see what you see? (sends no keys)
.\.venv\Scripts\python.exe run.py view
#    -> yellow trapezoid should bracket the road, red boxes on traffic cars.
#    q quits. `run.py shot` saves debug_frame.jpg / debug_bev.jpg instead.

# 2. Measure YOUR machine's input dead time (car cruising on open road first)
.\.venv\Scripts\python.exe run.py probe

# 3. Measure steering response (same setup)
.\.venv\Scripts\python.exe run.py calibrate

# 4. Drive: just double-click drive.bat with Roblox open. Autopilot starts
#    immediately, clicks through the death screen (Menu -> Play, never the
#    Robux Revive) and keeps farming runs. F8 pauses, F9 panic-quits.
.\.venv\Scripts\python.exe run.py drive
#    add --overlay to watch the bot's bird's-eye view live
#    add --no-autostart to arm it manually with F8
```

Every drive run logs `runs/<timestamp>/telemetry.jsonl` for offline diagnosis.

## Tuning

Everything lives in `bot/config.py` (screen regions as fractions, color
thresholds, planner distances). `run.py view` is the feedback loop: tweak,
watch, repeat. `calibration.json` holds measured latency + steering model.
