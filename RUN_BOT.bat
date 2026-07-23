@echo off
title HighwaySurfers autopilot
cd /d "%~dp0"

echo ============================================================
echo   HighwaySurfers autopilot
echo ============================================================
echo   Before this starts:
echo     1. Have the game open and IN A RUN (or on the menu).
echo     2. Close the People/leaderboard panel (the X on the right).
echo.
echo   Once it starts it drives on its own and restarts after a
echo   crash (Menu -^> Play, never the paid Revive).
echo.
echo     F8 = pause / resume the bot
echo     F9 = stop and quit (releases all keys)
echo ============================================================
echo.
pause

.venv\Scripts\python.exe run.py drive
pause
