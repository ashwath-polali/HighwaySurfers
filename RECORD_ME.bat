@echo off
title HighwaySurfers - record a human play session
cd /d "%~dp0"

echo ============================================================
echo   Record YOU playing (for the bot to learn from)
echo ============================================================
echo   This sends NO input. You play; it just watches and logs
echo   your keys + the screen while the game window is focused.
echo.
echo   1. Have the game open and start a run.
echo   2. Play normally. Crashing is fine, just keep playing.
echo   3. Press F9 when you are done to save the recording.
echo.
echo   Do a few runs. It all saves under records\.
echo ============================================================
echo.
pause

.venv\Scripts\python.exe run.py record
pause
