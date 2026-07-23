@echo off
rem Double-click me with Roblox open. F8 pauses the bot, F9 kills it.
cd /d "%~dp0"
.venv\Scripts\python.exe run.py drive %*
pause
