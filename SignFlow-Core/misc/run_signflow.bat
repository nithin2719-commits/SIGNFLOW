@echo off
setlocal
cd /d "%~dp0"

start "SignFlow Overlay" cmd /k python overlay.py
start "SignFlow Realtime Sender" cmd /k python realtime_sender.py