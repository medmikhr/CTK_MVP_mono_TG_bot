@echo off
cd /d %~dp0
call .venv\Scripts\activate
python bot_agent.py
pause 