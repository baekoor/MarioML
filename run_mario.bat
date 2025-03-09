@echo off
echo Running Mario Agent Demo
echo =======================

REM Activate virtual environment
call .venv\Scripts\activate

REM Check if virtual environment activated successfully
if errorlevel 1 (
    echo Failed to activate virtual environment.
    echo Make sure you have created the virtual environment.
    exit /b 1
)

REM Find latest checkpoint directory
for /f "tokens=*" %%a in ('dir /b /od /ad .\mario_checkpoints') do set LATEST_DIR=%%a

REM Find latest checkpoint file in that directory
for /f "tokens=*" %%a in ('dir /b /od .\mario_checkpoints\%LATEST_DIR%\mario_model_step_*.pt') do set CHECKPOINT=.\mario_checkpoints\%LATEST_DIR%\%%a

if "%CHECKPOINT%"=="" (
    echo No checkpoints found.
    exit /b 1
)

echo Latest checkpoint: %CHECKPOINT%
echo Running agent with latest checkpoint...

REM Run the agent with the found checkpoint
python run_mario_agent.py --checkpoint "%CHECKPOINT%" --episodes 5

echo Demo complete!
pause
