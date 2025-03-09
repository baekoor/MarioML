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

echo Finding latest checkpoint...
python -c "from run_mario_agent import find_latest_checkpoint; checkpoint = find_latest_checkpoint(); print(str(checkpoint) if checkpoint else 'None')" > temp_checkpoint.txt
set /p CHECKPOINT=<temp_checkpoint.txt
del temp_checkpoint.txt

if "%CHECKPOINT%"=="None" (
    echo No checkpoints found.
    exit /b 1
)

echo Latest checkpoint: %CHECKPOINT%
echo Running agent with latest checkpoint...

REM Run the agent with the found checkpoint
python run_mario_agent.py --checkpoint "%CHECKPOINT%" --episodes 10

echo Demo complete!
pause