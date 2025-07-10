@echo off
REM Batch script to run the improved pipeline from any directory

REM Get the directory of this batch file
SET script_dir=%~dp0

REM Run the Python script with all arguments passed to this batch file
python %script_dir%run_improved_pipeline.py %*

REM Pause to see the output
pause
