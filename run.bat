@echo off
REM Enable UTF-8 support in the console
chcp 65001 > nul

REM Set environment variables for Python UTF-8 mode
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Activate the virtual environment
call C:\Users\saite\OneDrive\Desktop\deep_research_final\.venv\Scripts\activate

REM Run the Python script with UTF-8 mode explicitly enabled
python -X utf8 final.py --generate-review --non-interactive --continue

REM Pause to see any errors
pause