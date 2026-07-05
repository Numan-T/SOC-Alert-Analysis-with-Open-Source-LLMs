@echo off
TITLE Docker Runner
echo ========================================================
echo Starting Python ^& Ollama Environment.
echo Results are saved under directory: /results
echo ========================================================

docker compose up --build

echo.
echo ========================================================
echo Finished all processes.
echo Press any key to close the window.
echo ========================================================
pause