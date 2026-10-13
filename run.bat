@echo off
rem Starts the Flask server and the wake word engine, opens the page, and closes the engine window when the server stops.

rem Run from the repository folder: both programs share manual_trigger.txt and the Excel files there
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:5000/"

rem First run: create the virtual environment and install the Python packages
if exist "%PY%" goto venv_ready
echo Creating .venv and installing the Python packages, first run only...
python -m venv .venv || goto install_failed
"%PY%" -m pip install --upgrade pip || goto install_failed
"%PY%" -m pip install -r requirements.txt || goto install_failed
:venv_ready

"%PY%" -c "import onnxruntime" 2>nul || goto onnx_failed

rem An engine left over from an earlier run would hold the mic, and a leftover trigger file would start a recording at once
call :stop_engine
if exist manual_trigger.txt del manual_trigger.txt

echo Starting Voice Command System Server...

rem Open the page once the server answers
start "" /b powershell -NoProfile -Command "for ($i = 0; $i -lt 60; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%URL%get_status' | Out-Null; break } catch { Start-Sleep -Milliseconds 500 } }; Start-Process '%URL%'"

rem Start the wake word engine in its own background terminal window
start "VCM Wake Word Engine" cmd /k %PY% voice_pipeline\wake_word_engine.py

echo To stop, press Ctrl+C here and answer N, so the engine window closes too.
"%PY%" main_ui.py

rem The server has stopped: close the engine window too
call :stop_engine
pause
exit /b

:stop_engine
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like '*wake_word_engine.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
exit /b

:install_failed
echo Package install failed. Check that Python 3 is installed and on PATH, then see the README.
pause
exit /b 1

:onnx_failed
echo onnxruntime does not load in .venv. Reinstall it with: %PY% -m pip install --force-reinstall onnxruntime
pause
exit /b 1
