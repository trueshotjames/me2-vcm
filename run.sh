#!/bin/bash
# Starts the Flask server and the wake word engine, opens the page, and stops both on exit.

# Run from the repository folder: both programs share manual_trigger.txt and the Excel files there
cd "$(dirname "$0")" || exit 1

PY=.venv/bin/python
URL="http://127.0.0.1:5000/?lite=1"   # ?lite=1 turns off the page's fade animations (saves CPU on the Pi)

# First run: create the virtual environment. Its onnxruntime comes from PyPI; Debian's
# python3-onnxruntime crashes with "Illegal instruction" on the Raspberry Pi 4.
if [ ! -x "$PY" ]; then
    echo "Creating .venv and installing the Python packages (first run only)..."
    python3 -m venv .venv && "$PY" -m pip install --upgrade pip && "$PY" -m pip install -r requirements.txt || {
        echo "Package install failed. See the README, section 2-3."
        read -p "Press Enter to exit..."
        exit 1
    }
fi

if ! "$PY" -c "import onnxruntime" 2>/dev/null; then
    echo "onnxruntime does not load in .venv. Reinstall it with: $PY -m pip install --force-reinstall onnxruntime"
    read -p "Press Enter to exit..."
    exit 1
fi

# A trigger file left over from an earlier run would start a recording at once
rm -f manual_trigger.txt

echo Starting Voice Command System Server...

SERVER_PID=
ENGINE_PID=
cleanup() {
    trap - EXIT INT TERM HUP
    echo
    echo "Stopping..."
    [ -n "$ENGINE_PID" ] && kill "$ENGINE_PID" 2>/dev/null
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

"$PY" main_ui.py &
SERVER_PID=$!

# Wait for the server before starting the engine and opening the page
for _ in $(seq 1 60); do
    if "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/get_status', timeout=1)" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "The server stopped while starting (is port 5000 already in use?)."
        read -p "Press Enter to exit..."
        exit 1
    fi
    sleep 0.5
done

# The wake word engine runs in the background of this terminal; its output appears here too
"$PY" voice_pipeline/wake_word_engine.py &
ENGINE_PID=$!

xdg-open "$URL" >/dev/null 2>&1 &

echo "Running. Press Ctrl+C to stop the server and the wake word engine."
wait "$SERVER_PID"
status=$?
if [ "$status" -ne 0 ] && [ "$status" -lt 128 ]; then
    echo "The server stopped with an error (exit code $status)."
    read -p "Press Enter to exit..."
fi
