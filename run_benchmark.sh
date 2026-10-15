#!/usr/bin/env bash
# run_benchmark.sh: start the wake word engine in benchmark mode.
#
# Benchmark mode is enabled via VCM_BENCHMARK=1 (see voice_pipeline/benchmark_adapter.py).
# The engine opens one log file, ~/vcm_benchmark/benchmark_<timestamp>.log, keeps it open,
# and appends one JSON line per wake word detection and per transcribed command.

set -euo pipefail

# Run from the repository folder (the engine uses manual_trigger.txt there, like run.sh)
cd "$(dirname "${BASH_SOURCE[0]}")"

# Same Python as run.sh: Debian's python3-onnxruntime crashes on the Raspberry Pi 4
PY=.venv/bin/python
if [ ! -x "$PY" ]; then
    echo "[Benchmark] .venv not found. Run ./run.sh once first to create it."
    exit 1
fi

# Kill any previously running wake word engine before starting a new one.
# (main_ui.py / Flask is NOT killed here; stop it separately to silence music.)
pkill -f "voice_pipeline/wake_word_engine.py" 2>/dev/null || true

# A trigger file left over from an earlier run would start a recording at once
rm -f manual_trigger.txt

export VCM_BENCHMARK=1

echo "[Benchmark] Mode enabled. Log file: ~/vcm_benchmark/benchmark_<timestamp>.log"
echo "[Benchmark] NOTE: stop main_ui.py (Flask) if music is playing; it is not killed here."

# Start the wake word engine (blocking; Ctrl-C to stop).
exec "$PY" voice_pipeline/wake_word_engine.py
