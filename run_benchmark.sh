#!/usr/bin/env bash
# run_benchmark.sh — start the wake word engine in benchmark mode.
#
# Benchmark mode is enabled via VCM_BENCHMARK=1 (see voice_pipeline/benchmark_adapter.py).
# Each transcribed command is appended as one JSON line to the newest *.log file
# under ~/vcm_benchmark/.

set -euo pipefail

# Kill any previously running wake word engine before starting a new one.
# (main_ui.py / Flask is NOT killed here — stop it separately to silence music.)
pkill -f "voice_pipeline/wake_word_engine.py" 2>/dev/null || true

# Self-locate the script directory so the script can be run from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$SCRIPT_DIR/voice_pipeline/wake_word_engine.py"

# Enable benchmark mode for the engine process.
export VCM_BENCHMARK=1

echo "[Benchmark] Mode enabled. Log files will be written to ~/vcm_benchmark/"
echo "[Benchmark] NOTE: stop main_ui.py (Flask) if music is playing — it is not killed here."

# Start the wake word engine (blocking; Ctrl-C to stop).
exec python "$ENGINE"
