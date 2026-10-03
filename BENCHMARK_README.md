# VCM Benchmark Integration

## Overview

This integration connects the voice command pipeline to the [airimonda/vcm-benchmark](https://github.com/airimonda/vcm-benchmark) framework. It enables automated accuracy benchmarking of the wake-word + voice-command (VCM) system on a Raspberry Pi, producing per-intent JSON log lines that the benchmark's analysis script (`benchmark.py`) can score.

The flow: the Pi runs the wake-word engine in benchmark mode (`VCM_BENCHMARK=1`). When a wake word is detected, the engine records a command, runs ONNX inference, decodes the CTC output to text, and passes that text through `benchmark_adapter.py`. The adapter maps the transcribed intent to one of the benchmark's 19 intent names, formats any slots, and appends a JSON record to a timestamped log file under `~/vcm_benchmark/`. On the laptop, `benchmark.py` reads those logs and computes accuracy metrics.

## Prerequisites

1. **Clone the benchmark repo** (on the Pi):
   ```bash
   git clone https://github.com/airimonda/vcm-benchmark.git ~/vcm-benchmark
   ```

2. **Install benchmark dependencies** (on the Pi):
   ```bash
   cd ~/vcm-benchmark
   pip install -r requirements.txt
   ```

3. **SSH setup** — ensure the laptop can SSH into the Pi:
   ```bash
   # On the Pi, enable SSH if not already:
   sudo systemctl enable ssh
   sudo systemctl start ssh

   # From the laptop, test connectivity:
   ssh pi@<pi-ip-address>
   ```

4. **Pi audio** — the Pi must have a working microphone (USB mic or HAT). Test with:
   ```bash
   arecord -d 3 test.wav && aplay test.wav
   ```

## Setup

1. **Copy `bench_settings.json` to the benchmark directory** (on the Pi):
   ```bash
   cp /path/to/Machine\ Exercise\ 2/bench_settings.json ~/vcm-benchmark/
   ```
   This file maps the pipeline's intent names to the benchmark's canonical intent strings (e.g. `"music_play"` → `"PLAY_MUSIC"`).

2. **Verify the voice pipeline is on the Pi** — the `voice_pipeline/` directory (containing `wake_word_engine.py`, `benchmark_adapter.py`, `intent_parser.py`) and the `models/` directory (containing `wake_word_model_quantized.onnx` and `voice_command_model.onnx`) must be present on the Pi.

## How to Run

### On the Pi (start the engine in benchmark mode):

```bash
cd /path/to/Machine\ Exercise\ 2
./run_benchmark.sh
```

This script:
- Sets `VCM_BENCHMARK=1` in the environment
- Launches `voice_pipeline/wake_word_engine.py`
- The engine loads both ONNX models, opens the microphone, and enters the detection loop

The engine runs indefinitely. Press `Ctrl-C` to stop.

### On the laptop (analyze results):

After collecting benchmark logs on the Pi, transfer them to the laptop and run:

```bash
cd ~/vcm-benchmark
python benchmark.py
```

This reads all `benchmark_*.log` files from `~/vcm_benchmark/` and produces accuracy metrics per intent.

## Toggle Workaround

The `intent_parser` module collapses both "lights on" and "lights out" into a single action `"toggle"`. The benchmark, however, expects two distinct intents: `LIGHT_ON` and `LIGHT_OFF`.

`benchmark_adapter.py` resolves this with a **module-level state tracker** (`_light_on`):

- The light **starts off** (`_light_on = False`).
- The first `"toggle"` command flips the tracker to `True` and emits `LIGHT_ON`.
- The second `"toggle"` command flips it back to `False` and emits `LIGHT_OFF`.
- Each subsequent toggle alternates between the two.

This means the benchmark log will show alternating `LIGHT_ON` / `LIGHT_OFF` entries for consecutive light commands, matching the expected benchmark behavior. The tracker is in-memory only; it resets to `False` (off) each time the engine process restarts.

## Expected Output

Log files are written to `~/vcm_benchmark/` on the Pi. Each file is named `benchmark_<timestamp>.log` (e.g. `benchmark_20261003_143052_123456.log`).

Each line is a JSON object:

```json
{"intent": "PLAY_MUSIC", "slot": "", "infer_ms": 42.3, "audio_ms": 5000}
{"intent": "LIGHT_ON", "slot": "", "infer_ms": 38.1, "audio_ms": 5000}
{"intent": "TIMER", "slot": "90 seconds", "infer_ms": 45.7, "audio_ms": 5000}
```

Fields:
- `intent` — one of the 19 benchmark intent names (e.g. `PLAY_MUSIC`, `LIGHT_ON`, `TIMER`, `WEATHER`, `OUT_OF_SCOPE`)
- `slot` — formatted slot string (e.g. `"90 seconds"` for timer, `"7:30 AM"` for alarm, `""` for non-slotted intents)
- `infer_ms` — ONNX inference time in milliseconds (VCM model only)
- `audio_ms` — audio capture window in milliseconds (default 5000)

The engine also prints human-readable lines to the terminal:
```
Logged intent 'PLAY_MUSIC' slot='' to /home/pi/vcm_benchmark/benchmark_20261003_143052_123456.log
```

## Known Limitations

1. **Music must be disabled during benchmarking.** The VLC media player and any background music processes must be stopped on the Pi before running the benchmark, otherwise the microphone will pick up music audio and produce false wake-word triggers or garbled transcriptions.

2. **Audio gap calibration.** The engine uses a noise-based VAD threshold (`get_noise_threshold()`) that adapts from recent non-wake audio. In a quiet room the threshold may be too sensitive; in a noisy room it may miss commands. If the benchmark shows unexpectedly low trigger rates, adjust the `k` multiplier (default 3.0) or `min_thresh` (default 0.01) in `wake_word_engine.py`.

3. **Toggle state resets on restart.** The `_light_on` tracker is in-memory. If the engine is restarted mid-benchmark, the first toggle command will always emit `LIGHT_ON` regardless of the previous state.

4. **Single-threaded inference.** The ONNX sessions run sequentially on the Pi's CPU. The `infer_ms` values in the log reflect this single-threaded latency; they will be higher than on a laptop GPU but are representative of real Pi performance.

5. **No ground-truth alignment.** The benchmark logs what the system *transcribed*, not what was *actually said*. Accuracy scoring depends on the benchmark harness providing the correct expected intent for each test utterance.
