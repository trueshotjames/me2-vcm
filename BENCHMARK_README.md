# VCM Benchmark Integration

## Overview

This connects the voice pipeline to the class benchmark, [airimonda/vcm-benchmark](https://github.com/airimonda/vcm-benchmark).

The benchmark runs **live**:

- **Laptop:** runs `benchmark.py`. It plays "wake word + command" out loud from the holdout set.
- **Pi:** runs this project's wake word engine in benchmark mode. It hears the audio, decodes the command, and writes one JSON line to a log file.
- The laptop reads that log over SSH while the test runs and scores it. It also records the Pi's CPU, RAM, temperature and latency.

Nothing has to be copied between the two by hand.

## What runs where

| | Laptop | Pi |
|---|---|---|
| Code | `vcm-benchmark` (clone it here) | this project (`voice_pipeline/`, `models/`) |
| Start | `python benchmark.py` | `./run_benchmark.sh` |
| Setup | `pip install -r requirements.txt` in the benchmark folder | `./run.sh` once, to create `.venv` |

## How to run

1. **Pi:** stop the UI if it is playing music, then start the engine in benchmark mode:
   ```bash
   cd /path/to/Machine\ Exercise\ 2
   ./run_benchmark.sh
   ```
   The engine runs with `.venv/bin/python` (the same Python as `run.sh`), sets `VCM_BENCHMARK=1`, and prints the path of its log file. Ctrl-C stops it.

2. **Laptop:**
   ```bash
   git clone https://github.com/airimonda/vcm-benchmark.git && cd vcm-benchmark
   python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python benchmark.py
   ```
   Follow its steps. Recommended answers:
   - **Wake word:** "Alexa".
   - **Pause after the wake word (step 3):** `1.0` s. The default 0.8 s is tight; see [Timing](#timing).
   - **ONNX model (step 5):** optional. Copy `models/voice_command_model.onnx` and `voice_command_model.onnx.data` to the laptop and give it that path to get FLOPs and model size in the report.

   The benchmark finds `~/vcm_benchmark/` on the Pi and the engine process (the one holding the log file open) by itself.

## What the engine logs

Log file: `~/vcm_benchmark/benchmark_<timestamp>.log` on the Pi. One file per engine run, opened at start and kept open.

```json
{"event": "wake"}
{"intent": "TEMPERATURE", "slot": "22 degrees", "infer_ms": 41.2, "audio_ms": 1500}
{"intent": "LIGHT_OFF", "slot": "", "infer_ms": 30.0, "audio_ms": 1200}
```

- `{"event": "wake"}`: the wake word fired. This gives the report's wake detection rate.
- `intent`: one of the benchmark's 19 intents, or `OUT_OF_SCOPE`.
- `slot`: e.g. `"90 seconds"`, `"7:30 AM"`, `"22 degrees"`, `"20 percent"`, `"blue"`, `"drink water"`; `""` for intents without a slot.
- `infer_ms`: VCM time on the final recording (features + ONNX).
- `audio_ms`: length of the recorded command (recording stops when the command is finished, at most 5000).

The command line is written as soon as the command is decoded, before the result is sent to the UI. The benchmark measures latency up to this line.

`voice_pipeline/benchmark_adapter.py` turns the decoded text into these lines. It uses `intent_parser.parse_text` and maps the action names to the benchmark's names (`INTENT_MAP`). Light phrases come out of `intent_parser` as `toggle` with the value `"on"` or `"off"`, which becomes `LIGHT_ON` or `LIGHT_OFF`. The names already match the benchmark, so no `aliases` in the benchmark's `bench_settings.json` are needed.

## Timing

After "Alexa" ends, the Pi needs time before it records the command:

1. The engine checks for the wake word every 0.2 s (`hop_duration` in `wake_word_engine.py`), on the last 1.5 s of audio.
2. It plays the beep (0.16 s), then waits 30 ms (`BEEP_GUARD_MS`).
3. It starts recording.

That is roughly 0.45 s plus the model's run time and audio start-up on the Pi. A 1.0 s pause in the benchmark leaves room for this. If the warm-up commands in the benchmark's sound check lose their first word, raise the pause.

## Results

Run `20261017-015001` (17 October 2026), full holdout, 218 trials. The files are in [`benchmark_results/20261017-015001/`](benchmark_results/20261017-015001/report.md): `report.md` (the benchmark's report), `metrics.json`, `trials.csv` (one row per trial), `pi_metrics.csv` (Pi resource samples) and `pi_specs.json`.

| Item | Value |
|---|---|
| Wake word detected (trials with "Alexa") | 82.7 % (167 / 202) |
| Intent accuracy (19 intents + reject) | 73.3 % (148 / 202) |
| Command accuracy (93 commands + reject) | 72.8 % |
| False-accept rate (out-of-scope fired) | 12.5 % (2 / 16) |
| False-wake rate (no wake word, fired) | 31.2 % (5 / 16) |
| Slot exact (intent correct) | 98.8 % (79 / 80) |
| Latency p50 / p95 (command end to log line) | 0.62 / 1.19 s |
| Inference time mean / p95 | 12.3 / 16.0 ms |
| Real-time factor mean / p95 | 0.007 / 0.012 |

Settings of this run: wake word "Alexa" (one recorded take), pause 1.0 s, 10 to 15 s between trials, shuffle seed 24992, Pi mic `plughw:3,0` (signal-to-noise 23.1 dB). The README (section 3) has the breakdown by real and synthetic voices and the Pi's CPU, RAM and temperature.

To reproduce, follow [How to run](#how-to-run) with the same answers; `python benchmark.py` writes the same files to `runs/<timestamp>/` on the laptop. Enter seed `24992` when asked (or pass `--seed 24992`) to play the same clips in the same order.

## Benchmark mode behaviour

With `VCM_BENCHMARK=1` (set by `run_benchmark.sh`) the engine differs from normal use (`run.sh`) in one way:

- **No UI, no music.** The engine sends nothing to `main_ui.py`: results are only written to the log. Commands like PLAY_MUSIC or NEXT never start playback, even if the UI is open.

In both modes the command recording also stops on silence: 0.6 s after the voice goes quiet (`SILENCE_STOP_MS`), or 1.5 s after the beep if no voice is heard at all (`NO_VOICE_SEC`), whether or not the voice model recognised any words. The 5 s cap (`MAX_COMMAND_SEC`) stays as a backstop.

## Known limitations

1. **Music started before the test.** Benchmark mode never starts music, but music already playing from earlier use keeps playing. Stop it (or `main_ui.py`) before the test, or the mic picks it up.
2. **`infer_ms` covers the final decode only.** While the command is spoken, the VCM also runs every 80 ms to decide when the command has ended (`DECODE_TICK_MS`). That work shows up in the benchmark's CPU numbers, not in `infer_ms`.
3. **Single-threaded inference.** The ONNX sessions run one after the other on the Pi's CPU, so `infer_ms` is real Pi latency.
