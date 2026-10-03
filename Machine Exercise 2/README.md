# Voice Command Model (VCM) — Machine Exercise 2

A lightweight, on-device speech recognition system designed to run on a **Raspberry Pi 4 (4GB RAM)** without cloud dependencies. The system uses a two-stage pipeline: a continuously-listening **wake-word detector** and an on-demand **voice-command model** that transcribes speech into 116 fixed tokens.

> **Training reference:** When training on the DGX GPU cluster, open this project directly in GitHub Codespaces:
> 🔗 [https://vscode.dev/github/sibyl-oracles/onit/blob/mainbox/Machine%20Exercise%202](https://vscode.dev/github/sibyl-oracles/onit/blob/mainbox/Machine%20Exercise%202)

---

## Table of Contents

1. [GPU Cluster Training (DGX)](#gpu-cluster-training-dgx)
2. [System Architecture](#system-architecture)
3. [Project Layout](#project-layout)
4. [Datasets](#datasets)
5. [Training](#training)
6. [Quantization & ONNX Export](#quantization--onnx-export)
7. [Deployment](#deployment)
8. [Usage](#usage)
9. [Key Design Decisions](#key-design-decisions)
10. [Troubleshooting](#troubleshooting)

---

## GPU Cluster Training (DGX)

The intended environment for training is the GPU cluster. Open this project in GitHub Codespaces/VSCode:

> 🔗 [https://vscode.dev/github/sibyl-oracles/onit/blob/mainbox/Machine%20Exercise%202](https://vscode.dev/github/sibyl-oracles/onit/blob/mainbox/Machine%20Exercise%202)

### Wake-Word Training

```bash
cd training_env_wake

# Train the wake-word detector (DS-CNN, ~24k params, BCE loss)
python train.py --device cuda --epochs 30 --batches 64 --patience 7

# Export + int8 quantization
python export_onnx.py
# → Produces: wake_word_model_quantized.onnx
```

**Training details:**
- Loss: Binary Cross-Entropy (BCELoss, sigmoid built into model)
- Optimizer: Adam (lr=0.001)
- Input: 1.5s clips at 16 kHz
- Early stopping patience: 7 epochs
- Logs to `wake_training_results.txt`

### Voice Command Model (VCM) Training

```bash
cd training_env

# Train the VCM (1D-CNN, CTC loss, 57-token vocab)
python train.py --device cuda --epochs 300 --batches 64 --patience 5 --min-epochs 50

# Export to ONNX (fp32 only, no quantization)
python export_onnx.py
# → Produces: voice_command_model.onnx
```

**Training details:**
- Loss: CTC Loss (blank = last vocab index, 115)
- Optimizer: AdamW (lr=0.005, weight_decay=1e-4)
- Input: 5s tensors at 16 kHz (3s audio + 2s trailing silence)
- Asian accent data weighted 5× heavier than standard data
- Early stopping patience: 5 epochs (minimum 50 epochs before stopping)
- Metrics: Val F1, Val Accuracy, Val False Alarm Rate (FAR)
- Logs to `training_results.txt`
- Resumes from `best_model.pth` if it exists

> **Important:** VCM training on the DGX cluster uses GPU 6 (38/40 GB utilized). The `val_auc` metric on the cluster is sourced from `my_voice` validation — see `docs/` for details.

### Data Preparation

#### VCM Dataset
Before training, ensure all 6 datasets are downloaded into `training_env/dataset/`. The CSVs (`train_data.csv`, `val_data.csv`, etc.) reference audio files by relative path. Use `docs/download.py` as a template for downloading from Hugging Face or Kaggle.

To add duration metadata to existing CSVs:
```bash
cd training_env/dataset
python add_duration.py    # adds duration_seconds column to all 4 CSVs
```
This is idempotent — re-running it skips rows that already have durations.

#### Wake-Word Dataset
Download the **Speech Noise Dataset** (negative class) and **Alexa/synthetic data** (positive class) into `training_env_wake/dataset/`. The `target_noise_mixing.py` script mixes noise into target audio at various SNRs for augmentation.

---

## System Architecture

The VCM system follows a **train-on-GPU → quantize/export → run-on-edge** pipeline. Two independent models are trained in separate environments and deployed as ONNX artifacts:

```
                    ┌──────────────────────────────────────────┐
                    │         TRAINING (GPU Server)             │
                    │                                          │
   Voice Commands   │  training_env/                          │
   (English speech) │    ├── train.py          (CTC loss)     │
   ────────────────►│    ├── model.py          (1D-CNN)       │
   ────────────────►│    ├── dataset_loader.py (5s padding)  │
                    │    ├── export_onnx.py    (fp32 ONNX)    │
                    │    └── vocab.txt         (57 tokens)    │
                    │                                          │
   Wake Word Audio  │  training_env_wake/                       │
   ("Alexa" trigger)│    ├── train.py          (BCE)          │
   ────────────────►│    ├── model.py          (DS-CNN)       │
   ────────────────►│    ├── dataset_loader.py (1.5s window)  │
                    │    └── export_onnx.py    (int8 ONNX)     │
                    └──────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │            RUNTIME (Edge Device)          │
                    │                                          │
                    │  models/                                │
                    │    ├── voice_command_model.onnx         │
                    │    └── wake_word_model_quantized.onnx   │
                    │                                          │
                    │  voice_pipeline/                        │
                    │    ├── wake_word_engine.py  (always on)   │
                    │    ├── intent_parser.py     (CTC decode)  │
                    │    └── benchmark_adapter.py               │
                    │                                          │
                    │  main_ui.py  ↔  index.html              │
                    │  run.bat                                 │
                    └──────────────────────────────────────────┘
```

### Two Pipelines

- **Pipeline A (Voice):** Wake-word engine listens continuously → detects trigger → records 3s command → VCM transcribes → intent parser maps to action → UI executes.
- **Pipeline B (Button):** Manual buttons on the UI bypass the microphone for testing the intent parser and execution logic.

---

## Project Layout

```
Machine Exercise 2/
├── README.md                              ← This file
├── plan.md                                ← Detailed project plan & design
├── main_ui.py                             ← Flask backend: command execution & API
├── index.html                             ← Frontend UI (house view, chatbox, widgets)
├── alarms.xlsx                            ← Persistent alarm storage
├── reminders.xlsx                         ← Persistent reminder storage
│
├── voice_pipeline/                        ← Runtime inference (Pi/laptop)
│   ├── wake_word_engine.py                ← Continuous wake-word listener + VCM relay
│   ├── intent_parser.py                   ← Rule-based intent mapping (178 entries)
│   └── transcription_vcm.py               ← VCM transcription wrapper (per docs)
│
├── models/                                ← ONNX artifacts (copied from GPU server)
│   ├── voice_command_model.onnx           ← fp32 VCM model (57-token CTC)
│   └── wake_word_model_quantized.onnx     ← int8 quantized wake-word model
│
├── static/
│   ├── house_objects/                     ← bookshelf.png, lamp.png, sofa.png, chair.png
│
├── playlist/                              ← Local MP3 files for music commands
│   ├── song1.mp3, song2.mp3, song3.mp3
│
├── training_env/                          ← VCM training (GPU server only)
│   ├── train.py                           ← CTC training loop with early stopping
│   ├── model.py                           ← 1D-CNN architecture (57-class CTC)
│   ├── dataset_loader.py                  ← Audio decoding, 5s padding, Asian weighting
│   ├── export_onnx.py                     ← ONNX export (fp32, no quantization)
│   ├── vocab.txt                          ← 57-token vocabulary (44 cmd + 10 num + [UNK])
│   └── dataset/                           ← Downloaded datasets (see Datasets below)
│       ├── train_data.csv                 ← ~104,852 training samples
│       ├── val_data.csv                   ← ~25,403 validation samples
│       ├── asian_train_data.csv           ← ~1,006 Asian-accent samples (5× weight)
│       └── asian_val_data.csv             ← 184 Asian-accent validation samples
│
├── training_env_wake/                     ← Wake-word training (GPU server only)
│   ├── train.py                           ← BCE training loop with early stopping
│   ├── model.py                           ← DS-CNN architecture (~24k params)
│   ├── dataset_loader.py                  ← 1.5s audio, random shift padding
│   ├── export_onnx.py                     ← ONNX export + int8 dynamic quantization
│   └── dataset/
│       ├── wake_train_data.csv            ← ~13,797 training samples (Alexa + Vox + noise)
│       ├── wake_val_data.csv              ← ~3,450 validation samples
│       ├── Alexa/                         ← Alexa wake-word audio (positive class)
│       └── speech-noise-dataset/          ← Noise/Speech noise (negative class)
```

---

## Datasets

All datasets are downloaded into `training_env/dataset/` (for VCM) or `training_env_wake/dataset/` (for wake word). The folders ship with `.gitkeep` placeholders. For the actual dataset used in training, refer to the DGX link above.

| Dataset | Purpose | Source |
|---|---|
| **MLEnd Spoken Numerals** | Number utterance detection in VCM | [Kaggle](https://www.kaggle.com/datasets/jesusrequena/mlend-spoken-numerals) |
| **Speech Noise Dataset** | Negative class (noise/background) in wake-word detection | [Kaggle](https://www.kaggle.com/datasets/abdullahhaydarkadolu/speech-noise-dataset) 
| **Common Voice Dataset** | Conversational speech recognition in VCM | [Kaggle](https://www.kaggle.com/datasets/mozillaorg/common-voice) 
| **Fluent Speech Commands Dataset** | Detection of many commands in VCM | [Kaggle](https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus) 
| **SynTTS Commands Media Dataset** | Command detection in VCM + Alexa wake-word data | [Hugging Face](https://huggingface.co/datasets/lugan/Syntts-Commands-Media-Dataset/tree/main) 
| **Classmate Dataset** | Supplementary classmate contributions for VCM | [Google Drive](https://drive.google.com/drive/folders/1_GcDuvaRnGlkdtGdoogQI7FflgWf6gF8?usp=sharing)

### Dataset Structure (CSV Schema)

The VCM training CSVs (`train_data.csv`, `val_data.csv`, `asian_train_data.csv`, `asian_val_data.csv`) share this schema:

| Column | Description |
|---|---|
| `filepath` | Relative path to the audio file (prefixed with `dataset/` at load time) |
| `transcription` | The spoken text |
| `dataset_source` | `common_voice`, `fluent_speech`, `mlend`, `syntts`, `classmate` |
| `speaker_id` | Speaker identifier (for speaker-disjoint splits where applicable) |
| `label_indices` | Python list literal of CTC token indices (e.g. `"[58, 30, 1, 33, 0, 1, 0]"`) |
| `known_word_count` | Number of recognized tokens in the transcription |
| `known_word_indices` | Subset of `label_indices` that map to known words |
| `significant_word_count` | Number of significant (non-skip) tokens |
| `significant_word_indices` | Subset of `label_indices` for significant words |
| `duration_seconds` | Audio duration, added by `add_duration.py` |

The wake-word CSVs (`wake_train_data.csv`, `wake_val_data.csv`) use a simpler schema:

| Column | Description |
|---|---|
| `path` | Relative path to the audio file |
| `wake` | Binary label: `1` = wake word ("Alexa"), `0` = background/noise |
| `duration` | Audio duration in seconds |

> **Note:** `training_env_wake/dataset_loader.py` automatically renames `path` → `filepath` and `wake` → `label` at load time for compatibility.

### Vocabulary

`training_env/vocab.txt` defines the 116-token vocabulary: command words (e.g. play, weather, lights, volume, etc.), 10 number tokens (0–100 in steps, 16–26°C for thermostat), and the `[UNK]` / `[BLANK]` token as the first and last entry (index 0 and 115, used as the CTC blank).

---

---

## Quantization & ONNX Export

| Pipeline | Export Script | Output | Quantization |
|---|---|---|---|
| **VCM** | `training_env/export_onnx.py` | `voice_command_model.onnx` | fp32 only (no quantization) |
| **Wake Word** | `training_env_wake/export_onnx.py` | `wake_word_model_quantized.onnx` | int8 dynamic quantization |

### Key Export Details

- **BatchNorm fusion:** All BatchNorm parameters are mathematically fused into convolutional weights during export — zero computation cost at inference.
- **Dropout:** Automatically disabled during export (PyTorch handles this in eval mode).
- **int8 quantization (wake word):** Uses `QuantType.QUInt8` for 4× size reduction and faster CPU inference. The base ONNX is deleted after quantization.

---

## Deployment

### Artifact Transfer

Only the ONNX files cross from the GPU server to the edge device. The `.pth` checkpoints stay on the server.

```bash
# Copy from GPU server to laptop/Pi
cp training_env/voice_command_model.onnx models/
cp training_env_wake/wake_word_model_quantized.onnx models/
```

### Running the System

```bash
# Windows
run.bat

# Or manually (Linux/macOS/Pi)
python main_ui.py &
python voice_pipeline/wake_word_engine.py &
```

- `main_ui.py` serves the Flask backend on `http://127.0.0.1:5000`
- `wake_word_engine.py` runs the always-listening wake-word loop
- Open `http://127.0.0.1:5000` in a browser to use the UI

---

## Usage

### Voice Pipeline (Pipeline A)

1. Say **"Alexa"** near the microphone.
2. The wake-word engine detects it and pauses media.
3. Record your command (up to 5 seconds, 16 kHz).
4. The VCM transcribes it into 116 fixed tokens.
5. `intent_parser.py` maps the text to one of 19 actions.
6. `main_ui.py` executes the action and updates the UI.

### Manual Pipeline (Pipeline B)

Click any button in the **Pipeline B Manual Action Bar** at the bottom of the UI to test commands without using the microphone. Each button sends the same intent signal that the voice pipeline would produce.

### Supported Commands

| Category | Examples |
|---|---|
| Media | Play music, Pause, Stop, Next song, Volume up/down |
| Info | Weather, Time |
| Lights | Lights on/off (toggle), Brightness %, Color |
| Climate | Temperature °C |
| Productivity | Timer (seconds/minutes), Alarm (HH:MM AM/PM), Reminders |
| Calls | Call Mom / Call Dad |

### Debug Clips

The wake-word engine saves the first 5 VCM clips as `vcm_debug_clip_1.wav` through `vcm_debug_clip_5.wav` at the project root for verification of the 5-second padding strategy.

---

---

## Key Design Decisions

### 1. 5-Second Training / 3-Second Deployment Window

Documented in `docs/model_input.md`. During training, audio shorter than 3s is center-padded with random silence shifts, then 2s of trailing silence is appended to reach 5s. At runtime, the 3s live recording is right-padded with 2s of silence to match the model's 5s input tensor.

### 2. Model Architecture (VCM)

Documented in `docs/implementation_report_cnn_based_vcm.md`. A 1D CNN with:
- **Stem:** Conv1d(1→64, k=256, s=160) → BatchNorm → ReLU (learned spectrogram, no raw waveform FFT)
- **Encoder:** 3× ResidualBlock (depthwise-separable conv → BN → ReLU → Dropout 15% → depthwise conv → BN → residual add → ReLU)
- **Head:** Conv1d(64→116, k=1) → CTC output

Total params: under 1M (well within the Pi's 4GB RAM budget).

### 3. Wake-Word Architecture (DS-CNN)

Documented in `docs/Wake_Word_Model_Research.docx` and `docs/wake_word_design.md`. A Depthwise-Separable CNN with ~23,000–25,000 parameters, trained with BCE loss, and exported with int8 dynamic quantization for sub-100ms inference on the Pi.

### 4. Asian Accent Weighting

The `VoiceCommandDataset` assigns a weight of **5.0** to all rows from `asian_train_data.csv` and `asian_val_data.csv`, compared to 1.0 for standard data. This ensures Asian-accented speech is not underrepresented during training.

### 5. Toggle State Tracking (Benchmark)

`intent_parser.py` collapses both "lights on" and "lights out" into a single `toggle` action. For benchmarking, `benchmark_adapter.py` uses an in-memory `_light_on` flag to alternate between `LIGHT_ON` and `LIGHT_OFF` intents.

### 6. CTC Decoding

The CTC decoder (`decode_ctc`) collapses repeated tokens and removes the blank token (index 115), then maps output indices to vocabulary words via `vocab.txt`.

---

## Troubleshooting

| Issue | Solution | Documented in |
|---|---|---|
| Browser opens `file:///index.html` → API requests blocked | Use `run.bat` to serve via Flask at `http://127.0.0.1:5000` | `docs/mistakes_prevention_fix.md` |
| `ModuleNotFoundError: No module named 'flask'` | `pip install flask` (use a venv per `docs/mistakes_prevention_fix.md`) | `docs/mistakes_prevention_fix.md` |
| Duplicate `@app.route('/')` definitions | Remove duplicates, review copy-paste edits | `docs/mistakes_prevention_fix.md` |
| `sqlite3.OperationalError: no such table` | Initialize DB before server start | `docs/mistakes_prevention_fix.md` |
| Strict datetime parsing fails (HH:MM vs HH:MM:SS) | Use flexible `strptime` with fallback formats | `docs/mistakes_prevention_fix.md` |
| Wake-word threshold too sensitive/insensitive | Calibrate `k` multiplier (default 3.0) or `min_thresh` (default 0.01) in `wake_word_engine.py` | `BENCHMARK_README.md` |
| Music plays during benchmarking → false triggers | Stop VLC/music before running `run_benchmark.sh` | `BENCHMARK_README.md` |
| ONNX inference crashes on Pi | Ensure ONNX Runtime is installed on the Pi; check wheel compatibility with Python/OS version | `docs/Wake_Word_Model_Research.docx` |

---

## Citation & Further Reading

- **Paper reference:** *A Lightweight End-to-End Speech Recognition System on Embedded Devices* — [ResearchGate](https://www.researchgate.net/publication/372045971_A_Lightweight_End-to-End_Speech_Recognition_System_on_Embedded_Devices)
- **DS-CNN (Hello Edge):** Zhang et al., [arXiv:1711.07128](https://arxiv.org/abs/1711.07128)
- **MLPerf Tiny KWS reference:** [GitHub](https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting)
- **Full research bibliography:** See `docs/Wake_Word_Model_Research.docx` (21 references)

---

*This project is part of AI 231 — MLOps (Machine Exercise 2) at the University of the Philippines.