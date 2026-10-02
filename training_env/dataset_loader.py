# dataset_loader.py

import torch
import random
import torch.nn.functional as F
import pandas as pd
import soundfile as sf
import ast
import numpy as np
import subprocess
import shutil


def _read_audio(audio_path):
    """Decode an audio file to (mono 1-D float32 numpy array, sample_rate).

    Tries soundfile first (wav/flac/ogg/mp3). If libsndfile cannot decode the
    format (e.g. m4a/aac), falls back to an ffmpeg subprocess that decodes to
    mono float32 PCM at 16 kHz and returns it as a numpy array.
    """
    try:
        audio_np, sr = sf.read(audio_path)
        return audio_np, sr
    except Exception as sf_err:
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", audio_path,
            "-f", "f32le", "-ac", "1", "-ar", "16000",
            "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"soundfile could not decode {audio_path} ({sf_err}) and "
                f"ffmpeg was not found on PATH"
            ) from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to decode {audio_path}: soundfile={sf_err}; "
                f"ffmpeg rc={proc.returncode}: "
                f"{proc.stderr.decode(errors='replace')}"
            )
        audio_np = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        return audio_np, 16000


def _resample_audio(audio, orig_sr, target_sr):
    """Resample a 1-D float32 tensor from ``orig_sr`` to ``target_sr``.

    Uses ``torchaudio.functional.resample`` when available, otherwise falls back
    to ``scipy.signal.resample_poly`` (anti-aliased polyphase resampling). No-op
    when the rates already match.
    """
    if orig_sr == target_sr:
        return audio
    try:
        import torchaudio
        return torchaudio.functional.resample(audio, orig_sr, target_sr)
    except Exception:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(orig_sr), int(target_sr))
        out = resample_poly(audio.numpy(), target_sr // g, orig_sr // g)
        return torch.tensor(np.asarray(out), dtype=torch.float32)


class VoiceCommandDataset(torch.utils.data.Dataset):
    def __init__(self, csv_filepath):
        # Load your real dataset CSV
        self.data = pd.read_csv(csv_filepath)
        
        # FIX: Add the "dataset/" folder to the beginning of every file path
        self.data['filepath'] = "dataset/" + self.data['filepath'].astype(str)
        
        # Automatically count words in the text file
        with open("vocab.txt", "r") as f:
            self.vocab_size = len([line for line in f.read().splitlines() if line.strip()])
        self.sample_rate = 16000
        self.max_recording_len = self.sample_rate * 3
        
        # ASSIGN WEIGHT: 5.0 heavy penalty for Asian accents, 1.0 for standard
        if "asian" in csv_filepath.lower():
            self.weight = 5.0
        else:
            self.weight = 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Read via soundfile, with an ffmpeg fallback for m4a/aac
        n = len(self.data)
        # Iterative skip (bounded, no recursion). Tries at most n rows starting
        # at idx; logs every unreadable file so the data problem is visible.
        for attempt in range(n):
            i = (idx + attempt) % n
            audio_path = self.data.iloc[i]['filepath']
            try:
                audio_np, native_sr = _read_audio(audio_path)
                break
            except Exception as e:
                print(f"[dataset_loader] WARNING: skipping unreadable file "
                      f"{audio_path} (row {i}): {e}", flush=True)
        else:
            raise RuntimeError(
                f"[dataset_loader] all {n} rows failed to decode from idx={idx}")
        idx = i   # align downstream label lookup with the row that decoded
            
        audio = torch.tensor(audio_np, dtype=torch.float32)
        
        # soundfile returns (Time, Channels). Flatten into a 1D squiggly line.
        if audio.dim() > 1:
            audio = audio.mean(dim=1)

        # Resample to the model sample rate BEFORE any length/padding math, so
        # max_recording_len (sample_rate*3) always means 3 real seconds,
        # regardless of the clip's native rate (22.05k/24k/44.1k/48k).
        if native_sr != self.sample_rate:
            audio = _resample_audio(audio, native_sr, self.sample_rate)

        audio_length = audio.shape[0]
        full_len = self.max_recording_len + self.sample_rate * 2  # 5 s = 80000

        # PRESERVED FEATURE: Random shift padding for short audio (< 3 s)
        if audio_length < self.max_recording_len:
            short_padding = self.max_recording_len - audio_length
            left_pad = random.randint(0, short_padding)
            right_pad_split = short_padding - left_pad
            trailing_silence = self.sample_rate * 2
            total_right_pad = right_pad_split + trailing_silence
        else:
            # 3-5 s clips: keep FULL audio, pad only enough trailing silence.
            # >5 s clips: truncate as a pure safety guard.
            if audio_length > full_len:
                audio = audio[:full_len]
            left_pad = 0
            total_right_pad = full_len - audio.shape[0]

        padded_audio = F.pad(audio, (left_pad, total_right_pad), "constant", 0)
        
        # Extract the text label array from the CSV
        label = torch.tensor(ast.literal_eval(self.data.iloc[idx]['label_indices']))
        
        # NEW: Output the weight alongside the audio and label
        return padded_audio, label, self.weight