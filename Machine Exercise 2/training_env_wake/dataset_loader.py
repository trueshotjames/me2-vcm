# training_env_wake/dataset_loader.py

import torch
import random
import torch.nn.functional as F
import pandas as pd
import soundfile as sf
import numpy as np
import subprocess
import shutil

def _read_audio(audio_path):
    """
    Decodes varying audio file types (.mp3, .wav, .m4a). 
    Falls back to a background decoder if the standard reader fails.
    """
    try:
        audio_np, sr = sf.read(audio_path)
        return audio_np, sr
    except Exception:
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", audio_path, "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return np.frombuffer(proc.stdout, dtype=np.float32).copy(), 16000

def _resample(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr: 
        return audio
    import torchaudio
    return torchaudio.functional.resample(audio, orig_sr, target_sr)

class WakeWordDataset(torch.utils.data.Dataset):
    def __init__(self, csv_filepath):
        self.data = pd.read_csv(csv_filepath)
        # Schema shim: accept either ('filepath','label') or the on-disk
        # ('path','wake') column names. Backward-compatible; CSVs untouched.
        rename = {}
        if 'filepath' not in self.data.columns and 'path' in self.data.columns:
            rename['path'] = 'filepath'
        if 'label' not in self.data.columns and 'wake' in self.data.columns:
            rename['wake'] = 'label'
        if rename:
            self.data = self.data.rename(columns=rename)
            print(f"[dataset_loader] renamed CSV columns {rename} for compatibility",
                  flush=True)
        missing = [c for c in ('filepath', 'label') if c not in self.data.columns]
        if missing:
            raise KeyError(
                f"[dataset_loader] {csv_filepath} missing required columns {missing}; "
                f"found {list(self.data.columns)}")
        
        # FIX: Add the "dataset/" folder to the beginning of every file path
        self.data['filepath'] = "dataset/" + self.data['filepath'].astype(str)
        
        self.sample_rate = 16000
        # Exactly 1.5 seconds as dictated by wake_word_design.md
        self.max_len = int(1.5 * self.sample_rate) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        n = len(self.data)
        # Iterative skip (bounded, no recursion); logs every unreadable file.
        for attempt in range(n):
            i = (idx + attempt) % n
            audio_path = self.data.iloc[i]['filepath']
            try:
                audio_np, native_sr = _read_audio(audio_path)
                audio = torch.tensor(audio_np, dtype=torch.float32)
                break
            except Exception as e:
                print(f"[dataset_loader] WARNING: skipping unreadable file "
                      f"{audio_path} (row {i}): {e}", flush=True)
        else:
            raise RuntimeError(
                f"[dataset_loader] all {n} rows failed to decode from idx={idx}")
        idx = i   # align downstream label lookup with the row that decoded
            
        if audio.dim() > 1: 
            audio = audio.mean(dim=1)
            
        audio = _resample(audio, native_sr, self.sample_rate)
        audio_length = audio.shape[0]

        # Direct handling: Crop if too long, or randomly shift/pad with silence if too short
        if audio_length > self.max_len:
            audio = audio[:self.max_len]
        elif audio_length < self.max_len:
            short_pad = self.max_len - audio_length
            left_pad = random.randint(0, short_pad)
            right_pad = short_pad - left_pad
            audio = F.pad(audio, (left_pad, right_pad), "constant", 0)
            
        # Binary label: 1 for wake word, 0 for background/noise
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.float32)
        return audio, label