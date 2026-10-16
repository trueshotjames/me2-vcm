# wake_check.py: live wake word score check (no UI, no commands, no logging).
#
# Runs the same steps as voice_pipeline/wake_word_engine.py: 48 kHz mic, last 1.5 s,
# resampled to 16 kHz, models/wake_word_model_quantized.onnx, "wake" when score > 0.5.
# Prints the score 5 times a second so you can see whether "Alexa" gets near 0.5.
#
# Stop run_benchmark.sh / run.sh first (they hold the mic), then:
#   .venv/bin/python wake_check.py            (Ctrl+C to stop)

import os
import time

import numpy as np
import onnxruntime as ort
import scipy.signal
import sounddevice as sd

NATIVE_SR, TARGET_SR = 48000, 16000
WINDOW_S, HOP_S = 1.5, 0.2

model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "wake_word_model_quantized.onnx")
session = ort.InferenceSession(model)
print("Mic:", sd.query_devices(kind="input")["name"])

ring = np.zeros(int(3.0 * NATIVE_SR), np.float32)
pos = 0

def callback(indata, frames, t, status):
    global pos
    x = indata[:, 0]
    idx = (pos + np.arange(frames)) % len(ring)
    ring[idx] = x
    pos = (pos + frames) % len(ring)

with sd.InputStream(samplerate=NATIVE_SR, channels=1, dtype="float32", callback=callback):
    print("Say 'Alexa'. Score > 0.50 = wake. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(HOP_S)
            n = int(WINDOW_S * NATIVE_SR)
            clip = np.roll(ring, -pos)[-n:]
            clip16 = scipy.signal.resample_poly(clip, TARGET_SR, NATIVE_SR).astype(np.float32).reshape(1, -1)
            score = float(session.run(["output"], {"input": clip16})[0][0][0])
            level = 20 * np.log10(np.sqrt(np.mean(clip ** 2)) + 1e-9)
            bar = "#" * int(score * 40)
            print(f"score {score:.2f} {'WAKE' if score > 0.5 else '    '}  mic {level:6.1f} dBFS  {bar}")
    except KeyboardInterrupt:
        pass
