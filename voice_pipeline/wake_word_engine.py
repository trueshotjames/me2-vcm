# voice_pipeline/wake_word_engine.py

import sounddevice as sd
import numpy as np
import scipy.signal
import time
import sys
import soundfile as sf
import os
import onnxruntime as ort
import vlc     
import requests  
from collections import deque

# Point the system path to the /models folder to access the attached model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

# --- Load the Wake Word Model ---
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'wake_word_model_quantized.onnx'))
session = ort.InferenceSession(model_path)

# --- NEW: Load the Voice Command Model (VCM) and Vocabulary ---
vcm_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'voice_command_model.onnx'))
vcm_session = ort.InferenceSession(vcm_model_path)

vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training_env', 'vocab.txt'))
with open(vocab_path, "r") as f:
    vocab = [line.strip() for line in f.readlines() if line.strip()]

class WakeWordEngine:
    def __init__(self):
        # Hardware native rate and target model rate[cite: 12]
        self.native_sr = 48000
        self.target_sr = 16000
        self.saved_clip_count = 0  # Tracks the number of saved VCM clips
        self.noise_hist = deque(maxlen=30)   # (timestamp, rms) of recent non-wake clips
        
        # Timing constraints
        self.window_duration = 1.5
        self.hop_duration = 0.5
        
        # Allocate a rolling 3-second recording loop buffer in RAM[cite: 12]
        self.buffer_duration = 3.0
        self.buffer_size = int(self.buffer_duration * self.native_sr)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0
        
        # Open a non-blocking background microphone stream
        self.stream = sd.InputStream(
            samplerate=self.native_sr, 
            channels=1, 
            dtype='float32', 
            callback=self.audio_callback
        )

    def audio_callback(self, indata, frames, time, status):
        """Continuously overwrites the oldest audio in the ring buffer."""
        end_idx = self.buffer_index + frames
        if end_idx < self.buffer_size:
            self.buffer[self.buffer_index:end_idx] = indata[:, 0]
        else:
            overflow = end_idx - self.buffer_size
            self.buffer[self.buffer_index:self.buffer_size] = indata[:frames - overflow, 0]
            self.buffer[0:overflow] = indata[frames - overflow:, 0]
        
        self.buffer_index = (self.buffer_index + frames) % self.buffer_size

    def get_noise_threshold(self, lag=3.0, window=3.0, k=3.0, min_thresh=0.01):
        """Threshold from non-wake clips that are 3-6 s old (before the wake word was spoken)."""
        now = time.monotonic()
        old = [r for t, r in self.noise_hist if lag <= now - t <= lag + window]
        if not old:
            return min_thresh
        return max(float(np.median(old)) * k, min_thresh)

    def record_command(self, threshold, max_sec=5.0, frame_ms=20,
                       end_silence_ms=300, no_speech_sec=3.0):
        """Records until speech ends. Returns 1D float32 array at 16 kHz."""
        sr = self.target_sr
        frame = int(sr * frame_ms / 1000)
        end_frames = end_silence_ms // frame_ms
        max_frames = int(max_sec * 1000 / frame_ms)
        no_speech_frames = int(no_speech_sec * 1000 / frame_ms)

        chunks = []
        speech_started = False
        silent_run = 0

        with sd.InputStream(samplerate=sr, channels=1, dtype='float32', blocksize=frame) as s:
            for n in range(max_frames):
                data, _ = s.read(frame)
                x = data[:, 0].copy()
                chunks.append(x)
                rms = float(np.sqrt(np.mean(x ** 2)))

                if rms > threshold:
                    speech_started = True
                    silent_run = 0
                elif speech_started:
                    silent_run += 1
                    if silent_run >= end_frames:
                        break
                elif n >= no_speech_frames:
                    break

        return np.concatenate(chunks)
    
    def start_listening(self):
        print("Wake Word Engine started silently in the background...")
        with self.stream:
            while True:
                # Wait for the exact hop duration
                time.sleep(self.hop_duration)
                
                # Extract the most recent 1.5 seconds from the rolling buffer
                clip_size = int(self.window_duration * self.native_sr)
                
                if self.buffer_index >= clip_size:
                    clip = self.buffer[self.buffer_index - clip_size : self.buffer_index]
                else:
                    # Stitch together the wrapped data if it crosses the buffer's edge
                    tail = clip_size - self.buffer_index
                    clip = np.concatenate((self.buffer[-tail:], self.buffer[:self.buffer_index]))
                    
                # Downsample from 48 kHz to 16 kHz
                clip_16k = scipy.signal.resample_poly(clip, self.target_sr, self.native_sr)
                
                # --- REVISED SECTION ---
                # 1. Format the audio array to (Batch=1, Time=24000) and cast to float32
                clip_16k_formatted = clip_16k.astype(np.float32).reshape(1, -1)
                
                # 2. Run the ONNX inference session
                raw_output = session.run(["output"], {"input": clip_16k_formatted})
                
                # 3. Transform the raw numerical output into your desired text labels
                # (Assuming a standard binary classification output format; adjust indexing if needed)
                score = raw_output[0][0][0]
                prediction = "wake" if score > 0.5 else "nonwake"
                
                # 4. Print horizontally for easy terminal monitoring
                print(f"[{prediction}] ", end="", flush=True)
                
                # Check if the manual record button in the UI was clicked
                manual_trigger = False
                if os.path.exists("manual_trigger.txt"):
                    os.remove("manual_trigger.txt")
                    manual_trigger = True

                if prediction == "nonwake":
                    self.noise_hist.append((time.monotonic(), float(np.sqrt(np.mean(clip_16k ** 2)))))
                
                if prediction == "wake" or manual_trigger:
                    # --- NEW: Tell UI we are in Voice Command Mode ---
                    try:
                        requests.post('http://127.0.0.1:5000/set_status', json={'status': 'recording'})
                    except Exception:
                        pass

                    print("\n[System] Command trigger detected. Pausing background listener...")
                    
                    # 1. Temporarily stop the 1.5-second clip recording
                    self.stream.stop()

                    # 2. Sound the beep
                    beep_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sounds', 'beep.mp3'))
                    player = vlc.MediaPlayer(beep_path)
                    player.play()
                    time.sleep(0.5) # Allow beep to finish

                    # 3. Record until the speaker stops (max 5 s)
                    print("[System] Listening for command...")
                    command_audio = self.record_command(self.get_noise_threshold())

                    # 4. Zero-pad to exactly 5 s (80000 samples) for the VCM
                    target_len = int(5.0 * self.target_sr)
                    padded_audio = np.zeros(target_len, dtype=np.float32)
                    padded_audio[:min(len(command_audio), target_len)] = command_audio[:target_len]

                    # NEW: Save the first 5 clips as WAV files to verify the 5-second padding
                    if self.saved_clip_count < 5:
                        self.saved_clip_count += 1
                        clip_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'vcm_debug_clip_{self.saved_clip_count}.wav'))
                        sf.write(clip_filename, padded_audio, self.target_sr)
                        print(f"[System] Saved VCM debug clip: vcm_debug_clip_{self.saved_clip_count}.wav")

                    # Reshape to (Batch=1, Channels=1, Time=80000)
                    vcm_input = padded_audio.reshape(1, 1, -1)

                    # Run ONNX inference
                    vcm_output = vcm_session.run(["output"], {"input": vcm_input})[0]

                    # Decode the CTC Output (Extract max probability and ignore blank tokens [index 0])
                    preds = np.argmax(vcm_output, axis=1)[0]
                    decoded_tokens = []
                    prev_token = -1
                    for p in preds:
                        if p != prev_token and p != 0: 
                            decoded_tokens.append(vocab[p])
                        prev_token = p

                    transcribed_command = " ".join(decoded_tokens)
                    
                    # 5. Display the final output in the terminal
                    print(f"[VCM Output]: {transcribed_command}")

                    # Pass the transcribed text back to the UI's intent parser to control the house
                    try:
                        requests.post('http://127.0.0.1:5000/voice_result', json={'text': transcribed_command}, timeout=3)
                    except Exception:
                        pass

                    # --- NEW: Tell UI to go back to Background Listening ---
                    try:
                        requests.post('http://127.0.0.1:5000/set_status', json={'status': 'listening'})
                    except Exception:
                        pass

                    # 6. Resume the 1.5-second continuous recordings
                    self.buffer.fill(0) # Clear stale audio to prevent instant re-triggering
                    self.buffer_index = 0
                    self.stream.start()

if __name__ == "__main__":
    try:
        engine = WakeWordEngine()
        engine.start_listening()
    except KeyboardInterrupt:
        print("\nWake Word Engine stopped.")
        sys.exit(0)