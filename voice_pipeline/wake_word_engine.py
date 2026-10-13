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
import queue
import threading
from intent_parser import BeamDecoder
# --- Command endpointing (all in milliseconds unless noted) ---
# A "word" is any VCM token that is neither [BLANK] nor [UNK].
END_WAIT_MS        = 200   # wait for another word after the last one (everywhere except below)
UNFINISHED_WAIT_MS = 500   # wait for the next word inside an unfinished command ("set the temperature to ...")
CONFIRM_STEPS      = 3     # 40 ms steps of no voice before a finished command stops the recording
NOISY_STOP_MS      = 500   # stop a finished command after this long even if the mic still hears sound
MIN_MARGIN         = 0.0   # how much the command must beat the "filler only" score to count as finished
NO_SPEECH_SEC      = 5.0   # give up if no word at all
MAX_COMMAND_SEC    = 5.0   # hard cap on the recording
BEEP_GUARD_MS      = 30    # skip this much mic audio after the beep ends (speaker echo)
BEEP_PEAK          = 0.7   # fixed beep loudness (0-1 of full scale), independent of the music volume
DECODE_TICK_MS     = 80    # how often the VCM re-runs on the audio so far while recording

try:
    import benchmark_adapter
except ImportError:
    benchmark_adapter = None

def _benchmark_mode():
    if benchmark_adapter is None:
        return False
    return benchmark_adapter.is_benchmark_mode()

# Point the system path to the /models folder to access the attached model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

# --- Load the Wake Word Model ---
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'wake_word_model_quantized.onnx'))
session = ort.InferenceSession(model_path)

# --- NEW: Load the Voice Command Model (VCM) and Vocabulary ---
vcm_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'voice_command_model.onnx'))
vcm_session = ort.InferenceSession(vcm_model_path)

vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'vocab.txt'))
with open(vocab_path, "r") as f:
    vocab = [line.strip() for line in f.readlines() if line.strip()]

# Updated VCM (causal conv, CTC): input [1, 40, frames] log-Mel, output [1, 120, steps] raw scores,
# one step per 40 ms. The CMVN (fixed mean/std) is built into the ONNX graph.
# vocab.txt: [UNK] first, [BLANK] last (CTC blank = 119).
VCM_BLANK_ID = vocab.index('[BLANK]') if '[BLANK]' in vocab else len(vocab) - 1
VCM_UNK_ID = vocab.index('[UNK]') if '[UNK]' in vocab else 0
_vcm_classes = vcm_session.get_outputs()[0].shape[1]
if len(vocab) != _vcm_classes:
    print(f"[VCM] Warning: vocab.txt has {len(vocab)} entries but the model outputs {_vcm_classes} classes.")
VCM_STEP_MS = 40          # one VCM output step per 40 ms of audio
VCM_UNSTABLE_STEPS = 1    # the newest step can still change when more audio arrives (causal model)
VCM_FLUSH_MS = 600        # silence added after the audio so far, so the VCM emits words it is still holding back
VOICE_RANGE_DB = 30       # mic frames within this many dB of the loudest one count as voice

# --- VCM front end: 40 log-Mel bands, 25 ms Hann window, 10 ms hop (blueprint Stage 2) ---
# Matches torchaudio MelSpectrogram(n_fft=400, hop_length=160, n_mels=40) defaults + log(x + 1e-6).
VCM_N_FFT, VCM_HOP, VCM_N_MELS = 400, 160, 40

def _vcm_mel_filterbank(sr=16000, n_fft=VCM_N_FFT, n_mels=VCM_N_MELS):
    hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    slopes = pts[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / np.diff(pts)[:-1]
    up = slopes[:, 2:] / np.diff(pts)[1:]
    return np.maximum(0.0, np.minimum(down, up)).T.astype(np.float32)   # [n_mels, n_fft//2+1]

_VCM_MEL_FB = _vcm_mel_filterbank()
_VCM_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(VCM_N_FFT) / VCM_N_FFT)).astype(np.float32)

def _vcm_mel_frames(x_padded, first, last):
    """Log-Mel frames first..last-1 of already reflect-padded audio -> [40, last - first]."""
    x = x_padded[first * VCM_HOP:]
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(last - first, VCM_N_FFT), strides=(x.strides[0] * VCM_HOP, x.strides[0]))
    power = np.abs(np.fft.rfft(frames * _VCM_WINDOW, n=VCM_N_FFT)) ** 2
    return np.log(_VCM_MEL_FB @ power.T + 1e-6).astype(np.float32)

def vcm_log_mel(audio_16k):
    """1D float32 audio at 16 kHz -> [1, 40, frames] log-Mel features for the VCM."""
    x = np.pad(audio_16k.astype(np.float32), (VCM_N_FFT // 2, VCM_N_FFT // 2), mode='reflect')
    n_frames = 1 + (len(x) - VCM_N_FFT) // VCM_HOP
    return _vcm_mel_frames(x, 0, n_frames)[None]

def log_softmax(vcm_output):
    """[1, classes, steps] raw VCM scores -> [steps, classes] log-probabilities."""
    scores = vcm_output[0].T
    scores = scores - scores.max(axis=1, keepdims=True)
    return scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))

class CommandDecoder:
    """Decodes a command while it is being spoken and decides when it is finished.

    add_audio() takes 16 kHz audio; tick() re-runs the VCM on the audio so far and returns a
    reason string once the recording should stop (None to keep going); finish() decodes
    everything and returns the result.

    The VCM emits a word up to ~0.5 s after it was spoken, often in a burst once the
    speaker pauses. So each tick also runs the model on the audio plus VCM_FLUSH_MS of
    silence (like the zero padding it was trained with): that shows every word spoken so
    far right away, and the stop rules below use that "flushed" decode.
    """

    def __init__(self, sr=16000):
        self.sr = sr
        self.audio = np.zeros(int(sr * MAX_COMMAND_SEC), dtype=np.float32)
        self.n = 0
        self.mel_done = np.zeros((VCM_N_MELS, 0), dtype=np.float32)   # frames that can no longer change
        self.decoder = BeamDecoder(vocab, VCM_BLANK_ID, VCM_UNK_ID)   # settled steps only
        self.pushed = 0                 # VCM steps already given to self.decoder
        self.words = []                 # words in the latest flushed decode (greedy, no [BLANK]/[UNK])
        self.last_word_ms = None        # audio time when the flushed words last changed
        self.voice = VoiceActivity(sr)
        self.finished_since_ms = None   # audio time since when a finished command has been on top
        self.flushed = None             # (log_probs, decoder) of the latest flushed decode

    def add_audio(self, x):
        x = x[:len(self.audio) - self.n]
        self.audio[self.n:self.n + len(x)] = x
        self.n += len(x)
        self.voice.add(x)

    @property
    def audio_ms(self):
        return self.n * 1000.0 / self.sr

    def _features(self, extra_zeros):
        """Log-Mel of the audio so far plus extra_zeros samples of silence.
        Frames that only cover recorded audio are computed once and reused."""
        half = VCM_N_FFT // 2
        x = np.pad(np.concatenate([self.audio[:self.n], np.zeros(extra_zeros, np.float32)]), (half, half), mode='reflect')
        total = 1 + (len(x) - VCM_N_FFT) // VCM_HOP
        settled = min(total, (self.n - half) // VCM_HOP + 1)   # frames inside the recorded audio
        done = self.mel_done.shape[1]
        if settled > done:
            self.mel_done = np.concatenate([self.mel_done, _vcm_mel_frames(x, done, settled)], axis=1)
            done = settled
        return np.concatenate([self.mel_done[:, :total], _vcm_mel_frames(x, done, total)], axis=1)[None]

    def _decode_flushed(self):
        """Runs the VCM on audio + silence; settles the steps that can't change and decodes the rest on a copy."""
        flush = int(self.sr * VCM_FLUSH_MS / 1000)
        log_probs = log_softmax(vcm_session.run(["output"], {"input": self._features(flush)})[0])
        real_steps = -(-(1 + self.n // VCM_HOP) // 4)   # VCM: one step per 4 frames (rounded up)
        settled = max(self.pushed, real_steps - VCM_UNSTABLE_STEPS)
        if settled > self.pushed:
            self.decoder.push(log_probs[self.pushed:settled])
            self.pushed = settled
        flushed = self.decoder.copy().push(log_probs[self.pushed:])
        self.flushed = (log_probs, flushed)
        return log_probs, flushed

    def tick(self):
        if self.audio_ms >= MAX_COMMAND_SEC * 1000:
            return "max length"
        if self.n <= VCM_N_FFT:
            return None
        log_probs, flushed = self._decode_flushed()
        now = self.audio_ms

        words = [int(k) for k in _collapse(np.argmax(log_probs, axis=1)) if k != VCM_UNK_ID]
        if words != self.words:
            self.words, self.last_word_ms = words, now
        if self.last_word_ms is None:
            return "no speech" if now >= NO_SPEECH_SEC * 1000 else None

        # The wait restarts on a new word, and while the mic still hears the voice
        # (a word in progress is not in the decode yet).
        quiet_ms = now - max(self.last_word_ms, self.voice.last_voice_ms)
        st = flushed.status()
        finished = st["state"] == "complete" and st["phrase"] and (st["margin"] is None or st["margin"] >= MIN_MARGIN)
        if finished and not st["can_continue"]:
            if self.finished_since_ms is None:
                self.finished_since_ms = now
            # Stop once the voice has paused, or after NOISY_STOP_MS if the room is too noisy to tell
            if now - self.voice.last_voice_ms >= CONFIRM_STEPS * VCM_STEP_MS or now - self.finished_since_ms >= NOISY_STOP_MS:
                return "command finished"
            return None
        self.finished_since_ms = None
        if st["state"] == "unfinished":
            return "pause in command" if quiet_ms >= UNFINISHED_WAIT_MS else None
        return "no more words" if quiet_ms >= END_WAIT_MS else None

    def finish(self):
        """Decodes all audio recorded so far (plus the flush silence) -> dict for the UI and the benchmark."""
        t0 = time.perf_counter()
        if self.n > VCM_N_FFT:
            log_probs, flushed = self._decode_flushed()
        else:
            log_probs, flushed = np.zeros((0, len(vocab)), dtype=np.float32), self.decoder
        infer_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        text, details = flushed.result(return_details=True)
        decode_ms = (time.perf_counter() - t1) * 1000.0
        heard = ' '.join(vocab[k] for k in _collapse(np.argmax(log_probs, axis=1)))   # greedy transcript, display only
        tail_ms = None if self.last_word_ms is None else self.audio_ms - max(self.last_word_ms, self.voice.last_voice_ms)
        return {"text": text, "heard": heard, "details": details, "infer_ms": infer_ms,
                "decode_ms": decode_ms, "audio_ms": self.audio_ms, "tail_ms": tail_ms,
                "audio": self.audio[:self.n].copy()}

def _collapse(tokens):
    """CTC collapse: merge repeats, drop [BLANK]."""
    out, prev = [], -1
    for t in tokens:
        if t != prev and t != VCM_BLANK_ID:
            out.append(t)
        prev = t
    return out

class VoiceActivity:
    """Tracks when the mic last heard a voice, from 40 ms frame loudness.

    A frame counts as voice when it is within VOICE_RANGE_DB of the loudest frame so far
    and clearly above the quietest frames (the room's noise floor)."""

    def __init__(self, sr):
        self.frame = int(sr * VCM_STEP_MS / 1000)
        self.sr = sr
        self.pending = np.zeros(0, np.float32)
        self.levels = []
        self.last_voice_ms = 0.0

    def add(self, x):
        self.pending = np.concatenate([self.pending, x])
        while len(self.pending) >= self.frame:
            f, self.pending = self.pending[:self.frame], self.pending[self.frame:]
            rms = float(np.sqrt(np.mean(f ** 2))) + 1e-9
            self.levels.append(rms)
            floor = float(np.percentile(self.levels, 10))
            if rms > max(self.levels) * 10 ** (-VOICE_RANGE_DB / 20) and rms > floor * 2.0:
                self.last_voice_ms = len(self.levels) * VCM_STEP_MS

class Decimator:
    """Streams native-rate mic audio down to 16 kHz in pieces without clicks at the piece edges."""

    def __init__(self, native_sr, target_sr):
        if native_sr % target_sr:
            raise ValueError(f"mic rate {native_sr} is not a multiple of {target_sr}")
        self.factor = native_sr // target_sr
        self.taps = scipy.signal.firwin(97, 0.45 * target_sr, fs=native_sr)
        self.zi = np.zeros(len(self.taps) - 1)
        self.offset = 0   # index of the next sample to keep in the next piece

    def process(self, x):
        y, self.zi = scipy.signal.lfilter(self.taps, 1.0, x, zi=self.zi)
        out = y[self.offset::self.factor]
        self.offset = (self.offset - len(x)) % self.factor
        return out.astype(np.float32)

# --- Beep before a VCM command: play it, wait until it has finished, then record ---
BEEP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sounds', 'beep.mp3'))
try:
    _beep_audio, _beep_sr = sf.read(BEEP_PATH, dtype='float32')
    _beep_audio = _beep_audio * (BEEP_PEAK / max(float(np.abs(_beep_audio).max()), 1e-6))   # fixed loudness
except Exception as e:
    _beep_audio, _beep_sr = None, None
    print(f"[System] Can't decode beep.mp3 with soundfile ({e}); the beep will use VLC.")

def play_beep():
    """Plays the beep and returns once it has finished."""
    if _beep_audio is not None:
        try:
            sd.play(_beep_audio, _beep_sr)
            sd.wait()
            return
        except Exception as e:
            print(f"[System] Beep playback failed ({e}); using VLC.")
    player = vlc.MediaPlayer(BEEP_PATH)
    player.audio_set_volume(100)   # its own player, so the music volume doesn't affect it
    player.play()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:   # wait for VLC to report the end, at most 1 s
        if player.get_state() in (vlc.State.Ended, vlc.State.Error):
            break
        time.sleep(0.01)

def post_async(route, payload, timeout=1):
    """Status updates for the UI go out on a side thread so they never hold up the audio."""
    def send():
        try:
            requests.post('http://127.0.0.1:5000' + route, json=payload, timeout=timeout)
        except Exception:
            pass
    threading.Thread(target=send, daemon=True).start()

class WakeWordEngine:
    def __init__(self):
        if len(vocab) != _vcm_classes:
            sys.exit(f"[VCM] vocab.txt has {len(vocab)} words but voice_command_model.onnx outputs "
                     f"{_vcm_classes} classes. Export a model trained with this vocab.txt.")

        # Hardware native rate and target model rate[cite: 12]
        self.native_sr = 48000
        self.target_sr = 16000
        self.saved_clip_count = 0  # Tracks the number of saved VCM clips

        # Timing constraints
        self.window_duration = 1.5
        self.hop_duration = 0.5

        # Allocate a rolling 3-second recording loop buffer in RAM[cite: 12]
        self.buffer_duration = 3.0
        self.buffer_size = int(self.buffer_duration * self.native_sr)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0

        # While a command is recorded, the same stream also feeds this queue
        self.capturing = False
        self.command_queue = queue.Queue()

        # Open a non-blocking background microphone stream (stays open, also used for commands)
        self.stream = sd.InputStream(
            samplerate=self.native_sr,
            channels=1,
            dtype='float32',
            callback=self.audio_callback
        )

    def audio_callback(self, indata, frames, time, status):
        """Continuously overwrites the oldest audio in the ring buffer."""
        if status.input_overflow:
            print("\n[System] Mic overflow: some audio was dropped.")
        if self.capturing:
            self.command_queue.put(indata[:, 0].copy())

        end_idx = self.buffer_index + frames
        if end_idx < self.buffer_size:
            self.buffer[self.buffer_index:end_idx] = indata[:, 0]
        else:
            overflow = end_idx - self.buffer_size
            self.buffer[self.buffer_index:self.buffer_size] = indata[:frames - overflow, 0]
            self.buffer[0:overflow] = indata[frames - overflow:, 0]

        self.buffer_index = (self.buffer_index + frames) % self.buffer_size

    def record_command(self):
        """Records from the open mic stream and decodes while the person speaks.

        Stops as soon as the command is finished (see the endpointing settings at the top)
        and returns (CommandDecoder.finish() result, reason the recording stopped).
        """
        decimator = Decimator(self.native_sr, self.target_sr)
        decoder = CommandDecoder(self.target_sr)
        while not self.command_queue.empty():   # drop anything from before the beep
            self.command_queue.get_nowait()
        self.capturing = True
        reason, next_tick = None, DECODE_TICK_MS
        try:
            while reason is None:
                try:
                    chunk = self.command_queue.get(timeout=1.0)
                except queue.Empty:
                    reason = "mic stopped"
                    break
                decoder.add_audio(decimator.process(chunk))
                if decoder.audio_ms >= next_tick:
                    next_tick = decoder.audio_ms + DECODE_TICK_MS
                    reason = decoder.tick()
        finally:
            self.capturing = False
        return decoder.finish(), reason

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

                if prediction == "wake" or manual_trigger:
                    try:
                        self.handle_command()
                    except Exception as e:
                        print(f"\n[System] Command failed: {e!r}")
                    finally:
                        # Tell UI to go back to Background Listening
                        post_async('/set_status', {'status': 'listening'})
                        self.capturing = False
                        self.buffer.fill(0) # Clear stale audio to prevent instant re-triggering
                        self.buffer_index = 0

    def handle_command(self):
        """Beep, record until the command is finished, decode, and send the result to the UI."""
        # Tell UI we are in Voice Command Mode
        post_async('/set_status', {'status': 'recording'})
        print("\n[System] Command trigger detected.")

        # 1. Sound the beep and wait until it has finished (the mic stream keeps running)
        play_beep()
        time.sleep(BEEP_GUARD_MS / 1000.0)

        # 2. Record and decode until the command is finished (max MAX_COMMAND_SEC)
        print("[System] Listening for command...")
        result, reason = self.record_command()
        post_async('/set_status', {'status': 'decoding'})

        transcribed_command, decode_details = result["text"], result["details"]
        print(f"[VCM Raw]: {result['heard']}")
        print(f"[VCM Output]: {transcribed_command}  ({reason}, {result['audio_ms']:.0f} ms of audio"
              + (f", stopped {result['tail_ms']:.0f} ms after the last word)" if result['tail_ms'] is not None else ")"))

        # 3. Pass the command to the UI, which runs it
        try:
            requests.post('http://127.0.0.1:5000/voice_result', json={
                'text': transcribed_command, 'heard': result['heard'],
                'top': decode_details['top'], 'margin': decode_details['margin'],
                'infer_ms': round(result['infer_ms'], 1), 'decode_ms': round(result['decode_ms'], 1),
                'audio_ms': round(result['audio_ms']), 'end_reason': reason}, timeout=3)
        except Exception:
            pass

        if _benchmark_mode():
            benchmark_adapter.log_benchmark_result(transcribed_command, result['infer_ms'],
                                                   audio_ms=round(result['audio_ms']))

        # Save the first 5 commands as 5 s WAV files (zero-padded) for checking
        if self.saved_clip_count < 5:
            self.saved_clip_count += 1
            target_len = int(5.0 * self.target_sr)
            padded_audio = np.zeros(target_len, dtype=np.float32)
            recorded = result['audio'][:target_len]
            padded_audio[:len(recorded)] = recorded
            clip_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'vcm_debug_clip_{self.saved_clip_count}.wav'))
            sf.write(clip_filename, padded_audio, self.target_sr)
            print(f"[System] Saved VCM debug clip: vcm_debug_clip_{self.saved_clip_count}.wav")

if __name__ == "__main__":
    try:
        engine = WakeWordEngine()
        engine.start_listening()
    except KeyboardInterrupt:
        print("\nWake Word Engine stopped.")
        sys.exit(0)
