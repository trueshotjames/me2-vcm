# Changes — 13 October 2026

The project was tested end to end on the Raspberry Pi 4 (Raspberry Pi OS, Python 3.13.5, M-930 USB mic) by running `run.sh`. Two problems stopped the voice pipeline. Both are fixed below, along with smaller fixes to the launchers (`run.sh` and `run.bat`), the server and the README.

## Problems found

1. **The wake word engine crashed on start.** `import onnxruntime` killed Python with `Illegal instruction` (exit code 132), before the engine printed anything. The onnxruntime in use was Debian's apt package `python3-onnxruntime` 1.21, which does not run on the Pi 4's CPU (Cortex-A72). The PyPI build of onnxruntime (1.31.0) runs fine. `run.sh` called the system `python3`, so it always picked up the broken apt copy.
2. **A command with no words hung the engine.** When a command recording reached the 5 s limit without the model hearing a word (a quiet room, or pressing *Record command* and saying nothing), the recording never ended. The buffer stops growing at 5000 ms, but the next decode check was scheduled just past it (5029–5040 ms), so the stop check never ran again. The engine stayed stuck in `record_command()`, and the wake word stopped working until restart. This was confirmed with a stack dump and a simulation on silent audio.
3. **`run.sh` had launcher problems:**
   - It did not change into the project folder, so starting it from anywhere else broke the record button and the alarm/reminder files.
   - It opened the browser before the server was up.
   - It left the wake word engine running (and holding the mic) after Ctrl+C.
4. **A leftover `manual_trigger.txt`** in the folder made the engine start recording a command the moment it launched.
5. **`main_ui.py`** still had hard-coded Windows paths (`C:\Users\King\...`) for the playlist and ringtone, and imported `os` twice.

## Changes made

### `run.sh` — new Pi / Linux launcher (the Linux counterpart of `run.bat`)

`run.sh` is the launcher for the Pi. It:

- changes into its own folder first, so it works from a desktop shortcut or autostart;
- creates `.venv` and installs `requirements.txt` on the first run if `.venv` is missing;
- checks that onnxruntime loads in `.venv`, and stops with a clear message if it does not;
- deletes a leftover `manual_trigger.txt`;
- starts the server, waits until it answers, then starts the wake word engine and opens `http://127.0.0.1:5000/?lite=1` in the default browser (`?lite=1` turns off the fade animations to save CPU on the Pi);
- stops the server and the engine together on Ctrl+C, when the terminal is closed, or when the server exits;
- keeps the window open with "Press Enter to exit..." if the server stops with an error, so the error can be read.

### `run.bat` — updated Windows launcher, same steps as `run.sh`

- Changes into its own folder (`cd /d "%~dp0"`), so it works from any folder.
- Creates `.venv` and installs `requirements.txt` on the first run if `.venv` is missing, then checks that onnxruntime loads.
- Stops a wake word engine left over from an earlier run (it would hold the mic), and deletes a leftover `manual_trigger.txt`.
- Starts the wake word engine in its own window, titled "VCM Wake Word Engine", as before. Opens `http://127.0.0.1:5000/` only once the server answers.
- When the server stops, it closes the engine window too.
- Windows has no way to catch Ctrl+C in a batch file. Pressing Ctrl+C in the server window asks *Terminate batch job (Y/N)?*: answer **N** so the engine window closes. If you answer Y, the engine window stays open, but the next `run.bat` stops it anyway.
- The engine is stopped by looking for `wake_word_engine.py` in each process's command line (PowerShell `Get-CimInstance`). That also stops an engine you started by hand in another terminal, which would conflict anyway.
- The file now has Windows (CRLF) line endings.

**Not tested:** this Pi has no Windows or PowerShell, so `run.bat` has not been run. See "What you need to do".

### `.gitattributes` — new

Makes Git check out `.bat` files with Windows (CRLF) line endings and `.sh` files with Linux (LF) line endings, on any computer. Without it, a batch file with LF endings can break `goto` and labels on Windows. A `run.sh` checked out on Windows with CRLF endings would fail on the Pi.

### `requirements.txt` — new

The Python packages for `.venv`: flask, openpyxl, requests, python-vlc, numpy, scipy, sounddevice, soundfile, onnxruntime, psutil.

### `.venv` — created on this Pi (not committed; `.gitignore` already excludes it)

Built from `requirements.txt` with packages from PyPI. Installed versions include onnxruntime 1.31.0, numpy 2.5.4 and scipy 1.18.1, all loading without errors.

### `voice_pipeline/wake_word_engine.py`

- `record_command()` now also runs the decode check once the 5 s buffer is full, so a recording with no words ends with "max length" instead of hanging (problem 2).

### `main_ui.py`

- Removed the hard-coded Windows playlist and ringtone paths; it now uses `playlist/` and `sounds/` in the repository directly.
- Removed the duplicate `import os`.
- Deletes a leftover `manual_trigger.txt` when the server starts. This covers starting the programs by hand without `run.sh`.

### `README.md`

- The file tree lists `run.sh`, `requirements.txt` and `CHANGE.md`.
- Pi setup step 3 installs with `pip install -r requirements.txt`, and warns not to use apt's `python3-onnxruntime`.
- The file tree describes the updated `run.bat` and lists `.gitattributes`. Pi step 5 explains how to start and stop `run.bat` on Windows.
- Pi step 5 starts everything with `./run.sh`. The manual commands now activate `.venv` first and use `chromium` (the browser command on current Raspberry Pi OS) instead of `chromium-browser`.

## Test results after the changes

`run.sh` was started from a different folder (`/tmp`) and then:

- the server answered `/get_status`, and every page, data, image and ringtone endpoint returned 200;
- the wake word engine started, loaded both models and listened to the microphone;
- a *Record command* press beeped, recorded for 5 s, decoded, and the result reached the server (`voice_result` id 0 → 1, end reason "max length", since nobody spoke);
- music playback through VLC worked;
- a simulated Ctrl+C stopped both the server and the engine, leaving no processes behind.

Correction to the first report: `models/vocab.txt` has **120** entries, matching the model and the README. `wc -l` showed 119 only because the file has no newline at the end.

## What you need to do

1. **Test the wake word with your voice.** I could not speak into the mic. Run `./run.sh`, say "Alexa", wait for the beep, and say a command such as "turn on the lights" or "set the temperature to twenty five degrees". The grammar needs "degrees" at the end.
2. **Commit and push the changes:** `run.sh`, `run.bat`, `.gitattributes`, `requirements.txt`, `CHANGE.md`, `main_ui.py`, `README.md` and `voice_pipeline/wake_word_engine.py`. `run.sh` was never committed before.
3. **On a fresh clone or another Pi,** the first `./run.sh` needs internet to build `.venv` (a few minutes). After that it starts offline.
4. **Optional:** `sudo apt remove python3-onnxruntime` removes the broken apt copy. It is not required, since `.venv` does not use it, but nothing else picks it up by mistake afterwards.
5. **Test `run.bat` on your Windows laptop.** Pull the changes first. If your clone already has `run.bat` checked out, run `git add --renormalize .` once, or delete `run.bat` and run `git checkout run.bat`, so it gets Windows line endings.
   - Python 3 must be on PATH (`python --version` in a terminal), and 64-bit VLC must be installed.
   - The first double-click builds `.venv`, which needs internet and takes a few minutes.
   - Check that the server window starts, a "VCM Wake Word Engine" window opens, and the browser opens the page once the server is up.
   - Then press Ctrl+C in the server window, answer N, and check that the engine window closes.
   - `main_ui.py` now always plays from the repo's `playlist/` folder instead of `C:\Users\King\...\Machine Exercise 2\playlist`. Copy any extra songs from the old folder into it.
