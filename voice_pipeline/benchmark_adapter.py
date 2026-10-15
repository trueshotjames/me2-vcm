# voice_pipeline/benchmark_adapter.py
#
# Integrates the airimonda/vcm-benchmark into the existing voice pipeline.
# Self-contained: depends only on the Python standard library plus the existing
# intent_parser module (same directory).
#
# The benchmark reads one JSON object per line from the newest *.log file in
# ~/vcm_benchmark/. Each command line records the mapped intent name, the formatted
# slot string, and the inference/audio timings; a {"event": "wake"} line marks
# each wake word detection.

import os
import sys
import json
from datetime import datetime

# Make sure the sibling intent_parser module is importable regardless of the
# caller's current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intent_parser import parse_text


# 1. Map intent_parser action names -> the benchmark's 19 intent names.
#    'toggle' is intentionally absent: it is resolved by the toggle tracker
#    below into LIGHT_ON / LIGHT_OFF.
INTENT_MAP = {
    "music_play": "PLAY_MUSIC",
    "music_pause": "PAUSE",
    "music_stop": "STOP",
    "music_skip": "NEXT",
    "music_vol_up": "VOLUME_UP",
    "music_vol_down": "VOLUME_DOWN",
    "weather": "WEATHER",
    "time": "TIME",
    "call_mom": "CALL",
    "call_dad": "CALL",
    "message": "MESSAGE",
    "get_reminders": "LIST_REMINDERS",
    "timer": "TIMER",
    "alarm": "ALARM",
    "temp": "TEMPERATURE",
    "dim": "BRIGHTNESS",
    "color": "COLOR",
    "reminder": "CREATE_REMINDER",
    "Unrecognized": "OUT_OF_SCOPE",
}


# 2. Toggle state tracker. intent_parser maps both "lights on" and
#    "lights out" to the action 'toggle', with value "on" or "off".
#    The value decides LIGHT_ON / LIGHT_OFF; without one, this module-level
#    flag alternates between them.
_light_on = False


def _resolve_toggle(value=None):
    """Return the toggle intent ("on"/"off" value first, else alternate) and update the tracker."""
    global _light_on
    if value in ("on", "off"):
        _light_on = value == "on"
        return "LIGHT_ON" if _light_on else "LIGHT_OFF"
    if not _light_on:
        _light_on = True
        return "LIGHT_ON"
    _light_on = False
    return "LIGHT_OFF"


# 3. Convert an intent_parser value into a benchmark-compatible slot string.
def format_slot(action, value):
    if action == "timer":
        # value is a dict like {"minutes": 1, "seconds": 30}
        minutes = int(value.get("minutes", 0))
        seconds = int(value.get("seconds", 0))
        return "{} seconds".format(minutes * 60 + seconds)

    if action == "alarm":
        # value is 'HH:MM:SS' -> 'H:MM AM/MM'
        parts = str(value).split(":")
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        period = "AM" if hours < 12 else "PM"
        hour12 = hours % 12
        if hour12 == 0:
            hour12 = 12
        return "{}:{:02d} {}".format(hour12, minutes, period)

    if action == "temp":
        return "{} degrees".format(int(value))

    if action == "dim":
        return "{} percent".format(int(value))

    if action in ("color", "reminder"):
        return str(value)

    # Non-slotted intents (and 'toggle') carry no slot.
    return ""


# 4. One log file per engine run, opened once and kept open. The benchmark follows the
#    newest *.log in ~/vcm_benchmark/ and finds this process by the log file it holds open
#    (for its CPU/RAM numbers), so the file must stay open for the whole run.
_log = None


def open_log():
    """Opens ~/vcm_benchmark/benchmark_<timestamp>.log (once) and returns it."""
    global _log
    if _log is None:
        log_dir = os.path.join(os.path.expanduser("~"), "vcm_benchmark")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, "benchmark_{}.log".format(timestamp))
        _log = open(path, "a", encoding="utf-8", buffering=1)
        print("[Benchmark] Logging to {}".format(path))
    return _log


def _write(record):
    fh = open_log()
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def log_wake():
    """One line when the wake word fires: gives the benchmark's wake detection rate."""
    _write({"event": "wake"})


# 5. Parse transcribed text, map it, and append one JSON line to the log.
def log_benchmark_result(transcribed_text, infer_ms, audio_ms=5000):
    action, value = parse_text(transcribed_text)

    if action == "toggle":
        intent = _resolve_toggle(value)
    else:
        intent = INTENT_MAP.get(action, "OUT_OF_SCOPE")

    slot = format_slot(action, value)

    _write({
        "intent": intent,
        "slot": slot,
        "infer_ms": round(infer_ms, 1),
        "audio_ms": audio_ms,
    })

    print("Logged intent '{}' slot='{}' to {}".format(intent, slot, _log.name))


# 6. Benchmark mode is enabled by the VCM_BENCHMARK environment variable.
def is_benchmark_mode():
    return os.environ.get("VCM_BENCHMARK") == "1"
