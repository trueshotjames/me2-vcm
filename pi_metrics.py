# pi_metrics.py
#
# Resource readings for the "Pi health" card: temperature, CPU (total and per part of the
# project), memory, clock speed and throttling.
# On the Pi (Linux) everything is read from the kernel's own files (/proc, /sys) and
# `vcgencmd`, so no extra library is needed. Elsewhere (the Windows laptop) psutil is used
# if it is installed; temperature and throttling are then reported as None.

import os
import subprocess
import sys
import time

try:
    import psutil
except ImportError:
    psutil = None

THROTTLE_CHECK_SEC = 10     # vcgencmd is the one slow call (~10-20 ms), so it runs less often
PROCESS_SCAN_SEC = 10       # how often to look for the engine / browser processes again

# Parts of the project, by what their command line contains
PARTS = {
    "engine": ("wake_word_engine.py",),
    "browser": ("chromium", "chrome", "msedge", "firefox"),
}

_HAS_PROC = os.path.exists("/proc/stat")
_CORES = os.cpu_count() or 1
_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") and _HAS_PROC else 100


def available():
    return _HAS_PROC or psutil is not None


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


class Collector:
    def __init__(self):
        self.prev_cpu = None          # (busy, total) jiffies, Linux
        self.prev_proc = {}           # pid -> (cpu seconds, timestamp)
        self.parts = {}               # pid -> part name
        self.last_scan = 0.0
        self.throttle = None
        self.last_throttle = 0.0
        self.ps_procs = {}            # pid -> psutil.Process (keeps cpu_percent state)

    # --- process lists ---
    def _scan(self):
        parts = {os.getpid(): "server"}
        if _HAS_PROC:
            for pid in os.listdir("/proc"):
                if not pid.isdigit() or int(pid) == os.getpid():
                    continue
                cmd = (_read(f"/proc/{pid}/cmdline") or "").replace("\0", " ").lower()
                for part, keys in PARTS.items():
                    if any(k in cmd for k in keys):
                        parts[int(pid)] = part
                        break
        elif psutil is not None:
            for p in psutil.process_iter(["pid", "name", "cmdline"]):
                if p.info["pid"] == os.getpid():
                    continue
                cmd = " ".join(p.info["cmdline"] or [p.info["name"] or ""]).lower()
                for part, keys in PARTS.items():
                    if any(k in cmd for k in keys):
                        parts[p.info["pid"]] = part
                        break
        self.parts = parts
        self.last_scan = time.monotonic()

    def _proc_cpu_linux(self, pid):
        stat = _read(f"/proc/{pid}/stat")
        if not stat:
            return None
        fields = stat.rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / _CLK_TCK   # utime + stime

    def _part_cpu(self):
        """CPU % of the whole machine used by each part since the last reading."""
        now = time.monotonic()
        usage = {"engine": 0.0, "server": 0.0, "browser": 0.0}
        if _HAS_PROC:
            seen = {}
            for pid, part in self.parts.items():
                t = self._proc_cpu_linux(pid)
                if t is None:
                    continue
                seen[pid] = (t, now)
                if pid in self.prev_proc:
                    t0, n0 = self.prev_proc[pid]
                    if now > n0:
                        usage[part] += 100.0 * (t - t0) / (now - n0) / _CORES
            self.prev_proc = seen
        else:
            for pid, part in self.parts.items():
                proc = self.ps_procs.get(pid)
                try:
                    if proc is None:
                        proc = self.ps_procs[pid] = psutil.Process(pid)
                        proc.cpu_percent(None)   # first call only starts the measurement
                        continue
                    usage[part] += proc.cpu_percent(None) / _CORES
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self.ps_procs.pop(pid, None)
            for pid in list(self.ps_procs):
                if pid not in self.parts:
                    del self.ps_procs[pid]
        return {k: round(v, 1) for k, v in usage.items()}

    # --- machine readings ---
    def _cpu_total(self):
        if _HAS_PROC:
            vals = [int(v) for v in (_read("/proc/stat") or "cpu 0").splitlines()[0].split()[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            pct = None
            if self.prev_cpu and total > self.prev_cpu[1]:
                pct = 100.0 * (1 - (idle - self.prev_cpu[0]) / (total - self.prev_cpu[1]))
            self.prev_cpu = (idle, total)
            return None if pct is None else round(pct, 1)
        return round(psutil.cpu_percent(None), 1)

    def _memory(self):
        if _HAS_PROC:
            info = {}
            for line in (_read("/proc/meminfo") or "").splitlines():
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
            total = info.get("MemTotal")
            avail = info.get("MemAvailable", info.get("MemFree"))
            return (total - avail, total) if total and avail is not None else (None, None)
        vm = psutil.virtual_memory()
        return vm.total - vm.available, vm.total

    def _temperature(self):
        t = _read("/sys/class/thermal/thermal_zone0/temp")
        if t and t.strip().lstrip("-").isdigit():
            return round(int(t) / 1000.0, 1)
        if psutil is not None and hasattr(psutil, "sensors_temperatures"):
            try:
                for entries in psutil.sensors_temperatures().values():
                    if entries:
                        return round(entries[0].current, 1)
            except Exception:
                pass
        return None

    def _clock_mhz(self):
        f = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if f and f.strip().isdigit():
            return round(int(f) / 1000)
        if psutil is not None:
            try:
                return round(psutil.cpu_freq().current)
            except Exception:
                pass
        return None

    def _throttled(self):
        """Pi only: {'now': [...], 'since_boot': [...]} from `vcgencmd get_throttled`, else None."""
        if time.monotonic() - self.last_throttle < THROTTLE_CHECK_SEC and self.last_throttle:
            return self.throttle
        self.last_throttle = time.monotonic()
        if not sys.platform.startswith("linux"):
            return None
        try:
            out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
            bits = int(out.strip().split("=")[1], 16)
        except Exception:
            self.throttle = None
            return None
        names = {0: "under-voltage", 1: "clock capped", 2: "throttled", 3: "soft temperature limit"}
        self.throttle = {"now": [n for b, n in names.items() if bits & (1 << b)],
                         "since_boot": [n for b, n in names.items() if bits & (1 << (b + 16))]}
        return self.throttle

    def sample(self):
        if time.monotonic() - self.last_scan > PROCESS_SCAN_SEC:
            self._scan()
        used, total = self._memory()
        return {
            "t": time.time(),
            "temp_c": self._temperature(),
            "cpu": self._cpu_total(),
            "cores": _CORES,
            "parts": self._part_cpu(),
            "mem_used": used,
            "mem_total": total,
            "clock_mhz": self._clock_mhz(),
            "throttle": self._throttled(),
        }
