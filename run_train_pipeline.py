"""One-run pipeline: dataset setup, preprocessing, training and export.

    python run_train_pipeline.py                  # menu: pick VCM, wake word or both
    python run_train_pipeline.py --model vcm      # no menu (for nohup / cluster jobs)
    python run_train_pipeline.py --dry-run        # show the plan and what is already done
    python run_train_pipeline.py --from preprocess --yes
    python run_train_pipeline.py --only train --gpus 3

Edit the CONFIG block below to change training settings and GPUs.
Stages are skipped when their outputs already exist (--force reruns them).
Options not listed in --help are passed straight to train.py (e.g. --lr 0.001).

Needs: pip install huggingface_hub, plus training_env/requirements.txt.
PLD (Philippine Languages Database) is not re-hosted; it is downloaded with
your own Mozilla Data Collective API key (env MDC_API_KEY, or typed in).
"""

# ============================================================== CONFIG ==
# Where the training code, split CSVs and Common Voice clips are published.
HF_REPO = "trueshotjames/ai231-king-me2"
HF_REVISION = "v0.1-vcm"                    # tag or commit
# Class dataset with the ME2 audio (parquet), pinned to the commit the CSVs were built from.
ME2_REPO = "airimonda/ai231-me2-voice-commands"
ME2_REVISION = "bfca0bbebfe17e00d6a685d1da573615d29ecdce"

# Voice command model. gpus: numbers as nvidia-smi shows them ("2,5"), or "all".
VCM = dict(
    gpus="all",
    epochs=300,
    patience=30,
    min_epochs=60,
    workers=None,          # CPU workers for preprocessing; None = all cores
    k=2,                   # augmented copies per train clip
    resume=False,          # True = continue from last_checkpoint.pt instead of --fresh
    extra_args=[],         # more train.py options, e.g. ["--lr", "0.001"]
)

# Wake word model: not added yet.
WAKE = dict(
    gpus="0",
    workers=None,
    extra_args=[],
)

# PLD from Mozilla Data Collective (only the ENG folder is kept).
MDC_DATASET_ID = "cmmxhw46c00tqnw07xyr94zjk"
MDC_API = "https://mozilladatacollective.com/api"
MDC_PAGE = f"https://mozilladatacollective.com/datasets/{MDC_DATASET_ID}"
PLD_STOP_AFTER_ENG = True  # stop streaming the 45.6 GB archive once the ENG folder has passed

# MUSAN (noise + music for augmentation), streamed from OpenSLR.
MUSAN_MIRRORS = ["https://openslr.trmal.net", "https://us.openslr.org", "https://openslr.elda.org",
                 "https://openslr.magicdatatech.com", "https://www.openslr.org"]
MUSAN_PATH = "/resources/17/musan.tar.gz"

# =========================================================== end CONFIG ==

import argparse
import csv
import getpass
import glob
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
VCM_DIR = os.path.join(ROOT, "training_env")
DATASET = os.path.join(VCM_DIR, "dataset")
ME2_DIR = os.path.join(DATASET, "ai231-me2-voice-commands")
PLD_ENG = os.path.join(DATASET, "philippine-language-database", "PLD", "ENG")
MUSAN_PARENT = os.path.join(DATASET, "musan-noise-dataset")
POP_DIR = os.path.join(DATASET, "pop-songs-noise")
MODELS_DIR = os.path.join(ROOT, "models")
STATE_DIR = os.path.join(VCM_DIR, ".pipeline")    # marker files for train / export
CACHE_META = os.path.join(DATASET, "cache", "cache_meta.json")

ME2_FOLDERS = ["data", "supplemental_fil", "supplemental_synth", "synthetic_negatives"]
SPLIT_CSVS = ["integrated_train_data.csv", "integrated_test_data.csv", "integrated_holdout_data.csv"]
CODE_FILES = ["train.py", "preprocess.py", "augment.py", "features.py", "model.py", "dataset_loader.py",
              "export_onnx.py", "vocab.txt", "dataset/ai231-me2-voice-commands/build_me2_csv.py",
              "dataset/ai231-me2-voice-commands/trim_long_clips.py"] + [f"dataset/{c}" for c in SPLIT_CSVS]


# ----------------------------------------------------------------- helpers --
def run(cmd, cwd):
    """Run a command, streaming its output. Stops the pipeline if it fails."""
    print(f"  $ (cd {os.path.relpath(cwd, ROOT) or '.'}) {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"\nStopped: command failed with exit code {result.returncode}.")


def csv_paths():
    """Every filepath in the three split CSVs (relative to training_env/dataset)."""
    paths = []
    for name in SPLIT_CSVS:
        path = os.path.join(DATASET, name)
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            paths += [row["filepath"].strip().replace("\\", "/") for row in csv.DictReader(f)]
    return paths


def missing_files(prefix=None):
    paths = [p for p in csv_paths() if prefix is None or p.startswith(prefix)]
    return [p for p in paths if not os.path.exists(os.path.join(DATASET, *p.split("/")))]


def mark(name):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, name), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def newer(path, than):
    """True if path exists and is at least as new as than (or than does not exist)."""
    if not os.path.exists(path):
        return False
    return not os.path.exists(than) or os.path.getmtime(path) >= os.path.getmtime(than)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def has_files(folder):
    return os.path.isdir(folder) and any(files for _, _, files in os.walk(folder))


def write_member(src, dest):
    """Copy an archive member to dest via a .part file, so a stopped run never leaves half files."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + ".part", "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    os.replace(dest + ".part", dest)


def safe_rel(parts):
    """Archive path parts -> safe relative path (no absolute paths, no '..')."""
    if not parts or any(p in ("", ".", "..") or ":" in p for p in parts):
        return None
    return os.path.join(*parts)


# ------------------------------------------------------------ HTTP streams --
class ResumableStream(io.RawIOBase):
    """Read-only stream over HTTP. On a dropped connection it reconnects with a
    Range header from the current byte, so tarfile/gzip never notice.
    get_url() is called again if the URL is refused (e.g. an expired presigned URL)."""

    def __init__(self, get_url, tries=8):
        self.get_url, self.tries = get_url, tries
        self.url, self.pos, self.resp, self.end = get_url(), 0, None, None

    def readable(self):
        return True

    def _open(self):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-"} if self.pos else {})
        self.resp = urllib.request.urlopen(req, timeout=60)
        if self.pos and self.resp.status != 206:
            raise IOError("server ignored the Range header; cannot resume")
        length = self.resp.headers.get("Content-Length")
        self.end = self.pos + int(length) if length else None

    def readinto(self, buf):
        for attempt in range(self.tries):
            try:
                if self.resp is None:
                    self._open()
                n = self.resp.readinto(buf)
                if n == 0 and len(buf) and self.end is not None and self.pos < self.end:
                    raise IOError("connection closed early")   # resume from self.pos
                self.pos += n
                return n
            except urllib.error.HTTPError as e:
                self.resp = None
                if e.code in (401, 403) and attempt < self.tries - 1:
                    print("  download link refused (expired?), asking for a new one")
                    self.url = self.get_url()
                    continue
                raise
            except (OSError, urllib.error.URLError) as e:
                self.resp = None
                if attempt == self.tries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  connection lost at {self.pos / 1e9:.2f} GB ({e}); resuming in {wait} s")
                time.sleep(wait)


class SeekableHTTP(io.RawIOBase):
    """Random-access file over HTTP Range requests (for reading a remote zip)."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size or len(buf) == 0:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


# -------------------------------------------------------- archive extraction --
def eng_rel(name):
    """Archive member name -> path inside PLD/ENG, or None if it is not under an ENG folder."""
    parts = name.replace("\\", "/").strip("/").split("/")
    for i, p in enumerate(parts[:-1]):
        if p == "ENG" and (i == 0 or parts[i - 1] == "PLD" or "PLD" not in parts[:i]):
            return safe_rel(parts[i + 1:])
    return None


def extract_eng_from_tar(fileobj, dest, needed=None, stop_after_eng=True):
    """Stream a .tar / .tar.gz and write only the ENG members. Returns files written."""
    written, inside = 0, False
    with tarfile.open(fileobj=fileobj, mode="r|*") as tar:
        for m in tar:
            rel = eng_rel(m.name)
            if rel is None:
                if inside and stop_after_eng and (needed is None or all(os.path.exists(p) for p in needed)):
                    print("  ENG folder done; stopping the download early")
                    break
                continue
            inside = True
            if not m.isfile():
                continue
            out = os.path.join(dest, rel)
            if not os.path.exists(out):
                write_member(tar.extractfile(m), out)
                written += 1
                if written % 500 == 0:
                    print(f"  {written} ENG files written")
    return written


def extract_eng_from_zip(fileobj, dest):
    """Read a zip's central directory and fetch only the ENG members."""
    written = 0
    with zipfile.ZipFile(fileobj) as z:
        for info in z.infolist():
            rel = eng_rel(info.filename)
            if rel is None or info.is_dir():
                continue
            out = os.path.join(dest, rel)
            if not os.path.exists(out):
                with z.open(info) as src:
                    write_member(src, out)
                written += 1
    return written


# --------------------------------------------------------------- VCM stages --
def fetch_code_done():
    return all(os.path.exists(os.path.join(VCM_DIR, *f.split("/"))) for f in CODE_FILES)


def fetch_code(args):
    if HF_REPO.startswith("<"):
        sys.exit("Stopped: set HF_REPO in the CONFIG block of run_train_pipeline.py.")
    from huggingface_hub import snapshot_download   # pip install huggingface_hub
    print(f"  downloading training_env/ from {HF_REPO} @ {HF_REVISION}")
    snapshot_download(HF_REPO, repo_type="dataset", revision=HF_REVISION,
                      allow_patterns=["training_env/**"], local_dir=ROOT)


def fetch_me2_done():
    return all(glob.glob(os.path.join(ME2_DIR, folder, "*.parquet")) for folder in ME2_FOLDERS)


def fetch_me2(args):
    from huggingface_hub import snapshot_download
    print(f"  downloading {ME2_REPO} @ {ME2_REVISION[:7]} into {os.path.relpath(ME2_DIR, ROOT)}")
    snapshot_download(ME2_REPO, repo_type="dataset", revision=ME2_REVISION, local_dir=ME2_DIR)


def unpack_me2_done():
    return (os.path.exists(os.path.join(DATASET, "ai231_me2_data.csv"))
            and not missing_files("ai231-me2-voice-commands/"))


def unpack_me2(args):
    # Trim clips over 5 s inside the parquet files (restarts from .orig copies, so reruns
    # give the same result), then write every clip as a wav file plus ai231_me2_data.csv.
    run([sys.executable, "trim_long_clips.py"], ME2_DIR)
    run([sys.executable, "build_me2_csv.py", "--extract-audio"], ME2_DIR)


def musan_done():
    musan = os.path.join(MUSAN_PARENT, "musan")
    return (has_files(os.path.join(musan, "noise")) and has_files(os.path.join(musan, "music"))
            and not os.path.exists(os.path.join(MUSAN_PARENT, ".extracting")))


def musan(args):
    os.makedirs(MUSAN_PARENT, exist_ok=True)
    flag = os.path.join(MUSAN_PARENT, ".extracting")
    open(flag, "w").close()
    for mirror in MUSAN_MIRRORS:
        url = mirror + MUSAN_PATH
        print(f"  streaming {url} (~11 GB) into {os.path.relpath(MUSAN_PARENT, ROOT)}")
        try:
            stream = io.BufferedReader(ResumableStream(lambda: url), 1 << 20)
            n = 0
            with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                for m in tar:
                    rel = safe_rel(m.name.replace("\\", "/").strip("/").split("/"))
                    if not m.isfile() or rel is None or not rel.startswith("musan" + os.sep):
                        continue
                    out = os.path.join(MUSAN_PARENT, rel)
                    if not os.path.exists(out):
                        write_member(tar.extractfile(m), out)
                    n += 1
                    if n % 500 == 0:
                        print(f"  {n} files")
            os.remove(flag)
            return
        except (OSError, tarfile.TarError) as e:
            print(f"  {mirror} failed: {e}")
    sys.exit("Stopped: could not download MUSAN from any mirror (https://www.openslr.org/17/).")


def pld_needed():
    return [os.path.join(DATASET, *p.split("/")) for p in csv_paths()
            if p.startswith("philippine-language-database/")]


def pld_done():
    return not missing_files("philippine-language-database/")


def pld_help(reason):
    sys.exit(f"\nStopped: {reason}\n"
             f"To get PLD (free, about 2 minutes of clicking):\n"
             f"  1. Log in at {MDC_PAGE} and accept the terms.\n"
             f"  2. Create an API key in your profile settings.\n"
             f"  3. Rerun this script; paste the key when asked (or set env MDC_API_KEY).\n"
             f"Or unzip the ENG folder yourself so this path exists: {os.path.relpath(PLD_ENG, ROOT)}\n"
             "Or run on the course cluster; see README.md (Reproduce training).")


def mdc_download_info(key):
    """Ask Mozilla Data Collective for a presigned download URL."""
    req = urllib.request.Request(f"{MDC_API}/datasets/{MDC_DATASET_ID}/download", data=b"", method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            pld_help("the Mozilla Data Collective API key was not accepted.")
        if e.code == 403:
            pld_help("the dataset terms are not accepted yet for this account.")
        if e.code == 429:
            pld_help(f"download quota reached (30 per day); retry after {e.headers.get('Retry-After', '?')} s.")
        raise


def pld(args):
    key = os.environ.get("MDC_API_KEY")
    if not key and sys.stdin.isatty():
        print(f"  PLD comes from {MDC_PAGE} (accept the terms there first).")
        key = getpass.getpass("  Mozilla Data Collective API key (Enter to skip): ").strip()
    if not key:
        pld_help(f"{len(missing_files('philippine-language-database/'))} PLD clips are missing.")

    info = mdc_download_info(key)
    name, size = info.get("filename", ""), info.get("sizeBytes") or 0
    print(f"  archive: {name} ({size / 1e9:.1f} GB); keeping only ENG/ in {os.path.relpath(PLD_ENG, ROOT)}")
    if name.lower().endswith(".zip"):
        n = extract_eng_from_zip(SeekableHTTP(info["downloadUrl"], int(size)), PLD_ENG)
    else:
        urls = iter([info["downloadUrl"]])
        stream = ResumableStream(lambda: next(urls, None) or mdc_download_info(key)["downloadUrl"])
        n = extract_eng_from_tar(io.BufferedReader(stream, 1 << 20), PLD_ENG, pld_needed(), PLD_STOP_AFTER_ENG)
    print(f"  {n} ENG files written")
    if not pld_done():
        pld_help(f"{len(missing_files('philippine-language-database/'))} PLD clips still missing after download.")


def pop_songs_done():
    return True   # optional; never blocks


def pop_songs_note():
    if not has_files(POP_DIR):
        return "pop-songs-noise/ is empty (copyrighted demo songs, not published); augment.py uses MUSAN music instead"
    return None


def build_csv_done():
    return all(os.path.exists(os.path.join(DATASET, name)) for name in SPLIT_CSVS)


def build_csv(args):
    # The integrated CSVs are the frozen splits, so this only runs with --force.
    # It needs asian_train_data.csv, asian_val_data.csv and ai231_me2_data.csv.
    run([sys.executable, "build_integrated_csv.py"], DATASET)


def verify_done():
    return False   # cheap, so it always runs


def verify(args):
    paths = csv_paths()
    missing = missing_files()
    print(f"  {len(paths) - len(missing)} ok, {len(missing)} missing (of {len(paths)} clips in the split CSVs)")
    if missing:
        for p in missing[:10]:
            print(f"    missing: {p}")
        sys.exit("\nStopped: preprocess.py would silently drop these clips.")


def preprocess_done():
    # preprocess.cache_ready() also checks the cache was built from the current CSVs/vocab.
    check = "import sys, preprocess; sys.exit(0 if preprocess.cache_ready() else 1)"
    try:
        return subprocess.run([sys.executable, "-c", check], cwd=VCM_DIR,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False


def preprocess(args):
    cmd = [sys.executable, "preprocess.py", "--workers", str(args.workers), "--k", str(args.k)]
    if args.force:
        cmd.append("--force")
    run(cmd, VCM_DIR)


def train_done():
    marker = os.path.join(STATE_DIR, "train.done")
    return newer(marker, CACHE_META) and os.path.exists(os.path.join(VCM_DIR, "best_model.pth"))


def train(args):
    cmd = [sys.executable, "train.py", "--gpus", args.gpus, "--epochs", str(args.epochs),
           "--patience", str(args.patience), "--min_epochs", str(args.min_epochs),
           "--workers", str(args.workers), "--k", str(args.k)]
    if not args.resume:
        cmd.append("--fresh")
    run(cmd + VCM["extra_args"] + args.train_args, VCM_DIR)
    mark("train.done")


def export_done():
    marker = os.path.join(STATE_DIR, "export.done")
    return newer(marker, os.path.join(STATE_DIR, "train.done")) and os.path.exists(
        os.path.join(VCM_DIR, "voice_command_model.onnx"))


def export(args):
    run([sys.executable, "export_onnx.py"], VCM_DIR)
    mark("export.done")


def install_files():
    return [("voice_command_model.onnx", os.path.join(VCM_DIR, "voice_command_model.onnx")),
            ("vocab.txt", os.path.join(VCM_DIR, "vocab.txt"))]


def install_done():
    for name, src in install_files():
        dst = os.path.join(MODELS_DIR, name)
        if not (os.path.exists(src) and os.path.exists(dst) and md5(src) == md5(dst)):
            return False
    return True


def install(args):
    # voice_pipeline/wake_word_engine.py loads the model and vocab from models/.
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, src in install_files():
        shutil.copy2(src, os.path.join(MODELS_DIR, name))
        print(f"  copied {os.path.relpath(src, ROOT)} -> models/{name}")


# (name, what it does, done check, action)
VCM_STAGES = [
    ("fetch_code", "training_env code, split CSVs, Common Voice clips (HF)", fetch_code_done, fetch_code),
    ("fetch_me2", "ME2 parquet files (HF)", fetch_me2_done, fetch_me2),
    ("unpack_me2", "trim long clips, extract ME2 wav files", unpack_me2_done, unpack_me2),
    ("musan", "MUSAN noise and music (OpenSLR, ~11 GB)", musan_done, musan),
    ("pld", "PLD English clips (Mozilla Data Collective, your API key)", pld_done, pld),
    ("pop_songs", "demo playlist for augmentation (optional)", pop_songs_done, None),
    ("build_csv", "integrated train/test/holdout CSVs (frozen)", build_csv_done, build_csv),
    ("verify", "every clip in the split CSVs exists", verify_done, verify),
    ("preprocess", "decode, augment, log-Mel cache", preprocess_done, preprocess),
    ("train", "CTC training (train.py)", train_done, train),
    ("export", "best_model.pth -> voice_command_model.onnx", export_done, export),
    ("install", "copy the ONNX model and vocab.txt to models/", install_done, install),
]

# Wake word stages go here later, in the same (name, description, done, action) form.
WAKE_STAGES = []

PIPELINES = {"vcm": VCM_STAGES, "wake": WAKE_STAGES}
NOTES = {"pop_songs": pop_songs_note}


# ------------------------------------------------------------------- main --
def get_args():
    p = argparse.ArgumentParser(description="Dataset setup, preprocessing, training and export in one run.")
    p.add_argument("--model", choices=["vcm", "wake", "all"], help="skip the menu")
    p.add_argument("--from", dest="start", help="start at this stage")
    p.add_argument("--only", help="run only this stage")
    p.add_argument("--dry-run", action="store_true", help="show the plan; download or run nothing")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.add_argument("--force", action="store_true", help="rerun stages even if they look done")
    p.add_argument("--workers", type=int, default=VCM["workers"] or os.cpu_count())
    p.add_argument("--k", type=int, default=VCM["k"])
    p.add_argument("--gpus", default=VCM["gpus"])
    p.add_argument("--epochs", type=int, default=VCM["epochs"])
    p.add_argument("--patience", type=int, default=VCM["patience"])
    p.add_argument("--min_epochs", type=int, default=VCM["min_epochs"])
    p.add_argument("--resume", action="store_true", default=VCM["resume"],
                   help="continue from last_checkpoint.pt instead of --fresh")
    args, args.train_args = p.parse_known_args()
    return args


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        sys.exit("\nNo answer (no terminal?): pass --model vcm, wake or all, and --yes.")


def choose_model():
    print("Which model do you want to set up and train?\n"
          "  1) Voice command model (VCM)\n"
          "  2) Wake word model (not available yet)\n"
          "  3) Both")
    while True:
        choice = ask("Choose 1, 2 or 3: ")
        if choice in ("1", "2", "3"):
            return {"1": "vcm", "2": "wake", "3": "all"}[choice]


def select(stages, args):
    names = [s[0] for s in stages]
    for opt in (args.start, args.only):
        if opt and opt not in names:
            sys.exit(f"Unknown stage '{opt}'. Stages: {', '.join(names)}")
    if args.only:
        return [s for s in stages if s[0] == args.only]
    if args.start:
        return stages[names.index(args.start):]
    return stages


def show_plan(model, stages, args):
    print(f"\n=== {model} pipeline ===")
    if not stages:
        print("wake word pipeline not added yet")
    for name, desc, done, _ in stages:
        is_done = done()
        status = "done" if is_done and not args.force else "todo"
        if is_done and args.force and name != "pop_songs":
            status = "todo (forced)"
        note = NOTES.get(name, lambda: None)()
        print(f"[{status}] {name}: {desc}" + (f"\n         note: {note}" if note else ""))


def run_stages(model, stages, args):
    print(f"\n=== {model} pipeline: running ===")
    ran, skipped = [], []
    for name, desc, done, action in stages:
        if action is None or (done() and not args.force):
            skipped.append(name)
            continue
        print(f"\n[run]  {name}: {desc}", flush=True)
        t0 = time.time()
        action(args)
        print(f"[ok]   {name} ({time.time() - t0:.0f} s)")
        ran.append(name)
    return ran, skipped


def main():
    args = get_args()
    if not args.model:
        if not sys.stdin.isatty():
            sys.exit("No terminal for the menu: pass --model vcm, wake or all.")
        args.model = choose_model()
    if args.train_args:
        print(f"Passing to train.py: {' '.join(args.train_args)}")

    models = ["vcm", "wake"] if args.model == "all" else [args.model]
    plans = [(m, select(PIPELINES[m], args) if PIPELINES[m] else []) for m in models]
    for model, stages in plans:
        show_plan(model, stages, args)
    if args.dry_run:
        print("\nDry run: nothing was downloaded or run.")
        return
    if not args.yes and sys.stdin.isatty() and ask("\nProceed? [Y/n] ").lower() not in ("", "y", "yes"):
        return

    t0 = time.time()
    summary = [(m,) + run_stages(m, s, args) for m, s in plans if s]
    print(f"\nSummary ({(time.time() - t0) / 60:.1f} min)")
    for model, ran, skipped in summary:
        print(f"  {model}: ran {', '.join(ran) or 'nothing'}; skipped (done) {', '.join(skipped) or 'nothing'}")


if __name__ == "__main__":
    main()
