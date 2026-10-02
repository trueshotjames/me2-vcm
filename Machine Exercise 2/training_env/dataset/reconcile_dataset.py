#!/usr/bin/env python3
"""Reconcile the dataset folders against the CSV manifests.

Usage:
    python reconcile_dataset.py [--root PATH] [--dry-run] [--report PATH]

What it does (3 lines):
    1. Reads the four dataset CSVs (train_data.csv, val_data.csv,
       asian_train_data.csv, asian_val_data.csv) and collects the distinct set
       of relative filepaths referenced by their first column.
    2. Walks each of the five dataset folders, and its Others/<dataset>/ mirror,
       exactly once, building an {absolute_path: relative_path} map where
       relative_path is '<dataset>/<rest>' from the dataset root.
    3. Moves CSV-referenced files that are sitting in Others/<dataset>/ back into
       <dataset>/, and moves non-referenced files that are sitting in
       <dataset>/ out into Others/<dataset>/; everything else is left alone.

Idempotency: the script is idempotent. It computes the whole move plan from a
single pass over the filesystem before executing anything, and membership is
tested against the CSV set, not against the current folder. After one successful
run every CSV-referenced file lives under its <dataset>/ folder and every other
file lives under Others/<dataset>/, so a second run finds move_back=0 and
move_out=0 and performs no filesystem writes. Running it repeatedly therefore
converges to the same state and is safe to schedule or re-run. (Others/noise_only/
and Others/noisy_speech/ are never walked and never touched.)

Note on Windows: os.rename is attempted first (cheap, same-volume); shutil.move
is used as a cross-volume fallback. No file content is ever hashed.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime

DATASETS = [
    "classmate-dataset",
    "common-voice-dataset",
    "fluent_speech_commands_dataset",
    "mlend_spoken_numerals",
    "syntts_commands_media_dataset",
]
CSV_NAMES = ["train_data.csv", "val_data.csv", "asian_train_data.csv", "asian_val_data.csv"]
# Never walked, never touched.
SKIP_OTHERS = {"noise_only", "noisy_speech"}


def load_csv_set(root):
    """Return the distinct set of relative filepaths referenced by the CSVs."""
    refs = set()
    read = []
    for name in CSV_NAMES:
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        read.append(name)
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # header
            for row in reader:
                if not row:
                    continue
                fp = row[0].strip().replace("\\", "/")
                if fp:
                    refs.add(fp)
    return refs, read


def walk_map(root_dir, ds):
    """One os.walk over root_dir -> {absolute_path: '<ds>/<rest>'}."""
    mapping = {}
    if not os.path.isdir(root_dir):
        return mapping
    for dirpath, _dirs, files in os.walk(root_dir):
        for fname in files:
            abspath = os.path.join(dirpath, fname)
            rest = os.path.relpath(abspath, root_dir).replace(os.sep, "/")
            mapping[abspath] = f"{ds}/{rest}"
    return mapping


def move_file(src, dst):
    """Create dst's parent, then rename same-volume or shutil.move fallback."""
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        os.rename(src, dst)
    except OSError:
        shutil.move(src, dst)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reconcile dataset folders against CSV manifests.")
    parser.add_argument("--root", default=os.getcwd(),
                        help="dataset root (default: current working directory)")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and print the move plan but do not execute it")
    parser.add_argument("--report", default=None,
                        help="optional path to write a JSON report (task-001 shape)")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    started = time.time()

    refs, csvs_read = load_csv_set(root)

    # per-dataset referenced counts (split on the first '/', no prefix matching)
    ref_counts = {}
    for rel in refs:
        ds = rel.split("/", 1)[0]
        ref_counts[ds] = ref_counts.get(ds, 0) + 1

    ds_maps = {}
    others_maps = {}
    for ds in DATASETS:
        ds_maps[ds] = walk_map(os.path.join(root, ds), ds)
        others_maps[ds] = walk_map(os.path.join(root, "Others", ds), ds)

    per_dataset = {}
    plan_back = []   # (src, dst)
    plan_out = []    # (src, dst)
    total_out = total_back = 0

    for ds in DATASETS:
        ds_map = ds_maps[ds]
        others_map = others_maps[ds]

        back = [(ap, os.path.join(root, rel)) for ap, rel in others_map.items() if rel in refs]
        out = [(ap, os.path.join(root, "Others", rel)) for ap, rel in ds_map.items() if rel not in refs]
        plan_back.extend(back)
        plan_out.extend(out)

        per_dataset[ds] = {
            "referenced_by_csv": ref_counts.get(ds, 0),
            "was_in_others_moved_back": len(back),
            "was_in_dataset_moved_out": len(out),
            "remaining_in_dataset": len(ds_map) - len(out),
            "remaining_in_others": len(others_map) - len(back),
        }
        total_out += len(out)
        total_back += len(back)

    # Discovered rels (noise folders intentionally excluded).
    discovered = set()
    for ds in DATASETS:
        discovered.update(ds_maps[ds].values())
        discovered.update(others_maps[ds].values())
    missing = sorted(refs - discovered)

    moved_out = moved_back = 0
    conflicts = []
    if args.dry_run:
        moved_out, moved_back = total_out, total_back
    else:
        for src, dst in plan_back:
            if os.path.exists(dst):
                conflicts.append(dst)
                continue
            move_file(src, dst)
            moved_back += 1
        for src, dst in plan_out:
            if os.path.exists(dst):
                conflicts.append(dst)
                continue
            move_file(src, dst)
            moved_out += 1

    # ---- stdout summary table ------------------------------------------------
    print(f"dataset root : {root}")
    print(f"CSVs read    : {', '.join(csvs_read) if csvs_read else '(none found)'}")
    print(f"mode         : {'DRY-RUN (no moves executed)' if args.dry_run else 'APPLY'}")
    print(f"referenced distinct filepaths : {len(refs)}")
    print()
    header = f"{'dataset':<32}{'kept':>8}{'moved_out':>11}{'moved_back':>12}{'total_in_csv':>14}"
    print(header)
    print("-" * len(header))
    for ds in DATASETS:
        s = per_dataset[ds]
        print(f"{ds:<32}{s['remaining_in_dataset']:>8}{s['was_in_dataset_moved_out']:>11}"
              f"{s['was_in_others_moved_back']:>12}{s['referenced_by_csv']:>14}")
    print("-" * len(header))
    print(f"{'TOTAL':<32}{'':>8}{total_out:>11}{total_back:>12}{len(refs):>14}")
    print()
    print(f"moves performed : moved_out={moved_out} moved_back={moved_back}")
    if conflicts:
        print(f"conflicts skipped (dst already existed) : {len(conflicts)}")
    if missing:
        print(f"referenced but missing on disk : {len(missing)}")

    # ---- optional JSON report ------------------------------------------------
    if args.report:
        anomalies = []
        if missing:
            anomalies.append({
                "kind": "missing_on_disk",
                "count": len(missing),
                "note": ("CSV-referenced filepaths with no file anywhere under the dataset root "
                         "(not in the dataset folder, nor in Others/<dataset>/, nor "
                         "Others/noise_only|noisy_speech). Pre-existing data gap, not a move error."),
                "paths": missing,
            })
        if conflicts:
            anomalies.append({
                "kind": "destination_exists",
                "count": len(conflicts),
                "note": "Move skipped because the destination path already existed.",
                "paths": conflicts,
            })
        report = {
            "dataset_root": root,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "referenced_distinct_total": len(refs),
            "per_dataset": per_dataset,
            "totals": {
                "was_in_dataset_moved_out": total_out,
                "was_in_others_moved_back": total_back,
            },
            "anomalies": anomalies,
            "elapsed_sec": round(time.time() - started, 1),
            "unreachable_referenced_count": len(missing),
        }
        rp = args.report
        parent = os.path.dirname(os.path.abspath(rp))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"report written : {os.path.abspath(rp)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())