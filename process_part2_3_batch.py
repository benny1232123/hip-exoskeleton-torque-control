"""
Camargo 2021 - Batch extraction of Part 2 / Part 3, then merge all Parts.

One subprocess per subject keeps memory bounded while staying far faster than
spawning a process per .mat file.

Usage
-----
    python process_part2_3_batch.py
    python process_part2_3_batch.py --parts Part2 Part3 --merge
    python process_part2_3_batch.py --python /path/to/venv/python

The batch reader reuses ``sys.executable`` unless you point it elsewhere with
--python / $GAIT_PYTHON. Every other path follows ``config.py``
(CLI > env > repo root).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

import config


# Reader executed once per subject inside a child interpreter. Arguments are
# passed through argv instead of being interpolated into the source, so Windows
# paths (backslashes) and non-ASCII characters cannot corrupt the code string.
BATCH_READER = r"""
import sys

import matio
import pandas as pd
import numpy as np

trial_list_file, output_csv, subject_id, target_hz = sys.argv[1:5]
target_hz = float(target_hz)

records = []
with open(trial_list_file, "r", encoding="utf-8") as fh:
    trials = [line.rstrip("\n").split("\t") for line in fh if line.strip()]

for trial_name, ik_path, id_path in trials:
    try:
        ik_df = matio.load_from_mat(ik_path)["data"]
        id_df = matio.load_from_mat(id_path)["data"]

        n = min(len(ik_df), len(id_df))
        combined = pd.concat(
            [
                ik_df.iloc[:n].reset_index(drop=True),
                id_df.iloc[:n].reset_index(drop=True),
            ],
            axis=1,
        )

        combined["subject"] = subject_id
        combined["trial"] = trial_name
        combined["sample_rate"] = target_hz
        combined["time_s"] = np.arange(n) / target_hz

        records.append(combined)
    except Exception as exc:
        print("ERROR %s: %s" % (trial_name, exc))

if records:
    df = pd.concat(records, ignore_index=True)
    df.to_csv(output_csv, index=False)
    print("OK: %d rows, %d trials" % (len(df), len(records)))
else:
    print("NO DATA")
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch-extract Part 2/3 of the Camargo 2021 dataset and merge all Parts.",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["Part2", "Part3"],
        choices=list(config.DEFAULT_PARTS),
        help="Parts to extract (default: Part2 Part3)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root containing Part1/Part2/Part3 (default: $DATA_ROOT or repo root)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write CSVs (default: $GAIT_OUTPUT_DIR or <repo>/processed)",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Interpreter used for the batch reader subprocess "
        "(default: $GAIT_PYTHON or sys.executable)",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=config.TARGET_HZ,
        help=f"Nominal sampling rate written into metadata (default: {config.TARGET_HZ:g})",
    )
    parser.add_argument(
        "--no-merge",
        dest="merge",
        action="store_false",
        help="Skip the final merge into levelground_all.csv",
    )
    parser.set_defaults(merge=True)
    return parser.parse_args(argv)


def batch_read_subject_mat_files(
    subject_dir, subject_id, tmp_csv_dir, python_exe, target_hz
):
    """Read every IK+ID .mat file of one subject in a single child process."""
    subject_dir = Path(subject_dir)

    lg_dirs = list(subject_dir.rglob("levelground"))
    if not lg_dirs:
        return None

    lg_dir = lg_dirs[0]
    ik_dir = lg_dir / "ik"
    id_dir = lg_dir / "id"

    if not ik_dir.exists() or not id_dir.exists():
        return None

    ik_files = {f.stem: str(f) for f in ik_dir.glob("*.mat")}
    id_files = {f.stem: str(f) for f in id_dir.glob("*.mat")}
    common = sorted(set(ik_files.keys()) & set(id_files.keys()))
    if not common:
        return None

    trial_list_file = Path(tmp_csv_dir) / f"trials_{subject_id}.txt"
    with open(trial_list_file, "w", encoding="utf-8") as f:
        for t in common:
            f.write(f"{t}\t{ik_files[t]}\t{id_files[t]}\n")

    output_csv = Path(tmp_csv_dir) / f"data_{subject_id}.csv"

    result = subprocess.run(
        [
            python_exe,
            "-c",
            BATCH_READER,
            str(trial_list_file),
            str(output_csv),
            str(subject_id),
            str(target_hz),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    print(f"  {subject_id}: {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[:300]}")

    if output_csv.exists():
        return pd.read_csv(output_csv)
    return None


def process_part(part_root, python_exe, tmp_csv_dir, target_hz):
    part_root = Path(part_root)
    print(f"\n{'=' * 60}")
    print(f"Processing {part_root.name}")
    print(f"{'=' * 60}")

    if not part_root.exists():
        print(f"  Skipped: {part_root} does not exist.")
        return None

    subject_dirs = sorted(
        d for d in part_root.iterdir() if d.is_dir() and d.name.startswith("AB")
    )
    if not subject_dirs:
        print(f"  Skipped: no AB* subject directories under {part_root}.")
        return None

    all_dfs = []
    for sd in subject_dirs:
        df = batch_read_subject_mat_files(
            sd, sd.name, tmp_csv_dir, python_exe, target_hz
        )
        if df is not None and len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        return None

    return pd.concat(all_dfs, ignore_index=True)


def merge_all(output_dir):
    print(f"\n{'=' * 60}")
    print("Merging all parts...")
    print(f"{'=' * 60}")

    all_dfs = []
    for part in config.DEFAULT_PARTS:
        p_csv = config.per_part_csv(output_dir, part)
        if p_csv.exists():
            df = pd.read_csv(p_csv)
            print(
                f"  {p_csv.name}: {len(df)} rows, "
                f"subjects: {sorted(df['subject'].unique())}"
            )
            all_dfs.append(df)
        else:
            print(f"  {p_csv.name}: MISSING (skipped)")

    if not all_dfs:
        print("\nNothing to merge.")
        return None

    common_cols = all_dfs[0].columns
    for df in all_dfs[1:]:
        common_cols = common_cols.intersection(df.columns)

    merged = pd.concat([df[list(common_cols)] for df in all_dfs], ignore_index=True)
    merged_csv = config.merged_csv(output_dir)
    merged.to_csv(merged_csv, index=False)

    print(f"\nMerged: {len(merged)} rows, {len(merged.columns)} cols")
    print(f"Output: {merged_csv}")
    print(f"All subjects: {sorted(merged['subject'].unique())}")
    print(f"Total subjects: {len(merged['subject'].unique())}")
    return merged


def main(argv=None):
    args = parse_args(argv)

    data_root = config.data_root(args.data_root)
    output_dir = config.output_dir(args.output_dir)
    python_exe = config.python_executable(args.python)

    print(f"Data root : {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Subprocess: {python_exe}")

    tmp_csv_dir = output_dir / config.TMP_DIR_NAME
    tmp_csv_dir.mkdir(parents=True, exist_ok=True)

    try:
        for part in args.parts:
            df_part = process_part(
                config.part_dir(data_root, part),
                python_exe,
                tmp_csv_dir,
                args.target_hz,
            )
            if df_part is not None:
                csv_out = config.per_part_csv(output_dir, part)
                df_part.to_csv(csv_out, index=False)
                print(
                    f"\nSaved {part}: {len(df_part)} rows, "
                    f"subjects: {sorted(df_part['subject'].unique())}"
                )
    finally:
        shutil.rmtree(tmp_csv_dir, ignore_errors=True)

    if args.merge:
        merge_all(output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
