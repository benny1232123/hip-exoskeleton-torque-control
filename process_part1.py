"""
Camargo 2021 - Level ground walking data extraction (single Part).

Extracts IK (inverse kinematics) + ID (inverse dynamics) tables for every
level-ground walking trial of one dataset Part, resamples them to a common
rate, and writes one CSV.

Usage
-----
    python process_part1.py
    python process_part1.py --part Part1 --data-root D:/data/camargo
    python process_part1.py --output-dir ./processed

All paths are resolved relative to the repository root by default; see
``config.py`` for the precedence rules (CLI flag > env var > repo root).
"""
from __future__ import annotations

import argparse
import time
import warnings
from fractions import Fraction
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract level-ground walking trials from one Camargo 2021 Part.",
    )
    parser.add_argument(
        "--part",
        default="Part1",
        choices=list(config.DEFAULT_PARTS),
        help="Dataset Part to extract (default: Part1)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root containing Part1/Part2/Part3 (default: $DATA_ROOT or repo root)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the CSV (default: $GAIT_OUTPUT_DIR or <repo>/processed)",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=config.TARGET_HZ,
        help=f"Resample target frequency in Hz (default: {config.TARGET_HZ:g})",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# .mat handling
# --------------------------------------------------------------------------
def read_mat_table(path):
    import matio

    return matio.load_from_mat(str(path))["data"]


def find_levelground_files(data_root, modality):
    """Return every ``<modality>/*.mat`` file sitting under a ``levelground`` dir."""
    files = []
    for mod_dir in Path(data_root).rglob(modality):
        parent = str(mod_dir.parent).lower()
        if "levelground" in parent:
            for f in sorted(mod_dir.glob("*.mat")):
                files.append(f)
    return sorted(set(files))


def get_subject(path):
    for part in Path(path).parts:
        if part.startswith("AB") and part[2:].isdigit():
            return part
    return "unknown"


def resample_df(df, source_hz, target_hz):
    """Resample numeric columns from ``source_hz`` to ``target_hz``."""
    if abs(source_hz - target_hz) < 1e-6:
        return df

    from scipy.signal import resample_poly

    ratio = Fraction(target_hz / source_hz).limit_denominator(1000)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    lengths = {col: len(df[col]) for col in numeric_cols}
    if not lengths:
        return df

    min_len = min(lengths.values())
    resampled = {}
    for col in df.columns:
        if col in numeric_cols:
            arr = df[col].values[:min_len].astype(np.float64)
            new_len = int(round(min_len * target_hz / source_hz))
            if new_len >= 2:
                resampled[col] = resample_poly(
                    arr, up=ratio.numerator, down=ratio.denominator
                )[:new_len]
            else:
                resampled[col] = arr
        else:
            resampled[col] = df[col].values[:min_len]
    return pd.DataFrame(resampled)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)

    data_root = config.data_root(args.data_root)
    output_dir = config.output_dir(args.output_dir)
    target_hz = args.target_hz
    part_root = config.part_dir(data_root, args.part)

    if not part_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {part_root}\n"
            f"Download {args.part} from the Mendeley link in README.md and unpack it "
            f"under the data root, or point --data-root at its actual location.\n"
            f"(current data root: {data_root})"
        )

    t0 = time.time()

    print("=" * 60)
    print(f"Camargo 2021 - Level Ground Walking Data Extraction [{args.part}]")
    print("=" * 60)
    print(f"Data root : {data_root}")
    print(f"Output dir: {output_dir}")

    ik_files = find_levelground_files(part_root, "ik")
    print(f"\nFound {len(ik_files)} levelground IK files")

    if not ik_files:
        print(f"Nothing to do: no levelground IK files found under {part_root}")
        return None

    all_rows = []
    errors = []

    for i, ik_path in enumerate(ik_files):
        subject = get_subject(ik_path)
        trial = ik_path.stem
        id_path = ik_path.parent.parent / "id" / ik_path.name

        try:
            ik_df = read_mat_table(ik_path)

            # Determine source Hz from the Header column.
            headers = pd.to_numeric(ik_df["Header"], errors="coerce").dropna().values
            if len(headers) > 1:
                dt = np.median(np.diff(headers))
                source_hz = 1.0 / dt if dt > 0 else target_hz
            else:
                source_hz = target_hz

            ik_cols = [c for c in ik_df.columns if c != "Header"]
            ik_data = ik_df[ik_cols].copy()

            if id_path.exists():
                id_df = read_mat_table(id_path)
                id_cols = [c for c in id_df.columns if c != "Header"]
                id_data = id_df[id_cols].copy()

                min_len = min(len(ik_data), len(id_data))
                ik_data = ik_data.iloc[:min_len].reset_index(drop=True)
                id_data = id_data.iloc[:min_len].reset_index(drop=True)

                row_data = pd.concat([ik_data, id_data], axis=1)
            else:
                id_cols = []
                row_data = ik_data

            row_data = resample_df(row_data, source_hz, target_hz)

            row_data["subject"] = subject
            row_data["trial"] = trial
            row_data["sample_rate"] = source_hz
            row_data["time_s"] = np.arange(len(row_data)) / source_hz

            all_rows.append(row_data)

            n_id = len(id_cols) if id_path.exists() else 0
            print(
                f"  [{i + 1}/{len(ik_files)}] {subject}/{trial}: "
                f"{len(row_data)} samples @ {source_hz:.0f}Hz -> {target_hz:.0f}Hz, "
                f"{len(row_data.columns)} cols (IK:{len(ik_cols)} ID:{n_id})"
            )

        except Exception as e:
            errors.append((str(ik_path.relative_to(part_root)), str(e)))
            print(f"  [{i + 1}/{len(ik_files)}] {subject}/{trial}: ERROR - {e}")

    if not all_rows:
        print("\nNo rows extracted.")
        return None

    combined = pd.concat(all_rows, ignore_index=True)
    out_path = config.per_part_csv(output_dir, args.part)
    combined.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Output: {out_path}")
    print(f"Total: {len(combined)} rows x {len(combined.columns)} columns")
    print(f"Subjects: {sorted(combined['subject'].unique())}")
    print(f"Columns: {list(combined.columns[:10])}...")

    print("\nPer-subject summary:")
    for subj in sorted(combined["subject"].unique()):
        sub = combined[combined["subject"] == subj]
        print(f"  {subj}: {sub['trial'].nunique()} trials, {len(sub)} total rows")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for path, err in errors:
            print(f"  {path}: {err}")

    return combined


if __name__ == "__main__":
    main()
