"""
Camargo 2021 - Level ground walking data extraction.
Run with: .venv_mat2csv\Scripts\python.exe process_part1.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT = Path(r"E:\外骨骼算法\gait_data\Part1")
OUTPUT_DIR = Path(r"E:\外骨骼算法\gait_data\processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_HZ = 200.0

def read_mat_table(path):
    import matio
    return matio.load_from_mat(str(path))["data"]

def find_levelground_files(data_root, modality):
    files = []
    for mod_dir in data_root.rglob(modality):
        parent = str(mod_dir.parent).lower()
        if "levelground" in parent:
            for f in sorted(mod_dir.glob("*.mat")):
                files.append(f)
    return sorted(set(files))

def get_subject(path):
    for part in path.parts:
        if part.startswith("AB") and part[2:].isdigit():
            return part
    return "unknown"

def resample_df(df, source_hz, target_hz):
    if abs(source_hz - target_hz) < 1e-6:
        return df
    from scipy.signal import resample_poly
    from fractions import Fraction
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
                resampled[col] = resample_poly(arr, up=ratio.numerator, down=ratio.denominator)[:new_len]
            else:
                resampled[col] = arr
        else:
            resampled[col] = df[col].values[:min_len]
    return pd.DataFrame(resampled)

def main():
    import time
    t0 = time.time()

    print("=" * 60)
    print("Camargo 2021 - Level Ground Walking Data Extraction")
    print("=" * 60)

    ik_files = find_levelground_files(DATA_ROOT, "ik")
    print(f"\nFound {len(ik_files)} levelground IK files")

    all_rows = []
    errors = []
    
    for i, ik_path in enumerate(ik_files):
        subject = get_subject(ik_path)
        trial = ik_path.stem
        id_path = ik_path.parent.parent / "id" / ik_path.name
        
        try:
            ik_df = read_mat_table(ik_path)
            
            # Determine source Hz from Header
            headers = pd.to_numeric(ik_df["Header"], errors="coerce").dropna().values
            if len(headers) > 1:
                dt = np.median(np.diff(headers))
                source_hz = 1.0 / dt if dt > 0 else 200.0
            else:
                source_hz = 200.0
            
            # IK data
            ik_cols = [c for c in ik_df.columns if c != "Header"]
            ik_data = ik_df[ik_cols].copy()
            
            # ID data
            if id_path.exists():
                id_df = read_mat_table(id_path)
                id_cols = [c for c in id_df.columns if c != "Header"]
                id_data = id_df[id_cols].copy()
                
                # Align lengths
                min_len = min(len(ik_data), len(id_data))
                ik_data = ik_data.iloc[:min_len].reset_index(drop=True)
                id_data = id_data.iloc[:min_len].reset_index(drop=True)
                
                # Combine IK + ID
                row_data = pd.concat([ik_data, id_data], axis=1)
            else:
                row_data = ik_data
            
            # Add metadata
            row_data["subject"] = subject
            row_data["trial"] = trial
            row_data["sample_rate"] = source_hz
            row_data["time_s"] = np.arange(len(row_data)) / source_hz
            
            all_rows.append(row_data)
            
            n_id = len(id_cols) if id_path.exists() else 0
            print(f"  [{i+1}/{len(ik_files)}] {subject}/{trial}: {len(row_data)} samples @ {source_hz:.0f}Hz, {len(row_data.columns)} cols (IK:{len(ik_cols)} ID:{n_id})")
            
        except Exception as e:
            errors.append((str(ik_path.relative_to(DATA_ROOT)), str(e)))
            print(f"  [{i+1}/{len(ik_files)}] {subject}/{trial}: ERROR - {e}")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out_path = OUTPUT_DIR / "levelground_part1.csv"
        combined.to_csv(out_path, index=False)
        
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f}s")
        print(f"Output: {out_path}")
        print(f"Total: {len(combined)} rows x {len(combined.columns)} columns")
        print(f"Subjects: {sorted(combined['subject'].unique())}")
        print(f"Columns: {list(combined.columns[:10])}...")
        
        # Summary per subject
        print(f"\nPer-subject summary:")
        for subj in sorted(combined["subject"].unique()):
            sub = combined[combined["subject"] == subj]
            trials = sub["trial"].nunique()
            print(f"  {subj}: {trials} trials, {len(sub)} total rows")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for path, err in errors:
            print(f"  {path}: {err}")

if __name__ == "__main__":
    main()
