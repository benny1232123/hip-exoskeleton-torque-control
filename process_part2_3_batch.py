"""
Batch process Part 2 and Part 3 - one Python process per subject
Much faster than one subprocess per file
"""
import zipfile
import os
import tempfile
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

PYTHON_VENV = r"E:\外骨骼算法\gait_data\.venv_mat2csv\Scripts\python.exe"
BASE_DIR = Path(r"E:\外骨骼算法")
OUTPUT_DIR = BASE_DIR / "gait_data" / "processed"

def batch_read_subject_mat_files(subject_dir, subject_id, tmp_csv_dir):
    """Read all IK+ID .mat files for a subject in one Python process"""
    # Build list of (trial_name, ik_path, id_path)
    lg_dirs = list(subject_dir.rglob("levelground"))
    if not lg_dirs:
        return []
    
    lg_dir = lg_dirs[0]
    ik_dir = lg_dir / "ik"
    id_dir = lg_dir / "id"
    
    if not ik_dir.exists() or not id_dir.exists():
        return []
    
    ik_files = {f.stem: str(f) for f in ik_dir.glob("*.mat")}
    id_files = {f.stem: str(f) for f in id_dir.glob("*.mat")}
    
    common = sorted(set(ik_files.keys()) & set(id_files.keys()))
    if not common:
        return []
    
    # Write trial list to temp file
    trial_list_file = tmp_csv_dir / f"trials_{subject_id}.txt"
    with open(trial_list_file, 'w') as f:
        for t in common:
            f.write(f"{t}\t{ik_files[t]}\t{id_files[t]}\n")
    
    # Run batch reader
    output_csv = tmp_csv_dir / f"data_{subject_id}.csv"
    script = f"""
import matio
import pandas as pd
import numpy as np

trials = []
with open(r'{trial_list_file}', 'r') as f:
    for line in f:
        parts = line.strip().split('\\t')
        trials.append(parts)

records = []
for trial_name, ik_path, id_path in trials:
    try:
        ik = matio.load_from_mat(ik_path)
        id_data = matio.load_from_mat(id_path)
        
        ik_df = ik['data']
        id_df = id_data['data']
        
        n = min(len(ik_df), len(id_df))
        combined = pd.concat([
            ik_df.iloc[:n].reset_index(drop=True),
            id_df.iloc[:n].reset_index(drop=True)
        ], axis=1)
        
        combined['subject'] = '{subject_id}'
        combined['trial'] = trial_name
        combined['sample_rate'] = 200
        combined['time_s'] = np.arange(n) / 200
        
        records.append(combined)
    except Exception as e:
        print(f'ERROR {{trial_name}}: {{e}}')

if records:
    df = pd.concat(records, ignore_index=True)
    df.to_csv(r'{output_csv}', index=False)
    print(f'OK: {{len(df)}} rows, {{len(records)}} trials')
else:
    print('NO DATA')
"""
    result = subprocess.run(
        [PYTHON_VENV, "-c", script],
        capture_output=True, text=True, timeout=300
    )
    
    print(f"  {subject_id}: {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[:300]}")
    
    if output_csv.exists():
        return pd.read_csv(output_csv)
    return None

def process_part(part_dir, part_name):
    print(f"\n{'='*60}")
    print(f"Processing {part_name}")
    print(f"{'='*60}")
    
    tmp_csv_dir = OUTPUT_DIR / "_tmp_batch"
    tmp_csv_dir.mkdir(exist_ok=True)
    
    subject_dirs = sorted([d for d in part_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('AB')])
    
    all_dfs = []
    for sd in subject_dirs:
        df = batch_read_subject_mat_files(sd, sd.name, tmp_csv_dir)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
    
    # Cleanup
    shutil.rmtree(tmp_csv_dir, ignore_errors=True)
    
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        csv_out = OUTPUT_DIR / f"levelground_{part_name.lower()}.csv"
        merged.to_csv(csv_out, index=False)
        print(f"\nSaved {part_name}: {len(merged)} rows, subjects: {sorted(merged['subject'].unique())}")
        return merged
    return None

# Process Part 2
part2_dir = BASE_DIR / "gait_data" / "Part2"
part2_df = process_part(part2_dir, "Part2")

# Process Part 3
part3_dir = BASE_DIR / "gait_data" / "Part3"
part3_df = process_part(part3_dir, "Part3")

# Merge all
print(f"\n{'='*60}")
print("Merging all parts...")
print(f"{'='*60}")

part1_csv = OUTPUT_DIR / "levelground_part1.csv"
all_dfs = []
for p_csv in [part1_csv, OUTPUT_DIR / "levelground_part2.csv", OUTPUT_DIR / "levelground_part3.csv"]:
    if p_csv.exists():
        df = pd.read_csv(p_csv)
        print(f"  {p_csv.name}: {len(df)} rows, subjects: {sorted(df['subject'].unique())}")
        all_dfs.append(df)

if all_dfs:
    common_cols = all_dfs[0].columns
    for df in all_dfs[1:]:
        common_cols = common_cols.intersection(df.columns)
    
    merged = pd.concat([df[list(common_cols)] for df in all_dfs], ignore_index=True)
    merged_csv = OUTPUT_DIR / "levelground_all.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"\nMerged: {len(merged)} rows, {len(merged.columns)} cols")
    print(f"All subjects: {sorted(merged['subject'].unique())}")
    print(f"Total subjects: {len(merged['subject'].unique())}")
