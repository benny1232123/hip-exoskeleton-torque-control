"""
Visualize healthy human gait curves from Camargo 2021 dataset.
Plot hip/knee/ankle flexion-extension angles and moments.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
csv_path = Path(r"E:\外骨骼算法\gait_data\processed\levelground_part1.csv")
df = pd.read_csv(csv_path)
print(f"Loaded: {len(df)} rows, {len(df.columns)} cols")
print(f"Subjects: {sorted(df['subject'].unique())}")

# ---- Normalize to gait cycle (0-100%) ----
# Each trial has different length; resample to 100 points (0-100%)

def normalize_to_gait_cycle(group):
    """Resample each trial to 100 points (0-100% gait cycle)"""
    n_points = 100
    result = {}
    numeric_cols = group.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['sample_rate', 'time_s']:
            continue
        values = group[col].values
        if len(values) > 1:
            x_old = np.linspace(0, 100, len(values))
            x_new = np.linspace(0, 100, n_points)
            result[col] = np.interp(x_new, x_old, values)
        else:
            result[col] = np.full(n_points, values[0])
    result['gait_pct'] = np.linspace(0, 100, n_points)
    return pd.DataFrame(result)

print("\nNormalizing to gait cycle (0-100%)...")
normalized = []
for (subject, trial), group in df.groupby(['subject', 'trial']):
    norm_df = normalize_to_gait_cycle(group)
    norm_df['subject'] = subject
    norm_df['trial'] = trial
    normalized.append(norm_df)

norm_df = pd.concat(normalized, ignore_index=True)
print(f"Normalized: {len(norm_df)} rows")

# ---- Compute mean and std across all trials ----
gait_pct = np.linspace(0, 100, 100)

# Right side joint angles
angle_cols_r = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
angle_labels_r = ['Hip Flexion (R)', 'Knee Flexion (R)', 'Ankle Dorsiflexion (R)']

# Left side joint angles
angle_cols_l = ['hip_flexion_l', 'knee_angle_l', 'ankle_angle_l']
angle_labels_l = ['Hip Flexion (L)', 'Knee Flexion (L)', 'Ankle Dorsiflexion (L)']

# Right side joint moments
moment_cols_r = ['hip_flexion_r_moment', 'knee_angle_r_moment', 'ankle_angle_r_moment']
moment_labels_r = ['Hip Moment (R)', 'Knee Moment (R)', 'Ankle Moment (R)']

# Left side joint moments
moment_cols_l = ['hip_flexion_l_moment', 'knee_angle_l_moment', 'ankle_angle_l_moment']
moment_labels_l = ['Hip Moment (L)', 'Knee Moment (L)', 'Ankle Moment (L)']

# ---- Plot ----
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
fig.suptitle('Healthy Human Gait Curves - Camargo 2021 (Level Ground Walking)\n'
             f'{df["subject"].nunique()} subjects, {df["trial"].nunique()} trials (mean ± std)',
             fontsize=14)

colors = ['#e74c3c', '#2ecc71', '#3498db']  # hip, knee, ankle

# Plot right side angles
for i, (col, label) in enumerate(zip(angle_cols_r, angle_labels_r)):
    vals = norm_df.pivot_table(values=col, index='gait_pct', aggfunc=['mean', 'std'])
    mean_vals = vals['mean'].values.ravel()
    std_vals = vals['std'].values.ravel()
    axes[i, 0].plot(gait_pct, mean_vals, color=colors[i], linewidth=2, label='Mean')
    axes[i, 0].fill_between(gait_pct, mean_vals - std_vals, mean_vals + std_vals,
                              color=colors[i], alpha=0.2, label='±1 Std')
    axes[i, 0].set_ylabel(label + '\n(deg)')
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].legend(loc='upper right', fontsize=8)

# Plot right side moments
for i, (col, label) in enumerate(zip(moment_cols_r, moment_labels_r)):
    vals = norm_df.pivot_table(values=col, index='gait_pct', aggfunc=['mean', 'std'])
    mean_vals = vals['mean'].values.ravel()
    std_vals = vals['std'].values.ravel()
    axes[i, 1].plot(gait_pct, mean_vals, color=colors[i], linewidth=2, label='Mean')
    axes[i, 1].fill_between(gait_pct, mean_vals - std_vals, mean_vals + std_vals,
                              color=colors[i], alpha=0.2, label='±1 Std')
    axes[i, 1].set_ylabel(label + '\n(Nm/kg)')
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].legend(loc='upper right', fontsize=8)

axes[2, 0].set_xlabel('Gait Cycle (%)')
axes[2, 1].set_xlabel('Gait Cycle (%)')
axes[0, 0].set_title('Joint Angles (Right Side)')
axes[0, 1].set_title('Joint Moments (Right Side)')

plt.tight_layout()
out_path = Path(r"E:\外骨骼算法\gait_data\processed\gait_curves_part1.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close()

# ---- Also plot per-subject comparison ----
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
fig.suptitle('Per-Subject Gait Curves - Hip/Knee/Ankle (Right Side)', fontsize=14)

subjects = sorted(norm_df['subject'].unique())
cmap = plt.cm.tab10

for i, (col, label) in enumerate(zip(angle_cols_r, angle_labels_r)):
    for j, subj in enumerate(subjects):
        sub_data = norm_df[norm_df['subject'] == subj]
        vals = sub_data.pivot_table(values=col, index='gait_pct', aggfunc='mean')
        axes[i, 0].plot(gait_pct, vals[col].values, color=cmap(j), linewidth=1.5,
                          label=subj, alpha=0.8)
    axes[i, 0].set_ylabel(label + '\n(deg)')
    axes[i, 0].grid(True, alpha=0.3)
    if i == 0:
        axes[i, 0].legend(loc='upper right', fontsize=7, ncol=3)

for i, (col, label) in enumerate(zip(moment_cols_r, moment_labels_r)):
    for j, subj in enumerate(subjects):
        sub_data = norm_df[norm_df['subject'] == subj]
        vals = sub_data.pivot_table(values=col, index='gait_pct', aggfunc='mean')
        axes[i, 1].plot(gait_pct, vals[col].values, color=cmap(j), linewidth=1.5,
                          label=subj, alpha=0.8)
    axes[i, 1].set_ylabel(label + '\n(Nm/kg)')
    axes[i, 1].grid(True, alpha=0.3)

axes[2, 0].set_xlabel('Gait Cycle (%)')
axes[2, 1].set_xlabel('Gait Cycle (%)')
axes[0, 0].set_title('Joint Angles (Right Side)')
axes[0, 1].set_title('Joint Moments (Right Side)')

plt.tight_layout()
out_path2 = Path(r"E:\外骨骼算法\gait_data\processed\gait_curves_per_subject.png")
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path2}")
plt.close()

print("\nDone!")
