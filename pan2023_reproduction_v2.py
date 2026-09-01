"""
Pan 2023 Biological Torque Controller - Fixed version
Uses original time-series data for gait phase estimation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path

# ============================================================
# 1. Load data
# ============================================================
csv_path = Path(r"E:\外骨骼算法\gait_data\processed\levelground_part1.csv")
df = pd.read_csv(csv_path)
print(f"Loaded: {len(df)} rows")

# ============================================================
# 2. Sum of Gaussians - Biological Torque Controller
# ============================================================
def sum_of_gaussians(t, peaks):
    result = np.zeros_like(t, dtype=float)
    for mean, std, amp in peaks:
        result += amp * np.exp(-0.5 * ((t - mean) / std) ** 2)
    return result

# ============================================================
# 3. Gait Phase Estimator - Fixed
# ============================================================
def estimate_gait_phase(hip_angle, sample_rate=200):
    """
    Gait phase estimator using hip angle peaks.
    Hip flexion peak corresponds to ~90% gait cycle (toe-off).
    """
    n_samples = len(hip_angle)
    gait_phase = np.zeros(n_samples)
    
    # Filter signal
    b, a = butter(4, 5/(sample_rate/2), btype='low')
    hip_filtered = filtfilt(b, a, hip_angle)
    
    # Find hip flexion peaks (max flexion = toe-off = 90% gait cycle)
    peaks, _ = find_peaks(hip_filtered, distance=int(sample_rate*0.5), prominence=3)
    
    if len(peaks) < 2:
        return gait_phase, peaks
    
    # For each gait cycle, linearly interpolate phase
    # Peak = 90% gait cycle, next peak = 90% of next cycle
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        cycle_len = end - start
        
        # Phase goes from 90% at peak to 90% at next peak
        # So phase at start = 90%, phase at end = 190% (or 90% of next cycle)
        phase_at_peak = 90.0  # Hip flexion peak = 90% gait cycle
        
        # Linear from phase_at_peak to phase_at_peak + 100%
        phase_values = np.linspace(phase_at_peak, phase_at_peak + 100, cycle_len)
        phase_values = phase_values % 100  # Wrap to 0-100%
        
        gait_phase[start:end] = phase_values
    
    return gait_phase, peaks

# ============================================================
# 4. Assistance Generator
# ============================================================
def generate_assistance(gait_phase, scale=0.05):
    """Generate assistance torque based on gait phase"""
    assistance = np.zeros_like(gait_phase)
    for i, phase in enumerate(gait_phase):
        if phase > 0:
            flex = sum_of_gaussians(np.array([phase]), [(60, 8, scale)])
            ext = sum_of_gaussians(np.array([phase]), [(10, 8, -scale)])
            assistance[i] = flex + ext
    return assistance

# ============================================================
# 5. Test on a sample trial (use full time series)
# ============================================================
sample_subject = 'AB06'
sample_trial = 'levelground_ccw_normal_01_01'
sample_data = df[(df['subject'] == sample_subject) & 
                 (df['trial'] == sample_trial)].copy()

print(f"\nTesting on {sample_subject}/{sample_trial}: {len(sample_data)} samples")

# Estimate gait phase
hip_angle = sample_data['hip_flexion_r'].values
estimated_phase, peaks = estimate_gait_phase(hip_angle)
sample_data['estimated_phase'] = estimated_phase

# Generate assistance
sample_data['assistance_torque'] = generate_assistance(estimated_phase, scale=0.05)

print(f"Detected {len(peaks)} gait cycles")
print(f"Mean stride duration: {np.mean(np.diff(peaks))/200:.2f}s")

# ============================================================
# 6. Plot results
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f'Biological Torque Controller Demo - {sample_subject}/{sample_trial}\n'
             f'Pan 2023 Reproduction using Camargo 2021 Dataset', fontsize=14)

time_s = np.arange(len(sample_data)) / 200

# Plot 1: Hip angle with detected peaks
axes[0].plot(time_s, hip_angle, 'b-', linewidth=1, alpha=0.7, label='Raw')
b, a = butter(4, 5/100, btype='low')
hip_filtered = filtfilt(b, a, hip_angle)
axes[0].plot(time_s, hip_filtered, 'b-', linewidth=2, label='Filtered')
axes[0].plot(time_s[peaks], hip_angle[peaks], 'rv', markersize=10, label='Detected peaks')
axes[0].set_ylabel('Hip Flexion (deg)')
axes[0].set_title('Hip Flexion Angle (with peak detection)')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot 2: Estimated gait phase
axes[1].plot(time_s, estimated_phase, 'g-', linewidth=1.5)
axes[1].set_ylabel('Gait Phase (%)')
axes[1].set_title('Estimated Gait Phase')
axes[1].set_ylim(-5, 105)
axes[1].grid(True, alpha=0.3)

# Plot 3: Assistance torque
axes[2].plot(time_s, sample_data['assistance_torque'].values, 'r-', linewidth=1.5)
axes[2].set_ylabel('Torque (Nm/kg)')
axes[2].set_title('Assistance Torque (Level 2 - 4 Nm)')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Plot 4: Overlay - hip angle and assistance
ax4_twin = axes[3].twinx()
axes[3].plot(time_s, hip_filtered, 'b-', linewidth=1.5, label='Hip angle')
ax4_twin.plot(time_s, sample_data['assistance_torque'].values, 'r-', linewidth=1.5, label='Assistance')
axes[3].set_ylabel('Hip Flexion (deg)', color='b')
ax4_twin.set_ylabel('Torque (Nm/kg)', color='r')
axes[3].set_xlabel('Time (s)')
axes[3].set_title('Hip Angle vs Assistance Torque')
axes[3].grid(True, alpha=0.3)

# Combined legend
lines1, labels1 = axes[3].get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
axes[3].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
out_path = Path(r"E:\外骨骼算法\gait_data\processed\biological_torque_controller_demo.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close()

# ============================================================
# 7. Multiple assistance levels comparison
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Biological Torque Controller - Assistance Profiles\n'
             'Pan 2023 Reproduction', fontsize=14)

gait_phase = np.linspace(0, 100, 200)

# Plot 1: All assistance levels
ax1 = axes[0, 0]
levels = {0: 0, 1: 0.03, 2: 0.05, 3: 0.08}
colors = ['gray', 'orange', 'green', 'red']
for (level, scale), color in zip(levels.items(), colors):
    assistance = sum_of_gaussians(gait_phase, [(60, 8, scale)]) + \
                 sum_of_gaussians(gait_phase, [(10, 8, -scale)])
    ax1.plot(gait_phase, assistance, color=color, linewidth=2, 
             label=f'Level {level} ({scale*75:.0f} Nm)')
ax1.set_xlabel('Gait Cycle (%)')
ax1.set_ylabel('Hip Torque (Nm/kg)')
ax1.set_title('Assistance Profiles by Level')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Plot 2: Timing diagram
ax2 = axes[0, 1]
# Stance and swing phases
ax2.axvspan(0, 60, alpha=0.2, color='blue', label='Stance (0-60%)')
ax2.axvspan(60, 100, alpha=0.2, color='green', label='Swing (60-100%)')
# Assistance timing from Pan 2023
ax2.axvline(x=45, color='red', linestyle='--', linewidth=2, label='Flexion onset (45%)')
ax2.axvline(x=60, color='red', linestyle=':', linewidth=2, label='Flexion peak (60%)')
ax2.axvline(x=90, color='blue', linestyle='--', linewidth=2, label='Extension onset (90%)')
ax2.axvline(x=10, color='blue', linestyle=':', linewidth=2, label='Extension peak (10%)')
ax2.set_xlabel('Gait Cycle (%)')
ax2.set_title('Assistance Timing (Pan 2023)')
ax2.legend(fontsize=8, loc='upper right')
ax2.set_xlim(0, 100)
ax2.grid(True, alpha=0.3)

# Plot 3: Normative vs controller
ax3 = axes[1, 0]
# Load normative data
norm_df = pd.read_csv(csv_path)
def normalize_to_gait_cycle(group, n_points=100):
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

normalized = []
for (subject, trial), group in norm_df.groupby(['subject', 'trial']):
    normalized.append(normalize_to_gait_cycle(group))
norm_df = pd.concat(normalized, ignore_index=True)

gait_pct = np.linspace(0, 100, 100)
hip_moment_r = norm_df.groupby('gait_pct')['hip_flexion_r_moment'].mean().values
hip_moment_l = norm_df.groupby('gait_pct')['hip_flexion_l_moment'].mean().values
hip_moment_avg = (hip_moment_r + hip_moment_l) / 2

# Scale to match controller magnitude
normative_scaled = hip_moment_avg / np.max(np.abs(hip_moment_avg)) * 0.05
controller_ref = sum_of_gaussians(gait_phase, [(60, 8, 0.05)]) + \
                 sum_of_gaussians(gait_phase, [(10, 8, -0.05)])

ax3.plot(gait_pct, normative_scaled, 'k-', linewidth=2, label='Normative (scaled)')
ax3.plot(gait_phase, controller_ref, 'r--', linewidth=2, label='Controller reference')
ax3.fill_between(gait_phase, 0, controller_ref, alpha=0.2, color='red')
ax3.set_xlabel('Gait Cycle (%)')
ax3.set_ylabel('Hip Torque (Nm/kg)')
ax3.set_title('Normative Hip Moment vs Controller Reference')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Plot 4: Gaussian components
ax4 = axes[1, 1]
# Show individual Gaussians
gauss_flex1 = sum_of_gaussians(gait_phase, [(60, 8, 1.0)])
gauss_flex2 = sum_of_gaussians(gait_phase, [(60, 12, 0.5)])
gauss_ext1 = sum_of_gaussians(gait_phase, [(10, 8, -1.0)])
gauss_ext2 = sum_of_gaussians(gait_phase, [(10, 12, -0.5)])

ax4.plot(gait_phase, gauss_flex1, 'r-', linewidth=1.5, alpha=0.5, label='Flexion (σ=8)')
ax4.plot(gait_phase, gauss_flex2, 'r--', linewidth=1.5, alpha=0.5, label='Flexion (σ=12)')
ax4.plot(gait_phase, gauss_ext1, 'b-', linewidth=1.5, alpha=0.5, label='Extension (σ=8)')
ax4.plot(gait_phase, gauss_ext2, 'b--', linewidth=1.5, alpha=0.5, label='Extension (σ=12)')
ax4.plot(gait_phase, gauss_flex1 + gauss_ext1, 'k-', linewidth=2, label='Total (σ=8)')
ax4.set_xlabel('Gait Cycle (%)')
ax4.set_ylabel('Amplitude')
ax4.set_title('Gaussian Components')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
out_path2 = Path(r"E:\外骨骼算法\gait_data\processed\biological_torque_controller_profiles.png")
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path2}")
plt.close()

print("\n" + "="*60)
print("Pan 2023 Biological Torque Controller Reproduction Complete!")
print("="*60)
