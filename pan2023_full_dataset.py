"""
Pan 2023 Biological Torque Controller - full dataset (22 subjects).

Reproduces the hip-exoskeleton biological torque controller on every
level-ground walking trial of the Camargo 2021 dataset and regenerates the
three figures used in the README.

Usage
-----
    python pan2023_full_dataset.py
    python pan2023_full_dataset.py --level 3
    python pan2023_full_dataset.py --csv /path/to/levelground_all.csv

``processed/levelground_all.csv`` is NOT tracked by git (~1.5 GB). Run
process_part1.py followed by process_part2_3_batch.py first, or point --csv at
an existing copy. All other paths follow ``config.py`` (CLI > env > repo root).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

import config


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the Pan 2023 biological torque controller on the full dataset.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Merged level-ground CSV (default: <repo>/processed/levelground_all.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write figures (default: $GAIT_OUTPUT_DIR or <repo>/processed)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=2,
        choices=config.ASSISTANCE_LEVEL_IDS,
        help=f"Assistance level for the demo figure (default: 2)",
    )
    parser.add_argument(
        "--subject",
        default=config.DEMO_SUBJECT,
        help=f"Subject used for the demo figure (default: {config.DEMO_SUBJECT})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figures interactively (disabled on headless machines)",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------
def sum_of_gaussians(t, peaks):
    """Sum of Gaussian bumps: peaks is a list of (mu, sigma, amplitude)."""
    result = np.zeros_like(t, dtype=float)
    for mean, std, amp in peaks:
        result += amp * np.exp(-0.5 * ((t - mean) / std) ** 2)
    return result


def controller_peaks(amplitude):
    """(mu, sigma, amp) pairs for the flexion and extension assistance bumps."""
    mu_f, sigma_f = config.FLEXION_GAUSSIAN
    mu_e, sigma_e = config.EXTENSION_GAUSSIAN
    return [(mu_f, sigma_f, amplitude), (mu_e, sigma_e, -amplitude)]


def estimate_gait_phase(hip_angle, sample_rate=config.TARGET_HZ):
    """Estimate gait phase (0-100%) from the hip flexion angle.

    A hip-flexion peak is treated as ~90% of the gait cycle (toe-off); phase is
    then linearly interpolated from peak to peak and wrapped into [0, 100).
    """
    n_samples = len(hip_angle)
    gait_phase = np.zeros(n_samples)

    b, a = butter(4, 5 / (sample_rate / 2), btype="low")
    hip_filtered = filtfilt(b, a, hip_angle)

    peaks, _ = find_peaks(
        hip_filtered, distance=int(sample_rate * 0.5), prominence=3
    )

    if len(peaks) < 2:
        return gait_phase, peaks

    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        cycle_len = end - start
        phase_values = np.linspace(
            config.PHASE_AT_FLEXION_PEAK,
            config.PHASE_AT_FLEXION_PEAK + 100,
            cycle_len,
        )
        gait_phase[start:end] = phase_values % 100

    return gait_phase, peaks


def generate_assistance(gait_phase, amplitude):
    """Torque commanded at each estimated gait phase."""
    assistance = np.zeros_like(gait_phase)
    for i, phase in enumerate(gait_phase):
        if phase > 0:
            assistance[i] = sum_of_gaussians(
                np.array([phase]), controller_peaks(amplitude)
            )[0]
    return assistance


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------
def normalize_to_gait_cycle(group, n_points=config.GAIT_CYCLE_POINTS):
    """Resample one trial onto a normalised 0-100% gait cycle."""
    result = {}
    numeric_cols = group.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in config.EXCLUDED_RESAMPLE_COLS:
            continue
        values = group[col].values
        if len(values) > 1:
            x_old = np.linspace(0, 100, len(values))
            x_new = np.linspace(0, 100, n_points)
            result[col] = np.interp(x_new, x_old, values)
        else:
            result[col] = np.full(n_points, values[0])
    result["gait_pct"] = np.linspace(0, 100, n_points)
    return pd.DataFrame(result)


def load_dataset(csv_path):
    config.require_input(csv_path, config.PIPELINE_HINT)
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} cols")
    print(f"Subjects: {sorted(df['subject'].unique())}")
    print(f"Total subjects: {len(df['subject'].unique())}")
    return df


def main(argv=None):
    args = parse_args(argv)

    output_dir = config.output_dir(args.output_dir)
    csv_path = Path(args.csv) if args.csv else config.merged_csv(output_dir)
    df = load_dataset(csv_path)

    amplitude = config.ASSISTANCE_LEVELS[args.level]

    # ------------------------------------------------------------------
    # Normative hip moment across all subjects
    # ------------------------------------------------------------------
    print("\nNormalizing all trials to gait cycle...")
    normalized = []
    for (subject, trial), group in df.groupby(["subject", "trial"]):
        norm_df = normalize_to_gait_cycle(group)
        norm_df["subject"] = subject
        norm_df["trial"] = trial
        normalized.append(norm_df)

    norm_df = pd.concat(normalized, ignore_index=True)
    print(f"Normalized: {len(norm_df)} rows")

    gait_pct = np.linspace(0, 100, config.GAIT_CYCLE_POINTS)
    hip_moment_r = norm_df.groupby("gait_pct")["hip_flexion_r_moment"].mean().values
    hip_moment_l = norm_df.groupby("gait_pct")["hip_flexion_l_moment"].mean().values
    hip_moment_avg = (hip_moment_r + hip_moment_l) / 2

    print("\nNormative hip moment (all subjects):")
    print(
        f"  Peak flexion: {hip_moment_avg.max():.2f} Nm/kg "
        f"at {gait_pct[hip_moment_avg.argmax()]:.0f}%"
    )
    print(
        f"  Peak extension: {hip_moment_avg.min():.2f} Nm/kg "
        f"at {gait_pct[hip_moment_avg.argmin()]:.0f}%"
    )

    # ------------------------------------------------------------------
    # Figure 1 - normative gait biomechanics
    # ------------------------------------------------------------------
    gait_phase = np.linspace(0, 100, 200)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Camargo 2021 Dataset - Healthy Gait Biomechanics\n"
        f"{df['subject'].nunique()} Subjects, Level Ground Walking "
        f"({len(df['trial'].unique())} trials)",
        fontsize=14,
    )

    ax = axes[0, 0]
    hip_angle_r = norm_df.groupby("gait_pct")["hip_flexion_r"].mean().values
    hip_angle_l = norm_df.groupby("gait_pct")["hip_flexion_l"].mean().values
    hip_angle_std_r = norm_df.groupby("gait_pct")["hip_flexion_r"].std().values
    ax.plot(gait_pct, hip_angle_r, "b-", linewidth=2, label="Right")
    ax.fill_between(
        gait_pct,
        hip_angle_r - hip_angle_std_r,
        hip_angle_r + hip_angle_std_r,
        alpha=0.2,
        color="blue",
    )
    ax.plot(gait_pct, hip_angle_l, "r-", linewidth=2, label="Left")
    ax.set_xlabel("Gait Cycle (%)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Hip Flexion/Extension")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    knee_r = norm_df.groupby("gait_pct")["knee_angle_r"].mean().values
    knee_l = norm_df.groupby("gait_pct")["knee_angle_l"].mean().values
    ax.plot(gait_pct, knee_r, "b-", linewidth=2, label="Right")
    ax.plot(gait_pct, knee_l, "r-", linewidth=2, label="Left")
    ax.set_xlabel("Gait Cycle (%)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Knee Flexion/Extension")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(gait_pct, hip_moment_avg, "k-", linewidth=2, label="Average")
    ax.fill_between(gait_pct, 0, hip_moment_avg, alpha=0.3)
    ax.set_xlabel("Gait Cycle (%)")
    ax.set_ylabel("Moment (Nm/kg)")
    ax.set_title("Hip Joint Moment (Normative)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    ax = axes[1, 1]
    ankle_r = norm_df.groupby("gait_pct")["ankle_angle_r"].mean().values
    ankle_l = norm_df.groupby("gait_pct")["ankle_angle_l"].mean().values
    ax.plot(gait_pct, ankle_r, "b-", linewidth=2, label="Right")
    ax.plot(gait_pct, ankle_l, "r-", linewidth=2, label="Left")
    ax.set_xlabel("Gait Cycle (%)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Ankle Dorsiflexion/Plantarflexion")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = output_dir / "gait_curves_all_subjects.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out1}")
    if args.show:
        plt.show()
    plt.close()

    # ------------------------------------------------------------------
    # Figure 2 - controller design parameters
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Biological Torque Controller - Pan 2023 Reproduction\n"
        "Reference Hip Torque Profiles using Sum of Gaussians",
        fontsize=14,
    )

    ax = axes[0, 0]
    for (level, scale), color in zip(
        config.ASSISTANCE_LEVELS.items(), config.ASSISTANCE_COLORS
    ):
        assistance = sum_of_gaussians(gait_phase, controller_peaks(scale))
        ax.plot(
            gait_phase,
            assistance,
            color=color,
            linewidth=2,
            label=f"Level {level} ({scale * config.REFERENCE_BODY_MASS_KG:.0f} Nm)",
        )
    ax.set_xlabel("Gait Cycle (%)")
    ax.set_ylabel("Hip Torque (Nm/kg)")
    ax.set_title("Assistance Profiles by Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    ax = axes[0, 1]
    ax.axvspan(0, 60, alpha=0.2, color="blue", label="Stance (0-60%)")
    ax.axvspan(60, 100, alpha=0.2, color="green", label="Swing (60-100%)")
    ax.axvline(x=45, color="red", linestyle="--", linewidth=2, label="Flexion onset (45%)")
    ax.axvline(x=60, color="red", linestyle=":", linewidth=2, label="Flexion peak (60%)")
    ax.axvline(x=90, color="blue", linestyle="--", linewidth=2, label="Extension onset (90%)")
    ax.axvline(x=10, color="blue", linestyle=":", linewidth=2, label="Extension peak (10%)")
    ax.set_xlabel("Gait Cycle (%)")
    ax.set_title("Assistance Timing (Pan 2023)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    normative_scaled = (
        hip_moment_avg / np.max(np.abs(hip_moment_avg)) * amplitude
    )
    controller_ref = sum_of_gaussians(gait_phase, controller_peaks(amplitude))
    ax.plot(gait_pct, normative_scaled, "k-", linewidth=2, label="Normative (scaled)")
    ax.plot(gait_phase, controller_ref, "r--", linewidth=2, label="Controller reference")
    ax.fill_between(gait_phase, 0, controller_ref, alpha=0.2, color="red")
    ax.set_xlabel("Gait Cycle (%)")
    ax.set_ylabel("Hip Torque (Nm/kg)")
    ax.set_title("Normative Hip Moment vs Controller Reference")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    ax = axes[1, 1]
    subjects = sorted(df["subject"].unique())
    cmap = plt.cm.tab20(np.linspace(0, 1, len(subjects)))
    for i, subj in enumerate(subjects):
        subj_data = norm_df[norm_df["subject"] == subj]
        if len(subj_data) > 0:
            moment = subj_data.groupby("gait_pct")["hip_flexion_r_moment"].mean().values
            ax.plot(gait_pct, moment, color=cmap[i], linewidth=1, alpha=0.5)
    ax.plot(gait_pct, hip_moment_avg, "k-", linewidth=2, label="Average")
    ax.set_xlabel("Gait Cycle (%)")
    ax.set_ylabel("Moment (Nm/kg)")
    ax.set_title("Per-Subject Hip Moments")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out2 = output_dir / "biological_torque_controller_all.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    if args.show:
        plt.show()
    plt.close()

    # ------------------------------------------------------------------
    # Figure 3 - single-trial controller demo
    # ------------------------------------------------------------------
    sample_trials = df[df["subject"] == args.subject]["trial"].unique()
    preferred = [t for t in sample_trials if "normal" in t and "ccw" in t]
    if not preferred:
        raise SystemExit(
            f"No level-ground 'normal ccw' trial found for subject {args.subject}. "
            f"Choose another one with --subject."
        )
    sample_trial = preferred[0]
    sample_data = df[
        (df["subject"] == args.subject) & (df["trial"] == sample_trial)
    ].copy()

    print(
        f"\nDemo trial: {args.subject}/{sample_trial} "
        f"({len(sample_data)} samples, level {args.level})"
    )

    hip_angle = sample_data["hip_flexion_r"].values
    estimated_phase, peaks = estimate_gait_phase(hip_angle)
    assistance = generate_assistance(estimated_phase, amplitude)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"Biological Torque Controller Demo - {args.subject}/{sample_trial}\n"
        f"Pan 2023 Reproduction using Camargo 2021 Dataset "
        f"({df['subject'].nunique()} subjects)",
        fontsize=14,
    )

    time_s = np.arange(len(sample_data)) / config.TARGET_HZ

    b, a = butter(4, 5 / 100, btype="low")
    hip_filtered = filtfilt(b, a, hip_angle)
    axes[0].plot(time_s, hip_angle, "b-", linewidth=1, alpha=0.5, label="Raw")
    axes[0].plot(time_s, hip_filtered, "b-", linewidth=2, label="Filtered")
    axes[0].plot(
        time_s[peaks], hip_angle[peaks], "rv", markersize=10, label="Detected peaks"
    )
    axes[0].set_ylabel("Hip Flexion (deg)")
    axes[0].set_title("Hip Flexion Angle (with peak detection)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, estimated_phase, "g-", linewidth=1.5)
    axes[1].set_ylabel("Gait Phase (%)")
    axes[1].set_title("Estimated Gait Phase")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_s, assistance, "r-", linewidth=1.5)
    axes[2].set_ylabel("Torque (Nm/kg)")
    axes[2].set_title(
        f"Assistance Torque (Level {args.level} - "
        f"{amplitude * config.REFERENCE_BODY_MASS_KG:.0f} Nm)"
    )
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color="k", linestyle="--", alpha=0.5)

    ax4_twin = axes[3].twinx()
    axes[3].plot(time_s, hip_filtered, "b-", linewidth=1.5, label="Hip angle")
    ax4_twin.plot(time_s, assistance, "r-", linewidth=1.5, label="Assistance")
    axes[3].set_ylabel("Hip Flexion (deg)", color="b")
    ax4_twin.set_ylabel("Torque (Nm/kg)", color="r")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Hip Angle vs Assistance Torque")
    axes[3].grid(True, alpha=0.3)
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    axes[3].legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    out3 = output_dir / "biological_torque_controller_demo_all.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")
    if args.show:
        plt.show()
    plt.close()

    print("\n" + "=" * 60)
    print("Full Dataset Reproduction Complete!")
    print("=" * 60)
    print(f"\nDataset: {df['subject'].nunique()} subjects, "
          f"{len(df['trial'].unique())} trials")
    print("Generated files:")
    print(f"  1. {out1.name}")
    print(f"  2. {out2.name}")
    print(f"  3. {out3.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
