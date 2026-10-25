"""
Central configuration for the hip-exoskeleton biological torque controller.

History
-------
Every script originally hardcoded an absolute Windows path
(``E:\\外骨骼算法\\gait_data\\...``) plus an absolute venv interpreter path,
which made the repository unusable for anyone who cloned it.

Now all paths are resolved relative to this file, and every default can be
overridden either from the command line or from the environment.

Expected layout under the data root::

    <DATA_ROOT>/Part1/        raw Camargo 2021 data (download: see README)
    <DATA_ROOT>/Part2/
    <DATA_ROOT>/Part3/
    <DATA_ROOT>/processed/    generated CSVs + figures

Environment variables
---------------------
DATA_ROOT        Root containing Part1/Part2/Part3.   Default: repository root.
GAIT_OUTPUT_DIR  Directory for generated artifacts.   Default: <repo>/processed.
GAIT_PYTHON      Interpreter for the .mat batch reader subprocess.
                 Default: sys.executable of the running interpreter.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_PARTS = ("Part1", "Part2", "Part3")

# --------------------------------------------------------------------------
# Signal / dataset constants (Pan 2023 + Camargo 2021)
# --------------------------------------------------------------------------
#: Target sampling rate of the IK/ID signals, in Hz.
TARGET_HZ = 200.0

#: Working directory used by process_part2_3_batch.py for intermediate CSVs.
TMP_DIR_NAME = "_tmp_batch"

#: CSV filename produced by each Part, e.g. Part1 -> levelground_part1.csv
PER_PART_CSV_TEMPLATE = "levelground_{part}.csv"

#: CSV filename holding all three Parts merged together.
MERGED_CSV_NAME = "levelground_all.csv"

#: Numeric columns excluded when resampling to a normalised gait cycle.
EXCLUDED_RESAMPLE_COLS = ("sample_rate", "time_s")

#: Points per normalised gait cycle.
GAIT_CYCLE_POINTS = 100

# --------------------------------------------------------------------------
# Controller constants (sum-of-Gaussians biological torque controller)
# --------------------------------------------------------------------------
#: Peak torque amplitude [Nm/kg] per assistance level.
ASSISTANCE_LEVELS = {0: 0.0, 1: 0.03, 2: 0.05, 3: 0.08}

#: Plot colours, kept in sync with ASSISTANCE_LEVELS.
ASSISTANCE_COLORS = ["gray", "orange", "green", "red"]

#: Reference body mass [kg] used to convert Nm/kg -> Nm in figure labels.
REFERENCE_BODY_MASS_KG = 75.0

#: Flexion-assistance Gaussian: (mu, sigma) in % of the gait cycle.
FLEXION_GAUSSIAN = (60.0, 8.0)

#: Extension-assistance Gaussian: (mu, sigma) in % of the gait cycle.
EXTENSION_GAUSSIAN = (10.0, 8.0)

#: Hip-flexion peak is treated as ~90% of the gait cycle (toe-off).
PHASE_AT_FLEXION_PEAK = 90.0

#: Subject used for the single-trial controller demo.
DEMO_SUBJECT = "AB06"

#: Valid assistance level ids, used for CLI validation.
ASSISTANCE_LEVEL_IDS = tuple(sorted(ASSISTANCE_LEVELS))

#: Message appended when a generated CSV is missing.
PIPELINE_HINT = (
    "This CSV is generated locally and is deliberately NOT tracked by git "
    "(see .gitignore) because it can reach ~1.5 GB.\n"
    "Run the extraction pipeline first:\n"
    "    python process_part1.py\n"
    "    python process_part2_3_batch.py\n"
    "then pass a custom location with --csv if it lives elsewhere."
)


# --------------------------------------------------------------------------
# Path resolution helpers
# --------------------------------------------------------------------------
def expand(value):
    """Expand ``~`` and environment variables; return None for falsy input."""
    if not value:
        return None
    return Path(os.path.expandvars(str(value))).expanduser()


def data_root(override=None):
    """Resolve the raw-data root.

    Priority: CLI argument > $DATA_ROOT > repository root.
    """
    candidate = expand(override) or expand(os.environ.get("DATA_ROOT")) or REPO_ROOT
    return candidate.resolve()


def output_dir(override=None):
    """Resolve the artifact output directory, creating it when missing.

    Priority: CLI argument > $GAIT_OUTPUT_DIR > <repo>/processed.
    """
    candidate = (
        expand(override)
        or expand(os.environ.get("GAIT_OUTPUT_DIR"))
        or (REPO_ROOT / "processed")
    )
    resolved = candidate.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def python_executable(override=None):
    """Interpreter used for the subprocess-based .mat batch reader.

    Priority: CLI argument > $GAIT_PYTHON > sys.executable.
    """
    return str(expand(override) or expand(os.environ.get("GAIT_PYTHON")) or sys.executable)


def part_dir(root, part):
    """Return ``<root>/<part>``."""
    return root / part


def per_part_csv(out_dir, part):
    """Return the per-Part CSV path, e.g. ``processed/levelground_part1.csv``."""
    return out_dir / PER_PART_CSV_TEMPLATE.format(part=part.lower())


def merged_csv(out_dir):
    """Return the merged three-Part CSV path."""
    return out_dir / MERGED_CSV_NAME


def require_input(path, hint=""):
    """Fail early with an actionable message when a required input is absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}\n{hint}".rstrip())
    return path
