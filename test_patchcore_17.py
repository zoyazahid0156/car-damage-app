from pathlib import Path
import sys

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


# ============================================================
# SETTINGS
# ============================================================

PRODUCTION_ROOT = Path("patchcore_models_17")

PARTS = [
    "front_bumper",
    "rear_bumper",
    "front_door",
    "rear_door",
    "front_window",
    "rear_window",
    "wheel",
    "front_fender",
    "rear_fender",
    "hood",
    "trunk_lid",
    "headlight",
    "taillight",
    "side_mirror",
    "windshield_front",
    "windshield_back",
    "license_plate",
]


# ============================================================
# CHECK MODEL FILES
# ============================================================

print("=" * 80)
print("PATCHCORE 17-MODEL PRODUCTION CHECK")
print("=" * 80)

missing = []

for part in PARTS:

    model_path = PRODUCTION_ROOT / part / "model.ckpt"

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)

        print(
            f"✅ {part:20s} "
            f"{size_mb:8.2f} MB"
        )

    else:
        print(f"❌ {part:20s} MODEL MISSING")
        missing.append(part)


print("=" * 80)

if missing:
    print("❌ Missing models:")
    for part in missing:
        print(f" {part}")

    print("\nStopping.")
    sys.exit(1)

print("✅ All 17 production models found.")
print("=" * 80)


# ============================================================
# IMPORTANT
# ============================================================
#
# This script currently performs ONLY the production-model
# validation.
#
# We intentionally do NOT load all 17 PatchCore models yet.
#
# Loading/training configuration must match the exact Anomalib
# version and checkpoint structure.
#
# After confirming the files, the next step will be the actual
# single-image scoring test.
# ============================================================

print()
print("Production model directory:")
print(PRODUCTION_ROOT.resolve())

print()
print("17-model check completed successfully.")
print()
print("Next step:")
print("We will test ONE clean part crop first.")
print("Then we will test a damaged crop.")
print("Then we will build the 17-part calibration system.")

print("=" * 80)