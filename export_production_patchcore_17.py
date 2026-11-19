from pathlib import Path
import shutil
from datetime import datetime

# ============================================================
# SETTINGS
# ============================================================

SOURCE_ROOT = Path("patchcore_results_17")
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
    "quarter_panel",
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
# FIND NEWEST CHECKPOINT
# ============================================================

def find_newest_checkpoint(part):
    part_folder = SOURCE_ROOT / part

    if not part_folder.exists():
        return None

    checkpoints = list(part_folder.rglob("*.ckpt"))

    if not checkpoints:
        return None

    return max(
        checkpoints,
        key=lambda p: p.stat().st_mtime
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("EXPORTING 17 PATCHCORE MODELS")
print("=" * 80)

PRODUCTION_ROOT.mkdir(parents=True, exist_ok=True)

missing = []
exported = []

for part in PARTS:

    print()
    print("-" * 80)
    print(f"PART: {part}")

    checkpoint = find_newest_checkpoint(part)

    if checkpoint is None:
        print("❌ No checkpoint found")
        missing.append(part)
        continue

    # Create production part folder
    part_output = PRODUCTION_ROOT / part
    part_output.mkdir(parents=True, exist_ok=True)

    # Final production filename
    destination = part_output / "model.ckpt"

    # Copy newest checkpoint
    shutil.copy2(checkpoint, destination)

    # Get checkpoint information
    timestamp = datetime.fromtimestamp(
        checkpoint.stat().st_mtime
    ).strftime("%Y-%m-%d %H:%M:%S")

    print(f"Source : {checkpoint}")
    print(f"Created : {timestamp}")
    print(f"Production : {destination}")
    print("✅ Exported")

    exported.append(part)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("EXPORT COMPLETE")
print("=" * 80)

print(f"Models exported : {len(exported)} / {len(PARTS)}")

if missing:
    print()
    print("❌ MISSING PARTS:")
    for part in missing:
        print(f" - {part}")
else:
    print()
    print("✅ ALL 17 PRODUCTION MODELS CREATED")


print()
print("Production model folder:")
print(PRODUCTION_ROOT.resolve())

print()
print("Expected structure:")
print()

for part in PARTS:
    print(f"patchcore_models_17/{part}/model.ckpt")

print("=" * 80)