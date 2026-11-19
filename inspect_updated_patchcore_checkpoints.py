from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "patchcore_results_17_updated"

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

print("=" * 70)
print("UPDATED PATCHCORE CHECKPOINT INSPECTION")
print("=" * 70)

if not RESULTS_ROOT.exists():
    raise SystemExit(f"ERROR: Results folder not found:\n{RESULTS_ROOT}")

successful = []
failed = []

for part in PARTS:
    part_dir = RESULTS_ROOT / part

# Search inside the part's folder, including nested folders.
checkpoints = list(part_dir.rglob("*.ckpt")) if part_dir.exists() else []

valid_checkpoints = [
    path for path in checkpoints
    if path.is_file() and path.stat().st_size > 0
]

if valid_checkpoints:
    successful.append(part)
    for checkpoint in valid_checkpoints:
        print(
            f"[OK] {part}: {checkpoint.relative_to(RESULTS_ROOT)} "
            f"({checkpoint.stat().st_size / (1024 * 1024):.2f} MB)"
        )
else:
    failed.append(part)
    print(f"[MISSING] {part}: no non-empty .ckpt file found")

print("\n" + "=" * 70)
print(f"Categories with checkpoints: {len(successful)}/17")
print(f"Categories missing checkpoints: {len(failed)}/17")

if failed:
    print("Needs investigation:", ", ".join(failed))
else:
    print("All 17 categories have non-empty checkpoint files.")

print("=" * 70)