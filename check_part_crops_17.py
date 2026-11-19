from pathlib import Path

PROJECT_ROOT = Path.cwd()
CROP_ROOT = PROJECT_ROOT / "part_crops_17"

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

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("=" * 80)
print("CURRENT 17-PART CROP DATASET")
print("=" * 80)

if not CROP_ROOT.exists():
    raise FileNotFoundError(
        f"Folder not found:\n{CROP_ROOT}"
    )

total = 0

for part in PARTS:

    folder = CROP_ROOT / part

    if not folder.exists():
        print(f"{part:<22}: MISSING FOLDER")
        continue

    files = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
    ]

    count = len(files)
    total += count

    print(f"{part:<22}: {count}")

print("-" * 80)
print(f"{'TOTAL':<22}: {total}")

print()
print("Expected: 17 part folders")
print()
print("These are the crops that will be used to update")
print("the existing patchcore_data_17 dataset.")
print("=" * 80)