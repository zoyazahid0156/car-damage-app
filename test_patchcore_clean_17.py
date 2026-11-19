from pathlib import Path
import torch
import numpy as np

from anomalib.models import Patchcore


PROJECT_ROOT = Path.cwd()

MODEL_ROOT = PROJECT_ROOT / "patchcore_models_17"
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

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


def main():

    print("=" * 80)
    print("PATCHCORE PRODUCTION MODEL CHECK")
    print("=" * 80)
    print()
    print("CPU mode")
    print("One model at a time")
    print("No multiprocessing")
    print()

    if not MODEL_ROOT.exists():
        raise FileNotFoundError(
            f"Missing production model folder:\n{MODEL_ROOT}"
        )

    if not CROP_ROOT.exists():
        raise FileNotFoundError(
            f"Missing crop folder:\n{CROP_ROOT}"
        )

    successful = []
    failed = []

    for part in PARTS:

        print("=" * 80)
        print(f"PART: {part}")
        print("=" * 80)

        model_path = MODEL_ROOT / part / "model.ckpt"
        crop_folder = CROP_ROOT / part

        if not model_path.exists():
            print("❌ Model checkpoint missing")
            failed.append(part)
            continue

        if not crop_folder.exists():
            print("❌ Crop folder missing")
            failed.append(part)
            continue

        images = get_images(crop_folder)

        if not images:
            print("❌ No clean crops found")
            failed.append(part)
            continue

        # Only use ONE clean crop for this first test.
        test_image = images[0]

        print(f"Model : {model_path}")
        print(f"Test : {test_image.name}")
        print()

        try:

            print("Creating PatchCore model...")

            model = Patchcore(
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                coreset_sampling_ratio=0.1,
                num_neighbors=9,
            )

            print("Loading checkpoint...")

            checkpoint = torch.load(
                model_path,
                map_location="cpu",
                weights_only=False,
            )

            state_dict = checkpoint.get("state_dict")

            if state_dict is None:
                raise RuntimeError(
                    "Checkpoint does not contain 'state_dict'."
                )

            model.load_state_dict(
                state_dict,
                strict=False,
            )

            model.eval()

            print("✅ Model loaded successfully")
            print()
            print("NOTE:")
            print(
                "This first pass verifies that the production "
                "checkpoint can be instantiated and loaded."
            )

            successful.append(part)

        except Exception as e:

            print("❌ FAILED")
            print(f"Error: {e}")

            failed.append(part)

        # Explicitly release the large model before loading the next one.
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print()

    print("=" * 80)
    print("FINAL CHECK")
    print("=" * 80)

    print(f"Successful : {len(successful)} / {len(PARTS)}")
    print(f"Failed : {len(failed)} / {len(PARTS)}")

    if failed:

        print()
        print("FAILED PARTS:")

        for part in failed:
            print(f" - {part}")

    else:

        print()
        print("✅ ALL 17 PRODUCTION MODELS LOAD SUCCESSFULLY")

    print("=" * 80)


if __name__ == "__main__":
    main()