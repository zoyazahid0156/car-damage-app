import os

# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "patchcore_data_17"


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_DIR):
    print("ERROR: patchcore_data folder was not found.")
    raise SystemExit


total_images = 0
total_parts = 0


print()
print("================================================")
print("PATCHCORE DATASET CHECK")
print("================================================")
print()


# ============================================================
# GO THROUGH VEHICLE VARIANTS
# ============================================================

for variant in sorted(os.listdir(DATASET_DIR)):

    variant_path = os.path.join(
        DATASET_DIR,
        variant
    )

    if not os.path.isdir(variant_path):
        continue


    print()
    print("-----------------------------------------------")
    print(variant)
    print("-----------------------------------------------")


    variant_total = 0


    # --------------------------------------------------------
    # Go through parts
    # --------------------------------------------------------

    for part in sorted(os.listdir(variant_path)):

        part_path = os.path.join(
            variant_path,
            part,
            "good"
        )

        if not os.path.isdir(part_path):
            continue


        # ----------------------------------------------------
        # Count images
        # ----------------------------------------------------

        image_count = 0

        for filename in os.listdir(part_path):

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            ]:
                image_count += 1


        print(
            f"{part:<30} {image_count:>4} good images"
        )


        variant_total += image_count
        total_images += image_count
        total_parts += 1


    print()
    print(
        f"Variant total: {variant_total} images"
    )


# ============================================================
# FINAL TOTAL
# ============================================================

print()
print("================================================")
print("TOTAL")
print("================================================")

print(
    f"Part folders : {total_parts}"
)

print(
    f"Good images : {total_images}"
)

print()