"""
Build a NEW PatchCore dataset with an 80% / 20% split of GOOD (undamaged) part crops.

For every part it pools
   - the good images already in  SOURCE_DATASET/<part>/train/good  and  .../test/good
   - the new clean crops listed in part_crops_17/crop_index.csv
removes exact duplicates (same file content), then splits 80% -> train/good and 20% -> test/good.

THE SPLIT IS DONE BY SOURCE PHOTO, not by crop: all crops cut from the same photo go to the same side.
(Crops of one photo look almost identical; splitting them would leak training images into the test set and make
the test results look better than they are.)

Nothing in SOURCE_DATASET is changed. The result is written to DEST_DATASET (must not exist yet).
Other test folders (e.g. test/damaged) are copied unchanged.

Run:  python make_patchcore_split.py
"""
import csv
import hashlib
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

SOURCE_DATASET = PROJECT / "patchcore_data_17_split_old"          # latest dataset (already has the earlier added crops)
DEST_DATASET = PROJECT / "patchcore_data_17_split"     # new folder, created by this script
CROP_INDEX_CSV = PROJECT / "part_crops_17" / "crop_index.csv"

TRAIN_FRACTION = 0.80
SEED = 42

# Which crops from crop_index.csv are CLEAN (undamaged)?  PatchCore must learn only from clean parts.
#   - If the csv has a column "review_status", only rows equal to "clean" are used.
#   - Otherwise ALL rows are used (set this to False to stop instead, so you can add the review column first).
ALLOW_UNREVIEWED_CROPS = True

EXPECTED_PARTS = {
    "front_bumper", "rear_bumper", "front_door", "rear_door", "front_window", "rear_window", "wheel",
    "front_fender", "quarter_panel", "hood", "trunk_lid", "headlight", "taillight", "side_mirror",
    "windshield_front", "windshield_back", "license_plate",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PART_COLS = ("part", "category", "class", "label")
PATH_COLS = ("crop_path", "crop_file", "crop", "path", "file", "filename", "crop_name")
SOURCE_COLS = ("source_image", "source", "image", "image_name", "file_name", "master_image")


def pick(row, names):
    for n in names:
        if n in row and row[n] is not None and row[n].strip():
            return row[n].strip()
    return None


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_from_name(stem):
    """'new_42_front_bumper_01' / '42_wheel_2' / 'img_42_hood' -> source photo '42'.  Otherwise the file itself."""
    m = re.match(r"^(?:new_)?(?:crop_|img_|image_)?(\d+)", stem, re.IGNORECASE)
    return m.group(1) if m else stem


def read_crop_index():
    if not CROP_INDEX_CSV.is_file():
        raise SystemExit(f"Crop index not found: {CROP_INDEX_CSV}")
    rows = []
    with CROP_INDEX_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        print(f"crop_index.csv columns: {cols}")
        has_review = "review_status" in cols
        if not has_review and not ALLOW_UNREVIEWED_CROPS:
            raise SystemExit("crop_index.csv has no review_status column and ALLOW_UNREVIEWED_CROPS = False.")
        if not has_review:
            print("WARNING: no review_status column - ALL crops are treated as clean. If some of these crops show\n"
                  "         damaged parts, PatchCore will learn damage as normal. Review them first if unsure.")
        for r in reader:
            if has_review and r["review_status"].strip().lower() != "clean":
                continue
            part, path, src = pick(r, PART_COLS), pick(r, PATH_COLS), pick(r, SOURCE_COLS)
            if not part or not path:
                raise SystemExit(f"Cannot find the part / crop path columns in crop_index.csv (columns: {cols}). "
                                 f"Tell me the column names and I will adjust PART_COLS / PATH_COLS.")
            p = Path(path.replace("\\", "/"))
            for base in (None, PROJECT, CROP_INDEX_CSV.parent, CROP_INDEX_CSV.parent / part):
                cand = p if base is None else base / p
                if cand.is_file():
                    p = cand
                    break
            else:
                raise SystemExit(f"Crop image not found: {path}")
            rows.append({"part": part, "path": p, "source": Path(src).stem if src else group_from_name(p.stem)})
    return rows


def main():
    if not SOURCE_DATASET.is_dir():
        raise SystemExit(f"Source dataset not found: {SOURCE_DATASET}")
    if DEST_DATASET.exists():
        raise SystemExit(f"Destination already exists: {DEST_DATASET}\nNothing was changed. Rename it or pick another DEST_DATASET.")
    actual = {p.name for p in SOURCE_DATASET.iterdir() if p.is_dir()}
    if actual != EXPECTED_PARTS:
        raise SystemExit(f"Source folders do not match the 17 parts.\nMissing: {sorted(EXPECTED_PARTS - actual)}\n"
                         f"Unexpected: {sorted(actual - EXPECTED_PARTS)}")

    new_rows = read_crop_index()
    for r in new_rows:
        if r["part"] not in EXPECTED_PARTS:
            raise SystemExit(f"Unexpected part in crop index: {r['part']}")
    # crop stem -> source photo, so the same photo is recognised even when its crop is already in the dataset
    stem_to_source = {r["path"].stem: r["source"] for r in new_rows}

    # ---- pool per part: (path, group) with duplicates removed by file content
    pool = defaultdict(list)
    seen = set()
    n_dups = 0

    def add(part, path, group):
        nonlocal n_dups
        digest = sha1(path)
        if (part, digest) in seen:
            n_dups += 1
            return
        seen.add((part, digest))
        pool[part].append((path, group))

    for part in sorted(EXPECTED_PARTS):
        for split in ("train", "test"):
            folder = SOURCE_DATASET / part / split / "good"
            if not folder.is_dir():
                continue
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXT:
                    stem = p.stem[4:] if p.stem.startswith("new_") else p.stem
                    add(part, p, stem_to_source.get(stem) or stem_to_source.get(p.stem) or group_from_name(p.stem))
    existing_total = sum(len(v) for v in pool.values())
    for r in new_rows:
        add(r["part"], r["path"], r["source"])

    # ---- build the destination: copy everything, then rebuild train/good and test/good
    print("Copying the source dataset (other folders such as test/damaged stay as they are)...")
    shutil.copytree(SOURCE_DATASET, DEST_DATASET)
    rng = random.Random(SEED)
    print(f"\nPooled {existing_total} existing good images + {len(new_rows)} new crops; "
          f"{n_dups} exact duplicates removed.\n")
    print(f"{'part':<18}{'photos':>7}{'images':>8}{'train':>8}{'test':>7}{'test %':>8}")
    tot_train = tot_test = 0
    for part in sorted(EXPECTED_PARTS):
        for split in ("train", "test"):
            g = DEST_DATASET / part / split / "good"
            if g.is_dir():
                shutil.rmtree(g)
            g.mkdir(parents=True, exist_ok=True)

        groups = defaultdict(list)
        for path, group in pool[part]:
            groups[group].append(path)
        ids = sorted(groups)
        rng.shuffle(ids)
        total = sum(len(v) for v in groups.values())
        want_test = round(total * (1 - TRAIN_FRACTION))
        test_ids, n_test = [], 0
        for gid in ids:
            if n_test >= want_test or len(test_ids) >= len(ids) - 1:     # always leave at least one photo for train
                break
            test_ids.append(gid)
            n_test += len(groups[gid])
        if not test_ids and len(ids) > 1:
            test_ids.append(ids[0]); n_test = len(groups[ids[0]])
        test_set = set(test_ids)

        n_train = 0
        used = set()
        for gid in ids:
            side = "test" if gid in test_set else "train"
            for p in groups[gid]:
                name = p.name
                if (side, name) in used:                       # two different files with the same name
                    name = f"{p.stem}_{sha1(p)[:6]}{p.suffix}"
                used.add((side, name))
                shutil.copy2(p, DEST_DATASET / part / side / "good" / name)
                n_train += side == "train"
        tot_train += n_train
        tot_test += n_test
        print(f"{part:<18}{len(ids):>7}{total:>8}{n_train:>8}{n_test:>7}{100 * n_test / max(1, total):>7.0f}%")

    print("-" * 56)
    print(f"{'TOTAL':<18}{'':>7}{tot_train + tot_test:>8}{tot_train:>8}{tot_test:>7}"
          f"{100 * tot_test / max(1, tot_train + tot_test):>7.0f}%")
    print(f"\nNew dataset: {DEST_DATASET}")
    print("The source dataset was not modified. 'test %' can differ slightly from 20% because whole photos move together.")


if __name__ == "__main__":
    main()