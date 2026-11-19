"""
Count images and annotations in a COCO json, per class, and flag problems.

  python check_coco.py                                  (checks car_parts_17_coco_final.json)
  python check_coco.py car_parts_17_coco_85_120.json    (any other file)
"""
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "car_parts_17_coco_final.json")
if not path.exists():
    sys.exit(f"File not found: {path.resolve()}")

d = json.load(open(path, encoding="utf-8"))
images, anns, cats = d.get("images", []), d.get("annotations", []), d.get("categories", [])
names = {c["id"]: c["name"] for c in cats}

print(f"File        : {path.resolve()}")
print(f"Images      : {len(images)}")
print(f"Annotations : {len(anns)}  (average {len(anns) / max(1, len(images)):.1f} boxes per image)")
print(f"Classes     : {len(cats)}\n")

per_class = Counter(a["category_id"] for a in anns)
print(f"{'class':<20}{'boxes':>7}")
for cid in sorted(names):
    flag = "   <-- few examples" if per_class[cid] < 20 else ""
    print(f"{names[cid]:<20}{per_class[cid]:>7}{flag}")
unknown = [c for c in per_class if c not in names]
if unknown:
    print("\nWARNING: boxes with an unknown class id:", unknown)

# ---- consistency checks
img_ids = [i["id"] for i in images]
ann_ids = [a["id"] for a in anns]
problems = []
if len(img_ids) != len(set(img_ids)):
    problems.append("duplicate image ids")
if len(ann_ids) != len(set(ann_ids)):
    problems.append("duplicate annotation ids")
fn = Counter(i["file_name"] for i in images)
dup_names = [n for n, c in fn.items() if c > 1]
if dup_names:
    problems.append(f"{len(dup_names)} file name(s) used by more than one image, e.g. {dup_names[:5]}")
orphans = [a["id"] for a in anns if a["image_id"] not in set(img_ids)]
if orphans:
    problems.append(f"{len(orphans)} annotation(s) point to a missing image")

with_boxes = {a["image_id"] for a in anns}
no_boxes = [i["file_name"] for i in images if i["id"] not in with_boxes]
print(f"\nImages with at least one box: {len(images) - len(no_boxes)} / {len(images)}")
if no_boxes:
    print(f"Images with NO boxes ({len(no_boxes)}): {no_boxes[:30]}{' ...' if len(no_boxes) > 30 else ''}")

size = {i["id"]: (i.get("width"), i.get("height")) for i in images}
bad_boxes = 0
for a in anns:
    x, y, w, h = a["bbox"]
    iw, ih = size.get(a["image_id"], (None, None))
    if w <= 0 or h <= 0 or (iw and ih and (x < 0 or y < 0 or x + w > iw + 1 or y + h > ih + 1)):
        bad_boxes += 1
if bad_boxes:
    problems.append(f"{bad_boxes} box(es) outside their image or with zero size")

print("\nProblems:" if problems else "\nNo problems found.")
for p in problems:
    print("  -", p)