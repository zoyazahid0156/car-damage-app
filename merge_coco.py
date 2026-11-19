"""
Merge the new COCO file (images 85-120) into your existing car_parts_17_coco_final.json.

  python merge_coco.py

What it does
  - makes a backup of the final file first (car_parts_17_coco_final.json.bak_YYYYMMDD_HHMMSS)
  - matches classes by NAME (the files may number the same 17 classes differently) and converts the new
    boxes to the final file's class ids; stops if the class NAMES differ
  - gives every new image and annotation a fresh id, so nothing collides with the old ones
  - STOPS if a new image has the same file_name as one already in the final file (a name clash means two
    different pictures would share one name). Set SKIP_DUPLICATES = True only if they really are the same pictures
  - checks every box is inside its image and every annotation points to an existing image
  - writes the merged result back into car_parts_17_coco_final.json

Put this file in the same folder as the two JSON files (or change the paths below).
Nothing is changed unless every check passes.
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

FINAL_JSON = Path("car_parts_17_coco_final.json")       # the big existing file (will be updated)
NEW_JSON = Path(sys.argv[1] if len(sys.argv) > 1 else "car_parts_17_coco_300_475.json")   # file from manual_labeler.py (or give its name: python merge_coco.py file.json)
OUT_JSON = FINAL_JSON                                   # set to another name to keep FINAL untouched
SKIP_DUPLICATES = False   # False = STOP if a new file name already exists in FINAL (safe). True = skip those images.


def load(path):
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("images", "annotations", "categories"):
        data.setdefault(key, [])
    return data


def cat_map(data):
    return {c["id"]: c["name"] for c in data["categories"]}


def main():
    final = load(FINAL_JSON)
    new = load(NEW_JSON)

    # ---- 1. classes are matched by NAME (the two files may number the same classes differently)
    a, b = cat_map(final), cat_map(new)                       # id -> name
    for label, m in (("final", a), ("new", b)):
        if len(set(m.values())) != len(m):
            sys.exit(f"ERROR: the {label} file lists the same class name twice. Nothing was changed.")
    if set(a.values()) != set(b.values()):
        print("ERROR: the two files do not have the same class NAMES. Nothing was changed.")
        print("  only in final:", sorted(set(a.values()) - set(b.values())))
        print("  only in new  :", sorted(set(b.values()) - set(a.values())))
        sys.exit(1)
    final_id_of_name = {name: cid for cid, name in a.items()}
    category_remap = {new_id: final_id_of_name[name] for new_id, name in b.items()}    # new id -> final id
    changed = {n: (i, category_remap[i]) for i, n in b.items() if category_remap[i] != i}
    if changed:
        print(f"NOTE: the files number their classes differently. New boxes are converted to the FINAL file's ids "
              f"({len(changed)} of {len(b)} classes change number), e.g.:")
        for name, (old, newid) in list(changed.items())[:5]:
            print(f"   {name}: new file id {old} -> final id {newid}")

    # ---- 2. images: skip names already present, give new ids to the rest
    existing_names = {im["file_name"] for im in final["images"]}
    next_image_id = max((im["id"] for im in final["images"] if isinstance(im.get("id"), int)), default=0) + 1
    next_ann_id = max((a["id"] for a in final["annotations"] if isinstance(a.get("id"), int)), default=0) + 1

    id_map, skipped, added_images = {}, [], []
    for im in new["images"]:
        if im["file_name"] in existing_names:
            skipped.append(im["file_name"])
            continue                              # (only reached before the clash check below stops the run)
        old_id = im["id"]
        record = dict(im)                       # keeps width, height, make, model, year, variant ...
        record["id"] = next_image_id
        id_map[old_id] = next_image_id
        added_images.append(record)
        existing_names.add(im["file_name"])
        next_image_id += 1

    if skipped and not SKIP_DUPLICATES:
        print(f"ERROR: {len(skipped)} new image name(s) already exist in {FINAL_JSON.name}. Nothing was changed.")
        print("  Clashing names:", skipped[:40], "..." if len(skipped) > 40 else "")
        print("  Two different pictures cannot share one file name. Either these are the same pictures\n"
              "  (then set SKIP_DUPLICATES = True at the top), or rename the new pictures AND their\n"
              "  file_name entries (e.g. 85.jpg -> n85.jpg) and run this again.")
        sys.exit(1)

    # ---- 3. annotations: remap ids, validate
    valid_cats = set(category_remap)                  # category ids that exist in the NEW file
    size = {r["id"]: (r.get("width"), r.get("height")) for r in added_images}
    added_anns, problems = [], []
    for ann in new["annotations"]:
        if ann["image_id"] not in id_map:
            if ann["image_id"] not in {im["id"] for im in new["images"]}:
                problems.append(f"annotation {ann.get('id')} points to unknown image id {ann['image_id']}")
            continue                            # belongs to a skipped (duplicate) image
        new_img_id = id_map[ann["image_id"]]
        if ann["category_id"] not in valid_cats:
            problems.append(f"annotation {ann.get('id')} has unknown category {ann['category_id']}")
            continue
        x, y, w, h = ann["bbox"]
        iw, ih = size[new_img_id]
        if w <= 0 or h <= 0 or (iw and ih and (x < 0 or y < 0 or x + w > iw + 1 or y + h > ih + 1)):
            problems.append(f"annotation {ann.get('id')} on image id {new_img_id}: box {ann['bbox']} is outside "
                            f"the {iw}x{ih} image")
            continue
        record = dict(ann)
        record["id"] = next_ann_id
        record["image_id"] = new_img_id
        record["category_id"] = category_remap[ann["category_id"]]      # same class NAME, final file's id
        added_anns.append(record)
        next_ann_id += 1

    if problems:
        print("ERROR: problems found in the new file. Nothing was changed:")
        for p in problems[:30]:
            print("  -", p)
        if len(problems) > 30:
            print(f"  ... and {len(problems) - 30} more")
        sys.exit(1)

    # ---- 4. report, backup, write
    boxes_per_img = {}
    for a in added_anns:
        boxes_per_img[a["image_id"]] = boxes_per_img.get(a["image_id"], 0) + 1
    no_boxes = [r["file_name"] for r in added_images if r["id"] not in boxes_per_img]

    print(f"Final file before : {len(final['images'])} images, {len(final['annotations'])} boxes")
    print(f"New file          : {len(new['images'])} images, {len(new['annotations'])} boxes")
    print(f"Adding            : {len(added_images)} images, {len(added_anns)} boxes")
    if skipped:
        print(f"Skipped (already in final, same file name): {skipped}")
    if no_boxes:
        print(f"WARNING - added images with NO boxes: {no_boxes}")

    if not added_images:
        print("Nothing to add. No file was changed.")
        return

    if OUT_JSON.exists():
        backup = OUT_JSON.with_name(f"{OUT_JSON.name}.bak_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(OUT_JSON, backup)
        print(f"Backup saved      : {backup}")

    final["images"].extend(added_images)
    final["annotations"].extend(added_anns)
    tmp = OUT_JSON.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    tmp.replace(OUT_JSON)
    print(f"Final file after  : {len(final['images'])} images, {len(final['annotations'])} boxes -> {OUT_JSON.resolve()}")
    print("Done. Remember: the actual image files (the new images) must be in the same image folder "
          "as the older ones when you train.")


if __name__ == "__main__":
    main()