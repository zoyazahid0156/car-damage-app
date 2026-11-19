"""
Rename ONE class in one or more COCO json files. The class id does not change, so every box keeps its class
and the number of classes stays the same (17).

  python rename_class.py
  python rename_class.py car_parts_17_coco_final.json car_parts_17_coco_85_120.json     (choose the files)

Edit OLD_NAME / NEW_NAME below. A backup of each file is made first. Files already using the new name are left alone.
"""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

OLD_NAME = "rear_fender"
NEW_NAME = "quarter_panel"
DEFAULT_FILES = ["car_parts_17_coco_final.json", "car_parts_17_coco_85_120.json"]


def main():
    files = [Path(p) for p in (sys.argv[1:] or DEFAULT_FILES)]
    for path in files:
        if not path.exists():
            print(f"- {path}: not found, skipped")
            continue
        data = json.load(open(path, encoding="utf-8"))
        cats = data.get("categories", [])
        names = [c["name"] for c in cats]

        if NEW_NAME in names and OLD_NAME not in names:
            print(f"- {path}: already uses '{NEW_NAME}', nothing to do")
            continue
        if OLD_NAME not in names:
            print(f"- {path}: no class called '{OLD_NAME}' (classes: {names}) - NOT changed")
            continue
        if NEW_NAME in names:
            print(f"- {path}: has BOTH '{OLD_NAME}' and '{NEW_NAME}' - NOT changed (they would need merging)")
            continue

        cat = next(c for c in cats if c["name"] == OLD_NAME)
        n_boxes = sum(1 for a in data.get("annotations", []) if a["category_id"] == cat["id"])

        backup = path.with_name(f"{path.name}.bak_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(path, backup)
        cat["name"] = NEW_NAME
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
        print(f"- {path}: '{OLD_NAME}' -> '{NEW_NAME}' (class id {cat['id']}, {n_boxes} boxes keep this class). "
              f"{len(cats)} classes. Backup: {backup.name}")


if __name__ == "__main__":
    main()