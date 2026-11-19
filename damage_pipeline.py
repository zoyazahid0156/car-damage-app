"""
Damaged-part detection: WHICH parts are damaged, HOW severe, and where (drawn on the images).
No prices, no web access.

TWO WAYS TO USE IT
  A) CLAIM mode   - everything you give it is ONE car (surveyor video, or photos from different angles).
                    Frames are combined per part (worst severity wins), and missing parts are checked.
        python damage_pipeline.py --input claim_video.mp4 --make Honda --model Civic --year 2017
        python damage_pipeline.py --input claim_photos

  B) BATCH mode   - a folder of photos of DIFFERENT cars (testing). Every image is judged on its own:
                    no combining, no missing-part check.
        python damage_pipeline.py --input damaged_images --mode batch

  --mode auto (default) picks CLAIM for a video, or when --make/--model is given; BATCH otherwise.
  make / model / year / variant are optional: they are only copied into assessment.json.

OUTPUT
  assessment.json      damaged parts + severity (+ suspected missing parts in claim mode)
  assessed_images/     each image with the damaged parts drawn on it:
                         box + mask colour = severity  (yellow minor, orange moderate, red severe)
                         label = part | damage type | severity
                         magenta = damage found but the part detector did not find a part there

Needs
  PART_MODEL   = your part detector (best.pt you already trained)
  DAMAGE_MODEL = damage segmentation model (CarDD best.pt)
"""
import argparse
import json
import os
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PART_MODEL = ROOT / "runs/detect/detector_results_17/car_parts_635/weights/best.pt"
DAMAGE_MODEL = ROOT / "damage_results/damage_seg/weights/best.pt"
# OPTIONAL second damage model (e.g. the one trained on 26k images, one class "damage").
# Used automatically if damage_26/best_26k.pt exists next to this file. Other path: set DAMAGE_MODEL_2=C:\path\best.pt
_default_m2 = ROOT / "damage_26" / "best_26k.pt"            # your 26k-image model, next to this file
_m2 = os.environ.get("DAMAGE_MODEL_2", "")
DAMAGE_MODEL_2 = Path(_m2) if _m2 else (_default_m2 if _default_m2.exists() else None)
EXTRA_CONF = 0.25          # confidence for the second damage model
EXTRA_MIN_ADD = 0.3        # a region from model 2 is added only if < this share of it overlaps CarDD regions

# Detection sensitivity. Lower conf / bigger imgsz / TTA find MORE damage (and a few more false alarms).
# Override without editing code:  set DAMAGE_CONF=0.15  /  DAMAGE_IMGSZ=960  /  DAMAGE_TTA=1  (or use the CLI flags)
import os
PART_CONF = float(os.environ.get("PART_CONF", 0.15))   # part detector is only ~47% recall: be generous
DAMAGE_CONF = float(os.environ.get("DAMAGE_CONF", 0.15))
DAMAGE_IMGSZ = int(os.environ.get("DAMAGE_IMGSZ", 960))
PART_IMGSZ = int(os.environ.get("PART_IMGSZ", 960))
DAMAGE_TTA = os.environ.get("DAMAGE_TTA", "1") == "1"      # test-time augmentation (flips/scales), slower
MIN_OVERLAP = 0.5          # share of a damage region inside a part box = confident match
WEAK_OVERLAP = 0.2         # below MIN_OVERLAP but >= this: still assigned to the best part (flagged 'weak')
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
SEV_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
MAX_VIDEO_IMAGES = 60      # a video saves at most this many annotated frames (only frames with damage)
VIDEO_SAVE_EVERY = 4       # ...plus every Nth damage frame; a frame showing a NEW damaged part is always saved

# BGR colours
SEV_COLOUR = {"minor": (0, 215, 255), "moderate": (0, 140, 255), "severe": (0, 0, 255)}
UNASSIGNED_COLOUR = (255, 0, 200)
OK_COLOUR = (0, 200, 0)

# Missing-part rule (CLAIM mode only): if the anchor parts are seen in at least N frames but the
# target never is, flag it as "suspected missing". Detector recall is only ~50%, and small parts like
# mirrors are often not visible from the anchor frames, so the numbers are deliberately strict.
MISSING_RULES = {            # target: (anchor parts, minimum frames showing the anchors)
    "side_mirror": (["front_door", "rear_door"], 6),
    "headlight": (["front_bumper", "hood"], 4),
    "taillight": (["rear_bumper", "trunk_lid"], 4),
    "license_plate": (["front_bumper", "rear_bumper"], 6),
}


# ----------------------------------------------------------------- frames
def blur_score(img):
    import cv2
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def load_frames(path, fps_sample=3, blur_min=40.0):
    import cv2
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                img = cv2.imread(str(p))
                if img is not None:
                    yield p.name, img
    elif path.suffix.lower() in IMG_EXT:
        img = cv2.imread(str(path))
        if img is not None:
            yield path.name, img
    else:  # video
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(fps / fps_sample)))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0 and blur_score(frame) >= blur_min:
                yield f"frame_{i:06d}", frame
            i += 1
        cap.release()


# ----------------------------------------------------------------- damage -> part
def severity_from(damage_name, ratio):
    n = damage_name.lower()
    if any(k in n for k in ("shatter", "broken", "lamp", "crack", "flat")):
        return "severe" if ratio > 0.10 else "moderate"
    if "dent" in n:
        return "minor" if ratio < 0.05 else "moderate" if ratio < 0.20 else "severe"
    if "scratch" in n:
        return "minor" if ratio < 0.10 else "moderate" if ratio < 0.30 else "severe"
    return "minor" if ratio < 0.05 else "moderate" if ratio < 0.20 else "severe"


def analyse_frame(img, part_model, dmg_model, damage_conf=None, extra_model=None):
    """Returns (results, unassigned).
    results    = one dict per detected part (with its box and damage mask, used for drawing)
    unassigned = damage regions that fall inside no detected part box"""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    pr = part_model.predict(img, conf=PART_CONF, imgsz=PART_IMGSZ, augment=DAMAGE_TTA, device="cpu", verbose=False)[0]
    dr = dmg_model.predict(img, conf=DAMAGE_CONF if damage_conf is None else damage_conf, imgsz=DAMAGE_IMGSZ,
                           augment=DAMAGE_TTA, retina_masks=True, agnostic_nms=True,
                           device="cpu", verbose=False)[0]

    parts = []
    for b in pr.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        parts.append({"name": pr.names[int(b.cls[0])], "conf": float(b.conf[0]),
                      "box": (max(0, x1), max(0, y1), min(w, x2), min(h, y2)),
                      "dmg_mask": np.zeros((h, w), np.uint8), "types": defaultdict(float)})

    # ---- damage regions: (mask, damage name, confidence) from CarDD, then extra ones from model 2
    regions = []
    if dr.masks is not None:
        for poly, c, cf in zip(dr.masks.xy, dr.boxes.cls, dr.boxes.conf):
            if len(poly) < 3:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            if m.any():
                regions.append((m, dr.names[int(c)], float(cf)))
    if extra_model is not None:
        er = extra_model.predict(img, conf=EXTRA_CONF, imgsz=DAMAGE_IMGSZ, augment=DAMAGE_TTA,
                                 retina_masks=True, device="cpu", verbose=False)[0]
        covered = np.zeros((h, w), np.uint8)
        for m, _n, _c in regions:
            covered |= m
        shapes = []
        if er.masks is not None:                                  # segmentation model
            for poly, cf in zip(er.masks.xy, er.boxes.conf):
                if len(poly) >= 3:
                    m = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
                    shapes.append((m, float(cf)))
        else:                                                     # box-only model: use an inscribed ellipse
            for b, cf in zip(er.boxes.xyxy.tolist(), er.boxes.conf):
                x1, y1, x2, y2 = [int(v) for v in b]
                m = np.zeros((h, w), np.uint8)
                cv2.ellipse(m, ((x1 + x2) // 2, (y1 + y2) // 2), (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)),
                            0, 0, 360, 1, -1)
                shapes.append((m, float(cf)))
        for m, cf in shapes:
            a = int(m.sum())
            if a and int((m & covered).sum()) / a < EXTRA_MIN_ADD:     # damage CarDD did not find
                regions.append((m, "damage", cf))
                covered |= m

    unassigned = []
    for m, dname, cf in regions:
            area = int(m.sum())
            best, best_score = None, 0.0
            for p in parts:
                x1, y1, x2, y2 = p["box"]
                inside = int(m[y1:y2, x1:x2].sum()) / area
                if inside < WEAK_OVERLAP:
                    continue
                box_area = max(1, (x2 - x1) * (y2 - y1))
                # confident matches (>= MIN_OVERLAP) always beat weak ones; then prefer the tighter box
                score = (10.0 if inside >= MIN_OVERLAP else 0.0) + inside + 1.0 / box_area ** 0.25
                if best is None or score > best_score:
                    best, best_score = p, score
            if best is not None:
                x1, y1, x2, y2 = best["box"]
                best["dmg_mask"][y1:y2, x1:x2] |= m[y1:y2, x1:x2]
                best["types"][dname] += float(cf)
            else:
                ys, xs = np.nonzero(m)
                unassigned.append({"damage_type": dname, "conf": float(cf), "mask": m,
                                   "area_pct_of_image": round(100.0 * area / (h * w), 2),
                                   "anchor": (int(xs.min()), int(ys.min()))})

    results = []
    for p in parts:
        x1, y1, x2, y2 = p["box"]
        box_area = max(1, (x2 - x1) * (y2 - y1))
        ratio = float(p["dmg_mask"][y1:y2, x1:x2].sum()) / box_area
        if p["types"]:
            specific = {k: v for k, v in p["types"].items() if k != "damage"} or p["types"]
            main = max(specific, key=specific.get)
            sev = severity_from(main, ratio)
        else:
            main, sev = None, "none"
        results.append({"part": p["name"], "part_conf": p["conf"], "damaged": bool(p["types"]),
                        "damage_type": main, "damage_ratio": ratio, "severity": sev,
                        "box": p["box"], "mask": p["dmg_mask"]})
    return results, unassigned


# ----------------------------------------------------------------- drawing
def _label(img, text, x, y, colour, scale, thick):
    import cv2
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    h, w = img.shape[:2]
    x = max(0, min(x, w - tw - 6))
    y = max(th + base + 4, min(y, h - 2))
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y), colour, -1)
    dark = sum(colour) > 380
    cv2.putText(img, text, (x + 3, y - base - 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0) if dark else (255, 255, 255), thick, cv2.LINE_AA)


def draw(img, results, unassigned, show_all_parts=False):
    """Damaged parts: coloured mask + box + 'part | damage | severity'. Returns a new image."""
    import cv2
    h, w = img.shape[:2]
    scale = max(0.5, w / 1400)
    thick = max(1, int(round(scale * 2)))
    layer = img.copy()
    for r in results:
        if r["damaged"]:
            layer[r["mask"] > 0] = SEV_COLOUR[r["severity"]]
    for u in unassigned:
        layer[u["mask"] > 0] = UNASSIGNED_COLOUR
    out = cv2.addWeighted(layer, 0.45, img, 0.55, 0)

    if show_all_parts:
        for r in results:
            if not r["damaged"]:
                x1, y1, x2, y2 = r["box"]
                cv2.rectangle(out, (x1, y1), (x2, y2), OK_COLOUR, 1)
                _label(out, r["part"], x1, y1, OK_COLOUR, scale * 0.7, max(1, thick - 1))

    # draw less severe first so the severest label ends up on top
    for r in sorted((r for r in results if r["damaged"]), key=lambda r: SEV_RANK[r["severity"]]):
        col = SEV_COLOUR[r["severity"]]
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, max(2, thick + 1))
        _label(out, f"{r['part']} | {r['damage_type']} | {r['severity']}", x1, y1, col, scale * 0.75, thick)

    for u in unassigned:
        x, y = u["anchor"]
        _label(out, f"{u['damage_type']} (part not identified)", x, y, UNASSIGNED_COLOUR, scale * 0.7, thick)

    # banner + legend at the bottom-left, so they never cover labels near the top of the image
    n = sum(1 for r in results if r["damaged"])
    banner = f"{n} damaged part(s)" if n else "no damaged part found"
    if unassigned:
        banner += f" + {len(unassigned)} unmatched damage region(s)"
    _label(out, banner, 0, h - 2, (40, 40, 40), scale * 0.8, thick)
    x = 0
    for name in ("minor", "moderate", "severe"):
        (tw, _), _b = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, 1)
        _label(out, name, x, h - 2 - int(36 * scale), SEV_COLOUR[name], scale * 0.6, 1)
        x += tw + 16
    return out


# ----------------------------------------------------------------- combining frames (pure logic)
def aggregate(frames):
    """frames = [(frame_name, [per-part results from analyse_frame])]
    -> (evidence per part, number of frames each part was seen in)"""
    agg, seen = {}, defaultdict(int)
    for name, results in frames:
        by_part = defaultdict(list)
        for r in results:
            by_part[r["part"]].append(r)
        for part, items in by_part.items():
            seen[part] += 1
            e = agg.setdefault(part, {"damage_frames": 0, "severity": "none", "damage_type": None,
                                      "damage_ratio": 0.0, "best_part_conf": 0.0,
                                      "max_damaged_instances": 0, "evidence": []})
            e["best_part_conf"] = max(e["best_part_conf"], max(r["part_conf"] for r in items))
            damaged = [r for r in items if r["damaged"]]
            if damaged:
                e["damage_frames"] += 1
                e["evidence"].append(name)
                e["max_damaged_instances"] = max(e["max_damaged_instances"], len(damaged))
                for r in damaged:
                    if SEV_RANK[r["severity"]] >= SEV_RANK[e["severity"]]:
                        e["severity"], e["damage_type"] = r["severity"], r["damage_type"]
                    e["damage_ratio"] = max(e["damage_ratio"], r["damage_ratio"])
    return agg, seen


def frames_needed(seen_frames, min_frames):
    """Frames that must show damage before a part is called CONFIRMED.
    min_frames > 0 forces a number. 0 = automatic: one or two frames seen -> a single hit is enough;
    otherwise 2 frames, or 10% of the frames where the part was seen (whichever is larger).
    Parts below this are NOT dropped - they are listed as 'possible' (see build_entries)."""
    if min_frames > 0:
        return min_frames
    return 1 if seen_frames < 3 else max(2, math.ceil(0.10 * seen_frames))


def build_entries(agg, seen, min_frames, include_possible=True):
    """One entry per damaged part. A part with damage in even ONE frame is listed, so the summary always
    matches the annotated images:  confidence = 'confirmed' (damage in enough frames) or
    'possible' (seen in too few frames - the surveyor should check that spot)."""
    damaged = []
    for part, e in agg.items():
        if e["damage_frames"] < 1:
            continue
        confirmed = e["damage_frames"] >= frames_needed(seen[part], min_frames)
        if not confirmed and not include_possible:
            continue
        damaged.append({"part": part, "count": max(1, e["max_damaged_instances"]),
                        "damage_type": e["damage_type"], "severity": e["severity"],
                        "confidence": "confirmed" if confirmed else "possible",
                        "damage_area_pct": round(e["damage_ratio"] * 100, 1),
                        "frames_with_damage": e["damage_frames"], "frames_seen": seen[part],
                        "evidence_frames": e["evidence"][:5]})
    damaged.sort(key=lambda d: (-SEV_RANK[d["severity"]], d["confidence"] != "confirmed"))
    return damaged


def find_missing(seen):
    missing = []
    for target, (anchors, min_anchor_frames) in MISSING_RULES.items():
        anchor_frames = sum(seen[x] for x in anchors)
        if seen[target] == 0 and anchor_frames >= min_anchor_frames:
            missing.append({"part": target, "status": "suspected_missing - surveyor must confirm",
                            "anchor_frames": anchor_frames})
    return missing


def severity_counts(parts):
    c = {"minor": 0, "moderate": 0, "severe": 0}
    for d in parts:
        c[d["severity"]] += d.get("count", 1)
    return c


# ----------------------------------------------------------------- modes
def resolve_mode(mode, input_path, make, model):
    is_video = Path(input_path).suffix.lower() in VIDEO_EXT
    if mode == "batch" and is_video:
        raise SystemExit("Batch mode is for folders of photos of different cars. A video is one car: use --mode claim.")
    if mode != "auto":
        return mode
    if is_video:
        return "claim"
    return "claim" if (make or model) else "batch"


def print_parts(parts, indent="  "):
    if not parts:
        print(f"{indent}no damaged parts found")
    for d in parts:
        n = f" x{d['count']}" if d.get("count", 1) > 1 else ""
        print(f"{indent}{d['part']}{n}: {d['severity']} {d['damage_type']} ({d['damage_area_pct']}% of the part)")


# ----------------------------------------------------------------- reusable entry points
def _load_extra(YOLO):
    if DAMAGE_MODEL_2 is None:
        return None
    if not DAMAGE_MODEL_2.exists():
        raise FileNotFoundError(f"Second damage model not found: {DAMAGE_MODEL_2}")
    return YOLO(str(DAMAGE_MODEL_2))


def load_models():
    """Load the part detector and the damage model once (the API does this at start-up)."""
    for p in (PART_MODEL, DAMAGE_MODEL):
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
    from ultralytics import YOLO
    return YOLO(str(PART_MODEL)), YOLO(str(DAMAGE_MODEL)), _load_extra(YOLO)


def frame_damage_list(results):
    """Damaged parts of ONE image: part, damage type, severity, how many of that part."""
    by = {}
    for r in results:
        if not r["damaged"]:
            continue
        e = by.setdefault(r["part"], {"part": r["part"], "damage_type": r["damage_type"],
                                      "severity": r["severity"], "count": 0,
                                      "damage_area_pct": round(r["damage_ratio"] * 100, 1)})
        e["count"] += 1
        if SEV_RANK[r["severity"]] > SEV_RANK[e["severity"]]:
            e.update(damage_type=r["damage_type"], severity=r["severity"],
                     damage_area_pct=round(r["damage_ratio"] * 100, 1))
    return sorted(by.values(), key=lambda d: -SEV_RANK[d["severity"]])


def assess_path(input_path, models, mode="auto", vehicle=None, min_frames=0,
                images_out=None, show_all_parts=False, damage_conf=None):
    """Run the whole pipeline on a video, a photo, or a folder of photos.
    Returns (result_dict, number_of_annotated_images_saved). Raises FileNotFoundError / ValueError."""
    import cv2
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp.resolve()}")
    vehicle = vehicle or {"make": "", "model": "", "year": "", "variant": ""}
    mode = resolve_mode(mode, str(inp), vehicle.get("make", ""), vehicle.get("model", ""))
    part_model, dmg_model = models[0], models[1]
    extra_model = models[2] if len(models) > 2 else None

    out_dir = Path(images_out) if images_out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    is_video = inp.is_file() and inp.suffix.lower() in VIDEO_EXT
    saved = 0

    frames, unmatched_by_frame, per_image = [], {}, []
    shown_parts, damage_frames_seen = set(), 0
    for name, img in load_frames(inp):
        results, unassigned = analyse_frame(img, part_model, dmg_model, damage_conf, extra_model)
        has_damage = any(r["damaged"] for r in results) or bool(unassigned)
        annotated_name = None
        save_it = not is_video
        if is_video and has_damage:
            damage_frames_seen += 1
            new_part = any(r["damaged"] and r["part"] not in shown_parts for r in results)
            save_it = saved < MAX_VIDEO_IMAGES and (new_part or damage_frames_seen % VIDEO_SAVE_EVERY == 1)
        if out_dir and save_it:
            shown_parts.update(r["part"] for r in results if r["damaged"])
            annotated_name = f"{Path(name).stem}_assessed.jpg"
            cv2.imwrite(str(out_dir / annotated_name), draw(img, results, unassigned, show_all_parts))
            saved += 1
        # keep only light-weight data (no masks) for the combining step
        frames.append((name, [{k: v for k, v in r.items() if k not in ("mask", "box")} for r in results]))
        if unassigned:
            unmatched_by_frame[name] = [{"damage_type": u["damage_type"], "conf": round(u["conf"], 2),
                                         "area_pct_of_image": u["area_pct_of_image"]} for u in unassigned]
        if not is_video or has_damage:          # a video lists only the frames that show damage
            dl = frame_damage_list(results)
            per_image.append({"image": name, "annotated_image": annotated_name,
                              "damaged_part_count": len(dl),
                              "damaged_instances": sum(d["count"] for d in dl),
                              "damaged_parts": dl,
                              "unmatched_damage_regions": len(unassigned)})
    if not frames:
        raise ValueError("No readable photos or video frames found in the input.")

    if mode == "claim":
        agg, seen = aggregate(frames)
        damaged = build_entries(agg, seen, min_frames)
        out = {"mode": "claim", "vehicle": vehicle, "frames_analysed": len(frames),
               "damaged_parts": damaged, "severity_counts": severity_counts(damaged),
               "confirmed_parts": sum(1 for d in damaged if d["confidence"] == "confirmed"),
               "possible_parts": sum(1 for d in damaged if d["confidence"] == "possible"),
               "suspected_missing_parts": find_missing(seen),
               "missing_part_check": "ran (valid because all frames are one car)",
               "damage_regions_not_matched_to_a_part": unmatched_by_frame,
               "per_image": per_image,
               "needs_surveyor_review": True}
    else:
        images = []
        for name, results in frames:
            agg, seen = aggregate([(name, results)])
            damaged = build_entries(agg, seen, 1)
            images.append({"image": name, "parts_detected": len(results), "damaged_parts": damaged,
                           "severity_counts": severity_counts(damaged),
                           "damage_regions_not_matched_to_a_part": unmatched_by_frame.get(name, [])})
        out = {"mode": "batch", "vehicle": vehicle,
               "vehicle_info_note": "images may be different cars; vehicle fields not used",
               "images_analysed": len(images),
               "images_with_damage": sum(1 for i in images if i["damaged_parts"]),
               "missing_part_check": "skipped - images may be of different cars",
               "images": images, "per_image": per_image, "needs_surveyor_review": True}
    return out, saved


# ----------------------------------------------------------------- main (command line)
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="video, photo, or folder of photos")
    ap.add_argument("--mode", choices=["auto", "claim", "batch"], default="auto")
    ap.add_argument("--make", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--year", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--min-frames", type=int, default=0,
                    help="claim mode: frames that must show damage (0 = automatic, recommended)")
    ap.add_argument("--out", default="assessment.json")
    ap.add_argument("--images-out", default="assessed_images", help="folder for the annotated images")
    ap.add_argument("--damage-conf", type=float, help="damage confidence threshold (lower = finds more, default 0.15)")
    ap.add_argument("--damage-imgsz", type=int, help="damage model image size (default 960)")
    ap.add_argument("--damage-model-2", help="path to a second damage model (e.g. the 26k-image one)")
    ap.add_argument("--no-tta", action="store_true", help="turn off test-time augmentation (faster)")
    ap.add_argument("--no-images", action="store_true", help="do not save annotated images")
    ap.add_argument("--show-all-parts", action="store_true", help="also draw undamaged parts (thin green boxes)")
    a = ap.parse_args()
    global DAMAGE_CONF, DAMAGE_IMGSZ, DAMAGE_TTA, DAMAGE_MODEL_2
    if a.damage_model_2:
        DAMAGE_MODEL_2 = Path(a.damage_model_2)
    if a.damage_conf is not None:
        DAMAGE_CONF = a.damage_conf
    if a.damage_imgsz is not None:
        DAMAGE_IMGSZ = a.damage_imgsz
    if a.no_tta:
        DAMAGE_TTA = False

    vehicle = {"make": a.make, "model": a.model, "year": a.year, "variant": a.variant}
    try:
        mode = resolve_mode(a.mode, a.input, a.make, a.model)
        print(f"Mode: {mode.upper()}")
        models = load_models()
        out, saved = assess_path(a.input, models, mode, vehicle, a.min_frames,
                                 None if a.no_images else a.images_out, a.show_all_parts)
    except (FileNotFoundError, ValueError) as ex:
        raise SystemExit(str(ex))

    if out["mode"] == "claim":
        print(f"\nCar ({out['frames_analysed']} frame(s)):")
        print_parts(out["damaged_parts"])
        for m in out["suspected_missing_parts"]:
            print(f"  SUSPECTED MISSING: {m['part']} (confirm on site)")
        c = out["severity_counts"]
        print(f"\nSeverity: {c['severe']} severe, {c['moderate']} moderate, {c['minor']} minor")
    else:
        print()
        for i in out["images"]:
            n = len(i["damage_regions_not_matched_to_a_part"])
            extra = f"  (+{n} damage region(s) not matched to a part)" if n else ""
            print(f"{i['image']}: {len(i['damaged_parts'])} damaged part(s){extra}")
            print_parts(i["damaged_parts"])

    Path(a.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {a.out}")
    if not a.no_images:
        print(f"Annotated images ({saved}) in: {Path(a.images_out).resolve()}")
    print("Severity comes from damage type + how much of the part is covered. A surveyor must review the result.")


if __name__ == "__main__":
    main()