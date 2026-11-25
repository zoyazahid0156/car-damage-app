import cv2
import json
import math
from pathlib import Path
import numpy as np

IMAGE_DIR = Path("car_images")

JSON_FILE = Path("car_parts_17_coco_476_684.json")

FIRST_IMAGE_NUMBER = 476
LAST_IMAGE_NUMBER = 684

MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 800
PANEL_HEIGHT = 125

WINDOW_NAME = "17-Part Car Labeler"

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

POLYGON_PARTS = {"front_fender", "quarter_panel", "front_door", "rear_door"}

CATEGORY_NAME_TO_ID = {
    part: index + 1
    for index, part in enumerate(PARTS)
}

CATEGORY_ID_TO_NAME = {
    index + 1: part
    for index, part in enumerate(PARTS)
}

EXPECTED_CATEGORIES = [
    {
        "id": index + 1,
        "name": part,
        "supercategory": "car_part",
    }
    for index, part in enumerate(PARTS)
]

DEFAULT_VEHICLE = {"make": "Unknown", "model": "Unknown", "year": 0, "variant": ""}

VEHICLE_INFO = {}

for i in range(FIRST_IMAGE_NUMBER, LAST_IMAGE_NUMBER + 1):    
    VEHICLE_INFO[f"{i}.jpg"] = {
        "make": "Toyota",
        "model": "Corolla",
        "year": 2012,
        "variant": "1.6 Altis",
    }

for i in range(95, 105):                                       
    VEHICLE_INFO[f"{i}.jpg"] = {
        "make": "Honda",
        "model": "Civic",
        "year": 2019,
        "variant": "Turbo",
    }
for i in range(476, 685):                                       
    VEHICLE_INFO[f"{i}.jpg"] = {
        "make": "Honda",
        "model": "Civic",
        "year": 2019,
        "variant": "Turbo",
    }
def vehicle_info(filename):
    """Vehicle details for an image; never crashes on an image that is not listed above."""
    return VEHICLE_INFO.get(filename, DEFAULT_VEHICLE)

if not IMAGE_DIR.exists():
    raise SystemExit(
        f"ERROR: Image folder not found: {IMAGE_DIR.resolve()}"
    )

image_files = []

for number in range(FIRST_IMAGE_NUMBER, LAST_IMAGE_NUMBER + 1):
    image_path = IMAGE_DIR / f"{number}.jpg"

    if image_path.exists():
        image_files.append(image_path)
    else:
        print(f"WARNING: Image not found: {image_path}")

if not image_files:
    raise SystemExit(
        f"ERROR: No images from {FIRST_IMAGE_NUMBER}.jpg to {LAST_IMAGE_NUMBER}.jpg were found."
    )

print(f"Images found: {len(image_files)} of {LAST_IMAGE_NUMBER - FIRST_IMAGE_NUMBER + 1}")

if JSON_FILE.exists():
    with open(JSON_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Resuming existing work from: {JSON_FILE}")

else:
    data = {
        "images": [],
        "annotations": [],
        "categories": EXPECTED_CATEGORIES.copy(),
    }

    print(f"Starting a new annotation file: {JSON_FILE}")

data.setdefault("images", [])
data.setdefault("annotations", [])

# Validate existing data before changing anything.
allowed_filenames = {path.name for path in image_files}

for record in data["images"]:
    if record.get("file_name") not in allowed_filenames:
        raise SystemExit(
            "ERROR: The output JSON contains an image outside "
            f"the {FIRST_IMAGE_NUMBER}-{LAST_IMAGE_NUMBER} range: "
            f"{record.get('file_name')}\n"
            "No changes have been saved. Check the output JSON."
        )

for annotation in data["annotations"]:
    if annotation.get("image_id") not in {
        record.get("id") for record in data["images"]
    }:
        raise SystemExit(
            "ERROR: An annotation refers to an unknown image ID. "
            "No changes have been saved."
        )

data["categories"] = EXPECTED_CATEGORIES.copy()


def save_json():
    # Write to a temporary file first, then replace the output.
    temp_file = JSON_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    temp_file.replace(JSON_FILE)
    print(f"Saved: {JSON_FILE}")


# ============================================================
# IMAGE AND ANNOTATION HELPERS
# ============================================================

def get_image_record(filename):
    for record in data["images"]:
        if record["file_name"] == filename:
            return record
    return None


def ensure_image_record(filename, width, height):
    record = get_image_record(filename)
    info = vehicle_info(filename)

    if record is None:
        existing_ids = [
            item["id"]
            for item in data["images"]
            if isinstance(item.get("id"), int)
        ]

        new_id = max(existing_ids, default=0) + 1

        record = {
            "id": new_id,
            "file_name": filename,
            "width": width,
            "height": height,
            **info,
        }

        data["images"].append(record)

    else:
        # Refresh image dimensions and known vehicle metadata.
        record["width"] = width
        record["height"] = height
        record.update(info)

    return record


def get_annotations_for_image(image_id):
    return [
        annotation
        for annotation in data["annotations"]
        if annotation["image_id"] == image_id
    ]


def next_annotation_id():
    ids = [
        annotation["id"]
        for annotation in data["annotations"]
        if isinstance(annotation.get("id"), int)
    ]
    return max(ids, default=0) + 1

def calculate_scale(width, height):
    return min(
        MAX_DISPLAY_WIDTH / width,
        MAX_DISPLAY_HEIGHT / height,
        1.0,
    )


def original_to_display(x, y, scale):
    return int(round(x * scale)), int(round(y * scale))


def polygon_area(points):
    """Shoelace formula; points = [(x, y), ...]."""
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def polygon_from_display(points, scale, width, height):
    """Display-window points -> (x, y, w, h, segmentation, area) in ORIGINAL image pixels, or None if too small."""
    original = [
        (min(max(x / scale, 0.0), width - 1.0), min(max(y / scale, 0.0), height - 1.0))
        for x, y in points
    ]
    area = polygon_area(original)
    if area < 4:
        return None
    xs = [p[0] for p in original]
    ys = [p[1] for p in original]
    x0, y0 = int(math.floor(min(xs))), int(math.floor(min(ys)))
    x1, y1 = int(math.ceil(max(xs))), int(math.ceil(max(ys)))
    w = max(1, min(x1 - x0, width - x0))
    h = max(1, min(y1 - y0, height - y0))
    segmentation = [[round(c, 1) for p in original for c in p]]
    return x0, y0, w, h, segmentation, round(area, 1)


def annotation_polygon_display(annotation, scale):
    """The annotation's polygon in display pixels (numpy int array), or None for a box-only annotation."""
    seg = annotation.get("segmentation")
    if seg and isinstance(seg, list) and isinstance(seg[0], list) and len(seg[0]) >= 6:
        flat = seg[0]
        return np.array(
            [[x * scale, y * scale] for x, y in zip(flat[0::2], flat[1::2])],
            dtype=np.int32,
        )
    return None

def draw_polygon_tool(base_image, window_title, context_image=None):
    """Let the user click polygon corners on base_image (a clean image). Returns [(x, y), ...] in display
    pixels, or None. Press H to show/hide the already-drawn labels (context_image) behind your polygon.
    Finish with ENTER or SPACE (at least 3 points); cancel with ESC."""
    points = []
    show_context = [False]
    mouse = [0, 0]
    message = [""]
    closed_frames = [0]
    frame = [0]

    def on_mouse(event, x, y, flags, param):
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            message[0] = ""
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, on_mouse)

    result = None
    while True:
        frame[0] += 1
        view = (context_image if (show_context[0] and context_image is not None) else base_image).copy()

        if len(points) >= 3:
            overlay = view.copy()
            cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], (0, 200, 255))
            view = cv2.addWeighted(overlay, 0.25, view, 0.75, 0)

        for i in range(1, len(points)):
            cv2.line(view, points[i - 1], points[i], (0, 200, 255), 2)
        if points:
            # live line from the last point to the mouse, and a faint closing line
            cv2.line(view, points[-1], (mouse[0], mouse[1]), (0, 200, 255), 1)
            if len(points) >= 2:
                cv2.line(view, (mouse[0], mouse[1]), points[0], (120, 120, 120), 1)
        for p in points:
            cv2.circle(view, p, 4, (0, 0, 255), -1)

        text = (f"Points: {len(points)}  |  Left-click: add  |  Right-click/BACKSPACE: undo  |  "
                f"ENTER or SPACE: finish & save  |  ESC: cancel  |  H: show/hide other labels")
        cv2.rectangle(view, (0, 0), (view.shape[1], 24), (30, 30, 30), -1)
        cv2.putText(view, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if message[0]:
            cv2.rectangle(view, (0, 26), (view.shape[1], 50), (0, 0, 160), -1)
            cv2.putText(view, message[0], (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_title, view)
        key = cv2.waitKey(20) & 0xFF

        if key in (10, 13, 32):                               # ENTER / SPACE = finish
            if len(points) >= 3:
                result = list(points)
                print(f"Polygon finished with {len(points)} points.")
                break
            message[0] = f"Need at least 3 points (you have {len(points)}). Keep clicking, or press ESC to cancel."
            print(message[0])
        elif key == 27:                                       # ESC = cancel
            print("Polygon cancelled (ESC).")
            break
        elif key in (8, ord("u"), ord("U")) and points:       # BACKSPACE / U = undo last point
            points.pop()
        elif key in (ord("h"), ord("H")):                     # H = show/hide the other labels
            show_context[0] = not show_context[0]

        # window closed with the X button? Only believe it after many consecutive frames (the property can
        # briefly read 0 while a window is still opening).
        try:
            visible = cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0
        closed_frames[0] = closed_frames[0] + 1 if (visible < 1 and frame[0] > 30) else 0
        if closed_frames[0] >= 15:
            print("Polygon window was closed - nothing saved.")
            break

    try:
        cv2.destroyWindow(window_title)
    except cv2.error:
        pass

    for _ in range(12):
        cv2.waitKey(25)
    return result

def draw_annotations(display_image, annotations, scale, selected_index=None):
    result = display_image.copy()

    for index, annotation in enumerate(annotations):
        x, y, w, h = annotation["bbox"]

        x1, y1 = original_to_display(x, y, scale)
        x2, y2 = original_to_display(x + w, y + h, scale)

        part_name = CATEGORY_ID_TO_NAME.get(
            annotation["category_id"], "unknown"
        )

        if index == selected_index:
            color = (0, 255, 255) # Selected: yellow
            thickness = 3
        else:
            color = (0, 255, 0) # Other boxes: green
            thickness = 2

        polygon = annotation_polygon_display(annotation, scale)

        if polygon is not None:
            overlay = result.copy()
            cv2.fillPoly(overlay, [polygon], color)
            result = cv2.addWeighted(overlay, 0.22, result, 0.78, 0)
            cv2.polylines(result, [polygon], True, color, thickness)
            kind = " [poly]"
        else:
            cv2.rectangle(
                result, (x1, y1), (x2, y2), color, thickness
            )
            kind = ""

        label = f"{index}: {part_name}{kind}"

        label_y = max(18, y1 - 6)

        cv2.putText(
            result,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return result

def find_annotation_at(mouse_x, mouse_y, annotations, scale):
    # Search backwards so the topmost/latest annotation is selected first.
    for index in range(len(annotations) - 1, -1, -1):
        polygon = annotation_polygon_display(annotations[index], scale)

        if polygon is not None:
            if cv2.pointPolygonTest(polygon.reshape(-1, 1, 2), (float(mouse_x), float(mouse_y)), False) >= 0:
                return index
            continue

        x, y, w, h = annotations[index]["bbox"]

        x1, y1 = original_to_display(x, y, scale)
        x2, y2 = original_to_display(x + w, y + h, scale)

        if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
            return index

    return None

mouse_x = 0
mouse_y = 0
mouse_clicked = False


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked

    mouse_x = x
    mouse_y = y

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True

image_index = 0
selected_part = None
selected_annotation_index = None
status_message = ""

try:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
except cv2.error:
    raise SystemExit(
        "\nERROR: your OpenCV has no window support (it is the 'headless' build, or another package\n"
        "replaced it). Fix it with these commands, then run this script again:\n\n"
        "    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python\n"
        "    pip install opencv-python==4.10.0.84\n"
    )
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

try:
    while 0 <= image_index < len(image_files):

        image_path = image_files[image_index]
        filename = image_path.name

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"ERROR: Could not read {image_path}")
            image_index += 1
            continue

        height, width = image.shape[:2]

        image_record = ensure_image_record(filename, width, height)
        image_id = image_record["id"]

        annotations = get_annotations_for_image(image_id)

        scale = calculate_scale(width, height)
        display_width = max(1, int(round(width * scale)))
        display_height = max(1, int(round(height * scale)))

        display_image = cv2.resize(
            image,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA,
        )

        display_image = draw_annotations(
            display_image,
            annotations,
            scale,
            selected_annotation_index,
        )

        # ----------------------------------------------------
        # HEADER AND STATUS PANEL
        # ----------------------------------------------------

        canvas = cv2.copyMakeBorder(
            display_image,
            PANEL_HEIGHT,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )

        info = vehicle_info(filename)

        header_lines = [
            f"Image {image_index + 1}/{len(image_files)}: {filename}",
            f"Vehicle: {info['make']} {info['model']} {info['year']}",
            f"Variant: {info['variant']}",
        ]

        for line_number, line in enumerate(header_lines):
            cv2.putText(
                canvas,
                line,
                (10, 24 + line_number * 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        selected_text = (
            f"Selected part: {selected_part} "
            f"({'POLYGON' if selected_part in POLYGON_PARTS else 'BOX'})"
            if selected_part
            else "Selected part: NONE"
        )

        selected_box_text = (
            f"Selected box: #{selected_annotation_index}"
            if selected_annotation_index is not None
            else "Selected box: NONE"
        )

        cv2.putText(
            canvas, selected_text, (530, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        cv2.putText(
            canvas, selected_box_text, (530, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        if status_message:
            cv2.putText(
                canvas, status_message, (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (80, 255, 80), 1, cv2.LINE_AA,
            )

        controls = (
            "1-9: parts 1-9 | 0: parts 10-17 | "
            "ENTER: draw (polygon or box) | D: delete selected | "
            "X: delete last | N/P: next/previous | Q: quit"
        )

        cv2.putText(
            canvas, controls, (10, 103),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(30) & 0xFF

        # HANDLE MOUSE CLICK

        if mouse_clicked:
            # Ignore clicks on the header panel.
            if mouse_y >= PANEL_HEIGHT:
                clicked_y = mouse_y - PANEL_HEIGHT

                clicked_index = find_annotation_at(
                    mouse_x,
                    clicked_y,
                    annotations,
                    scale,
                )

                if clicked_index is not None:
                    selected_annotation_index = clicked_index

                    selected_part = CATEGORY_ID_TO_NAME.get(
                        annotations[clicked_index]["category_id"]
                    )

                    print(
                        f"\nSelected box #{clicked_index}: "
                        f"{selected_part}"
                    )
                    print("Press ENTER to replace, or D to delete.")

                else:
                    selected_annotation_index = None

            mouse_clicked = False

        # SELECT PARTS 1-9


        if ord("1") <= key <= ord("9"):
            number = key - ord("0")
            selected_part = PARTS[number - 1]
            selected_annotation_index = None
            print(f"\nSelected part: {selected_part}")

        # SELECT PARTS 10-17

        elif key == ord("0"):
            print("\nChoose part number:")
            for number in range(10, 18):
                print(f"{number}: {PARTS[number - 1]}")

            try:
                choice = int(input("Enter a number from 10 to 17: ").strip())

                if 10 <= choice <= 17:
                    selected_part = PARTS[choice - 1]
                    selected_annotation_index = None
                    print(f"Selected part: {selected_part}")
                else:
                    print("Invalid part number.")

            except ValueError:
                print("Please enter a whole number from 10 to 17.")

        # ENTER = DRAW NEW BOX OR REPLACE SELECTED BOX

        elif key in (10, 13): # Enter key

            if selected_part is None:
                print("\nSelect a part first.")
                continue

            use_polygon = selected_part in POLYGON_PARTS

            roi_image = cv2.resize(
                image,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA,
            )

            if use_polygon:
                print(f"\nOutline the {selected_part} with a POLYGON.")
                print("Left-click = add a corner | right-click or BACKSPACE = undo | ENTER or SPACE = finish and save | ESC = cancel")

                # clean image (no other boxes); press H in the window to show the other labels if you need them
                context = draw_annotations(roi_image, annotations, scale)
                points = draw_polygon_tool(roi_image, "Draw Polygon", context)

                if points is None:
                    status_message = "Polygon NOT saved (cancelled or window closed)."
                    continue

                outline = polygon_from_display(points, scale, width, height)

                if outline is None:
                    print("Polygon too small - nothing saved.")
                    status_message = "Polygon too small - NOT saved."
                    continue

                original_x, original_y, original_w, original_h, segmentation, area = outline

            else:
                print(f"\nDraw a box around the entire {selected_part}.")
                print("Click and drag in the box-selection window.")

                roi = cv2.selectROI(
                    "Draw Bounding Box",
                    roi_image,
                    showCrosshair=True,
                    fromCenter=False,
                )

                cv2.destroyWindow("Draw Bounding Box")

                rx, ry, rw, rh = [int(value) for value in roi]

                if rw <= 2 or rh <= 2:
                    print("No valid box selected.")
                    continue

                # Convert display coordinates to original image coordinates.
                original_x = max(0, min(int(round(rx / scale)), width - 1))
                original_y = max(0, min(int(round(ry / scale)), height - 1))

                original_w = min(
                    int(round(rw / scale)),
                    width - original_x,
                )
                original_h = min(
                    int(round(rh / scale)),
                    height - original_y,
                )

                if original_w <= 0 or original_h <= 0:
                    print("Invalid bounding box.")
                    continue

                segmentation = []                       # box-only part: no polygon
                area = original_w * original_h

            category_id = CATEGORY_NAME_TO_ID[selected_part]
            bbox = [original_x, original_y, original_w, original_h]

            # Replace the selected annotation only if it still exists.
            if selected_annotation_index is not None:
                if 0 <= selected_annotation_index < len(annotations):
                    old_annotation = annotations[selected_annotation_index]

                    old_annotation["category_id"] = category_id
                    old_annotation["bbox"] = bbox
                    old_annotation["segmentation"] = segmentation
                    old_annotation["area"] = area
                    old_annotation["iscrowd"] = 0

                    print("\nExisting annotation replaced.")

                else:
                    print("Selected annotation no longer exists; adding a new one.")
                    selected_annotation_index = None

            if selected_annotation_index is None:
                new_annotation = {
                    "id": next_annotation_id(),
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "area": area,
                    "iscrowd": 0,
                }

                data["annotations"].append(new_annotation)
                print("\nNew annotation added.")

            print(f"Part: {selected_part} ({'polygon' if use_polygon else 'box'})")
            print("Box:", bbox)

            save_json()
            status_message = f"SAVED: {selected_part} ({'polygon' if use_polygon else 'box'})"

            annotations = get_annotations_for_image(image_id)
            selected_annotation_index = None

        # D = DELETE SELECTED BOX

        elif key in (ord("d"), ord("D")):
            if selected_annotation_index is None:
                print("\nNo box selected. Click inside a green box first.")
            elif 0 <= selected_annotation_index < len(annotations):
                selected_annotation = annotations[selected_annotation_index]
                part_name = CATEGORY_ID_TO_NAME.get(
                    selected_annotation["category_id"], "unknown"
                )

                data["annotations"].remove(selected_annotation)
                save_json()

                print(f"\nDeleted selected box: {part_name}")

                annotations = get_annotations_for_image(image_id)
                selected_annotation_index = None
            else:
                print("The selected box is no longer available.")
                selected_annotation_index = None

        # ----------------------------------------------------
        # X = DELETE LAST BOX ON CURRENT IMAGE
        # ----------------------------------------------------

        elif key in (ord("x"), ord("X")):
            annotations = get_annotations_for_image(image_id)

            if annotations:
                last_annotation = annotations[-1]
                part_name = CATEGORY_ID_TO_NAME.get(
                    last_annotation["category_id"], "unknown"
                )

                data["annotations"].remove(last_annotation)
                save_json()

                print(f"\nDeleted last box: {part_name}")

                annotations = get_annotations_for_image(image_id)
                selected_annotation_index = None
            else:
                print("\nThere are no boxes on this image.")

        # N = NEXT IMAGE

        elif key in (ord("n"), ord("N")):
            save_json()
            status_message = ""
            selected_part = None
            selected_annotation_index = None
            mouse_clicked = False

            if image_index < len(image_files) - 1:
                image_index += 1
            else:
                print("\nYou are already on the last image.")

        # P = PREVIOUS IMAGE

        elif key in (ord("p"), ord("P")):
            save_json()
            status_message = ""
            selected_part = None
            selected_annotation_index = None
            mouse_clicked = False

            if image_index > 0:
                image_index -= 1
            else:
                print("\nYou are already on the first image.")

        # ----------------------------------------------------
        # Q = SAVE AND QUIT
        # ----------------------------------------------------

        elif key in (ord("q"), ord("Q")):
            save_json()
            print("\nLabeling saved. Exiting.")
            break

finally:
    save_json()
    cv2.destroyAllWindows()

print("\n======================================")
print("LABELING SESSION ENDED")
print("======================================")
print(f"Output JSON: {JSON_FILE.resolve()}")
print(f"Images in output: {len(data['images'])}")
print(f"Annotations in output: {len(data['annotations'])}")
print("Categories:", len(data["categories"]))
print("======================================")