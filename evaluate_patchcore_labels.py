from pathlib import Path
import csv
from collections import Counter

PROJECT = Path(
    r"C:\Users\zoya.zahid\Documents\car dataset project"
)

INPUT_CSV = (
    PROJECT / "damage_calibration_17" / "review_labels.csv"
)

OUTPUT_CSV = (
    PROJECT / "damage_calibration_17" / "threshold_evaluation.csv"
)

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Review file not found: {INPUT_CSV}")

with open(INPUT_CSV, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

required = {"part", "patchcore_anomaly_score", "verified_label"}
if not rows or not required.issubset(rows[0]):
    raise ValueError(
        "CSV must contain part, patchcore_anomaly_score, "
        "and verified_label columns."
    )

# Only verified damaged and undamaged crops are used.
data = []
excluded = Counter()

for row in rows:
    label = (row.get("verified_label") or "").strip().lower()

    if label not in {"damaged", "undamaged"}:
        excluded[label or "blank"] += 1
        continue

    try:
        score = float(row["patchcore_anomaly_score"])
    except (ValueError, TypeError):
        excluded["invalid_score"] += 1
        continue

    if not 0 <= score <= 1:
        excluded["score_outside_0_to_1"] += 1
        continue

    data.append({
        "part": row["part"].strip(),
        "score": score,
        "actual": label == "damaged",
    })

if not data:
    raise RuntimeError("No usable verified labels found.")

print(f"Usable crops: {len(data)}")
print(f"Excluded rows: {dict(excluded)}")
print("Higher scores are treated as stronger damage signals.")

def evaluate(items, threshold):
    tp = fp = tn = fn = 0

    for item in items:
        predicted = item["score"] >= threshold
        actual = item["actual"]

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and not actual:
            tn += 1
        else:
            fn += 1

    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    balanced_accuracy = (recall + specificity) / 2

    return {
        "threshold": threshold,
        "n": len(items),
        "damaged": sum(x["actual"] for x in items),
        "undamaged": sum(not x["actual"] for x in items),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "balanced_accuracy": balanced_accuracy,
    }

def find_best_threshold(items):
    # Candidate thresholds include values between adjacent scores.
    scores = sorted(set(x["score"] for x in items))
    candidates = [0.0, 1.000001]

    for score in scores:
        candidates.append(score)

    for left, right in zip(scores, scores[1:]):
        candidates.append((left + right) / 2)

    results = [evaluate(items, t) for t in sorted(set(candidates))]

    # Prefer the threshold with the best balanced accuracy.
    # Break ties by choosing the one with higher specificity.
    return max(
        results,
        key=lambda r: (r["balanced_accuracy"], r["specificity"])
    )

reports = []

overall = find_best_threshold(data)
overall["scope"] = "ALL_PARTS"
reports.append(overall)

print("\nOVERALL CANDIDATE THRESHOLD")
for key in (
    "threshold", "n", "damaged", "undamaged",
    "recall", "specificity", "precision", "balanced_accuracy"
):
    print(f"{key}: {overall[key]:.4f}" if isinstance(
        overall[key], float
    ) else f"{key}: {overall[key]}")

# Part-specific thresholds are exploratory and require at least
# five verified examples of each class for that part.
parts = sorted(set(x["part"] for x in data))

print("\nPER-PART RESULTS")
for part in parts:
    subset = [x for x in data if x["part"] == part]
    damaged_n = sum(x["actual"] for x in subset)
    undamaged_n = sum(not x["actual"] for x in subset)

    if damaged_n < 5 or undamaged_n < 5:
        print(
            f"{part}: insufficient examples "
            f"(damaged={damaged_n}, undamaged={undamaged_n})"
        )
        continue

    result = find_best_threshold(subset)
    result["scope"] = part
    reports.append(result)

    print(
        f"{part}: threshold={result['threshold']:.4f}, "
        f"recall={result['recall']:.3f}, "
        f"specificity={result['specificity']:.3f}, "
        f"balanced_accuracy={result['balanced_accuracy']:.3f}, "
        f"n={result['n']}"
    )

columns = [
    "scope", "threshold", "n", "damaged", "undamaged",
    "tp", "fp", "tn", "fn", "recall", "specificity",
    "precision", "balanced_accuracy",
]

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for result in reports:
        writer.writerow({
            key: (
                round(result[key], 6)
                if isinstance(result[key], float)
                else result.get(key, "")
            )
            for key in columns
        })

print(f"\nSaved results to: {OUTPUT_CSV}")
print(
    "These are exploratory thresholds measured on the same labelled "
    "crops used to choose them; independent testing is still needed."
)