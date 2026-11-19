from pathlib import Path
import gc

import torch
from torch.utils.data import DataLoader

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\zoya.zahid\Documents\car dataset project"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "patchcore_models_17"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "patchcore_data_17_split"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "patchcore_results_17_split"
)


# ============================================================
# EXACTLY 17 FIXED PARTS
# ============================================================

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


# ============================================================
# LOAD MODEL CONFIGURATION
# ============================================================

def create_patchcore():

    model = Patchcore(
        backbone="wide_resnet50_2",

        layers=[
            "layer2",
            "layer3",
        ],

        coreset_sampling_ratio=0.1,

        num_neighbors=9,

        pre_trained=False,
    )

    return model


# ============================================================
# TEST ONE PART
# ============================================================

def test_part(part):

    print()
    print("=" * 70)
    print(f"TESTING: {part}")
    print("=" * 70)


    # --------------------------------------------------------
    # MODEL CHECKPOINT
    # --------------------------------------------------------

    model_path = (
        MODEL_ROOT
        / part
        / "model.ckpt"
    )


    # --------------------------------------------------------
    # TEST IMAGE DIRECTORY
    # --------------------------------------------------------

    test_folder = (
        DATA_ROOT
        / part
        / "test"
        / "good"
    )


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not model_path.exists():

        raise FileNotFoundError(
            f"Production model not found:\n"
            f"{model_path}"
        )


    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if not test_folder.exists():

        raise FileNotFoundError(
            f"Clean test folder not found:\n"
            f"{test_folder}"
        )


    # --------------------------------------------------------
    # CREATE ANOMALIB PREDICT DATASET
    # --------------------------------------------------------

    print(
        "Creating Anomalib PredictDataset..."
    )


    dataset = PredictDataset(
        path=test_folder,

        image_size=(
            256,
            256,
        ),
    )


    print(
        f"Clean test images: {len(dataset)}"
    )


    if len(dataset) == 0:

        raise RuntimeError(
            f"No images found in:\n"
            f"{test_folder}"
        )


    # --------------------------------------------------------
    # USE ANOMALIB'S OWN COLLATE FUNCTION
    # --------------------------------------------------------
    #
    # THIS IS IMPORTANT.
    #
    # We do NOT manually return ImageItem.
    #
    # PredictDataset provides the correct collate_fn for
    # Anomalib's ImageBatch structure.
    #
    # --------------------------------------------------------

    loader = DataLoader(
        dataset=dataset,

        batch_size=1,

        shuffle=False,

        num_workers=0,

        pin_memory=False,

        collate_fn=dataset.collate_fn,
    )


    print(
        "DataLoader workers: 0"
    )

    print(
        "Anomalib collate function: enabled"
    )


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print(
        "Creating PatchCore model..."
    )


    model = create_patchcore()


    print(
        "PatchCore model created."
    )


    # --------------------------------------------------------
    # CREATE ENGINE
    # --------------------------------------------------------

    engine = Engine(
        default_root_dir=RESULT_ROOT,

        accelerator="cpu",

        devices=1,
    )


    # --------------------------------------------------------
    # RUN INFERENCE
    # --------------------------------------------------------

    print()
    print(
        "Running inference..."
    )

    print(
        "Production checkpoint will be loaded."
    )


    predictions = engine.predict(
        model=model,

        dataloaders=[loader],

        ckpt_path=str(
            model_path
        ),

        return_predictions=True,
    )


    print()
    print(
        "Inference finished."
    )


    # --------------------------------------------------------
    # CHECK PREDICTIONS
    # --------------------------------------------------------

    if predictions is None:

        raise RuntimeError(
            f"No predictions returned for {part}."
        )


    # --------------------------------------------------------
    # EXTRACT PATCHCORE SCORES
    # --------------------------------------------------------

    scores = []


    for prediction in predictions:

        if not hasattr(
            prediction,
            "pred_score",
        ):

            continue


        raw_score = (
            prediction.pred_score
        )


        if torch.is_tensor(
            raw_score
        ):

            values = (
                raw_score
                .detach()
                .cpu()
                .flatten()
                .tolist()
            )

        else:

            values = [
                float(raw_score)
            ]


        for value in values:

            scores.append(
                float(value)
            )


    # --------------------------------------------------------
    # CLEAN MEMORY
    # --------------------------------------------------------

    del predictions

    del loader

    del dataset

    del model

    del engine

    gc.collect()


    # --------------------------------------------------------
    # CHECK SCORES
    # --------------------------------------------------------

    if not scores:

        raise RuntimeError(
            f"No PatchCore scores returned for {part}."
        )


    # --------------------------------------------------------
    # CALCULATE RESULTS
    # --------------------------------------------------------

    minimum = min(
        scores
    )

    maximum = max(
        scores
    )

    average = (
        sum(scores)
        / len(scores)
    )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print(
        f"{part} CLEAN RESULTS"
    )

    print(
        "-" * 60
    )

    print(
        f"Images tested : {len(scores)}"
    )

    print(
        f"Minimum : {minimum:.6f}"
    )

    print(
        f"Maximum : {maximum:.6f}"
    )

    print(
        f"Average : {average:.6f}"
    )


    return {
        "part": part,

        "count": len(scores),

        "minimum": minimum,

        "maximum": maximum,

        "average": average,

        "scores": scores,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CPU SETTINGS
    # --------------------------------------------------------

    torch.set_num_threads(2)

    torch.set_grad_enabled(
        False
    )


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "PATCHCORE 17-MODEL CLEAN INFERENCE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Models:"
    )

    print(
        MODEL_ROOT
    )

    print()

    print(
        "Data:"
    )

    print(
        DATA_ROOT
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Models will be tested ONE AT A TIME."
    )

    print(
        "No training or model replacement will occur."
    )

    print()


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    failed = []


    # ========================================================
    # TEST ALL 17 MODELS
    # ========================================================

    for number, part in enumerate(
        PARTS,
        start=1,
    ):

        print()

        print(
            f"[{number}/17] Starting {part}"
        )


        try:

            result = test_part(
                part
            )


            results.append(
                result
            )


        except Exception as error:

            print()

            print(
                f"FAILED: {part}"
            )

            print()

            print(
                f"ERROR: {error}"
            )


            failed.append(
                (
                    part,
                    str(error),
                )
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print()

    print(
        "=" * 70
    )

    print(
        "FINAL 17-MODEL SUMMARY"
    )

    print(
        "=" * 70
    )

    print()


    print(
        f"{'Part':<20}"
        f"{'N':>5}"
        f"{'Min':>12}"
        f"{'Max':>12}"
        f"{'Average':>12}"
    )


    print(
        "-" * 70
    )


    for result in results:

        print(
            f"{result['part']:<20}"
            f"{result['count']:>5}"
            f"{result['minimum']:>12.6f}"
            f"{result['maximum']:>12.6f}"
            f"{result['average']:>12.6f}"
        )


    print()

    print(
        f"Successful: {len(results)} / 17"
    )

    print(
        f"Failed: {len(failed)} / 17"
    )


    # --------------------------------------------------------
    # FAILED PARTS
    # --------------------------------------------------------

    if failed:

        print()

        print(
            "FAILED PARTS"
        )

        print(
            "-" * 70
        )


        for part, error in failed:

            print(
                f"{part}: {error}"
            )


    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )


    if len(results) == 17:

        print(
            "ALL 17 MODELS COMPLETED CLEAN INFERENCE."
        )

    else:

        print(
            "SOME MODELS FAILED."
        )


    print(
        "=" * 70
    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()