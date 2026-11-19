from pathlib import Path
import torch

from anomalib.models import Patchcore
from anomalib.data import Folder
from anomalib.engine import Engine


PROJECT_ROOT = Path.cwd()

MODEL_PATH = (
    PROJECT_ROOT
    / "patchcore_models_17"
    / "front_bumper"
    / "model.ckpt"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "patchcore_data_17_split"
    / "front_bumper"
)


def main():

    print("=" * 70)
    print("PATCHCORE FRONT BUMPER INFERENCE TEST")
    print("=" * 70)

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)

    # ---------------------------------------------------------
    # 1. Check files
    # ---------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Data not found:\n{DATA_ROOT}"
        )

    print("\nModel:")
    print(MODEL_PATH)

    print("\nData:")
    print(DATA_ROOT)

    # ---------------------------------------------------------
    # 2. Create model
    # ---------------------------------------------------------

    print("\nCreating PatchCore...")

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        pre_trained=False,
    )

    # ---------------------------------------------------------
    # 3. Load checkpoint
    # ---------------------------------------------------------

    print("Loading checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint["state_dict"]

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    model.eval()

    print("Checkpoint loaded successfully.")

    # ---------------------------------------------------------
    # 4. Prepare data
    # ---------------------------------------------------------

    print("\nPreparing test data...")

    datamodule = Folder(
        name="front_bumper_test",
        root=DATA_ROOT,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=None,
        normal_split_ratio=0.0,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
    )

    datamodule.setup()

    test_loader = datamodule.test_dataloader()

    print(
        "Test batches:",
        len(test_loader)
    )

    print(
        "DataLoader workers:",
        test_loader.num_workers
    )

    # ---------------------------------------------------------
    # 5. IMPORTANT
    # ---------------------------------------------------------

    # We create scores HERE before inference.
    scores = []

    print("\nScores list created successfully.")

    # ---------------------------------------------------------
    # 6. Run inference
    # ---------------------------------------------------------

    print("\nStarting PatchCore inference...")
    print("Please wait...")

    engine = Engine(
        default_root_dir=(
            PROJECT_ROOT / "patchcore_results_17_split"
        )
    )

    predictions = engine.predict(
        model=model,
        dataloaders=test_loader,
        ckpt_path=None,
    )

    print("\nPatchCore inference finished.")

    # ---------------------------------------------------------
    # 7. Extract scores
    # ---------------------------------------------------------

    print("\nReading anomaly scores...")

    for prediction in predictions:

        print(
            "Prediction object received:",
            type(prediction)
        )

        if not hasattr(
            prediction,
            "pred_score"
        ):
            print(
                "WARNING: prediction has no pred_score"
            )
            continue

        score = prediction.pred_score

        print(
            "Raw pred_score:",
            score
        )

        if torch.is_tensor(score):

            score_values = (
                score
                .detach()
                .cpu()
                .flatten()
                .tolist()
            )

        else:

            score_values = [
                float(score)
            ]

        for value in score_values:

            scores.append(
                float(value)
            )

    # ---------------------------------------------------------
    # 8. Final results
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        "Number of scores:",
        len(scores)
    )

    if len(scores) == 0:

        print(
            "\nNO SCORES WERE RETURNED."
        )

        print(
            "The checkpoint loaded successfully, "
            "but Anomalib did not return pred_score."
        )

        return

    print("\nCLEAN FRONT_BUMPER SCORES:")

    for number, score in enumerate(
        scores,
        start=1
    ):

        print(
            f"Image {number:02d}: "
            f"{score:.6f}"
        )

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Minimum: {min(scores):.6f}"
    )

    print(
        f"Maximum: {max(scores):.6f}"
    )

    print(
        f"Average: "
        f"{sum(scores) / len(scores):.6f}"
    )

    print("\nTEST COMPLETE.")


if __name__ == "__main__":
    main()