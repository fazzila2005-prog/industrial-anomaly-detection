import os

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

from torchvision.transforms import v2


if __name__ == "__main__":

    category = "bottle"

    print("=" * 70)
    print("PATCHCORE - AUGMENTED BOTTLE TRAINING")
    print("=" * 70)

    # ------------------------------------------------------------
    # TRAINING AUGMENTATIONS
    # ------------------------------------------------------------

    train_augmentations = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),

        v2.RandomRotation(
            degrees=10
        ),

        v2.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.90, 1.10)
        ),

        v2.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.20,
            hue=0.03
        ),
    ])

    # ------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------

    datamodule = MVTecAD(
        root="./data",
        category=category,

        train_batch_size=2,
        eval_batch_size=2,
        num_workers=0,

        train_augmentations=train_augmentations,

        # VERY IMPORTANT:
        # No augmentation on validation/test.
        val_augmentations=None,
        test_augmentations=None,
    )

    # ------------------------------------------------------------
    # PATCHCORE
    # ------------------------------------------------------------

    model = Patchcore(
        backbone="wide_resnet50_2",

        layers=[
            "layer2",
            "layer3"
        ],

        pre_trained=True,

        coreset_sampling_ratio=0.01,

        num_neighbors=9,
    )

    # ------------------------------------------------------------
    # ENGINE
    # ------------------------------------------------------------

    engine = Engine(
        accelerator="cpu",
        devices=1,

        default_root_dir=(
            "results/patchcore_bottle_augmented"
        )
    )

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    print("\nStarting augmented PatchCore training...")

    engine.fit(
        model=model,
        datamodule=datamodule
    )

    # ------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------

    print("\nRunning MVTec bottle test...")

    results = engine.test(
        model=model,
        datamodule=datamodule
    )

    print("\n" + "=" * 70)
    print("AUGMENTED PATCHCORE COMPLETE")
    print("=" * 70)

    print("\nResults:")
    print(results)