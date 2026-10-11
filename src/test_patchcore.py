import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

if __name__ == '__main__':

    # --------------------------------------------------
    # 1. LOAD DATASET (Clean Isolated Path)
    # --------------------------------------------------
    print("Initializing MVTec AD dataset...")
    datamodule = MVTecAD(
        root="./data",        # Points directly to your clean new data folder!
        category="bottle",
        train_batch_size=2,
        eval_batch_size=2,
        num_workers=0,        # Ensures stable execution on Windows CPU architectures
    )

    # --------------------------------------------------
    # 2. CREATE OPTIMIZED PATCHCORE
    # --------------------------------------------------
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.01,  # Keeps your 1% coreset optimization active for CPU efficiency
        num_neighbors=9,
    )

    # --------------------------------------------------
    # 3. CREATE STABLE TRAINING ENGINE
    # --------------------------------------------------
    # Note: We do NOT need limit_val_batches=0 anymore! 
    # Separating your folders has completely fixed the data loader bug.
    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir="results/patchcore_bottle"
    )

    # --------------------------------------------------
    # 4. RUN PIPELINE (TRAIN + TEST)
    # --------------------------------------------------
    print("\nStarting PatchCore pipeline on CPU...")
    print("Running training phase (Expect 10-15 minutes on CPU)...")
    engine.fit(model=model, datamodule=datamodule)

    print("\n--- Running Final Evaluation ---")
    results = engine.test(model=model, datamodule=datamodule)
    
    print("\nFinal Performance Results:")
    print(results)
