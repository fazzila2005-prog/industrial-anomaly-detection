import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

if __name__ == '__main__':
    # List of all remaining MVTec categories (excluding bottle)
    categories = [
        "cable", "capsule", "carpet", "grid", "hazelnut", 
        "leather", "metal_nut", "pill", "screw", "tile", 
        "toothbrush", "transistor", "wood", "zipper"
    ]

    for category in categories:
        print(f"\n==================================================")
        # Modified to keep log output concise during automated runs
        print(f"🚀 STARTING PIPELINE FOR CATEGORY: {category.upper()}")
        print(f"==================================================")
        
        try:
            # 1. Load Dataset
            datamodule = MVTecAD(
                root="./data",
                category=category,
                train_batch_size=2,
                eval_batch_size=2,
                num_workers=0,
            )

            # 2. Create Optimized PatchCore
            model = Patchcore(
                backbone="wide_resnet50_2",
                layers=["layer2", "layer3"],
                pre_trained=True,
                coreset_sampling_ratio=0.01,  # 1% coreset for CPU stability
                num_neighbors=9,
            )

            # 3. Create Engine (Saves results in unique folder per object)
            engine = Engine(
                accelerator="cpu",
                devices=1,
                default_root_dir=f"results/patchcore_{category}"
            )

            # 4. Fit & Test
            engine.fit(model=model, datamodule=datamodule)
            results = engine.test(model=model, datamodule=datamodule)
            
            print(f"✅ Finished {category}. Results: {results}")
            
        except Exception as e:
            print(f"❌ Error processing category {category}: {str(e)}")
            continue
