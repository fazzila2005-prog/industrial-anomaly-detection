import os
import pandas as pd
import torch
from anomalib.data import MVTecAD
from anomalib.models import Padim
from anomalib.engine import Engine

if __name__ == '__main__':
    # List of all 15 MVTec categories to benchmark using PaDiM
    categories = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", 
        "leather", "metal_nut", "pill", "screw", "tile", 
        "toothbrush", "transistor", "wood", "zipper"
    ]

    print(f"Starting automated PaDiM evaluation across {len(categories)} categories...")
    summary_data = []

    for category in categories:
        print(f"\n==================================================")
        print(f"🚀 PROCESSING CATEGORY (PADIM): {category.upper()}")
        print(f"==================================================")
        
        try:
            # 1. Load Dataset from your clean data folder layout
            datamodule = MVTecAD(
                root="./data",
                category=category,
                train_batch_size=4,   # PaDiM easily handles larger batch sizes on CPU
                eval_batch_size=4,
                num_workers=0,        # Keeps execution stable on Windows CPU
            )

            # 2. Create PaDiM Model (Uses ResNet-18 backbone out of the box)
            model = Padim(
                backbone="resnet18",  # Standard benchmark default for PaDiM
                layers=["layer1", "layer2", "layer3"]
            )

            # 3. Create Engine
            engine = Engine(
                accelerator="cpu",
                devices=1,
                default_root_dir=f"results/padim_{category}"  # Kept separate from PatchCore
            )

            # 4. Fit Model (Calculates Gaussian stats profiles)
            print(f"[{category.upper()}] Running statistical feature grouping...")
            engine.fit(model=model, datamodule=datamodule)
            
            # 5. Evaluate Performance
            print(f"[{category.upper()}] Running final testing phase...")
            results = engine.test(model=model, datamodule=datamodule)
            
            print(f"✅ SUCCESSFULLY COMPLETED {category.upper()}")
            print(f"Metrics: {results}\n")
            
            # Extract raw metrics from output list safely
            if results and len(results) > 0:
                metrics_dict = results[0] if isinstance(results, list) else results
                summary_data.append({
                    "Category": category,
                    "Image_AUROC": metrics_dict.get("image_AUROC", 0.0),
                    "Image_F1Score": metrics_dict.get("image_F1Score", 0.0),
                    "Pixel_AUROC": metrics_dict.get("pixel_AUROC", 0.0),
                    "Pixel_F1Score": metrics_dict.get("pixel_F1Score", 0.0)
                })
            
        except Exception as e:
            print(f"❌ Error occurred while processing category {category}: {str(e)}")
            continue

    # 5. Save PaDiM Summary Report to a distinct CSV file
    if summary_data:
        print("\n📊 Compiling all PaDiM scores into a summary report...")
        df = pd.DataFrame(summary_data)
        os.makedirs("results", exist_ok=True)
        csv_path = "results/padim_results_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"📁 Summary report saved successfully at: {csv_path}")

    print("\n==================================================")
    print("🎉 ALL MVTEC CATEGORIES COMPLETED VIA PADIM!")
    print("==================================================")
