import csv
import json
from pathlib import Path

results = [
    ["bottle", 1.0000, 0.9853],
    ["cable", 0.9925, 0.9836],
    ["capsule", 0.9908, 0.9899],
    ["carpet", 0.9827, 0.9902],
    ["grid", 0.9841, 0.9797],
    ["hazelnut", 0.9999, 0.9876],
    ["leather", 1.0000, 0.9921],
    ["metal_nut", 0.9965, 0.9878],
    ["pill", 0.9604, 0.9818],
    ["screw", 0.9625, 0.9792],
    ["tile", 1.0000, 0.9563],
    ["toothbrush", 0.9250, 0.9879],
    ["transistor", 0.9933, 0.9606],
    ["wood", 0.9868, 0.9335],
    ["zipper", 0.9761, 0.9813],
]

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "patchcore_all_categories.csv"

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "category",
        "image_AUROC",
        "pixel_AUROC"
    ])

    writer.writerows(results)

print(f"Saved: {csv_path}")
print("PatchCore results recorded for all 15 categories.")