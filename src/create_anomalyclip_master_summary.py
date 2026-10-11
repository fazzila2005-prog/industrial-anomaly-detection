import os
import re
import pandas as pd

# ============================================================
# ANOMALYCLIP - MASTER EXTERNAL EVALUATION SUMMARY
# ============================================================

PROJECT_DIR = r"C:\Users\Fazzila\OneDrive\Desktop\industrial-anomaly-project"

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "anomalyclip_external"
)

# Special Cable results are stored separately
CABLE_RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "anomalyclip_external_cable"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "all_categories_summary.csv"
)

# ============================================================
# ALL 15 MVTec CATEGORIES
# ============================================================

categories = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper"
]

all_results = []

print("=" * 70)
print("ANOMALYCLIP - MASTER EXTERNAL EVALUATION SUMMARY")
print("=" * 70)
print()


# ============================================================
# FUNCTION: EXTRACT METRIC FROM SUMMARY TXT
# ============================================================

def extract_from_txt(text, label):

    # Handles formats such as:
    # Accuracy : 0.8500
    # AUROC : 0.9500
    # False-positive rate : 0.3000

    pattern = (
        rf"{re.escape(label)}"
        rf"\s*:\s*([0-9]*\.?[0-9]+)"
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return None


# ============================================================
# FUNCTION: READ ONE CATEGORY
# ============================================================

def read_category(category):

    # --------------------------------------------------------
    # CABLE HAS A DIFFERENT FOLDER AND FILE NAME
    # --------------------------------------------------------

    if category == "cable":

        category_dir = CABLE_RESULTS_DIR

        csv_file = os.path.join(
            category_dir,
            "anomalyclip_external_cable_results.csv"
        )

        txt_file = os.path.join(
            category_dir,
            "anomalyclip_external_cable_summary.txt"
        )

    else:

        category_dir = os.path.join(
            RESULTS_DIR,
            category
        )

        csv_file = os.path.join(
            category_dir,
            f"{category}_external_results.csv"
        )

        txt_file = os.path.join(
            category_dir,
            f"{category}_external_summary.txt"
        )

    print(f"Reading: {category}")

    row = {
        "Category": category.replace("_", " ").title(),
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1 Score": None,
        "False Positive Rate": None,
        "AUROC": None,
        "Average Precision": None,
        "Locked Threshold": None
    }

    # ========================================================
    # READ CSV
    # ========================================================

    if os.path.exists(csv_file):

        try:

            df = pd.read_csv(csv_file)

            print(f"  CSV found: {os.path.basename(csv_file)}")

            # ------------------------------------------------
            # Normalize column names
            # ------------------------------------------------

            column_map = {
                col.strip().lower().replace(" ", "_"): col
                for col in df.columns
            }

            # ------------------------------------------------
            # Helper to get metric from CSV
            # ------------------------------------------------

            def get_value(possible_names):

                for name in possible_names:

                    key = name.lower().replace(" ", "_")

                    if key in column_map:

                        value = df[column_map[key]].iloc[0]

                        try:
                            return float(value)
                        except:
                            return value

                return None

            row["Accuracy"] = get_value([
                "accuracy"
            ])

            row["Precision"] = get_value([
                "precision"
            ])

            row["Recall"] = get_value([
                "recall"
            ])

            row["F1 Score"] = get_value([
                "f1_score",
                "f1"
            ])

            row["False Positive Rate"] = get_value([
                "false_positive_rate",
                "fpr"
            ])

            row["AUROC"] = get_value([
                "auroc",
                "image_auroc"
            ])

            row["Average Precision"] = get_value([
                "average_precision",
                "ap"
            ])

            row["Locked Threshold"] = get_value([
                "locked_threshold",
                "threshold"
            ])

            # =================================================
            # IF CSV CONTAINS IMAGE-LEVEL RESULTS
            # =================================================

            if row["Accuracy"] is None:

                actual_col = None
                prediction_col = None

                for col in df.columns:

                    clean = col.lower().replace(" ", "_")

                    if clean in [
                        "actual",
                        "actual_label",
                        "true_label",
                        "ground_truth"
                    ]:
                        actual_col = col

                    if clean in [
                        "prediction",
                        "predicted",
                        "predicted_label",
                        "predicted_class"
                    ]:
                        prediction_col = col

                # --------------------------------------------
                # Calculate classification metrics
                # --------------------------------------------

                if actual_col and prediction_col:

                    actual = (
                        df[actual_col]
                        .astype(str)
                        .str.upper()
                    )

                    predicted = (
                        df[prediction_col]
                        .astype(str)
                        .str.upper()
                    )

                    correct = (
                        actual == predicted
                    )

                    row["Accuracy"] = correct.mean()

                    actual_anomaly = (
                        actual == "ANOMALY"
                    )

                    predicted_anomaly = (
                        predicted == "ANOMALY"
                    )

                    tp = (
                        actual_anomaly &
                        predicted_anomaly
                    ).sum()

                    tn = (
                        ~actual_anomaly &
                        ~predicted_anomaly
                    ).sum()

                    fp = (
                        ~actual_anomaly &
                        predicted_anomaly
                    ).sum()

                    fn = (
                        actual_anomaly &
                        ~predicted_anomaly
                    ).sum()

                    precision = (
                        tp / (tp + fp)
                        if (tp + fp) > 0
                        else 0
                    )

                    recall = (
                        tp / (tp + fn)
                        if (tp + fn) > 0
                        else 0
                    )

                    f1 = (
                        2 * precision * recall /
                        (precision + recall)
                        if (precision + recall) > 0
                        else 0
                    )

                    fpr = (
                        fp / (fp + tn)
                        if (fp + tn) > 0
                        else 0
                    )

                    row["Precision"] = precision
                    row["Recall"] = recall
                    row["F1 Score"] = f1
                    row["False Positive Rate"] = fpr

        except Exception as e:

            print(f"  CSV ERROR: {e}")

    else:

        print("  CSV not found.")

    # ========================================================
    # READ SUMMARY TXT
    # ========================================================

    if os.path.exists(txt_file):

        try:

            with open(
                txt_file,
                "r",
                encoding="utf-8"
            ) as f:

                text = f.read()

            print(
                f"  TXT found: {os.path.basename(txt_file)}"
            )

            # ------------------------------------------------
            # Fill ONLY missing values from TXT
            # ------------------------------------------------

            if row["Accuracy"] is None:
                row["Accuracy"] = extract_from_txt(
                    text,
                    "Accuracy"
                )

            if row["Precision"] is None:
                row["Precision"] = extract_from_txt(
                    text,
                    "Precision"
                )

            if row["Recall"] is None:
                row["Recall"] = extract_from_txt(
                    text,
                    "Recall"
                )

            if row["F1 Score"] is None:
                row["F1 Score"] = extract_from_txt(
                    text,
                    "F1 Score"
                )

            if row["False Positive Rate"] is None:
                row["False Positive Rate"] = extract_from_txt(
                    text,
                    "False-positive rate"
                )

            if row["AUROC"] is None:
                row["AUROC"] = extract_from_txt(
                    text,
                    "AUROC"
                )

            if row["Average Precision"] is None:
                row["Average Precision"] = extract_from_txt(
                    text,
                    "Average Precision"
                )

            if row["Locked Threshold"] is None:
                row["Locked Threshold"] = extract_from_txt(
                    text,
                    "LOCKED THRESHOLD"
                )

        except Exception as e:

            print(f"  TXT ERROR: {e}")

    else:

        print("  TXT not found.")

    # ========================================================
    # CHECK WHETHER CATEGORY IS COMPLETE
    # ========================================================

    required_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "False Positive Rate",
        "AUROC",
        "Average Precision",
        "Locked Threshold"
    ]

    missing = [
        metric
        for metric in required_metrics
        if row[metric] is None
    ]

    if missing:

        print(
            "  WARNING - Missing:",
            ", ".join(missing)
        )

    else:

        print("  SUCCESS - All metrics found.")

    print()

    return row


# ============================================================
# READ ALL 15 CATEGORIES
# ============================================================

for category in categories:

    row = read_category(category)

    all_results.append(row)


# ============================================================
# CREATE MASTER DATAFRAME
# ============================================================

summary_df = pd.DataFrame(
    all_results
)


# ============================================================
# CORRECT MVTec CATEGORY ORDER
# ============================================================

category_order = [
    "Bottle",
    "Cable",
    "Capsule",
    "Carpet",
    "Grid",
    "Hazelnut",
    "Leather",
    "Metal Nut",
    "Pill",
    "Screw",
    "Tile",
    "Toothbrush",
    "Transistor",
    "Wood",
    "Zipper"
]

summary_df["Category"] = pd.Categorical(
    summary_df["Category"],
    categories=category_order,
    ordered=True
)

summary_df = summary_df.sort_values(
    "Category"
)


# ============================================================
# ROUND NUMBERS
# ============================================================

numeric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "False Positive Rate",
    "AUROC",
    "Average Precision",
    "Locked Threshold"
]

for column in numeric_columns:

    summary_df[column] = pd.to_numeric(
        summary_df[column],
        errors="coerce"
    ).round(4)


# ============================================================
# SAVE MASTER CSV
# ============================================================

summary_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print("=" * 70)
print("MASTER SUMMARY CREATED")
print("=" * 70)
print()

print(
    summary_df.to_string(
        index=False
    )
)

print()

print("=" * 70)

complete_count = (
    summary_df[numeric_columns]
    .notna()
    .all(axis=1)
    .sum()
)

print(
    f"Complete categories: {complete_count} / 15"
)

print("=" * 70)

print()

print("Saved to:")
print(OUTPUT_FILE)

print()

if complete_count == 15:

    print("SUCCESS: ALL 15 CATEGORIES ARE COMPLETE.")

else:

    print(
        "WARNING: Some categories still have missing metrics."
    )

print()

print("DONE!")