import os
import json
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "results/consensus/per_image_evidence_fusion.csv"

OUTPUT_FILE = "results/consensus/genai_evidence.json"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("CREATING STRUCTURED EVIDENCE FOR GENAI")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check input file
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print(f"\n❌ Input file not found:")
        print(INPUT_FILE)
        exit()

    # --------------------------------------------------------
    # 2. Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"\n✓ Loaded {len(df)} evidence records.")


    # --------------------------------------------------------
    # 3. Create structured evidence records
    # --------------------------------------------------------

    evidence_records = []

    for _, row in df.iterrows():

        # Convert model labels
        patchcore_prediction = (
            "ANOMALY"
            if int(row["PatchCore_Label"]) == 1
            else "NORMAL"
        )

        padim_prediction = (
            "ANOMALY"
            if int(row["PaDiM_Label"]) == 1
            else "NORMAL"
        )


        # Create structured record
        record = {

            "image": {
                "path": str(row["Image_Path"]),
                "category": str(row["Category"]),
                "defect_type": str(row["Defect_Type"])
            },

            "models": {

                "PatchCore": {
                    "anomaly_score": float(
                        row["PatchCore_Raw_Score"]
                    ),
                    "prediction": patchcore_prediction
                },

                "PaDiM": {
                    "anomaly_score": float(
                        row["PaDiM_Raw_Score"]
                    ),
                    "prediction": padim_prediction
                }
            },

            "evidence": {

                "score_agreement": float(
                    row["Score_Agreement"]
                ),

                "prediction_agreement": int(
                    row["Prediction_Agreement"]
                ),

                "heatmap_overlap_iou": float(
                    row["Heatmap_Agreement_IoU"]
                )
            },

            "consensus": {

                "score": float(
                    row["Consensus_Score"]
                ),

                "level": str(
                    row["Consensus_Level"]
                ),

                "uncertainty": float(
                    row["Uncertainty_Score"]
                )
            }
        }

        evidence_records.append(record)


    # --------------------------------------------------------
    # 4. Create output directory
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )


    # --------------------------------------------------------
    # 5. Save JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evidence_records,
            f,
            indent=4
        )


    # --------------------------------------------------------
    # 6. Success message
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "🎉 SUCCESS: Structured GenAI evidence created!"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print(
        f"\nTotal evidence records: {len(evidence_records)}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # 7. Show first record
    # --------------------------------------------------------

    print("\nExample structured record:\n")

    print(
        json.dumps(
            evidence_records[0],
            indent=4
        )
    )