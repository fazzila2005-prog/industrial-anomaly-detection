import pandas as pd

FILE = "results/consensus/per_image_evidence_fusion.csv"

df = pd.read_csv(FILE)

print("\n" + "=" * 70)
print("EVIDENCE FUSION CSV CHECK")
print("=" * 70)

# ------------------------------------------------------------
# 1. Total records
# ------------------------------------------------------------

print(f"\nTotal records: {len(df)}")


# ------------------------------------------------------------
# 2. Categories present
# ------------------------------------------------------------

print("\nCategories present:")

category_counts = df["Category"].value_counts().sort_index()

print(category_counts.to_string())

print(f"\nNumber of categories: {df['Category'].nunique()}")


# ------------------------------------------------------------
# 3. Check for duplicate image paths
# ------------------------------------------------------------

duplicate_paths = df[df["Image_Path"].duplicated(keep=False)]

print("\nDuplicate Image_Path check:")

if duplicate_paths.empty:
    print("✓ No duplicate Image_Path values found.")
else:
    print(
        f"⚠️ Found {len(duplicate_paths)} rows with duplicate paths."
    )
    print(
        duplicate_paths[
            ["Category", "Defect_Type", "Image_Path"]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 4. Check missing values
# ------------------------------------------------------------

print("\nMissing-value check:")

missing = df.isnull().sum()

missing = missing[missing > 0]

if missing.empty:
    print("✓ No missing values found.")
else:
    print(missing.to_string())


# ------------------------------------------------------------
# 5. Check consensus levels
# ------------------------------------------------------------

print("\nConsensus levels:")

print(
    df["Consensus_Level"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 6. Check uncertainty range
# ------------------------------------------------------------

print("\nUncertainty score range:")

print(
    f"Minimum: {df['Uncertainty_Score'].min():.4f}"
)

print(
    f"Maximum: {df['Uncertainty_Score'].max():.4f}"
)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)