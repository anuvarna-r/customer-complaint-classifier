from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
input_path = BASE_DIR / "data" / "complaints_processed.csv"
output_path = BASE_DIR / "data" / "complaints_cleaned.csv"

df = pd.read_csv(input_path, usecols=["product", "narrative"])
original_rows = len(df)

df = df.dropna(subset=["product", "narrative"]).copy()

df["product"] = df["product"].str.strip()
df["narrative"] = (
    df["narrative"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df = df[
    df["product"].ne("") & df["narrative"].ne("")
].copy()

print("Missing or empty rows removed:", original_rows - len(df))

# Remove text assigned to multiple categories.
label_counts = df.groupby("narrative")["product"].nunique()
conflicting_texts = label_counts[label_counts > 1].index
conflict_mask = df["narrative"].isin(conflicting_texts)

print("Rows with conflicting labels removed:", int(conflict_mask.sum()))
df = df.loc[~conflict_mask].copy()

# Keep identical complaints out of both training and testing.
before_duplicates = len(df)
df = df.drop_duplicates(subset=["narrative"]).reset_index(drop=True)

print("Duplicate complaints removed:", before_duplicates - len(df))

df.to_csv(output_path, index=False)

print("Cleaned dataset rows:", len(df))
print("\nCategory counts:")
print(df["product"].value_counts())
print("\nSaved to:", output_path)