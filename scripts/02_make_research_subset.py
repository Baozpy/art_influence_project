from pathlib import Path
import pandas as pd

META_PATH = Path("data/metadata/wikiart_metadata_clean_v1.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_STYLES = [
    "Early_Renaissance",
    "High_Renaissance",
    "Northern_Renaissance",
    "Mannerism_Late_Renaissance",
    "Baroque",
    "Rococo",
]

MAX_PER_STYLE = 1200
RANDOM_SEED = 42


def main():
    df = pd.read_csv(META_PATH)

    print("Full metadata shape:", df.shape)
    print("\nAvailable styles:")
    print(df["style_name"].value_counts().head(30))

    subset = df[df["style_name"].isin(TARGET_STYLES)].copy()

    print("\nRaw selected subset:")
    print(subset["style_name"].value_counts())

    # stratified sampling: avoid one style dominating the whole subset
    sampled_parts = []
    for style, group in subset.groupby("style_name"):
        n = min(len(group), MAX_PER_STYLE)
        sampled = group.sample(n=n, random_state=RANDOM_SEED)
        sampled_parts.append(sampled)

    subset = pd.concat(sampled_parts, axis=0)
    subset = subset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    out_csv = OUT_DIR / "wikiart_renaissance_subset_v1.csv"
    subset.to_csv(out_csv, index=False)

    print("\nFinal subset shape:", subset.shape)
    print("\nFinal style distribution:")
    print(subset["style_name"].value_counts())

    print("\nTop artists in subset:")
    print(subset["artist_name"].value_counts().head(30))

    print("\nSaved:")
    print(out_csv)


if __name__ == "__main__":
    main()