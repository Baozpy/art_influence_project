from datasets import load_dataset
import pandas as pd
from collections import Counter
from pathlib import Path

OUT_DIR = Path("outputs/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading WikiArt dataset...")
    ds = load_dataset("huggan/wikiart", split="train")

    print(ds)
    print("Number of samples:", len(ds))
    print("Columns:", ds.column_names)

    rows = []

    for i in range(min(len(ds), 5000)):
        item = ds[i]
        rows.append({
            "idx": i,
            "artist": item.get("artist", None),
            "genre": item.get("genre", None),
            "style": item.get("style", None),
        })

    df = pd.DataFrame(rows)

    print("\nSample metadata:")
    print(df.head())

    print("\nArtist count:")
    print(df["artist"].value_counts().head(20))

    print("\nGenre count:")
    print(df["genre"].value_counts().head(20))

    print("\nStyle count:")
    print(df["style"].value_counts().head(20))

    df.to_csv(OUT_DIR / "wikiart_sample_metadata_5000.csv", index=False)

    summary = {
        "num_total_samples": len(ds),
        "columns": ", ".join(ds.column_names),
        "num_sampled": len(df),
        "num_artists_in_sample": df["artist"].nunique(),
        "num_genres_in_sample": df["genre"].nunique(),
        "num_styles_in_sample": df["style"].nunique(),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUT_DIR / "wikiart_inventory_summary.csv", index=False)

    print("\nSaved:")
    print(OUT_DIR / "wikiart_sample_metadata_5000.csv")
    print(OUT_DIR / "wikiart_inventory_summary.csv")

if __name__ == "__main__":
    main()