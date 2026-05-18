from datasets import load_dataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm

OUT_DIR = Path("data/metadata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = Path("outputs/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def get_label_name(feature, value):
    """
    Convert integer ClassLabel id to readable string.
    If conversion fails, return the original value.
    """
    if value is None:
        return None

    try:
        if hasattr(feature, "int2str"):
            return feature.int2str(value)
    except Exception:
        pass

    return value


def main():
    print("Loading WikiArt dataset...")
    ds = load_dataset("huggan/wikiart", split="train")

    print(ds)
    print("Columns:", ds.column_names)
    print("Features:", ds.features)

    artist_feature = ds.features.get("artist")
    genre_feature = ds.features.get("genre")
    style_feature = ds.features.get("style")

    rows = []

    for i in tqdm(range(len(ds)), desc="Building clean metadata"):
        item = ds[i]

        artist_id = item.get("artist", None)
        genre_id = item.get("genre", None)
        style_id = item.get("style", None)

        img = item.get("image", None)

        if img is not None:
            width, height = img.size
            image_mode = img.mode
        else:
            width, height = None, None
            image_mode = None

        rows.append({
            "idx": i,

            "artist_id": artist_id,
            "artist_name": get_label_name(artist_feature, artist_id),

            "genre_id": genre_id,
            "genre_name": get_label_name(genre_feature, genre_id),

            "style_id": style_id,
            "style_name": get_label_name(style_feature, style_id),

            "width": width,
            "height": height,
            "image_mode": image_mode,
        })

    df = pd.DataFrame(rows)

    out_csv = OUT_DIR / "wikiart_metadata_clean_v1.csv"
    df.to_csv(out_csv, index=False)

    print("\nSaved clean metadata:")
    print(out_csv)

    print("\nShape:")
    print(df.shape)

    print("\nSample:")
    print(df.head())

    print("\nTop artists:")
    print(df["artist_name"].value_counts().head(20))

    print("\nTop genres:")
    print(df["genre_name"].value_counts().head(20))

    print("\nTop styles:")
    print(df["style_name"].value_counts().head(20))

    # Save distribution tables for paper/project report
    df["artist_name"].value_counts().reset_index().rename(
        columns={"artist_name": "artist_name", "count": "count"}
    ).to_csv(TABLE_DIR / "artist_distribution_v1.csv", index=False)

    df["genre_name"].value_counts().reset_index().rename(
        columns={"genre_name": "genre_name", "count": "count"}
    ).to_csv(TABLE_DIR / "genre_distribution_v1.csv", index=False)

    df["style_name"].value_counts().reset_index().rename(
        columns={"style_name": "style_name", "count": "count"}
    ).to_csv(TABLE_DIR / "style_distribution_v1.csv", index=False)

    print("\nSaved distribution tables:")
    print(TABLE_DIR / "artist_distribution_v1.csv")
    print(TABLE_DIR / "genre_distribution_v1.csv")
    print(TABLE_DIR / "style_distribution_v1.csv")


if __name__ == "__main__":
    main()