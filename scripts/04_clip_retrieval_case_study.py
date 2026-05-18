from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

EMB_PATH = Path("data/processed/embeddings/clip_renaissance_subset_v1.npy")
META_PATH = Path("data/processed/embeddings/clip_renaissance_subset_metadata_v1.csv")

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 8

# 你可以先设成 None，让程序自动选一张
# 后面也可以改成某个 subset row number，比如 QUERY_ROW = 100
QUERY_ROW = None


def safe_title(row):
    artist = row.get("artist_name", "Unknown")
    style = row.get("style_name", "Unknown")
    genre = row.get("genre_name", "Unknown")
    idx = row.get("idx", "NA")
    return f"idx={idx}\n{artist}\n{style}\n{genre}"


def main():
    print("Loading embeddings and metadata...")
    emb = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)

    print("Embedding shape:", emb.shape)
    print("Metadata shape:", meta.shape)

    if QUERY_ROW is None:
        # 先选一张 High Renaissance，如果没有就选第 0 张
        candidates = meta[meta["style_name"] == "High_Renaissance"]
        if len(candidates) > 0:
            query_row = candidates.sample(1, random_state=42).index[0]
        else:
            query_row = 0
    else:
        query_row = QUERY_ROW

    print("Query row:", query_row)

    query_vec = emb[query_row:query_row + 1]
    sims = cosine_similarity(query_vec, emb)[0]

    # 排除自己
    sims[query_row] = -1

    top_indices = np.argsort(-sims)[:TOP_K]

    result_rows = [query_row] + list(top_indices)

    print("\nQuery:")
    print(meta.iloc[query_row][["idx", "artist_name", "style_name", "genre_name"]])

    print("\nTop matches:")
    for rank, i in enumerate(top_indices, start=1):
        row = meta.iloc[i]
        print(
            rank,
            "score=", round(float(sims[i]), 4),
            "idx=", row["idx"],
            "artist=", row["artist_name"],
            "style=", row["style_name"],
            "genre=", row["genre_name"],
        )

    print("\nLoading WikiArt images...")
    ds = load_dataset("huggan/wikiart", split="train")

    fig, axes = plt.subplots(1, TOP_K + 1, figsize=(3.2 * (TOP_K + 1), 4.8))

    for pos, row_idx in enumerate(result_rows):
        row = meta.iloc[row_idx]
        original_idx = int(row["idx"])
        img = ds[original_idx]["image"].convert("RGB")

        axes[pos].imshow(img)
        axes[pos].axis("off")

        if pos == 0:
            title = "QUERY\n" + safe_title(row)
        else:
            score = sims[row_idx]
            title = f"Top {pos}\nscore={score:.3f}\n" + safe_title(row)

        axes[pos].set_title(title, fontsize=8)

    plt.tight_layout()

    out_path = OUT_DIR / "clip_retrieval_case_study_v1.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print("\nSaved figure:")
    print(out_path)


if __name__ == "__main__":
    main()