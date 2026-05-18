from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

EMB_PATH = Path("data/processed/embeddings/clip_renaissance_subset_v1.npy")
META_PATH = Path("data/processed/embeddings/clip_renaissance_subset_metadata_v1.csv")

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5


def main():
    emb = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)

    styles = sorted(meta["style_name"].dropna().unique().tolist())
    style_to_idx = {s: i for i, s in enumerate(styles)}

    print("Styles:", styles)
    print("TOP_K:", TOP_K)

    sim = cosine_similarity(emb, emb)

    n = len(meta)
    flow = np.zeros((len(styles), len(styles)), dtype=np.float64)

    for i in range(n):
        query_style = meta.iloc[i]["style_name"]
        if pd.isna(query_style):
            continue

        sims = sim[i].copy()
        sims[i] = -1  # remove self

        top_idx = np.argsort(-sims)[:TOP_K]

        for j in top_idx:
            neighbor_style = meta.iloc[j]["style_name"]
            if pd.isna(neighbor_style):
                continue

            flow[style_to_idx[query_style], style_to_idx[neighbor_style]] += 1

    # row normalize
    row_sums = flow.sum(axis=1, keepdims=True)
    flow_norm = np.divide(flow, row_sums, where=row_sums != 0)

    flow_df = pd.DataFrame(flow_norm, index=styles, columns=styles)

    out_csv = Path("outputs/tables/cross_style_retrieval_flow_v1.csv")
    flow_df.to_csv(out_csv)

    plt.figure(figsize=(9, 7))
    plt.imshow(flow_norm, interpolation="nearest")
    plt.colorbar(label="Proportion of top-k neighbors")
    plt.xticks(range(len(styles)), styles, rotation=45, ha="right")
    plt.yticks(range(len(styles)), styles)
    plt.title(f"Cross-Style Retrieval Flow (Top-{TOP_K} Neighbors)")
    plt.tight_layout()

    out_fig = OUT_DIR / "cross_style_retrieval_flow_v1.png"
    plt.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(out_csv)
    print(out_fig)

    print("\nFlow matrix (row-normalized):")
    print(flow_df.round(4))


if __name__ == "__main__":
    main()