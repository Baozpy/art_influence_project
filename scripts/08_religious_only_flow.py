from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

EMB_PATH = Path("data/processed/embeddings/clip_renaissance_subset_v1.npy")
META_PATH = Path("data/processed/embeddings/clip_renaissance_subset_metadata_v1.csv")

OUT_TABLE_DIR = Path("outputs/tables")
OUT_FIG_DIR = Path("outputs/figures")
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
TARGET_GENRE = "religious_painting"

STYLE_ORDER = {
    "Early_Renaissance": 0,
    "High_Renaissance": 1,
    "Northern_Renaissance": 1,
    "Mannerism_Late_Renaissance": 2,
    "Baroque": 3,
    "Rococo": 4,
}


def main():
    emb = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)

    # filter religious_painting only
    mask = (meta["genre_name"] == TARGET_GENRE) & (meta["style_name"].isin(STYLE_ORDER.keys()))
    meta_sub = meta[mask].reset_index(drop=True)
    emb_sub = emb[mask.values]

    print("Original shape:", meta.shape, emb.shape)
    print("Filtered shape:", meta_sub.shape, emb_sub.shape)

    print("\nStyle counts in religious_painting subset:")
    print(meta_sub["style_name"].value_counts())

    styles = sorted(meta_sub["style_name"].dropna().unique().tolist())
    style_to_idx = {s: i for i, s in enumerate(styles)}

    # -------- Part A: row-normalized cross-style retrieval flow --------
    sim = cosine_similarity(emb_sub, emb_sub)
    n = len(meta_sub)
    flow = np.zeros((len(styles), len(styles)), dtype=np.float64)

    for i in range(n):
        query_style = meta_sub.iloc[i]["style_name"]
        sims = sim[i].copy()
        sims[i] = -1
        top_idx = np.argsort(-sims)[:TOP_K]

        for j in top_idx:
            neighbor_style = meta_sub.iloc[j]["style_name"]
            flow[style_to_idx[query_style], style_to_idx[neighbor_style]] += 1

    row_sums = flow.sum(axis=1, keepdims=True)
    flow_norm = np.divide(flow, row_sums, where=row_sums != 0)

    flow_df = pd.DataFrame(flow_norm, index=styles, columns=styles)

    out_csv1 = OUT_TABLE_DIR / "religious_only_cross_style_retrieval_flow_v1.csv"
    flow_df.to_csv(out_csv1)

    plt.figure(figsize=(9, 7))
    plt.imshow(flow_norm, interpolation="nearest")
    plt.colorbar(label=f"Proportion of top-{TOP_K} neighbors")
    plt.xticks(range(len(styles)), styles, rotation=45, ha="right")
    plt.yticks(range(len(styles)), styles)
    plt.title("Religious-Painting Only: Cross-Style Retrieval Flow")
    plt.tight_layout()

    out_fig1 = OUT_FIG_DIR / "religious_only_cross_style_retrieval_flow_v1.png"
    plt.savefig(out_fig1, dpi=220, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(out_csv1)
    print(out_fig1)

    print("\nRow-normalized flow matrix:")
    print(flow_df.round(4))

    # -------- Part B: style-order constrained summary --------
    edges = []

    for target_i in range(n):
        target_style = meta_sub.iloc[target_i]["style_name"]
        target_order = STYLE_ORDER[target_style]

        sims = sim[target_i].copy()
        sims[target_i] = -1

        top_idx = np.argsort(-sims)[:TOP_K * 10]

        kept = 0
        for source_i in top_idx:
            source_style = meta_sub.iloc[source_i]["style_name"]
            source_order = STYLE_ORDER[source_style]

            if source_order > target_order:
                continue
            if source_style == target_style:
                continue

            edges.append({
                "source_style": source_style,
                "target_style": target_style,
                "similarity": float(sims[source_i]),
                "source_idx": int(meta_sub.iloc[source_i]["idx"]),
                "target_idx": int(meta_sub.iloc[target_i]["idx"]),
                "source_artist": meta_sub.iloc[source_i]["artist_name"],
                "target_artist": meta_sub.iloc[target_i]["artist_name"],
            })

            kept += 1
            if kept >= TOP_K:
                break

    edge_df = pd.DataFrame(edges)
    out_csv2 = OUT_TABLE_DIR / "religious_only_style_order_constrained_edges_v1.csv"
    edge_df.to_csv(out_csv2, index=False)

    agg = edge_df.groupby(["source_style", "target_style"]).agg(
        edge_count=("similarity", "count"),
        mean_similarity=("similarity", "mean"),
    ).reset_index()

    target_counts = meta_sub["style_name"].value_counts().to_dict()
    agg["target_count"] = agg["target_style"].map(target_counts)
    agg["edge_rate_per_target"] = agg["edge_count"] / agg["target_count"]

    out_csv3 = OUT_TABLE_DIR / "religious_only_style_order_constrained_flow_summary_v1.csv"
    agg.to_csv(out_csv3, index=False)

    print("\nSaved:")
    print(out_csv2)
    print(out_csv3)

    print("\nReligious-only constrained summary:")
    print(agg.sort_values("edge_rate_per_target", ascending=False).round(4))


if __name__ == "__main__":
    main()