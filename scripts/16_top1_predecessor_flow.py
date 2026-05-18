from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

EMB_PATH = Path("data/processed/embeddings/clip_renaissance_subset_v1.npy")
META_PATH = Path("data/processed/embeddings/clip_renaissance_subset_metadata_with_phash_v1.csv")

OUT_TABLE_DIR = Path("outputs/tables")
OUT_FIG_DIR = Path("outputs/figures")
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

STYLE_ORDER = {
    "Early_Renaissance": 0,
    "High_Renaissance": 1,
    "Northern_Renaissance": 1,
    "Mannerism_Late_Renaissance": 2,
    "Baroque": 3,
    "Rococo": 4,
}

STYLES = list(STYLE_ORDER.keys())

SEARCH_TOP_N = 300


def main():
    emb = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)

    print("Embedding shape:", emb.shape)
    print("Metadata shape:", meta.shape)

    sim = cosine_similarity(emb, emb)

    top1_edges = []

    for target_i in range(len(meta)):
        target_style = meta.iloc[target_i]["style_name"]
        if target_style not in STYLE_ORDER:
            continue

        target_order = STYLE_ORDER[target_style]
        target_dup = meta.iloc[target_i]["duplicate_group"]

        sims = sim[target_i].copy()
        sims[target_i] = -1

        ranked = np.argsort(-sims)[:SEARCH_TOP_N]

        best = None

        for source_i in ranked:
            source_style = meta.iloc[source_i]["style_name"]
            if source_style not in STYLE_ORDER:
                continue

            source_order = STYLE_ORDER[source_style]
            source_dup = meta.iloc[source_i]["duplicate_group"]

            # predecessor must be same or earlier order, but cross-style only
            if source_order > target_order:
                continue
            if source_style == target_style:
                continue
            if source_dup == target_dup:
                continue

            best = {
                "source_style": source_style,
                "target_style": target_style,
                "similarity": float(sims[source_i]),
                "source_idx": int(meta.iloc[source_i]["idx"]),
                "target_idx": int(meta.iloc[target_i]["idx"]),
                "source_artist": meta.iloc[source_i]["artist_name"],
                "target_artist": meta.iloc[target_i]["artist_name"],
                "source_genre": meta.iloc[source_i]["genre_name"],
                "target_genre": meta.iloc[target_i]["genre_name"],
            }
            break

        if best is not None:
            top1_edges.append(best)

    edge_df = pd.DataFrame(top1_edges)

    out_edges = OUT_TABLE_DIR / "top1_predecessor_edges_v1.csv"
    edge_df.to_csv(out_edges, index=False)

    print("\nSaved:")
    print(out_edges)
    print("Top-1 edge count:", len(edge_df))

    summary = edge_df.groupby(["source_style", "target_style"]).agg(
        edge_count=("similarity", "count"),
        mean_similarity=("similarity", "mean"),
    ).reset_index()

    target_counts = meta["style_name"].value_counts().to_dict()
    summary["target_count"] = summary["target_style"].map(target_counts)
    summary["target_coverage"] = summary["edge_count"] / summary["target_count"]

    out_summary = OUT_TABLE_DIR / "top1_predecessor_flow_summary_v1.csv"
    summary.to_csv(out_summary, index=False)

    print("\nSaved:")
    print(out_summary)

    print("\nTop-1 predecessor summary:")
    print(summary.sort_values("target_coverage", ascending=False).round(4))

    # matrix: source x target
    mat = pd.DataFrame(0.0, index=STYLES, columns=STYLES)

    for _, row in summary.iterrows():
        s = row["source_style"]
        t = row["target_style"]
        mat.loc[s, t] = row["target_coverage"]

    out_mat = OUT_TABLE_DIR / "top1_predecessor_flow_matrix_v1.csv"
    mat.to_csv(out_mat)

    plt.figure(figsize=(9, 7))
    plt.imshow(mat.values, interpolation="nearest")
    plt.colorbar(label="Target coverage by top-1 predecessor")
    plt.xticks(range(len(STYLES)), [s.replace("_", " ") for s in STYLES], rotation=45, ha="right")
    plt.yticks(range(len(STYLES)), [s.replace("_", " ") for s in STYLES])
    plt.xlabel("Target style")
    plt.ylabel("Source style")
    plt.title("Top-1 Cross-Style Predecessor Flow")
    plt.tight_layout()

    out_fig = OUT_FIG_DIR / "top1_predecessor_flow_matrix_v1.png"
    plt.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(out_mat)
    print(out_fig)


if __name__ == "__main__":
    main()