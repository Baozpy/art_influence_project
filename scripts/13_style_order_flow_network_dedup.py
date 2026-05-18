from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EMB_PATH = Path("data/processed/embeddings/clip_renaissance_subset_v1.npy")
META_PATH = Path("data/processed/embeddings/clip_renaissance_subset_metadata_with_phash_v1.csv")

OUT_TABLE_DIR = Path("outputs/tables")
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5

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

    print("Embedding shape:", emb.shape)
    print("Metadata shape:", meta.shape)

    sim = cosine_similarity(emb, emb)

    edges = []

    for target_i in range(len(meta)):
        target_style = meta.iloc[target_i]["style_name"]

        if target_style not in STYLE_ORDER:
            continue

        target_order = STYLE_ORDER[target_style]
        target_dup = meta.iloc[target_i]["duplicate_group"]

        sims = sim[target_i].copy()
        sims[target_i] = -1

        # search wider pool because many candidates may be removed
        top_idx = np.argsort(-sims)[:TOP_K * 30]

        kept = 0

        for source_i in top_idx:
            source_style = meta.iloc[source_i]["style_name"]

            if source_style not in STYLE_ORDER:
                continue

            source_order = STYLE_ORDER[source_style]
            source_dup = meta.iloc[source_i]["duplicate_group"]

            # only earlier/same-stage source to later target
            if source_order > target_order:
                continue

            # remove same style
            if source_style == target_style:
                continue

            # remove duplicate or near-duplicate images
            if source_dup == target_dup:
                continue

            edges.append({
                "source_style": source_style,
                "target_style": target_style,
                "similarity": float(sims[source_i]),
                "source_idx": int(meta.iloc[source_i]["idx"]),
                "target_idx": int(meta.iloc[target_i]["idx"]),
                "source_artist": meta.iloc[source_i]["artist_name"],
                "target_artist": meta.iloc[target_i]["artist_name"],
                "source_genre": meta.iloc[source_i]["genre_name"],
                "target_genre": meta.iloc[target_i]["genre_name"],
                "source_duplicate_group": int(source_dup),
                "target_duplicate_group": int(target_dup),
            })

            kept += 1
            if kept >= TOP_K:
                break

    edge_df = pd.DataFrame(edges)

    out_edges = OUT_TABLE_DIR / "style_order_constrained_edges_dedup_v1.csv"
    edge_df.to_csv(out_edges, index=False)

    print("\nSaved dedup edge table:")
    print(out_edges)
    print("Number of edges:", len(edge_df))

    agg = edge_df.groupby(["source_style", "target_style"]).agg(
        edge_count=("similarity", "count"),
        mean_similarity=("similarity", "mean"),
    ).reset_index()

    target_counts = meta["style_name"].value_counts().to_dict()
    agg["target_count"] = agg["target_style"].map(target_counts)
    agg["edge_rate_per_target"] = agg["edge_count"] / agg["target_count"]

    out_agg = OUT_TABLE_DIR / "style_order_constrained_flow_summary_dedup_v1.csv"
    agg.to_csv(out_agg, index=False)

    print("\nSaved dedup summary:")
    print(out_agg)

    print("\nDedup summary:")
    print(agg.sort_values("edge_rate_per_target", ascending=False).round(4))


if __name__ == "__main__":
    main()