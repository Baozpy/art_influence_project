from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

EMB_PATH = Path("data/processed/embeddings/clip_renaissance_subset_v1.npy")
META_PATH = Path("data/processed/embeddings/clip_renaissance_subset_metadata_v1.csv")

OUT_FIG_DIR = Path("outputs/figures")
OUT_TABLE_DIR = Path("outputs/tables")
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
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

    styles = list(STYLE_ORDER.keys())

    print("Embedding shape:", emb.shape)
    print("Metadata shape:", meta.shape)

    sim = cosine_similarity(emb, emb)

    edges = []

    for target_i in range(len(meta)):
        target_style = meta.iloc[target_i]["style_name"]

        if target_style not in STYLE_ORDER:
            continue

        target_order = STYLE_ORDER[target_style]

        sims = sim[target_i].copy()
        sims[target_i] = -1

        top_idx = np.argsort(-sims)[:TOP_K * 10]

        kept = 0

        for source_i in top_idx:
            source_style = meta.iloc[source_i]["style_name"]

            if source_style not in STYLE_ORDER:
                continue

            source_order = STYLE_ORDER[source_style]

            # only allow earlier or same-stage styles to point to later styles
            if source_order > target_order:
                continue

            if source_style == target_style:
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
            })

            kept += 1
            if kept >= TOP_K:
                break

    edge_df = pd.DataFrame(edges)

    out_edges = OUT_TABLE_DIR / "style_order_constrained_edges_v1.csv"
    edge_df.to_csv(out_edges, index=False)

    print("\nSaved edge table:")
    print(out_edges)
    print("Number of edges:", len(edge_df))

    agg = edge_df.groupby(["source_style", "target_style"]).agg(
        edge_count=("similarity", "count"),
        mean_similarity=("similarity", "mean"),
    ).reset_index()

    # normalize by target style counts
    target_counts = meta["style_name"].value_counts().to_dict()
    agg["target_count"] = agg["target_style"].map(target_counts)
    agg["edge_rate_per_target"] = agg["edge_count"] / agg["target_count"]

    out_agg = OUT_TABLE_DIR / "style_order_constrained_flow_summary_v1.csv"
    agg.to_csv(out_agg, index=False)

    print("\nSaved flow summary:")
    print(out_agg)
    print(agg.sort_values("edge_rate_per_target", ascending=False))

    # build graph
    G = nx.DiGraph()

    for style in styles:
        G.add_node(style)

    for _, row in agg.iterrows():
        G.add_edge(
            row["source_style"],
            row["target_style"],
            weight=row["edge_rate_per_target"],
            label=f'{row["edge_rate_per_target"]:.2f}',
        )

    pos = {
        "Early_Renaissance": (0, 0),
        "High_Renaissance": (1, 0.5),
        "Northern_Renaissance": (1, -0.5),
        "Mannerism_Late_Renaissance": (2, 0),
        "Baroque": (3, 0),
        "Rococo": (4, 0),
    }

    plt.figure(figsize=(13, 5))

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    widths = [1 + 8 * w for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=2500)
    nx.draw_networkx_labels(G, pos, font_size=9)

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.08",
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Style-Order Constrained Visual Flow Network")
    plt.axis("off")
    plt.tight_layout()

    out_fig = OUT_FIG_DIR / "style_order_constrained_flow_network_v1.png"
    plt.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close()

    print("\nSaved figure:")
    print(out_fig)


if __name__ == "__main__":
    main()