from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

SUMMARY_PATH = Path("outputs/tables/religious_top1_predecessor_flow_summary_v1.csv")
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COVERAGE = 0.20

DISPLAY_NAME = {
    "Early_Renaissance": "Early\nRenaissance",
    "High_Renaissance": "High\nRenaissance",
    "Northern_Renaissance": "Northern\nRenaissance",
    "Mannerism_Late_Renaissance": "Mannerism /\nLate Renaissance",
    "Baroque": "Baroque",
    "Rococo": "Rococo",
}

NODE_COLOR = {
    "Early_Renaissance": "#d8c3a5",
    "High_Renaissance": "#c97b63",
    "Northern_Renaissance": "#8aa29e",
    "Mannerism_Late_Renaissance": "#b692c2",
    "Baroque": "#6d8299",
    "Rococo": "#e8b4bc",
}


def main():
    df = pd.read_csv(SUMMARY_PATH)

    # Rococo religious sample is small, so keep it only if needed.
    # For the main religious motif network, focus on stronger and more stable edges.
    df = df[df["target_coverage"] >= MIN_COVERAGE].copy()

    print("Edges kept:")
    print(df[["source_style", "target_style", "target_coverage", "mean_similarity"]])

    G = nx.DiGraph()

    for s in DISPLAY_NAME:
        G.add_node(s)

    for _, row in df.iterrows():
        G.add_edge(
            row["source_style"],
            row["target_style"],
            weight=row["target_coverage"],
            sim=row["mean_similarity"],
        )

    pos = {
        "Early_Renaissance": (0.0, 0.0),
        "High_Renaissance": (1.4, 0.55),
        "Northern_Renaissance": (1.4, -0.55),
        "Mannerism_Late_Renaissance": (2.9, 0.25),
        "Baroque": (4.2, 0.0),
        "Rococo": (5.5, 0.0),
    }

    plt.figure(figsize=(13, 5.2), facecolor="#faf8f4")
    ax = plt.gca()
    ax.set_facecolor("#faf8f4")

    nodes_to_draw = [n for n in G.nodes() if G.degree(n) > 0]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes_to_draw,
        node_size=3400,
        node_color=[NODE_COLOR[n] for n in nodes_to_draw],
        edgecolors="#3f3f3f",
        linewidths=1.2,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: DISPLAY_NAME[n] for n in nodes_to_draw},
        font_size=10,
        font_family="serif",
    )

    edges = list(G.edges())
    widths = [1.0 + 5.5 * G[u][v]["weight"] for u, v in edges]
    edge_colors = [NODE_COLOR[u] for u, v in edges]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=widths,
        edge_color=edge_colors,
        alpha=0.72,
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.10",
    )

    edge_labels = {
        (u, v): f'{G[u][v]["weight"]:.2f}'
        for u, v in edges
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=9,
        font_family="serif",
        rotate=False,
        bbox=dict(facecolor="#faf8f4", edgecolor="none", pad=0.15),
    )

    plt.title(
        "Religious-Motif Visual Flow Across Renaissance-Related Styles",
        fontsize=18,
        fontfamily="serif",
        pad=18,
    )

    plt.text(
        0.0,
        -1.15,
        "Edge labels indicate the fraction of target-style religious paintings whose strongest cross-style predecessor comes from the source style.",
        fontsize=9,
        fontfamily="serif",
        ha="left",
    )

    plt.axis("off")
    plt.tight_layout()

    out_path = OUT_DIR / "religious_top1_flow_network_pretty_v1.png"
    plt.savefig(out_path, dpi=260, bbox_inches="tight", facecolor="#faf8f4")
    plt.close()

    print("\nSaved:")
    print(out_path)


if __name__ == "__main__":
    main()