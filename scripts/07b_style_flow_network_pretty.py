from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

SUMMARY_PATH = Path("outputs/tables/style_order_constrained_flow_summary_v1.csv")
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 只保留较强的边，避免太乱
MIN_EDGE_RATE = 0.8

# 用更适合展示的名字
DISPLAY_NAME = {
    "Early_Renaissance": "Early Renaissance",
    "High_Renaissance": "High Renaissance",
    "Northern_Renaissance": "Northern Renaissance",
    "Mannerism_Late_Renaissance": "Mannerism / Late Renaissance",
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

    # 只保留强边
    df = df[df["edge_rate_per_target"] >= MIN_EDGE_RATE].copy()

    print("Filtered edges:")
    print(df[["source_style", "target_style", "edge_rate_per_target"]])

    G = nx.DiGraph()

    styles = [
        "Early_Renaissance",
        "High_Renaissance",
        "Northern_Renaissance",
        "Mannerism_Late_Renaissance",
        "Baroque",
        "Rococo",
    ]

    for s in styles:
        G.add_node(s)

    for _, row in df.iterrows():
        G.add_edge(
            row["source_style"],
            row["target_style"],
            weight=row["edge_rate_per_target"]
        )

    # 更艺术史感的时间轴布局
    pos = {
        "Early_Renaissance": (0.0, 0.0),
        "High_Renaissance": (1.2, 0.45),
        "Northern_Renaissance": (1.2, -0.45),
        "Mannerism_Late_Renaissance": (2.5, 0.0),
        "Baroque": (3.8, 0.0),
        "Rococo": (5.1, 0.0),
    }

    plt.figure(figsize=(13, 4.8), facecolor="#faf8f4")
    ax = plt.gca()
    ax.set_facecolor("#faf8f4")

    # 节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=3600,
        node_color=[NODE_COLOR[n] for n in G.nodes()],
        edgecolors="#444444",
        linewidths=1.2
    )

    # 标签
    nx.draw_networkx_labels(
        G, pos,
        labels={k: DISPLAY_NAME[k] for k in G.nodes()},
        font_size=11,
        font_family="serif"
    )

    # 边
    edges = list(G.edges())
    widths = [1.2 + 2.2 * G[u][v]["weight"] for u, v in edges]

    edge_colors = []
    for u, v in edges:
        edge_colors.append(NODE_COLOR[u])

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        width=widths,
        edge_color=edge_colors,
        alpha=0.72,
        arrows=True,
        arrowsize=18,
        arrowstyle='-|>',
        connectionstyle="arc3,rad=0.10"
    )

    # 只保留简洁标签
    edge_labels = {
        (u, v): f'{G[u][v]["weight"]:.2f}'
        for u, v in edges
    }

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=9,
        font_family="serif",
        rotate=False,
        bbox=dict(facecolor="#faf8f4", edgecolor="none", pad=0.2)
    )

    plt.title(
        "Directed Visual Flow Among Renaissance-Related Styles",
        fontsize=18,
        fontfamily="serif",
        pad=18
    )

    plt.axis("off")
    plt.tight_layout()

    out_path = OUT_DIR / "style_flow_network_pretty_v1.png"
    plt.savefig(out_path, dpi=260, bbox_inches="tight", facecolor="#faf8f4")
    plt.close()

    print("\nSaved:")
    print(out_path)


if __name__ == "__main__":
    main()