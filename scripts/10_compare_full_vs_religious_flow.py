from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FULL_PATH = Path("outputs/tables/style_order_constrained_flow_summary_v1.csv")
REL_PATH = Path("outputs/tables/religious_only_style_order_constrained_flow_summary_v1.csv")

OUT_TABLE_DIR = Path("outputs/tables")
OUT_FIG_DIR = Path("outputs/figures")
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    full = pd.read_csv(FULL_PATH)
    rel = pd.read_csv(REL_PATH)

    full_small = full[[
        "source_style",
        "target_style",
        "edge_rate_per_target",
        "mean_similarity"
    ]].rename(columns={
        "edge_rate_per_target": "full_edge_rate",
        "mean_similarity": "full_mean_similarity"
    })

    rel_small = rel[[
        "source_style",
        "target_style",
        "edge_rate_per_target",
        "mean_similarity"
    ]].rename(columns={
        "edge_rate_per_target": "religious_edge_rate",
        "mean_similarity": "religious_mean_similarity"
    })

    merged = pd.merge(
        full_small,
        rel_small,
        on=["source_style", "target_style"],
        how="outer"
    ).fillna(0)

    merged["edge_rate_diff"] = merged["religious_edge_rate"] - merged["full_edge_rate"]
    merged["edge_rate_ratio"] = merged.apply(
        lambda r: r["religious_edge_rate"] / r["full_edge_rate"]
        if r["full_edge_rate"] > 0 else np.nan,
        axis=1
    )

    out_csv = OUT_TABLE_DIR / "full_vs_religious_flow_comparison_v1.csv"
    merged.to_csv(out_csv, index=False)

    print("\nSaved:")
    print(out_csv)

    print("\nComparison table:")
    print(
        merged.sort_values("full_edge_rate", ascending=False)
        .round(4)
        .to_string(index=False)
    )

    # plot top full-flow edges
    plot_df = merged.sort_values("full_edge_rate", ascending=False).head(12).copy()

    labels = [
        f"{s.replace('_', ' ')} →\n{t.replace('_', ' ')}"
        for s, t in zip(plot_df["source_style"], plot_df["target_style"])
    ]

    x = np.arange(len(plot_df))
    width = 0.38

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, plot_df["full_edge_rate"], width, label="Full subset")
    plt.bar(x + width / 2, plot_df["religious_edge_rate"], width, label="Religious paintings only")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Edge rate per target")
    plt.title("Full vs. Religious-Painting-Only Visual Flow")
    plt.legend()
    plt.tight_layout()

    out_fig = OUT_FIG_DIR / "full_vs_religious_flow_comparison_v1.png"
    plt.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(out_fig)


if __name__ == "__main__":
    main()