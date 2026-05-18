from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SUMMARY_PATH = Path("outputs/tables/style_order_constrained_flow_summary_v1.csv")
OUT_TABLE_DIR = Path("outputs/tables")
OUT_FIG_DIR = Path("outputs/figures")
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

STYLES = [
    "Early_Renaissance",
    "High_Renaissance",
    "Northern_Renaissance",
    "Mannerism_Late_Renaissance",
    "Baroque",
    "Rococo",
]


def main():
    df = pd.read_csv(SUMMARY_PATH)

    print("Loaded summary:")
    print(df.head())

    # For each target style, normalize incoming edge_count across all source styles
    df["incoming_share"] = df.groupby("target_style")["edge_count"].transform(
        lambda x: x / x.sum()
    )

    out_csv1 = OUT_TABLE_DIR / "normalized_incoming_flow_v1.csv"
    df.to_csv(out_csv1, index=False)

    print("\nSaved:")
    print(out_csv1)

    print("\nNormalized incoming flow:")
    print(df[["source_style", "target_style", "edge_count", "incoming_share"]]
          .sort_values(["target_style", "incoming_share"], ascending=[True, False])
          .round(4))

    # strongest predecessor for each target
    strongest = df.loc[df.groupby("target_style")["incoming_share"].idxmax()].copy()
    strongest = strongest.sort_values("target_style")

    out_csv2 = OUT_TABLE_DIR / "strongest_predecessor_per_target_v1.csv"
    strongest.to_csv(out_csv2, index=False)

    print("\nStrongest predecessor per target:")
    print(strongest[["target_style", "source_style", "incoming_share", "mean_similarity"]].round(4))

    print("\nSaved:")
    print(out_csv2)

    # matrix for heatmap
    mat = pd.DataFrame(0.0, index=STYLES, columns=STYLES)

    for _, row in df.iterrows():
        s = row["source_style"]
        t = row["target_style"]
        if s in STYLES and t in STYLES:
            mat.loc[s, t] = row["incoming_share"]

    out_csv3 = OUT_TABLE_DIR / "normalized_incoming_flow_matrix_v1.csv"
    mat.to_csv(out_csv3)

    plt.figure(figsize=(9, 7))
    plt.imshow(mat.values, interpolation="nearest")
    plt.colorbar(label="Share of incoming cross-style edges")
    plt.xticks(range(len(STYLES)), STYLES, rotation=45, ha="right")
    plt.yticks(range(len(STYLES)), STYLES)
    plt.xlabel("Target style")
    plt.ylabel("Source style")
    plt.title("Normalized Incoming Visual Flow")
    plt.tight_layout()

    out_fig = OUT_FIG_DIR / "normalized_incoming_flow_matrix_v1.png"
    plt.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(out_csv3)
    print(out_fig)


if __name__ == "__main__":
    main()