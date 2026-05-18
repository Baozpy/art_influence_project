from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

EDGE_PATH = Path("outputs/tables/top1_predecessor_edges_v1.csv")

OUT_TABLE_DIR = Path("outputs/tables")
OUT_FIG_DIR = Path("outputs/figures")
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EDGES = [
    ("Early_Renaissance", "High_Renaissance"),
    ("Early_Renaissance", "Northern_Renaissance"),
    ("High_Renaissance", "Mannerism_Late_Renaissance"),
    ("Mannerism_Late_Renaissance", "Baroque"),
    ("Baroque", "Rococo"),
]

TOP_GENRE_PAIRS = 8


def display_name(s):
    return s.replace("_", " ")


def main():
    edges = pd.read_csv(EDGE_PATH)

    all_rows = []

    for source_style, target_style in TARGET_EDGES:
        sub = edges[
            (edges["source_style"] == source_style)
            & (edges["target_style"] == target_style)
        ].copy()

        sub["genre_pair"] = sub["source_genre"].astype(str) + " → " + sub["target_genre"].astype(str)

        counts = sub["genre_pair"].value_counts().reset_index()
        counts.columns = ["genre_pair", "count"]
        counts["share"] = counts["count"] / counts["count"].sum()
        counts["source_style"] = source_style
        counts["target_style"] = target_style
        counts["edge_label"] = f"{display_name(source_style)} → {display_name(target_style)}"

        all_rows.append(counts)

        print(f"\n{source_style} -> {target_style}")
        print(counts.head(TOP_GENRE_PAIRS).round(4))

        plot_df = counts.head(TOP_GENRE_PAIRS).copy()
        plot_df = plot_df.sort_values("share", ascending=True)

        plt.figure(figsize=(8, 4.8))
        plt.barh(plot_df["genre_pair"], plot_df["share"])
        plt.xlabel("Share among top-1 predecessor edges")
        plt.title(f"Genre Composition: {display_name(source_style)} → {display_name(target_style)}")
        plt.tight_layout()

        out_fig = OUT_FIG_DIR / f"genre_composition_{source_style}_to_{target_style}.png"
        plt.savefig(out_fig, dpi=220, bbox_inches="tight")
        plt.close()

        print("Saved:", out_fig)

    all_df = pd.concat(all_rows, axis=0)

    out_csv = OUT_TABLE_DIR / "edge_genre_composition_v1.csv"
    all_df.to_csv(out_csv, index=False)

    print("\nSaved:")
    print(out_csv)


if __name__ == "__main__":
    main()