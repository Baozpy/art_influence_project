from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

EDGE_PATH = Path("outputs/tables/style_order_constrained_edges_v1.csv")
OUT_DIR = Path("outputs/figures/edge_cases")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 5

TARGET_EDGES = [
    ("Early_Renaissance", "High_Renaissance"),
    ("Early_Renaissance", "Northern_Renaissance"),
    ("High_Renaissance", "Mannerism_Late_Renaissance"),
    ("Mannerism_Late_Renaissance", "Baroque"),
    ("Baroque", "Rococo"),
]


def display_name(s):
    return s.replace("_", " ")


def safe_label(row, side):
    artist = row.get(f"{side}_artist", "Unknown")
    genre = row.get(f"{side}_genre", "Unknown")
    idx = row.get(f"{side}_idx", "NA")
    return f"idx={idx}\n{artist}\n{genre}"


def main():
    edges = pd.read_csv(EDGE_PATH)
    ds = load_dataset("huggan/wikiart", split="train")

    for source_style, target_style in TARGET_EDGES:
        sub = edges[
            (edges["source_style"] == source_style)
            & (edges["target_style"] == target_style)
        ].copy()

        if len(sub) == 0:
            print(f"No edges found for {source_style} -> {target_style}")
            continue

        sub = sub.sort_values("similarity", ascending=False).head(TOP_N)

        fig, axes = plt.subplots(
            TOP_N,
            2,
            figsize=(7, 3.2 * TOP_N)
        )

        if TOP_N == 1:
            axes = [axes]

        for r, (_, row) in enumerate(sub.iterrows()):
            source_idx = int(row["source_idx"])
            target_idx = int(row["target_idx"])

            source_img = ds[source_idx]["image"].convert("RGB")
            target_img = ds[target_idx]["image"].convert("RGB")

            axes[r][0].imshow(source_img)
            axes[r][0].axis("off")
            axes[r][0].set_title(
                "Source\n" + safe_label(row, "source"),
                fontsize=8
            )

            axes[r][1].imshow(target_img)
            axes[r][1].axis("off")
            axes[r][1].set_title(
                f"Target\nsim={row['similarity']:.3f}\n" + safe_label(row, "target"),
                fontsize=8
            )

        fig.suptitle(
            f"{display_name(source_style)} → {display_name(target_style)}",
            fontsize=15
        )

        plt.tight_layout()

        out_name = f"edge_case_{source_style}_to_{target_style}.png"
        out_path = OUT_DIR / out_name
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()

        print("Saved:", out_path)


if __name__ == "__main__":
    main()