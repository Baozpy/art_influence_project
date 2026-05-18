from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

EDGE_PATH = Path("outputs/tables/style_order_constrained_edges_dedup_v1.csv")
OUT_DIR = Path("outputs/figures/edge_cases_diverse")
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
    artist = row.get(f"{side}_artist", "Unknown Artist")
    genre = row.get(f"{side}_genre", "Unknown Genre")
    idx = row.get(f"{side}_idx", "NA")
    return f"idx={idx}\n{artist}\n{genre}"


def select_diverse_edges(sub, top_n=5):
    """
    Select visually strong but less repetitive examples.
    Avoid repeated source images, target images, and repeated source-target pairs.
    """
    sub = sub.sort_values("similarity", ascending=False).copy()

    selected = []
    used_source_idx = set()
    used_target_idx = set()
    used_pairs = set()
    used_source_artist = set()
    used_target_artist = set()

    # first pass: strict diversity
    for _, row in sub.iterrows():
        s_idx = int(row["source_idx"])
        t_idx = int(row["target_idx"])
        s_artist = str(row.get("source_artist", "Unknown Artist"))
        t_artist = str(row.get("target_artist", "Unknown Artist"))
        pair = (s_idx, t_idx)

        if s_idx in used_source_idx:
            continue
        if t_idx in used_target_idx:
            continue
        if pair in used_pairs:
            continue

        # avoid repeated known artists, but do not over-penalize Unknown Artist
        if s_artist != "Unknown Artist" and s_artist in used_source_artist:
            continue
        if t_artist != "Unknown Artist" and t_artist in used_target_artist:
            continue

        selected.append(row)
        used_source_idx.add(s_idx)
        used_target_idx.add(t_idx)
        used_pairs.add(pair)

        if s_artist != "Unknown Artist":
            used_source_artist.add(s_artist)
        if t_artist != "Unknown Artist":
            used_target_artist.add(t_artist)

        if len(selected) >= top_n:
            break

    # second pass: relax artist constraint if not enough
    if len(selected) < top_n:
        for _, row in sub.iterrows():
            s_idx = int(row["source_idx"])
            t_idx = int(row["target_idx"])
            pair = (s_idx, t_idx)

            if s_idx in used_source_idx:
                continue
            if t_idx in used_target_idx:
                continue
            if pair in used_pairs:
                continue

            selected.append(row)
            used_source_idx.add(s_idx)
            used_target_idx.add(t_idx)
            used_pairs.add(pair)

            if len(selected) >= top_n:
                break

    return pd.DataFrame(selected)


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

        sub = select_diverse_edges(sub, top_n=TOP_N)

        print(f"\n{source_style} -> {target_style}")
        print(sub[["source_idx", "target_idx", "similarity", "source_artist", "target_artist", "source_genre", "target_genre"]])

        fig, axes = plt.subplots(
            len(sub),
            2,
            figsize=(7.2, 3.3 * len(sub))
        )

        if len(sub) == 1:
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

        out_name = f"diverse_edge_case_{source_style}_to_{target_style}.png"
        out_path = OUT_DIR / out_name
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()

        print("Saved:", out_path)


if __name__ == "__main__":
    main()