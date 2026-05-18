from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EMB_PATH = Path("data/processed/embeddings/clip_renaissance_subset_v1.npy")
META_PATH = Path("data/processed/embeddings/clip_renaissance_subset_metadata_v1.csv")

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cosine_sim_matrix(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


def main():
    emb = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)

    print("Embedding shape:", emb.shape)
    print("Metadata shape:", meta.shape)

    styles = sorted(meta["style_name"].dropna().unique().tolist())
    print("\nStyles in subset:")
    print(styles)

    style_centroids = []
    style_names = []

    for style in styles:
        mask = meta["style_name"] == style
        style_emb = emb[mask.values]
        centroid = style_emb.mean(axis=0)
        style_centroids.append(centroid)
        style_names.append(style)

        print(f"{style}: {style_emb.shape[0]} images")

    style_centroids = np.stack(style_centroids, axis=0)
    sim = cosine_sim_matrix(style_centroids)

    sim_df = pd.DataFrame(sim, index=style_names, columns=style_names)

    out_csv = Path("outputs/tables/style_similarity_matrix_v1.csv")
    sim_df.to_csv(out_csv)

    plt.figure(figsize=(9, 7))
    plt.imshow(sim, interpolation="nearest")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(len(style_names)), style_names, rotation=45, ha="right")
    plt.yticks(range(len(style_names)), style_names)
    plt.title("Style-to-Style Similarity Matrix (CLIP Centroids)")
    plt.tight_layout()

    out_fig = OUT_DIR / "style_similarity_matrix_v1.png"
    plt.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(out_csv)
    print(out_fig)

    print("\nSimilarity matrix:")
    print(sim_df.round(4))


if __name__ == "__main__":
    main()