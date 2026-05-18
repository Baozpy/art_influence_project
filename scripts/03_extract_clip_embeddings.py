from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

SUBSET_PATH = Path("data/processed/wikiart_renaissance_subset_v1.csv")
OUT_DIR = Path("data/processed/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = get_device()
    print("Using device:", device)

    subset = pd.read_csv(SUBSET_PATH)
    indices = subset["idx"].tolist()

    print("Subset size:", len(indices))

    print("Loading WikiArt dataset...")
    ds = load_dataset("huggan/wikiart", split="train")

    print("Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    all_embeddings = []
    valid_rows = []

    for start in tqdm(range(0, len(indices), BATCH_SIZE), desc="Extracting CLIP embeddings"):
        batch_indices = indices[start:start + BATCH_SIZE]

        images = []
        real_indices = []

        for idx in batch_indices:
            try:
                img = ds[int(idx)]["image"].convert("RGB")
                images.append(img)
                real_indices.append(idx)
            except Exception as e:
                print(f"Failed to load image idx={idx}: {e}")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        emb = image_features.cpu().numpy()
        all_embeddings.append(emb)

        batch_meta = subset[subset["idx"].isin(real_indices)].copy()
        valid_rows.append(batch_meta)

    embeddings = np.concatenate(all_embeddings, axis=0)
    valid_meta = pd.concat(valid_rows, axis=0).reset_index(drop=True)

    emb_path = OUT_DIR / "clip_renaissance_subset_v1.npy"
    meta_path = OUT_DIR / "clip_renaissance_subset_metadata_v1.csv"

    np.save(emb_path, embeddings)
    valid_meta.to_csv(meta_path, index=False)

    print("\nSaved embeddings:")
    print(emb_path)
    print("Shape:", embeddings.shape)

    print("\nSaved metadata:")
    print(meta_path)
    print("Metadata shape:", valid_meta.shape)


if __name__ == "__main__":
    main()