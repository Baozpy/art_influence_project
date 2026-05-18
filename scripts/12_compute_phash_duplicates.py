from pathlib import Path
import pandas as pd
from datasets import load_dataset
from PIL import Image
import imagehash
from tqdm import tqdm

META_PATH = Path("data/processed/embeddings/clip_renaissance_subset_metadata_v1.csv")
OUT_PATH = Path("data/processed/embeddings/clip_renaissance_subset_metadata_with_phash_v1.csv")

HASH_SIZE = 8
DUP_THRESHOLD = 6


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main():
    meta = pd.read_csv(META_PATH)
    ds = load_dataset("huggan/wikiart", split="train")

    print("Metadata shape:", meta.shape)

    hashes = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Computing pHash"):
        idx = int(row["idx"])
        img = ds[idx]["image"].convert("RGB")
        h = imagehash.phash(img, hash_size=HASH_SIZE)
        hashes.append(h)

    uf = UnionFind(len(hashes))

    # O(n^2) for 7200 is okay-ish, but still may take a little time
    print("Finding near-duplicates...")
    for i in tqdm(range(len(hashes))):
        hi = hashes[i]
        for j in range(i + 1, len(hashes)):
            dist = hi - hashes[j]
            if dist <= DUP_THRESHOLD:
                uf.union(i, j)

    groups = [uf.find(i) for i in range(len(hashes))]

    meta["phash"] = [str(h) for h in hashes]
    meta["duplicate_group"] = groups

    # compress group ids
    unique_groups = {g: k for k, g in enumerate(sorted(set(groups)))}
    meta["duplicate_group"] = meta["duplicate_group"].map(unique_groups)

    group_sizes = meta["duplicate_group"].value_counts()
    meta["duplicate_group_size"] = meta["duplicate_group"].map(group_sizes)

    meta.to_csv(OUT_PATH, index=False)

    print("\nSaved:")
    print(OUT_PATH)

    print("\nDuplicate group size distribution:")
    print(group_sizes.value_counts().sort_index())

    print("\nLargest duplicate groups:")
    print(group_sizes.head(20))


if __name__ == "__main__":
    main()