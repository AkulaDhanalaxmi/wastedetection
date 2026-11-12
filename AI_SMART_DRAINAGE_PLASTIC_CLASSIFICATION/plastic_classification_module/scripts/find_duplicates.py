"""Exact (MD5) and near-duplicate (dHash) detection. Clusters are used by the splitter to prevent leakage."""
from __future__ import annotations
import argparse, sys
from collections import defaultdict
import numpy as np
import _path  # noqa: F401
from plastic_common import IMG_EXT, dhash, md5_file, open_rgb, load_yaml, resolve, MODULE_ROOT


class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def near_dup_pairs(hashes: list[int], max_hamming: int):
    """All index pairs with Hamming distance <= max_hamming (vectorised, 64-bit hashes)."""
    h = np.array(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(h) - 1):
        x = (h[i + 1:] ^ h[i]).view(np.uint8).reshape(-1, 8)
        d = np.unpackbits(x, axis=1).sum(axis=1)
        for j in np.nonzero(d <= max_hamming)[0]:
            pairs.append((i, i + 1 + int(j)))
    return pairs


def analyse(paths, hash_size=8, max_hamming=4):
    md5s, hashes, ok = [], [], []
    for p in paths:
        try:
            im = open_rgb(p); hashes.append(dhash(im, hash_size)); md5s.append(md5_file(p)); ok.append(p)
        except Exception:
            continue
    exact = defaultdict(list)
    for p, m in zip(ok, md5s): exact[m].append(p)
    uf = UnionFind(len(ok))
    for a, b in near_dup_pairs(hashes, max_hamming): uf.union(a, b)
    return ok, md5s, hashes, [uf.find(i) for i in range(len(ok))], {m: v for m, v in exact.items() if len(v) > 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None); ap.add_argument("--config", default="config/dataset_plastic.yaml")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    root = resolve(a.dir or cfg["paths"]["images_dir"])
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    ok, md5s, _, cl, exact = analyse(paths, cfg["dedupe"]["hash_size"], cfg["dedupe"]["max_hamming"])
    sizes = defaultdict(int)
    for c in cl: sizes[c] += 1
    print(f"images={len(ok)} exact-duplicate sets={len(exact)} near-duplicate clusters(size>1)={sum(1 for s in sizes.values() if s>1)}")
    for m, v in list(exact.items())[:20]: print("  exact:", [str(x.relative_to(MODULE_ROOT)) if MODULE_ROOT in x.parents else str(x) for x in v])

if __name__ == "__main__":
    main()
