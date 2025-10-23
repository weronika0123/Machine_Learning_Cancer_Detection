#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import re
from typing import List, Tuple
import numpy as np
import pandas as pd

BASE_GENE_NAMES_STR = """
ENSG00000000419,ENSG00000000460,ENSG00000000938,ENSG00000001036,ENSG00000001461,ENSG00000001497,ENSG00000001629,ENSG00000001631,ENSG00000002330,ENSG00000002549,ENSG00000002586,ENSG00000002822,ENSG00000002834,ENSG00000003056,ENSG00000003402,ENSG00000003436,ENSG00000003756,ENSG00000004059,ENSG00000004142,ENSG00000004455,ENSG00000004487,ENSG00000004534,ENSG00000004700,ENSG00000004766,ENSG00000004779,ENSG00000004866,ENSG00000004897,ENSG00000005007,ENSG00000005020,ENSG00000005022,ENSG00000005059,ENSG00000005075,ENSG00000005175,ENSG00000005194,ENSG00000005238,ENSG00000005249,ENSG00000005302,ENSG00000005339,ENSG00000005483,ENSG00000005486,ENSG00000005700,ENSG00000005810,ENSG00000005812,ENSG00000005844,ENSG00000005882,ENSG00000005893,ENSG00000005955,ENSG00000005961,ENSG00000006007,ENSG00000006015,ENSG00000006114,ENSG00000006125
""".strip()

def parse_base_names() -> Tuple[List[str], int, int]:
    raw = [x.strip() for x in BASE_GENE_NAMES_STR.split(",") if x.strip()]
    pat = re.compile(r"^ENSG(\d+)$")
    seen = set()
    names: List[str] = []
    pad_len = 0
    max_num = 0
    for x in raw:
        m = pat.match(x)
        if not m or x in seen:
            continue
        seen.add(x)
        names.append(x)
        num = int(m.group(1))
        pad_len = max(pad_len, len(m.group(1)))
        max_num = max(max_num, num)
    if pad_len == 0:
        pad_len = 11
    return names, pad_len, max_num

def choose_feature_names(k: int) -> List[str]:
    base, pad_len, max_num = parse_base_names()
    if k <= len(base):
        return base[:k]
    need = k - len(base)
    extra = []
    curr = max_num + 1
    used = set(base)
    while len(extra) < need:
        cand = f"ENSG{str(curr).zfill(pad_len)}"
        if cand not in used:
            extra.append(cand)
            used.add(cand)
        curr += 1
    return base + extra

def one_hot_splits(n: int, p_train: float, p_val: float, p_test: float, rng: np.random.Generator):
    probs = np.array([p_train, p_val, p_test], dtype=float)
    probs = probs / probs.sum()
    one_hot = rng.multinomial(1, probs, size=n)
    return one_hot[:, 0].astype(int), one_hot[:, 1].astype(int), one_hot[:, 2].astype(int)

def main():
    ap = argparse.ArgumentParser(description="Synthetic RNA dataset generator (cancer last).")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--out", type=str, default="synthetic_rna.csv")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--pos-rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min", dest="vmin", type=float, default=3.14)
    ap.add_argument("--max", dest="vmax", type=float, default=17.0)
    args = ap.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        print("ERROR: --rows and --cols must be > 0.", file=sys.stderr)
        sys.exit(2)
    if args.vmin >= args.vmax:
        print("ERROR: --min must be < --max.", file=sys.stderr)
        sys.exit(2)
    if any(x < 0 for x in (args.train, args.val, args.test)) or (args.train + args.val + args.test) == 0:
        print("ERROR: Split proportions must be valid.", file=sys.stderr)
        sys.exit(2)

    rng = np.random.default_rng(args.seed)

    feature_names = choose_feature_names(args.cols)
    X = rng.uniform(low=args.vmin, high=args.vmax, size=(args.rows, args.cols))
    df = pd.DataFrame(X, columns=feature_names)

    # Generate flags + cancer
    isTr, isVa, isTe = one_hot_splits(args.rows, args.train, args.val, args.test, rng)
    df["isTraining"] = isTr
    df["isValidation"] = isVa
    df["isTest"] = isTe
    df["cancer"] = rng.binomial(1, args.pos_rate, size=args.rows).astype(int)

    # Order: features, then splits, then cancer
    df = df[feature_names + ["isTraining", "isValidation", "isTest", "cancer"]]

    # Validate one-hot
    assert (df[["isTraining", "isValidation", "isTest"]].sum(axis=1) == 1).all(), "Split flags must sum to 1"

    df.to_csv(args.out, index=False)
    print(f"✅ Saved {args.rows} rows × {args.cols} features -> {args.out}")

if __name__ == "__main__":
    main()
