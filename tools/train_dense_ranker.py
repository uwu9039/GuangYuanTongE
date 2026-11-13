"""训练「带 dense 通道」的排序器：`python -m tools.train_dense_ranker [--max-queries 3000]`

与 `tools/train_ranker.py` 同一套正负例构造（别名/标题当查询，BM25 前列当硬负例），
区别只有两点：

1. 特征里多了 `dense_score` / `dense_rank_recip`（dense 通道召回的分与名次）；
2. 查询向量要真调云端，所以先**批量预编码**（带缓存），再逐条跑特征。

产物 `retrieval/model/ranker_dense.json`；字面模型 `ranker.json` 不动，两套权重共用
同一套特征位（dense 关闭时那两维恒为 0）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.dense import VectorCache, embed_texts  # noqa: E402
from retrieval.engine import Engine                   # noqa: E402
from retrieval.features import FEATURE_NAMES          # noqa: E402
from retrieval.ranker import Ranker                   # noqa: E402
from retrieval.text import query_text                 # noqa: E402
from tools.train_ranker import (N_HARD, N_RANDOM, build_queries,  # noqa: E402
                                eval_strings, evaluate)

ART = ROOT / "retrieval" / "artifacts"
MODEL = ROOT / "retrieval" / "model"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rng = random.Random(17)
    t0 = time.time()

    eng = Engine.load(dense=True)
    if not eng.use_dense:
        raise SystemExit("dense 索引或 dense 权重缺失：先跑 python -m tools.build_dense")
    print(f"dense 索引 {eng.dense.mat.shape} model={eng.dense.model}", flush=True)

    queries = build_queries(eng, args.max_queries, rng)
    texts = [q for q, _ in queries]
    print(f"训练查询 {len(texts)} 条，先批量编码查询向量…", flush=True)
    cache = VectorCache(ART / "dense" / "cache.sqlite")
    vecs = embed_texts(texts, text_type="query", model=eng.dense.model, dim=eng.dense.dim,
                       workers=args.workers, cache=cache,
                       on_progress=lambda d, n: print(f"    {d}/{n}", flush=True) if d % 500 < args.workers else None)
    for t, v in zip(texts, vecs):
        eng._qv[t] = v
    print(f"查询向量就绪 {time.time()-t0:.0f}s", flush=True)

    Xs, ys = [], []
    found = 0
    for n, (core, gold) in enumerate(queries, 1):
        ctx, _fs, cand, X = eng.retrieve(core, use_dense=True, raw=core)
        if not cand or gold not in cand:
            continue
        pos = cand.index(gold)
        found += 1
        rank_feat = X[:, FEATURE_NAMES.index("bm25_rank_recip")]
        order = np.argsort(-rank_feat)
        negs = [int(i) for i in order if i != pos][:N_HARD]
        if len(order) > N_HARD:
            negs += rng.sample([int(i) for i in order[N_HARD:]], min(N_RANDOM, len(order) - N_HARD))
        rows = [pos] + negs
        Xs.append(X[rows])
        ys.append(np.asarray([1.0] + [0.0] * (len(rows) - 1), dtype=np.float32))
        if n % 500 == 0:
            print(f"    {n}/{len(queries)} 命中正例 {found}  {time.time()-t0:.0f}s", flush=True)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    print(f"训练对 {len(y)}（正例 {int(y.sum())}），候选召回 {found/len(queries):.4f}", flush=True)
    ranker = Ranker.train(X, y, l2=0.6, iters=14)
    ranker.save(MODEL / "ranker_dense.json")
    print("权重前 12：", ranker.top_features(12), flush=True)
    eng.ranker_dense = ranker
    eng.use_dense = True
    print("评测：", evaluate(eng), flush=True)
    print(f"完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
