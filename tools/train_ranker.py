"""训练检索排序器。

    python -m tools.train_ranker [--rounds 2] [--max-queries 20000]

训练信号完全来自条目库自身：每条人工/自动条目都带 `aliases`，那些就是真实用户说法。
正例 = (别名 / 标题 → 本条目)，负例 = 同一查询的 BM25 前列其它条目（硬负例）+ 少量随机。
人工整理的服务条目按 HAND_WEIGHT 放大采样——线上绝大多数问题问的是它们。

不拿评测集（presets / 手写改写）训练：训练里出现的串会从评测集剔掉，避免背题。
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

from retrieval.engine import Engine            # noqa: E402
from retrieval.features import FEATURE_NAMES   # noqa: E402
from retrieval.gate import Gate                # noqa: E402
from retrieval.ranker import Ranker            # noqa: E402
from retrieval.text import query_text          # noqa: E402

ART = ROOT / "retrieval" / "artifacts"
MODEL = ROOT / "retrieval" / "model"
SETS = ROOT / "_eval" / "sets"

N_HARD = 24
N_RANDOM = 4
HAND_WEIGHT = 6


def eval_strings() -> set[str]:
    out: set[str] = set()
    p = ROOT / "presets.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for qs in d.get("coverage", {}).values():
            out.update(q.strip() for q in qs)
    pq = SETS / "paraphrase.tsv"
    if pq.exists():
        for ln in pq.read_text(encoding="utf-8").splitlines():
            if "\t" in ln and not ln.startswith("#"):
                out.add(ln.split("\t", 1)[1].strip())
    return out


def build_queries(eng: Engine, max_queries: int, rng: random.Random) -> list[tuple[str, int]]:
    blocked = {query_text(q) for q in eval_strings()}
    twin_of = getattr(eng.cache, "twin_of", None)
    buckets: dict[str, list[tuple[str, int]]] = {"hand": [], "other": []}
    for di, doc in enumerate(eng.ix.docs):
        bucket = buckets["hand"] if doc.kind == "hand" else buckets["other"]
        # 同名条目（人工 + 自动/条款多份）以人工版为「规范答案」：
        # 用户问「学生电子交通卡怎么申领」时，要的是那条服务，不是同名原文碎片
        gold = int(twin_of[di]) if (twin_of is not None and doc.kind != "hand"
                                    and twin_of[di] >= 0) else di
        for cand in (*doc.aliases, doc.title):
            core = query_text(cand)
            if 2 <= len(core) and core not in blocked:
                bucket.append((core, gold))
    hand = buckets["hand"] * HAND_WEIGHT
    other = buckets["other"]
    rng.shuffle(hand)
    rng.shuffle(other)
    quota_hand = int(max_queries * 0.45)
    queries = hand[:quota_hand] + other[: max(max_queries - quota_hand, 0)]
    rng.shuffle(queries)
    return queries


def build_pairs(eng: Engine, max_queries: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    queries = build_queries(eng, max_queries, rng)
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    found = 0
    for n, (core, gold) in enumerate(queries, 1):
        _ctx, _fs, cand, X = eng.retrieve(core)
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
        if n % 4000 == 0:
            print(f"    {n}/{len(queries)} 查询，命中正例 {found}", flush=True)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return X, y, {"queries": len(queries), "usable": found, "pairs": int(len(y)),
                  "positives": int(y.sum()), "candidate_recall": round(found / max(len(queries), 1), 4)}


def load_set(name: str) -> list[dict]:
    if name == "paraphrase":
        rows = []
        for ln in (SETS / "paraphrase.tsv").read_text(encoding="utf-8").splitlines():
            if "\t" in ln and not ln.startswith("#"):
                iid, q = ln.split("\t", 1)
                rows.append({"q": q.strip(), "gold": [iid.strip()]})
        return rows
    return [json.loads(ln) for ln in (SETS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def evaluate(eng: Engine, k: int = 5) -> dict[str, dict]:
    out = {}
    for name in ("presets", "paraphrase", "negatives", "negatives_hard"):
        try:
            rows = load_set(name)
        except FileNotFoundError:
            continue
        h1 = h5 = 0
        refused = 0
        for r in rows:
            res = eng.search(r["q"], k=k)
            ids = [h.id for h in res.hits]
            refused += int(res.refused)
            if r["gold"]:
                pos = next((i for i, x in enumerate(ids) if x in r["gold"]), None)
                h1 += int(pos == 0)
                h5 += int(pos is not None and pos < k)
        d = {"n": len(rows), "refused_rate": round(refused / len(rows), 4)}
        if rows[0]["gold"]:
            d["hit@1"] = round(h1 / len(rows), 4)
            d[f"hit@{k}"] = round(h5 / len(rows), 4)
        out[name] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-queries", type=int, default=20000)
    ap.add_argument("--l2", type=float, default=0.6)
    args = ap.parse_args()
    rng = random.Random(13)
    t0 = time.time()
    eng = Engine.load(with_ranker=False)
    print(f"索引 {eng.ix.stats}", flush=True)
    for rnd in range(args.rounds):
        X, y, info = build_pairs(eng, args.max_queries, rng)
        print(f"[round {rnd+1}] 训练对 {info}  ({time.time()-t0:.0f}s)", flush=True)
        ranker = Ranker.train(X, y, l2=args.l2, iters=14)
        ranker.save(MODEL / "ranker.json")
        eng.ranker = ranker
        print("  权重前 12：", ranker.top_features(12), flush=True)
        print("  评测：", evaluate(eng), flush=True)
    Gate.default().save(MODEL / "gate.json")
    print(f"完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
