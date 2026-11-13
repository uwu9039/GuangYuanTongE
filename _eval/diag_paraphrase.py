"""诊断：手写改写集里没命中的，到底是「候选没召回」还是「排序排错了」。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from retrieval.engine import Engine
from retrieval.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "_eval" / "sets"


def main() -> None:
    eng = Engine.load()
    rows = []
    for ln in (SETS / "paraphrase.tsv").read_text(encoding="utf-8").splitlines():
        if "\t" in ln and not ln.startswith("#"):
            iid, q = ln.split("\t", 1)
            rows.append((iid.strip(), q.strip()))
    idx_of = {d.id: i for i, d in enumerate(eng.ix.docs)}
    lost_cand = 0
    bad_rank = []
    for iid, q in rows:
        res = eng.search(q, k=10)
        ids = [h.id for h in res.hits]
        gold_di = idx_of.get(iid)
        ctx, _fs, cand, X = eng.retrieve(res.core)
        in_cand = gold_di in cand
        if not in_cand:
            lost_cand += 1
            print(f"[候选丢失] {q}  <=  {iid}")
            continue
        if iid not in ids[:5]:
            pos = ids.index(iid) if iid in ids else None
            bad_rank.append((q, iid, pos, ids[:3]))
    print(f"\n候选未召回 {lost_cand}/{len(rows)}；排序掉出前5 {len(bad_rank)} 条")
    for q, iid, pos, top in bad_rank[:30]:
        gold = eng.items.get(iid, {})
        print(f"  {q}\n     gold={iid} {gold.get('title','')[:28]}  pos={pos}  top={top[:3]}")


if __name__ == "__main__":
    main()
