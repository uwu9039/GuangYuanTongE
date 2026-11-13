"""负例体检：看每条「库里没有」的问题在库里离得最近的是什么。

负例必须真的答不出来，否则「100% 拒答」就是自欺欺人。这里用纯 BM25（不经过排序器）
把每条负例的前 3 名和字面覆盖率打出来，覆盖率高的那几条人工复核后删掉或改写。

    python -m _eval.check_negatives
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from retrieval.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "_eval" / "sets"


def main() -> None:
    eng = Engine.load(with_ranker=False)
    rows = [json.loads(ln) for ln in (SETS / "negatives.jsonl").read_text(encoding="utf-8").splitlines()]
    flagged = []
    for r in rows:
        res = eng.search(r["q"], k=3, explain=True)
        top = res.hits[0]
        cov = float(top.feats[16]) if top.feats is not None else 0.0
        flag = cov >= 0.5
        if flag:
            flagged.append((r["q"], cov, [(h.title[:30], round(float(h.prob), 2)) for h in res.hits[:3]]))
        print(f"{cov:4.2f} {'!' if flag else ' '} {r['q'][:30]:32s} -> {top.title[:34]}")
    print(f"\n覆盖率 >= 0.5 的负例 {len(flagged)} 条（需要复核/删除）：")
    for q, cov, hits in flagged:
        print(f"  {cov:.2f} {q}  ->  {hits}")


if __name__ == "__main__":
    main()
