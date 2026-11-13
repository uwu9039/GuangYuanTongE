"""看这几道为什么被拒/没被拒：拒答决策 + 关键特征。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from retrieval import Engine
from retrieval.gate import (context_features, other_schools_hit, out_of_scope_hit,
                            personal_hit)
from retrieval.text import query_text

SETS = Path("_eval/sets")
QS = ["学费在哪交怎么交", "宿舍里能打水吗", "心情不好想找人聊聊去哪",
      "清华大学校园卡怎么补办", "北京大学图书馆怎么进"]


def main() -> None:
    eng = Engine.load(dense=True)
    negs = [json.loads(ln) for ln in (SETS / "negatives.jsonl").read_text(encoding="utf-8").splitlines()]
    missed = [r["q"] for r in negs if not eng.search(r["q"], k=1).refused]
    print("strict 负例里没拒的：", missed)

    for q in QS:
        res = eng.search(q, k=1)
        ctx, _fs, cand, X = eng.retrieve(query_text(q), use_dense=True, raw=q)
        logits = (eng.ranker_dense or eng.ranker).score(X)
        order = np.argsort(-logits)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits[order], -30, 30)))
        f = context_features(eng.ix, eng.cache, ctx, X[order[0]], probs[:5])
        c = query_text(q)
        print(f"\n{q}  refused={res.refused} cat={res.category if hasattr(res,'category') else res.reason[:14]}")
        print(f"   core={ctx.core!r} len={len(ctx.core)} prob={f[0]:.2f} covS={f[2]:.2f} "
              f"exact={f[3]:.0f} covC={f[5]:.2f} run={f[7]*max(len(ctx.core),1):.1f}")
        print(f"   personal={bool(personal_hit(c))} school={other_schools_hit(c)!r} "
              f"oos={out_of_scope_hit(c)!r}")


if __name__ == "__main__":
    main()
