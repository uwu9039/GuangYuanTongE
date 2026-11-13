"""拒答边界特征：把「该拒 / 该答」几道题的关键维摊在一起看。"""
from __future__ import annotations

import numpy as np

from retrieval import Engine
from retrieval.gate import context_features
from retrieval.text import query_text

CASES = [
    ("卡的事怎么弄", "refuse"),
    ("研究生能换导师吗", "refuse"),
    ("我要办个证明", "refuse"),
    ("校医院能拔智齿吗？要不要转诊？", "answer"),
    ("饿了怎么办", "refuse"),
    ("饭卡丢了怎么办", "answer"),
    ("宿舍能不能用吹风机", "answer"),
    ("图书馆几点关门", "answer"),
    ("转专业要什么条件", "answer"),
    ("毕业了校园卡还能延期吗", "answer"),
    ("学生电子交通卡怎么申领？", "answer"),
    ("推免要求计算机等级证书吗？没过会影响报名吗？", "answer"),
]


def main() -> None:
    eng = Engine.load(dense=True)
    print(f"{'问题':34s} {'应':>3s} {'prob':>5s} {'margin':>6s} {'covS':>5s} "
          f"{'exact':>5s} {'covC':>5s} {'run':>4s} {'len':>3s}")
    for q, want in CASES:
        ctx, _fs, cand, X = eng.retrieve(query_text(q), use_dense=True, raw=q)
        r = eng.ranker_dense or eng.ranker
        logits = r.score(X)
        order = np.argsort(-logits)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits[order], -30, 30)))
        f = context_features(eng.ix, eng.cache, ctx, X[order[0]], probs[:5])
        run = f[7] * max(len(ctx.core), 1)
        print(f"{q[:32]:34s} {want[:1]:>3s} {f[0]:5.2f} {f[1]:6.2f} {f[2]:5.2f} "
              f"{f[3]:5.0f} {f[5]:5.2f} {run:4.1f} {len(ctx.core):3d}")


if __name__ == "__main__":
    main()
