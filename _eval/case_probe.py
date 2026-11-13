"""把几道典型案例的特征摊开，看是哪一维把正确答案压下去的。"""
from __future__ import annotations

import numpy as np

from retrieval import Engine
from retrieval.features import FEATURE_NAMES

CASES = [
    ("图书馆能续借吗", "lib-jieyue"),
    ("统一身份认证账号被锁了申诉", "it-shensu"),
    ("学生电子交通卡怎么申领？", "it-jiaotong"),
    ("捡到一张校园卡交到哪", "xyk-yishi"),
]
SHOW = ("log_bm25_total", "log_bm25_short", "log_g_title", "log_g_body", "title_eq",
        "title_in_q", "alias_eq", "alias_in_q", "cov_short", "cov_body", "kind_hand",
        "kind_auto", "kind_para", "hand_twin", "twin_nonhand", "log_doc_len",
        "dense_score", "dense_rank_recip")


def main() -> None:
    eng = Engine.load(dense=True)
    for q, gold in CASES:
        res = eng.search(q, k=3)
        ctx, _fs, cand, x = eng.retrieve(res.core, use_dense=True, raw=q)
        logits = (eng.ranker_dense or eng.ranker).score(x)
        order = np.argsort(-logits)[:3]
        print(f"\n=== {q}   (期望 {gold}: {eng.items.get(gold, {}).get('title', '?')[:28]})")
        for i in order:
            di = cand[int(i)]
            d = eng.ix.docs[di]
            feats = {n: round(float(x[int(i), FEATURE_NAMES.index(n)]), 2) for n in SHOW}
            print(f"  {d.id:22s} {d.title[:24]:26s} logit={logits[int(i)]:.2f} "
                  f"kind={d.kind} arch={d.archive}")
            print(f"     {feats}")
        if gold in {eng.ix.docs[cand[int(i)]].id for i in order}:
            print("  （gold 在前 3）")
        else:
            di = eng._doc_of.get(gold, -1)
            print(f"  gold 是否在候选池：{di in cand}")


if __name__ == "__main__":
    main()
