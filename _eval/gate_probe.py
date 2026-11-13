"""拒答特征分析：把负例和正例的特征摊开，看哪几档能分开。

    python -m _eval.gate_probe
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from retrieval.engine import Engine
from retrieval.gate import GATE_FEATURES, context_features
from retrieval.text import query_text

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "_eval" / "sets"


def feats(eng: Engine, q: str) -> np.ndarray:
    ctx, _fs, cand, X = eng.retrieve(query_text(q))
    if not cand:
        return np.zeros(len(GATE_FEATURES), dtype=np.float32)
    logits = eng.ranker.score(X)
    order = np.argsort(-logits)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits[order], -30, 30)))
    return context_features(eng.ix, eng.cache, ctx, X[order[0]], probs[:5])


def main() -> None:
    eng = Engine.load()
    negs = [json.loads(ln)["q"] for ln in (SETS / "negatives.jsonl").read_text(encoding="utf-8").splitlines()]
    presets = [json.loads(ln)["q"] for ln in (SETS / "presets.jsonl").read_text(encoding="utf-8").splitlines()]
    para = [ln.split("\t", 1)[1].strip() for ln in
            (SETS / "paraphrase.tsv").read_text(encoding="utf-8").splitlines()
            if "\t" in ln and not ln.startswith("#")]
    print("列：top_prob margin cov_short exact phrase cov_corpus oov longest n_above")
    rows = []
    for name, qs in (("neg", negs), ("preset", presets), ("para", para)):
        for q in qs:
            f = feats(eng, q)
            rows.append((name, q, f))
    for name in ("neg", "preset", "para"):
        sub = [f for n, _q, f in rows if n == name]
        a = np.asarray(sub)
        print(f"\n[{name}] n={len(sub)}")
        for i, fn in enumerate(GATE_FEATURES):
            col = a[:, i]
            print(f"  {fn:14s} min={col.min():7.3f} p10={np.percentile(col,10):7.3f} "
                  f"med={np.median(col):7.3f} p90={np.percentile(col,90):7.3f} max={col.max():7.3f}")
    # 逐条看最难的负例
    print("\n负例里最像正例的 20 条（按 cov_short + top_prob 排）：")
    negrows = [(q, f) for n, q, f in rows if n == "neg"]
    negrows.sort(key=lambda x: -(x[1][2] + x[1][0]))
    for q, f in negrows[:20]:
        print(f"  cov_s={f[2]:.2f} prob={f[0]:.2f} exact={f[3]:.0f} corpus={f[5]:.2f} "
              f"oov={f[6]:.2f} run={f[7]:.2f}  {q}")
    print("\n正例里最像负例的 15 条：")
    posrows = [(q, f) for n, q, f in rows if n in ("preset", "para")]
    posrows.sort(key=lambda x: (x[1][2] + x[1][0]))
    for q, f in posrows[:15]:
        print(f"  cov_s={f[2]:.2f} prob={f[0]:.2f} exact={f[3]:.0f} corpus={f[5]:.2f} "
              f"oov={f[6]:.2f} run={f[7]:.2f}  {q}")


if __name__ == "__main__":
    main()
