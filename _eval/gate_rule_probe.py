"""拒答规则归因：每条问句被哪条规则拦下，用来标定阈值。

    python -m _eval.gate_rule_probe
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from retrieval.engine import Engine
from retrieval.gate import context_features, other_schools_hit, out_of_scope_hit, personal_hit
from retrieval.text import query_text

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "_eval" / "sets"


def load(name: str) -> list[dict]:
    p = SETS / f"{name}.jsonl"
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    eng = Engine.load()
    sets = {n: load(n) for n in ("presets", "paraphrase", "negatives", "negatives_hard")}
    for name, rows in sets.items():
        cats: Counter[str] = Counter()
        rule_hits: Counter[str] = Counter()
        vals = []
        for r in rows:
            res = eng.search(r["q"], k=5)
            core = query_text(r["q"])
            if personal_hit(core):
                rule_hits["personal"] += 1
            elif other_schools_hit(core):
                rule_hits["other_school"] += 1
            elif out_of_scope_hit(core):
                rule_hits["out_of_scope"] += 1
            else:
                ctx, _fs, cand, X = eng.retrieve(core)
                if not cand:
                    rule_hits["empty"] += 1
                    continue
                logits = eng.ranker.score(X)
                order = __import__("numpy").argsort(-logits)
                probs = 1.0 / (1.0 + __import__("numpy").exp(-__import__("numpy").clip(logits[order], -30, 30)))
                f = context_features(eng.ix, eng.cache, ctx, X[order[0]], probs[:5])
                vals.append(f)
                if f[5] < 0.55 or f[7] * max(len(core), 1) >= 3:
                    rule_hits["off_domain"] += 1
                elif f[2] < 0.22 and f[3] < 0.5 and f[4] < 1:
                    rule_hits["no_match_cov"] += 1
                elif f[0] < 0.18:
                    rule_hits["no_match_prob"] += 1
            cats[res.refused and (res.reason or "?")[:6] or "ok"] += 1
        print(f"\n[{name}] n={len(rows)} 拒答 {sum(1 for r in rows if eng.search(r['q'], k=1).refused)}")
        print("  规则命中：", dict(rule_hits))
        if vals:
            import numpy as np
            a = np.asarray(vals, dtype=float)
            for i, key in enumerate(("top_prob", "margin", "cov_short", "exact", "phrase_hit",
                                     "cov_corpus", "oov_ratio", "longest_oov")):
                print(f"    {key:12s} p05={np.percentile(a[:,i],5):6.3f} p25={np.percentile(a[:,i],25):6.3f}"
                      f" med={np.median(a[:,i]):6.3f} p75={np.percentile(a[:,i],75):6.3f}"
                      f" p95={np.percentile(a[:,i],95):6.3f}")


if __name__ == "__main__":
    main()
