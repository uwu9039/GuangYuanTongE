"""拒答阈值扫描：在「不误拒正例」的前提下，找能把负例全拒掉的阈值。

    python -m _eval.gate_sweep
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from retrieval.engine import Engine
from retrieval.gate import (context_features, other_schools_hit, out_of_scope_hit,
                            personal_hit)
from retrieval.text import query_text

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "_eval" / "sets"


def load(name: str) -> list[dict]:
    return [json.loads(ln) for ln in (SETS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def main() -> None:
    eng = Engine.load()
    data = {}
    for name in ("presets", "paraphrase", "negatives", "negatives_hard"):
        rows = []
        for r in load(name):
            core = query_text(r["q"])
            if personal_hit(core):
                rows.append({"rule": "personal", "q": r["q"], "f": None})
                continue
            if other_schools_hit(core):
                rows.append({"rule": "other_school", "q": r["q"], "f": None})
                continue
            if out_of_scope_hit(core):
                rows.append({"rule": "out_of_scope", "q": r["q"], "f": None})
                continue
            ctx, _fs, cand, X = eng.retrieve(core)
            if not cand:
                rows.append({"rule": "empty", "q": r["q"], "f": None})
                continue
            logits = eng.ranker.score(X)
            order = np.argsort(-logits)
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits[order], -30, 30)))
            f = context_features(eng.ix, eng.cache, ctx, X[order[0]], probs[:5])
            rows.append({"rule": "", "q": r["q"], "f": f, "len": len(core)})
        data[name] = rows
        rules = {}
        for r in rows:
            rules[r["rule"] or "none"] = rules.get(r["rule"] or "none", 0) + 1
        print(f"[{name}] n={len(rows)} 规则命中 {rules}")

    print("\n扫阈值（只看没被规则拦下的那些）：")
    print("cov<  prob<   presets误拒  para误拒  neg拒  hard拒")
    for cov_thr in (0.04, 0.08, 0.12, 0.18, 0.22):
        for prob_thr in (0.04, 0.08, 0.12, 0.16):
            stats = {}
            for name in data:
                refused = 0
                for r in data[name]:
                    if r["rule"]:
                        refused += 1
                        continue
                    f = r["f"]
                    if f[5] < 0.45 or f[7] * max(r["len"], 1) >= 3:
                        refused += 1
                    elif f[2] < cov_thr and f[3] < 0.5 and f[4] < 1:
                        refused += 1
                    elif f[0] < prob_thr:
                        refused += 1
                stats[name] = refused / len(data[name])
            print(f"{cov_thr:4.2f} {prob_thr:5.2f}   {stats['presets']:7.3f}   "
                  f"{stats['paraphrase']:7.3f}  {stats['negatives']:5.3f}  {stats['negatives_hard']:5.3f}")

    # 负例里漏网的
    print("\nstrict 负例里还没被拦下的：")
    for r in data["negatives"]:
        if r["rule"]:
            continue
        f = r["f"]
        if not (f[5] < 0.45 or f[7] * max(r["len"], 1) >= 3 or
                (f[2] < 0.12 and f[3] < 0.5 and f[4] < 1) or f[0] < 0.12):
            print(f"   cov={f[2]:.2f} prob={f[0]:.2f} exact={f[3]:.0f} corpus={f[5]:.2f}  {r['q']}")
    print("\nparaphrase 里会被误拒的：")
    for r in data["paraphrase"]:
        if r["rule"]:
            if r["rule"] not in ("empty",):
                print(f"   [规则 {r['rule']}] {r['q']}")
            continue
        f = r["f"]
        if f[5] < 0.45 or f[7] * max(r["len"], 1) >= 3 or \
                (f[2] < 0.12 and f[3] < 0.5 and f[4] < 1) or f[0] < 0.12:
            print(f"   cov={f[2]:.2f} prob={f[0]:.2f}  {r['q']}")


if __name__ == "__main__":
    main()
