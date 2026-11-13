"""快速探针：只靠 BM25F（不训练）在 presets 上看命中率，用来判断候选召回够不够。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from retrieval.index import FIELD_W, FIELDS, Index
from retrieval.text import Lexicon, bigrams, query_text

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "retrieval" / "artifacts"


def term_groups(q: str, lex: Lexicon) -> dict[str, dict[str, float]]:
    core = query_text(q)
    grams = {"G:" + b: 1.0 for b in dict.fromkeys(bigrams(core))}
    phrases = {"P:" + w: 1.6 for w in dict.fromkeys(lex.match(core)) if len(w) >= 3}
    return {"g": grams, "p": phrases}


def main() -> None:
    ix = Index.load(ART)
    lex = Lexicon(json.loads((ART / "lexicon.json").read_text(encoding="utf-8")))
    d = json.loads((ROOT / "presets.json").read_text(encoding="utf-8"))
    rows = [(q, iid) for iid, qs in d["coverage"].items() for q in qs]
    idx_of = {doc.id: n for n, doc in enumerate(ix.docs)}

    t0 = time.time()
    hit1 = hit5 = hit10 = 0
    misses = []
    from collections import Counter
    ranks = Counter()
    for q, gold in rows:
        fs = ix.field_scores(term_groups(q, lex))
        total = np.zeros(ix.n, dtype=np.float32)
        for chan, per in fs.items():
            cw = 1.0 if chan == "g" else 1.25
            for f in FIELDS:
                total += cw * FIELD_W[f] * per[f]
        order = ix.top_k(total, 10)
        ids = [ix.docs[i].id for i in order]
        r = ids.index(gold) if gold in ids else None
        if r == 0:
            hit1 += 1
        if r is not None and r < 5:
            hit5 += 1
        if r is not None:
            hit10 += 1
            ranks[r] += 1
        else:
            misses.append((q, gold, ids[:5]))
    n = len(rows)
    print(f"n={n} hit@1={hit1/n:.3f} hit@5={hit5/n:.3f} hit@10={hit10/n:.3f} "
          f"{(time.time()-t0)/n*1000:.1f}ms/q")
    print("rank dist", dict(sorted(ranks.items())))
    print(f"misses {len(misses)}")
    for m in misses[:25]:
        print("  ", m[0], "| gold", m[1], "| top", m[2][:3])


if __name__ == "__main__":
    main()
