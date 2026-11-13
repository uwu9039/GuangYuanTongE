"""旧检索器在全部评测集上的基线（presets 已有，这里补 paraphrase / negatives）。

    python -m _eval.baseline_current --sets paraphrase,negatives,negatives_hard
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from _eval import stub_fastapi

stub_fastapi.install()

import app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "_eval" / "sets"


def load(name: str) -> list[dict]:
    return [json.loads(ln) for ln in (SETS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="paraphrase,negatives,negatives_hard")
    args = ap.parse_args()
    out = {}
    for name in [s for s in args.sets.split(",") if s]:
        rows = load(name)
        h1 = h5 = ref = 0
        lat = []
        misses = []
        for r in rows:
            t = time.perf_counter()
            # 必须走旧实现：app.rank 现在已经是新检索器
            ranked = app._rank_legacy(r["q"])
            lat.append((time.perf_counter() - t) * 1000)
            ids = [it["id"] for it, _ in ranked[:5]]
            refused = (not ranked) or ranked[0][1] < app.MATCH_THRESHOLD
            ref += int(refused)
            if r["gold"]:
                pos = next((i for i, x in enumerate(ids) if x in r["gold"]), None)
                h1 += int(pos == 0)
                h5 += int(pos is not None and pos < 5)
                if pos is None or pos >= 5:
                    misses.append({"q": r["q"], "gold": r["gold"], "top": ids[:3], "score": ranked[0][1] if ranked else 0})
        n = len(rows)
        out[name] = {"n": n, "hit@1": round(h1 / n, 4), "hit@5": round(h5 / n, 4),
                     "refused_rate": round(ref / n, 4),
                     "p50_ms": round(statistics.median(lat), 1),
                     "misses": misses[:30]}
        print(f"[old] {name}: n={n} hit@1={out[name]['hit@1']} hit@5={out[name]['hit@5']} "
              f"refused={out[name]['refused_rate']} p50={out[name]['p50_ms']}ms")
    (ROOT / "_eval" / "out" / "baseline_other.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
