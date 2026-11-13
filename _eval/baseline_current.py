"""基线测量：当前 app.rank() 在 presets 763 条问法上的表现。

用法：
    python -m _eval.baseline_current [--limit N] [--out path]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _eval import stub_fastapi

stub_fastapi.install()

import app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_presets() -> list[tuple[str, str]]:
    d = json.loads((ROOT / "presets.json").read_text(encoding="utf-8"))
    rows: list[tuple[str, str]] = []
    for iid, qs in d["coverage"].items():
        for q in qs:
            rows.append((q, iid))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="_eval/out/baseline_presets.json")
    args = ap.parse_args()

    rows = load_presets()
    if args.limit:
        rows = rows[: args.limit]

    out = []
    t0 = time.time()
    for n, (q, gold) in enumerate(rows, 1):
        t = time.time()
        ranked = app.rank(q)
        dt = (time.time() - t) * 1000
        top = [(it["id"], int(sc)) for it, sc in ranked[:10]]
        hit = next((i for i, (iid, _) in enumerate(top) if iid == gold), None)
        out.append({"q": q, "gold": gold, "top": top, "rank": hit, "ms": dt})
        if n % 20 == 0:
            print(f"{n}/{len(rows)}  elapsed {time.time()-t0:.0f}s  "
                  f"hit@1 {sum(1 for r in out if r['rank']==0)/len(out):.3f}  "
                  f"hit@5 {sum(1 for r in out if r['rank'] is not None and r['rank']<5)/len(out):.3f}",
                  flush=True)

    n = len(out)
    metrics = {
        "n": n,
        "hit@1": sum(1 for r in out if r["rank"] == 0) / n,
        "hit@5": sum(1 for r in out if r["rank"] is not None and r["rank"] < 5) / n,
        "mrr@10": sum(1 for r in out if r["rank"] is not None and r["rank"] < 10) / n
                  if False else sum((1 / (r["rank"] + 1)) if r["rank"] is not None and r["rank"] < 10 else 0 for r in out) / n,
        "no_signal@5": sum(1 for r in out if r["rank"] is None or r["rank"] >= 5) / n,
        "avg_ms": sum(r["ms"] for r in out) / n,
        "p95_ms": sorted(r["ms"] for r in out)[int(n * 0.95)],
    }
    payload = {"metrics": metrics, "rows": out, "elapsed_s": round(time.time() - t0, 1)}
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
