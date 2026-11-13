"""评测：新检索器 vs 旧 `app.rank()`。

    python -m _eval.run_eval --engine new --sets presets,paraphrase,negatives
    python -m _eval.run_eval --engine old --sets presets        # 基线

指标：
    hit@1 / hit@5   —— gold 条目是否落在前 1 / 前 5
    MRR@10          —— 平均倒数排名
    回答率          —— 系统判定「可答」的比例（正例集会因此暴露误拒）
    拒答率          —— 负例集被判定「不可答」的比例
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "_eval" / "sets"


def load_set(name: str) -> list[dict[str, Any]]:
    if name == "paraphrase":
        rows = []
        for ln in (SETS / "paraphrase.tsv").read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "\t" not in ln:
                continue
            iid, q = ln.split("\t", 1)
            rows.append({"q": q.strip(), "gold": [iid.strip()], "source": "paraphrase"})
        return rows
    path = SETS / f"{name}.jsonl"
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def evaluate(engine: str, sets: list[str], k: int = 5, answer_check: bool = False) -> dict[str, Any]:
    if engine == "old":
        from _eval import stub_fastapi
        stub_fastapi.install()
        import app as old_app
        def run(q: str):
            ranked = old_app.rank(q)
            ids = [it["id"] for it, _ in ranked[:k]]
            refused = not ranked or ranked[0][1] < old_app.MATCH_THRESHOLD
            return ids, refused, None
    else:
        from retrieval import Engine
        eng = Engine.load()
        def run(q: str):
            res = eng.search(q, k=k, with_answer=answer_check)
            return [h.id for h in res.hits], res.refused, res.answer

    out: dict[str, Any] = {"engine": engine, "k": k, "sets": {}}
    for name in sets:
        rows = load_set(name)
        n = len(rows)
        hits1 = hits5 = 0
        mrr = 0.0
        answered = 0
        refused = 0
        lat: list[float] = []
        misses: list[dict[str, Any]] = []
        over_refused: list[str] = []
        for r in rows:
            t = time.perf_counter()
            ids, ref, ans = run(r["q"])
            dt = (time.perf_counter() - t) * 1000
            lat.append(dt)
            refused += int(ref)
            answered += int(not ref)
            if r["gold"]:
                pos = next((i for i, x in enumerate(ids) if x in r["gold"]), None)
                if pos == 0:
                    hits1 += 1
                if pos is not None and pos < k:
                    hits5 += 1
                if pos is not None and pos < 10:
                    mrr += 1.0 / (pos + 1)
                if pos is None or pos >= k:
                    misses.append({"q": r["q"], "gold": r["gold"], "top": ids[:3]})
                    if ref:
                        over_refused.append(r["q"])
                elif ref and pos == 0:
                    over_refused.append(r["q"])
        metrics = {
            "n": n,
            "hit@1": round(hits1 / n, 4),
            f"hit@{k}": round(hits5 / n, 4),
            "mrr@10": round(mrr / n, 4),
            "answered_rate": round(answered / n, 4),
            "refused_rate": round(refused / n, 4),
            "p50_ms": round(statistics.median(lat), 1),
            "p95_ms": round(sorted(lat)[int(n * 0.95)], 1),
            "misses": misses[:30],
            "over_refused": over_refused[:30],
        }
        out["sets"][name] = metrics
        print(f"\n[{engine}] {name}: n={n} hit@1={metrics['hit@1']} hit@{k}={metrics[f'hit@{k}']} "
              f"mrr={metrics['mrr@10']} refused={metrics['refused_rate']} "
              f"p50={metrics['p50_ms']}ms p95={metrics['p95_ms']}ms")
        if misses:
            print(f"  未命中 {len(misses)} 条，前 8：")
            for m in misses[:8]:
                print(f"    {m['q']}  gold={m['gold']}  top={m['top'][:2]}")
        if over_refused:
            print(f"  误拒 {len(over_refused)} 条，前 8：{over_refused[:8]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="new", choices=["new", "old"])
    ap.add_argument("--sets", default="presets,paraphrase,negatives")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--answer", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    res = evaluate(args.engine, [s for s in args.sets.split(",") if s], args.k, args.answer)
    out = Path(args.out) if args.out else ROOT / "_eval" / "out" / f"eval_{args.engine}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
