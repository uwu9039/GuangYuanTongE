"""题库两跑对照：字面 vs dense。

    python -m _eval.bank_compare

读 `_eval/out/bank_sparse.json` 与 `_eval/out/bank_dense.json`，输出：
1. 各类型指标对照；
2. dense 救回来的题（字面没中、dense 中了）；
3. dense 弄坏的题（字面中了、dense 没中）；
4. 两跑都没中的题（**漏**，逐条列出来）；
5. 拒答对照（该拒没拒 / 误拒）。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_eval" / "out"
KINDS = ("item", "section", "refuse", "focus", "soft")


def load(tag: str) -> dict | None:
    p = OUT / f"bank_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def index_rows(data: dict) -> dict[str, dict]:
    return {r["q"]: r for r in data["rows"]}


def fmt(kind: str, s: dict) -> str:
    if kind == "refuse":
        return f"拒答 {s.get('refused_rate', 0):.1%} (n={s['n']})"
    if kind in ("focus", "soft"):
        return f"作答 {s.get('answer_rate', 0):.1%} (n={s['n']})"
    return (f"hit@1 {s.get('hit@1', 0):.1%} / hit@5 {s.get('hit@5', 0):.1%} (n={s['n']})")


def main() -> None:
    a, b = load("sparse"), load("dense")
    if a is None:
        raise SystemExit("缺 bank_sparse.json，先跑 python -m _eval.bank_eval")
    rows_a = index_rows(a)
    rows_b = index_rows(b) if b else {}
    print("=" * 78)
    print(f"{'类型':6s} {'字面（无 embedding）':38s} {'dense':38s}")
    for k in KINDS:
        sa = a["summary"].get(k)
        if not sa:
            continue
        sb = (b["summary"].get(k) if b else None)
        print(f"{k:6s} {fmt(k, sa):38s} {fmt(k, sb) if sb else '—':38s}")
    print("延迟：", a["summary"]["_meta"]["p50_ms"], "ms ->",
          (b["summary"]["_meta"]["p50_ms"] if b else "—"), "ms")

    if not b:
        print("\n（还没有 dense 那一跑）")
        return

    def hit(r: dict) -> bool:
        return bool(r.get("hit@k")) if r["kind"] in ("item", "section") else True

    fixed = [q for q, ra in rows_a.items()
             if ra["kind"] in ("item", "section") and not hit(ra) and hit(rows_b[q])]
    broke = [q for q, ra in rows_a.items()
             if ra["kind"] in ("item", "section") and hit(ra) and not hit(rows_b[q])]
    both_miss = [q for q, ra in rows_a.items()
                 if ra["kind"] in ("item", "section") and not hit(ra) and not hit(rows_b[q])]
    print(f"\ndense 救回 {len(fixed)} 条，弄坏 {len(broke)} 条，两跑都没中 {len(both_miss)} 条")
    if fixed:
        print("\n-- dense 救回（前 20）--")
        for q in fixed[:20]:
            print(f"  ✓ {q[:50]}")
    if broke:
        print("\n-- dense 弄坏（前 20）--")
        for q in broke[:20]:
            print(f"  ✗ {q[:50]}")
    if both_miss:
        print("\n-- 两跑都没中（这就是「漏」）--")
        for q in both_miss:
            ra = rows_a[q]
            print(f"  · {q[:56]}")
            print(f"      期望 {ra['expect']}  字面 top={ra['top'][:2]}  dense top={rows_b[q]['top'][:2]}")

    print("\n-- 拒答对照 --")
    for q, ra in rows_a.items():
        if ra["kind"] != "refuse":
            continue
        rb = rows_b[q]
        print(f"  {q[:28]:30s} 字面 refused={ra['refused']}  dense refused={rb['refused']}")


if __name__ == "__main__":
    main()
