"""统计「两跑都没中」的 30 条里，返回的 top1 与期望条目是不是同一件事。

判据：两条标题有 >=4 字的最长公共子串（保守），或 top1 就是期望条目的父/子（同前缀）。
只用于报告里的定性说明，不作为指标。
"""
from __future__ import annotations

import json
from pathlib import Path

from retrieval import Engine

ROOT = Path(__file__).resolve().parents[1]


def lcs4(a: str, b: str) -> str:
    best = ""
    for i in range(len(a)):
        for L in range(len(best) + 1, min(len(a) - i, 12) + 1):
            seg = a[i:i + L]
            if seg in b:
                best = seg
            else:
                break
    return best


def main() -> None:
    a = json.loads((ROOT / "_eval/out/bank_sparse.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "_eval/out/bank_dense.json").read_text(encoding="utf-8"))
    ra = {r["q"]: r for r in a["rows"]}
    rb = {r["q"]: r for r in b["rows"]}
    eng = Engine.load()
    title = lambda i: eng.items.get(i, {}).get("title") or i
    miss = [q for q, r in ra.items()
            if r["kind"] in ("item", "section") and not r.get("hit@k")
            and not rb[q].get("hit@k")]
    same = []
    for q in miss:
        r = rb[q]
        exp = r["expect"][0] if r["expect"] else ""
        top = r["top"][0] if r["top"] else ""
        common = lcs4(title(exp), title(top)) if exp else ""
        if len(common) >= 4:
            same.append((q, common, title(top)))
    print(f"两跑都没中 {len(miss)} 条，其中 {len(same)} 条的 top1 与期望条目共享 >=4 字主题串：")
    for q, c, t in same:
        print(f"  [{c}] {q[:40]:42s} -> {t[:30]}")
    print("\n剩下真正跑偏的：")
    for q in miss:
        r = rb[q]
        exp = r["expect"][0] if r["expect"] else ""
        top = r["top"][0] if r["top"] else ""
        if len(lcs4(title(exp), title(top))) < 4:
            print(f"  {q[:44]:46s} 期望 {title(exp)[:22]:24s} 实际 {title(top)[:22]}")


if __name__ == "__main__":
    main()
