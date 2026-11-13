"""看题库评测里没命中的题：期望什么、实际返回什么。"""
from __future__ import annotations

import json
from pathlib import Path

from retrieval import Engine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_eval" / "out" / "bank_sparse.json"


def main() -> None:
    data = json.loads(OUT.read_text(encoding="utf-8"))
    eng = Engine.load()
    title = lambda i: (eng.items.get(i, {}).get("title") or "?")[:26]
    misses = [r for r in data["rows"] if r["kind"] == "item" and not r.get("hit@k")]
    print(f"item 类未命中 {len(misses)} 条\n")
    for r in misses[:45]:
        print(f"Q {r['q'][:46]}")
        print(f"   期望 {r['expect']} -> {'; '.join(title(i) for i in r['expect'][:2])}")
        print(f"   实际 {[(x, title(x)) for x in r['top'][:3]]}")
    print("\n拒答题：")
    for r in data["rows"]:
        if r["kind"] == "refuse":
            print(f"   refused={r['refused']} {r['q'][:40]} | top={[title(x) for x in r['top'][:2]]}")
    print("\nsoft 里被拒的：")
    for r in data["rows"]:
        if r["kind"] == "soft" and r["refused"]:
            print(f"   {r['q'][:50]} | reason={r.get('reason','')[:40]}")


if __name__ == "__main__":
    main()
