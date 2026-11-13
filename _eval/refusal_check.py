"""拒答回归：一次跑完四套正/负例，确认「该拒的拒了、该答的没误拒」。

    python -m _eval.refusal_check [--dense]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval import Engine

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "_eval" / "sets"


def load(name: str) -> list[dict]:
    return [json.loads(ln) for ln in (SETS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true")
    args = ap.parse_args()
    eng = Engine.load(dense=args.dense)
    print(f"engine dense={eng.use_dense}")
    for name in ("presets", "paraphrase", "negatives", "negatives_hard"):
        rows = load(name)
        refused = [r["q"] for r in rows if eng.search(r["q"], k=1).refused]
        rate = len(refused) / len(rows)
        tag = "拒答" if name.startswith("neg") else "误拒"
        print(f"{name:15s} n={len(rows):4d} {tag} {len(refused):3d} ({rate:.1%})")
        if refused[:6]:
            print("   ", " | ".join(q[:16] for q in refused[:6]))
    # 题库里的应拒题（题库不在仓库里，用 BANK_JSON 指过去；没有就跳过这一段）
    from .bank_path import bank_json

    bank_file = bank_json()
    if not bank_file.exists():
        print(f"\n（题库不在 {bank_file}：设 BANK_JSON 后再跑这一段）")
        return
    bank = json.loads(bank_file.read_text(encoding="utf-8"))
    ref = [r["q"] for rows in bank["sets"].values() for r in rows
           if "拒答" in (r.get("expect") or "")]
    print("\n题库应拒题：")
    for q in dict.fromkeys(ref):
        res = eng.search(q, k=1)
        print(f"  {q[:24]:26s} refused={res.refused} {res.reason[:26]}")


if __name__ == "__main__":
    main()
