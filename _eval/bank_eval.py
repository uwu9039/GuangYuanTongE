"""题库评测：跑委托方给的题库 JSON（位置见 `_eval/bank_path.py`，用 `BANK_JSON` 指过去）。

题库的 `expect` 口径不统一，这里先归一成四类：

* `item`    —— 期望落到某个条目 id（`expect` 是 id，或 `expect=single` 而 id 在 `extra`）
* `section` —— 期望落到某份文档的第 X 节/条（`zhuanzhuanye-2025 · 第三章/第七条`），
                判分看前 k 条里有没有这篇文档的条目
* `refuse`  —— 期望拒答
* `soft`    —— 复合题 / 无固定期望（compound9/30、golden48），只统计「有没有被误拒」

    python -m _eval.bank_eval --engine new            # 不带 embedding
    python -m _eval.bank_eval --engine new --dense    # 带 dense 通道
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from .bank_path import bank_json  # noqa: E402  （放这儿：得先有 ROOT）

BANK = bank_json()
OUT = ROOT / "_eval" / "out"

ITEM_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*$")
SOFT_LABELS = {"plain", "vague", "single", "golden"}


def load_bank() -> list[dict]:
    """按问题去重；同一问题在多个集合出现时合并期望（并集）。"""
    d = json.loads(BANK.read_text(encoding="utf-8"))
    merged: dict[str, dict] = {}
    order: list[str] = []
    for name, rows in d["sets"].items():
        for r in rows:
            q = (r.get("q") or "").strip()
            if not q:
                continue
            if q not in merged:
                merged[q] = {"q": q, "sets": [], "expects": [], "extras": []}
                order.append(q)
            merged[q]["sets"].append(name)
            merged[q]["expects"].append((r.get("expect") or "").strip())
            merged[q]["extras"].append((r.get("extra") or "").strip())
    return [merged[q] for q in order]


def classify(row: dict) -> tuple[str, list[str]]:
    """返回 (类型, 期望 id 列表)。"""
    kinds: list[str] = []
    ids: list[str] = []
    for e, x in zip(row["expects"], row["extras"]):
        if not e:
            kinds.append("soft")
            continue
        if "拒答" in e:
            kinds.append("refuse")
        elif e == "single":
            cand = x.split("|")[0].strip()
            if ITEM_RE.match(cand):
                ids.append(cand)
                kinds.append("item")
            else:
                kinds.append("soft")
        elif "·" in e or e.startswith("第"):
            doc = e.split("·")[0].strip()
            ids.append(doc)
            kinds.append("section")
        elif e.startswith("聚焦="):
            kinds.append("focus")
        elif "子问题" in e or "golden" in e:
            kinds.append("soft")
        elif e in SOFT_LABELS:
            kinds.append("soft")
        elif ITEM_RE.match(e):
            ids.append(e)
            kinds.append("item")
        else:
            kinds.append("soft")
    # 以最严格的口径定类型：有 item 就按 item，其次 section / refuse
    for k in ("item", "section", "refuse", "focus", "soft"):
        if k in kinds:
            return k, ids
    return "soft", ids


def run(engine: str, dense: bool, k: int) -> dict:
    rows = load_bank()
    from retrieval import Engine

    eng = Engine.load(dense=dense) if dense else Engine.load()
    known_items = set(eng.items)
    missing_labels = sorted({i for r in rows for i in classify(r)[1]
                             if classify(r)[0] in ("item",) and i not in known_items})
    # 条目 -> 它落在哪几份文档（章节索引）
    doc_of: dict[str, set[str]] = {}
    try:
        import sections_index as si

        for iid in eng.items:
            for s in si.section_of(iid) or []:
                if s.get("slug"):
                    doc_of.setdefault(iid, set()).add(str(s["slug"]))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 章节索引不可用：{e}")

    per: dict[str, list[dict]] = defaultdict(list)
    t_all = []
    for r in rows:
        kind, ids = classify(r)
        t0 = time.perf_counter()
        res = eng.search(r["q"], k=k)
        dt = (time.perf_counter() - t0) * 1000
        t_all.append(dt)
        top_ids = [h.id for h in res.hits]
        rec = {"q": r["q"], "kind": kind, "expect": ids, "refused": res.refused,
               "reason": res.reason, "top": top_ids, "ms": round(dt, 1),
               "sets": r["sets"]}
        if kind == "item":
            pos = next((i for i, x in enumerate(top_ids) if x in ids), None)
            rec["hit@1"] = pos == 0
            rec["hit@k"] = pos is not None
            rec["rank"] = pos
        elif kind == "section":
            pos = next((i for i, x in enumerate(top_ids) if ids[0] in doc_of.get(x, ())), None)
            rec["hit@k"] = pos is not None
            rec["rank"] = pos
        per[kind].append(rec)

    summary = {}
    for kind, recs in per.items():
        n = len(recs)
        s = {"n": n, "refused": sum(1 for r in recs if r["refused"])}
        if kind in ("item", "section"):
            s["hit@1"] = round(sum(1 for r in recs if r.get("hit@1")) / n, 4)
            s[f"hit@{k}"] = round(sum(1 for r in recs if r.get("hit@k")) / n, 4)
            s["misses"] = [r["q"] for r in recs if not r.get("hit@k")][:40]
        if kind == "refuse":
            s["refused_rate"] = round(s["refused"] / n, 4)
        if kind in ("soft", "focus"):
            s["answer_rate"] = round(1 - s["refused"] / n, 4)
        summary[kind] = s
    summary["_meta"] = {
        "engine": engine, "dense": dense, "total": len(rows),
        "p50_ms": round(statistics.median(t_all), 1),
        "p95_ms": round(sorted(t_all)[int(len(t_all) * 0.95)], 1),
        "missing_label_ids": missing_labels,
        "set_counts": dict(Counter(r["sets"][0] for r in rows)),
    }
    return {"summary": summary, "rows": [r for rs in per.values() for r in rs]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="new")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    res = run(args.engine, args.dense, args.k)
    tag = "dense" if args.dense else "sparse"
    dest = Path(args.out) if args.out else OUT / f"bank_{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    print("->", dest)


if __name__ == "__main__":
    main()
