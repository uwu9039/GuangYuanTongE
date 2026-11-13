"""一次跑全：验收口径的关键指标（P0/P1/P2/G1/G2/G5 + hub 计数）。

    python -m _eval.quick_eval            # 字面
    python -m _eval.quick_eval --dense    # dense

比 `_acceptance.check` 轻：不校验指纹、不跑启动计时，只报我改代码时要盯的数字。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .bank_path import bank_json

ROOT = Path(__file__).resolve().parents[1]
BANK = bank_json()
SETS = ROOT / "_eval" / "sets"

P0 = [("选课什么时候开始？在哪里选？", "选课"), ("课程免修怎么申请？需要什么条件？", "免修"),
      ("毕业审核在哪里查？学分不够怎么办？", "毕业"), ("档案在哪里", "档案")]
HUB_RE = re.compile(r"常见问题|常见问答|汇总|一览|清单|问答|合集|攻略|宝典|指南|faq|FAQ")
ITEM_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*$")
SOFT_LABELS = {"plain", "vague", "single", "golden"}


def load_set(name: str) -> list[dict]:
    if name == "paraphrase":
        return [{"q": q.strip(), "gold": [iid.strip()]}
                for iid, q in (ln.split("\t", 1)
                               for ln in (SETS / "paraphrase.tsv").read_text(encoding="utf-8").splitlines()
                               if "\t" in ln and not ln.startswith("#"))]
    return [json.loads(x) for x in (SETS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def bank_rows() -> list[dict]:
    d = json.loads(BANK.read_text(encoding="utf-8"))
    merged: dict[str, dict] = {}
    order: list[str] = []
    for name, rows in d["sets"].items():
        for r in rows:
            q = (r.get("q") or "").strip()
            if not q:
                continue
            if q not in merged:
                merged[q] = {"q": q, "kinds": [], "ids": []}
                order.append(q)
            e = (r.get("expect") or "").strip()
            x = (r.get("extra") or "").strip()
            if not e:
                merged[q]["kinds"].append("soft")
            elif "拒答" in e:
                merged[q]["kinds"].append("refuse")
            elif e == "single":
                c = x.split("|")[0].strip()
                if ITEM_RE.match(c):
                    merged[q]["kinds"].append("item")
                    merged[q]["ids"].append(c)
                else:
                    merged[q]["kinds"].append("soft")
            elif "·" in e or e.startswith("第"):
                merged[q]["kinds"].append("section")
                merged[q]["ids"].append(e.split("·")[0].strip())
            elif e.startswith("聚焦="):
                merged[q]["kinds"].append("focus")
            elif "子问题" in e or "golden" in e:
                merged[q]["kinds"].append("soft")
            elif e in SOFT_LABELS:
                merged[q]["kinds"].append("soft")
            elif ITEM_RE.match(e):
                merged[q]["kinds"].append("item")
                merged[q]["ids"].append(e)
            else:
                merged[q]["kinds"].append("soft")
    out = []
    for q in order:
        m = merged[q]
        for k in ("item", "section", "refuse", "focus", "soft"):
            if k in m["kinds"]:
                out.append({"q": q, "kind": k, "ids": m["ids"]})
                break
    return out


def _doc_of(eng) -> dict[str, set[str]]:
    import sections_index as si
    doc_of: dict[str, set[str]] = {}
    for iid in eng.items:
        for s in si.section_of(iid) or []:
            if s.get("slug"):
                doc_of.setdefault(iid, set()).add(str(s["slug"]))
    return doc_of


def metrics(eng, k: int = 5, verbose: bool = False) -> dict:
    out: dict = {}

    rows = load_set("presets")
    h1 = h5 = ref = 0
    for r in rows:
        res = eng.search(r["q"], k=k)
        ids = [h.id for h in res.hits]
        ref += int(res.refused)
        pos = next((i for i, x in enumerate(ids) if x in r["gold"]), None)
        h1 += int(pos == 0)
        h5 += int(pos is not None and pos < k)
    out["presets"] = {"hit@1": h1, "hit@5": h5, "refused": ref, "n": len(rows)}

    rows = load_set("paraphrase")
    h5 = 0
    for r in rows:
        res = eng.search(r["q"], k=k)
        ids = [h.id for h in res.hits]
        pos = next((i for i, x in enumerate(ids) if x in r["gold"]), None)
        h5 += int(pos is not None and pos < k)
    out["paraphrase"] = {"hit@5": h5, "n": len(rows)}

    for nm in ("negatives", "negatives_hard"):
        rows = load_set(nm)
        r_ = sum(1 for r in rows if eng.search(r["q"], k=1).refused)
        out[nm] = {"refused": r_, "n": len(rows)}

    p0 = []
    for q, need in P0:
        res = eng.search(q, k=1)
        t = res.best.title if res.best else ""
        p0.append((q, need, need in t, t))
    out["p0"] = p0

    doc_of = _doc_of(eng)
    rows = bank_rows()
    item = [r for r in rows if r["kind"] == "item"]
    sect = [r for r in rows if r["kind"] == "section"]
    soft = [r for r in rows if r["kind"] == "soft"]
    ih1 = ih5 = hub = 0
    hub_qs = []
    for r in item:
        res = eng.search(r["q"], k=k)
        ids = [h.id for h in res.hits]
        pos = next((i for i, x in enumerate(ids) if x in r["ids"]), None)
        ih1 += int(pos == 0)
        ih5 += int(pos is not None and pos < k)
        if res.best is not None and HUB_RE.search(res.best.title):
            hub += 1
            hub_qs.append((r["q"], res.best.id, res.best.title[:34]))
    out["item"] = {"hit@1": ih1, "hit@5": ih5, "n": len(item), "hub": hub, "hub_qs": hub_qs}
    sh5 = 0
    sect_rows = []
    for r in sect:
        res = eng.search(r["q"], k=k)
        pos = next((i for i, h in enumerate(res.hits) if r["ids"][0] in doc_of.get(h.id, ())), None)
        sh5 += int(pos is not None)
        sect_rows.append((r["q"], r["ids"][0], pos))
    out["section"] = {"hit@5": sh5, "n": len(sect), "rows": sect_rows}
    sref = sum(1 for r in soft if eng.search(r["q"], k=1).refused)
    out["soft"] = {"answer": len(soft) - sref, "n": len(soft)}
    return out


def report(m: dict) -> str:
    p, pa = m["presets"], m["paraphrase"]
    it, se, so = m["item"], m["section"], m["soft"]
    lines = [
        f"G1 presets   hit@1 {p['hit@1']}/{p['n']}={p['hit@1']/p['n']:.2%}  "
        f"hit@5 {p['hit@5']}/{p['n']}={p['hit@5']/p['n']:.2%}  误拒 {p['refused']}",
        f"G2 paraphrase hit@5 {pa['hit@5']}/{pa['n']}={pa['hit@5']/pa['n']:.2%}",
        f"negatives   拒答 {m['negatives']['refused']}/{m['negatives']['n']}   "
        f"negatives_hard 拒答 {m['negatives_hard']['refused']}/{m['negatives_hard']['n']}"
        f"={m['negatives_hard']['refused']/m['negatives_hard']['n']:.1%}",
    ]
    for q, need, ok, t in m["p0"]:
        lines.append(f"  P0 {'✓' if ok else '✗'} {q[:24]:26s} top1={t[:30]}")
    lines += [
        f"G5 item      hit@1 {it['hit@1']}/{it['n']}={it['hit@1']/it['n']:.2%}  "
        f"hit@5 {it['hit@5']}/{it['n']}={it['hit@5']/it['n']:.2%}",
        f"P1 hub top1  {it['hub']}/{it['n']}",
        f"P2 section   hit@5 {se['hit@5']}/{se['n']}={se['hit@5']/se['n']:.2%}",
        f"soft 作答率   {so['answer']}/{so['n']}={so['answer']/so['n']:.2%}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    from retrieval import Engine

    eng = Engine.load(dense=args.dense)
    m = metrics(eng, k=args.k)
    print(f"=== engine dense={eng.use_dense}")
    print(report(m))
    if args.verbose:
        print("\n-- hub top1 --")
        for q, i, t in m["item"]["hub_qs"]:
            print(f"   {i:24s} {t:36s} <- {q[:30]}")
        print("\n-- section miss --")
        for q, slug, pos in m["section"]["rows"]:
            if pos is None:
                print(f"   {q[:44]:46s} want={slug}")


if __name__ == "__main__":
    main()
