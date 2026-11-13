"""答案完整性评测：命中条目之后，条目里有的信息是不是都进了答复。

    python -m _eval.eval_answers

对每条 presets 的 (问法, 条目)，用 `Engine.search(with_answer=True)` 装配答案，检查
条目里每个非空字段（步骤 / 材料 / 费用 / 地点 / 时间 / 时限 / 窗口 / 注意事项 /
联系方式 / 适用对象 / 渠道 / 出处）是否真的出现在答复文本里。
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from retrieval import Engine
from retrieval.compose import build as compose_build

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "_eval" / "sets"

FACETS = ("steps", "materials", "fee", "location", "hours", "duration", "window",
          "pitfalls", "contact", "audience", "channel", "source")


def item_facets(item: dict) -> dict[str, bool]:
    out = {}
    for k in FACETS:
        v = item.get(k)
        if isinstance(v, list):
            out[k] = bool(v)
        elif isinstance(v, dict):
            out[k] = bool(v.get("value") or v.get("name"))
        else:
            out[k] = bool(v) if isinstance(v, str) else False
    ans = item.get("answer")
    out["answer"] = bool(ans) if isinstance(ans, str) else False
    return out


LABELS = {"fee": "费用", "location": "办理地点", "hours": "办理时间", "window": "窗口",
          "duration": "办结时限", "audience": "适用对象", "channel": "办理渠道",
          "contact": "联系方式"}


def main() -> None:
    eng = Engine.load()
    rows = [json.loads(ln) for ln in (SETS / "presets.jsonl").read_text(encoding="utf-8").splitlines()]
    total_facets = covered = 0
    full = 0
    bad: list[tuple[str, str, list[str]]] = []
    for r in rows:
        item = eng.items.get(r["gold"][0], {})
        have = item_facets(item)
        # 只评「装配」这一层：就用 gold 条目装配，不受召回/排序误差影响
        doc = SimpleNamespace(id=r["gold"][0], title=item.get("title", ""))
        fake = SimpleNamespace(hits=[SimpleNamespace(doc=doc, item=item, id=r["gold"][0],
                                                    title=item.get("title", ""), prob=0.9)])
        ans = compose_build(fake, eng.items)
        labels = {f.label for f in (ans.facets if ans else [])}
        missing = []
        for k, present in have.items():
            if not present:
                continue
            total_facets += 1
            if k == "steps":
                ok = bool(ans and ans.steps)
            elif k == "materials":
                ok = bool(ans and ans.materials)
            elif k == "pitfalls":
                ok = bool(ans and ans.pitfalls)
            elif k == "source":
                ok = bool(ans and ans.sources)
            elif k == "answer":
                ok = bool(ans and ans.conclusion)
            else:
                ok = LABELS[k] in labels
            if ok:
                covered += 1
            else:
                missing.append(k)
        if missing:
            bad.append((r["q"], r["gold"][0], missing))
        else:
            full += 1
    n = len(rows)
    print(f"问法 {n} 条；字段级完整率 {covered}/{total_facets} = {covered/total_facets:.4f}；"
          f"整条无缺失 {full}/{n} = {full/n:.4f}")
    for q, iid, miss in bad[:20]:
        print(f"  {q[:24]:26s} {iid:22s} 缺 {miss}")


if __name__ == "__main__":
    main()
