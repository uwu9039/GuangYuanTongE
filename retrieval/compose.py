"""答案装配：把命中的条目摊成「一条不缺」的答复。

要求是「回答全面」，所以这里不是取一条最相关就完事：

* 条目的每个有内容的字段都要出现在答复里（步骤 / 材料 / 费用 / 地点 / 时间 / 时限 /
  窗口 / 注意事项 / 联系方式 / 适用人群 / 渠道）；
* 每个字段尽量挂上它自己的原文引文（`item["evidence"]`），便于核对；
* 同批命中里分值接近的其它条目作为「相关事项」列出，避免「只说了一半」；
* 条目没有的字段不编，`missing` 里如实留空。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# 字段 -> 中文标签。顺序即答复里的出现顺序。
FACET_LABELS: tuple[tuple[str, str], ...] = (
    ("audience", "适用对象"),
    ("channel", "办理渠道"),
    ("fee", "费用"),
    ("location", "办理地点"),
    ("hours", "办理时间"),
    ("window", "窗口"),
    ("duration", "办结时限"),
    ("contact", "联系方式"),
)

_NULLISH = {"", "无", "none", "null", "—", "-"}


@dataclass
class Facet:
    label: str
    text: str
    quote: str = ""
    src: str = ""


@dataclass
class Answer:
    title: str
    item_id: str
    conclusion: str
    steps: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    pitfalls: list[str] = field(default_factory=list)
    facets: list[Facet] = field(default_factory=list)
    quotes: list[dict[str, str]] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    related: list[dict[str, str]] = field(default_factory=list)
    text: str = ""
    filled: dict[str, bool] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def _clean(v: Any) -> str:
    if isinstance(v, (int, float)):
        return str(v)
    if not isinstance(v, str):
        return ""
    v = v.strip()
    return "" if v.lower() in _NULLISH else v


def _step_text(s: dict[str, Any]) -> str:
    what = _clean(s.get("what")) or _clean(s.get("where"))
    where = _clean(s.get("where"))
    cond = s.get("if")
    prefix = ""
    if isinstance(cond, dict) and cond:
        # 条件只写取值（「需要」「生源地贷款」），不把内部的键名（kind / sealed）露给用户
        prefix = "（" + "、".join(str(v) for v in cond.values() if v) + "时）"
    when = _clean(s.get("when"))
    if where and where not in what:
        what = f"{what}（{where}）"
    # 「第 1 步 / 第 2 步」交给列表编号，正文里不再重复
    head = "" if re.fullmatch(r"第\s*\d+\s*步", when or "") else (f"{when}：" if when else "")
    return f"{prefix}{head}{what}"


def _materials(item: dict[str, Any]) -> list[str]:
    out = []
    for m in item.get("materials") or []:
        if isinstance(m, dict):
            t = _clean(m.get("name")) or _clean(m.get("what"))
        else:
            t = _clean(m)
        if t:
            out.append(t)
    return out


def _quotes_for(item: dict[str, Any], key: str) -> tuple[str, str]:
    ev = item.get("evidence")
    if isinstance(ev, dict):
        e = ev.get(key)
        if isinstance(e, dict):
            return _clean(e.get("quote")), _clean(e.get("src"))
    return "", ""


def build(res: Any, items: dict[str, dict[str, Any]]) -> Answer:
    hit = res.hits[0]
    item = hit.item or items.get(hit.id, {})
    title = _clean(item.get("title")) or hit.title
    steps = [_step_text(s) for s in (item.get("steps") or []) if isinstance(s, dict)]
    steps = [s for s in steps if s]
    materials = _materials(item)
    pitfalls = [t for t in (_clean(p) for p in (item.get("pitfalls") or [])) if t]

    ans = _clean(item.get("answer"))
    if ans:
        conclusion = ans
    elif steps:
        # 没有「一句话答案」时不硬造：直接进步骤，避免和第一条步骤重复
        conclusion = f"共 {len(steps)} 个步骤。"
    else:
        conclusion = title

    facets: list[Facet] = []
    filled: dict[str, bool] = {"steps": bool(steps), "materials": bool(materials),
                               "pitfalls": bool(pitfalls), "conclusion": bool(conclusion)}
    for key, label in FACET_LABELS:
        v = item.get(key)
        if isinstance(v, list):
            text = "、".join(t for t in (_clean(x) for x in v) if t)
        else:
            text = _clean(v)
        if not text:
            filled[key] = False
            continue
        filled[key] = True
        quote, src = _quotes_for(item, key)
        facets.append(Facet(label=label, text=text, quote=quote, src=src))

    quotes: list[dict[str, str]] = []
    ev = item.get("evidence")
    if isinstance(ev, dict):
        for key, e in ev.items():
            if isinstance(e, dict) and _clean(e.get("quote")):
                quotes.append({"field": str(key), "quote": _clean(e.get("quote")),
                               "src": _clean(e.get("src"))})

    sources = []
    for s in item.get("source") or []:
        if isinstance(s, dict):
            sources.append({"title": _clean(s.get("title")), "url": _clean(s.get("url")),
                            "checked": _clean(s.get("checked"))})

    related = []
    for h in res.hits[1:4]:
        if h.prob < 0.35:
            continue
        t = _clean((h.item or {}).get("title")) or h.title
        if t and t != title:
            related.append({"id": h.id, "title": t, "score": round(float(h.prob), 3)})

    missing = [label for key, label in FACET_LABELS if not filled.get(key)]
    if not steps:
        missing.insert(0, "办理步骤")

    ans_obj = Answer(title=title, item_id=hit.id, conclusion=conclusion, steps=steps,
                     materials=materials, pitfalls=pitfalls, facets=facets, quotes=quotes,
                     sources=sources, related=related, filled=filled, missing=missing)
    ans_obj.text = render(ans_obj)
    return ans_obj


def render(a: Answer) -> str:
    lines: list[str] = [f"【{a.title}】", a.conclusion]
    if a.steps:
        lines.append("办理步骤：")
        lines.extend(f"  {n}. {s}" for n, s in enumerate(a.steps, 1))
    if a.materials:
        lines.append("需要带的材料：" + "；".join(a.materials))
    for f in a.facets:
        lines.append(f"{f.label}：{f.text}")
    if a.pitfalls:
        lines.append("注意事项：" + "；".join(a.pitfalls))
    if a.related:
        lines.append("相关事项：" + "；".join(r["title"] for r in a.related))
    if a.sources:
        src = "；".join(s["title"] for s in a.sources if s["title"])
        if src:
            lines.append("出处：" + src)
    return "\n".join(lines)
