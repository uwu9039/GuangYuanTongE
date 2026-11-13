"""章条锚定：问句点了「第X章 / 第X节 / 第X条」时，把对得上的那一节抬上来。

为什么需要这一层
----------------
题库里有 20 道题直接点了章条（「选课办法第十四条选课的限制条件有哪些」）。
`第X条` 在 531 份文档里命中 150 处以上，光按号加分等于给几百个条目一起加，
排序器仍然得自己猜是哪一份；猜错时 top1 会落进同类制度文件（甚至是别校的办法）。

这里的做法是把「号」和「这一节讲什么」两条一起用：

1. `sections_index.chapter_refs` 解析问句里的章条号；
2. `sections_index.numbered` 取出这些号命中的**每一节**（文档标题 + 节名 + 该节条目）；
3. 用**主题串**（`text.topic_text`，剥掉问句框架词、也剥掉问句里那个章条号本身）
   分别和「文档标题」「节名 + 该节条目标题」算二元组覆盖；
4. 覆盖高的那一节下的条目按比例加分，其余不动。

口径是**加分不改答案形态**：它只把「用户点名的那一条」往前抬，不把答案换成条款
碎片——「转专业需要什么条件」这类没点章条的问题，这一层完全不介入。
"""
from __future__ import annotations

import re
from typing import Any

from .text import bigrams, core_title, topic_text

# 问句里的章条号本体：算主题时要去掉，否则每个「第七条」都会共享「第七」「七条」
_REF = re.compile(r"第[一二三四五六七八九十百零〇\d]+[章节条款编篇]")
# 一节的「讲什么」= 节名 + 这一节下的条目名
MAX_SECTIONS = 600
MIN_SCORE = 0.15
MAX_BONUS = 4.0


def _grams(s: str) -> set[str]:
    s = topic_text(s)
    return set(bigrams(s)) if len(s) >= 2 else set()


def _cover(qg: set[str], dg: set[str]) -> float:
    if not qg:
        return 0.0
    return len(qg & dg) / len(qg)


def item_bonus(engine: Any, ctx: Any) -> dict[str, float]:
    """{item_id: 加分}；问句没点章条、或章节索引不可用时返回 {}（完全等于没接这层）。"""
    core = getattr(ctx, "core", "") or ""
    if "第" not in core:
        return {}
    try:
        import sections_index as si
    except Exception:  # noqa: BLE001  章节索引不在也不能拖垮检索
        return {}
    refs = si.chapter_refs(core)
    if not refs:
        return {}
    secs = si.numbered(refs)
    if not secs:
        return {}

    q = _REF.sub("", ctx.topic or core)
    qg = _grams(q)
    if not qg:
        return {}

    ix = engine.ix
    doc_of = engine._doc_of
    out: dict[str, float] = {}
    for s in secs[:MAX_SECTIONS]:
        doc_t = _grams(core_title(s.get("doc_title") or ""))
        body = [s.get("heading") or ""]
        for iid in s.get("items") or ():
            d = doc_of.get(iid)
            if d is not None:
                body.append(ix.docs[d].title)
        sec_t = _grams("".join(body))
        score = 0.4 * _cover(qg, doc_t) + 0.6 * _cover(qg, sec_t)
        if score < MIN_SCORE:
            continue
        for iid in s.get("items") or ():
            if score > out.get(iid, 0.0):
                out[iid] = score
    if not out:
        return {}
    return {k: MAX_BONUS * min(v, 1.0) for k, v in out.items()}
