"""建检索索引：`python -m retrieval.build [--out retrieval/artifacts]`

产物：
    artifacts/reach/   倒排索引（.npy + json，可 mmap）
    artifacts/lexicon.json   领域词表（最长匹配用）

三步都是确定性的：只读 `items.json.gz` 与仓库内的 contacts / departments / campus-pois
等静态表，不联网、不调模型。
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .index import DocMeta, Index
from .text import STOP_FRAGMENTS, bigrams, mine_ngrams, norm

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "retrieval" / "artifacts"


# --------------------------------------------------------------------- 读取条目


def load_items() -> list[dict[str, Any]]:
    p = ROOT / "items.json"
    raw = p.read_bytes() if p.exists() else gzip.decompress((ROOT / "items.json.gz").read_bytes())
    return json.loads(raw.decode("utf-8"))["items"]


def _s(x: Any) -> str:
    return x if isinstance(x, str) else ""


def item_kind(it: dict[str, Any]) -> str:
    if it.get("clause"):
        return "clause"
    if it.get("para"):
        return "para"
    return "auto" if it.get("auto") else "hand"


def item_fields(it: dict[str, Any]) -> dict[str, str]:
    """把一条条目摊成 5 个字段的原始文本。"""
    title = _s(it.get("title"))
    aliases = [a for a in (it.get("aliases") or []) if isinstance(a, str)]
    keywords = [k for k in (it.get("keywords") or []) if isinstance(k, str)]

    body: list[str] = []
    ans = it.get("answer")
    if isinstance(ans, str) and ans.strip():
        body.append(ans)
    for s in it.get("steps") or []:
        if isinstance(s, dict):
            body.extend(str(v) for v in s.values() if isinstance(v, (str, int, float)))
        elif isinstance(s, str):
            body.append(s)
    for key in ("fee", "location", "hours", "duration", "window", "note", "advisory",
                "scope_note", "amount"):
        v = it.get(key)
        if isinstance(v, str) and v.strip():
            body.append(v)
    for key in ("materials", "pitfalls", "answer_lines"):
        for x in it.get(key) or []:
            if isinstance(x, dict):
                body.extend(str(v) for v in x.values() if isinstance(v, str))
            elif isinstance(x, str):
                body.append(x)
    c = it.get("contact")
    if isinstance(c, dict):
        body.extend(str(v) for v in c.values() if isinstance(v, str))
    elif isinstance(c, str):
        body.append(c)

    meta: list[str] = []
    for key in ("dept", "channel", "audience"):
        v = it.get(key)
        if isinstance(v, str):
            meta.append(v)
        elif isinstance(v, list):
            meta.extend(str(x) for x in v if isinstance(x, str))

    return {
        "title": title,
        "alias": " ".join(aliases),
        "kw": " ".join(keywords),
        "body": " ".join(x for x in body if x),
        "meta": " ".join(meta),
    }


# ----------------------------------------------------------------------- 词表


def build_lexicon(items: list[dict[str, Any]]) -> list[str]:
    words: set[str] = set()
    shorts: list[str] = []
    for it in items:
        t = norm(_s(it.get("title")))
        if 2 <= len(t) <= 24:
            words.add(t)
            shorts.append(t)
        for a in it.get("aliases") or []:
            a = norm(_s(a))
            if 2 <= len(a) <= 24:
                words.add(a)
                shorts.append(a)
        for k in it.get("keywords") or []:
            k = norm(_s(k))
            if 2 <= len(k) <= 12:
                words.add(k)
    # 高频 n-gram：把「校园卡」「统一身份认证」「推免」这类没有被整条标题覆盖的词捞出来
    words |= mine_ngrams(shorts, lo=2, hi=6, min_freq=5)
    # 静态表里的机构名、地点名
    for name in _static_names():
        n = norm(name)
        if 2 <= len(n) <= 24:
            words.add(n)
    words -= STOP_FRAGMENTS
    return sorted(w for w in words if len(w) >= 2)


def _static_names() -> Iterable[str]:
    for fn, key in (("departments.json", None), ("contacts.json", None),
                    ("sources/campus-pois.json", None)):
        p = ROOT / fn
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        yield from _walk_strings(data, depth=0)


def _walk_strings(obj: Any, depth: int) -> Iterable[str]:
    if depth > 4:
        return
    if isinstance(obj, str):
        if 2 <= len(obj) <= 30:
            yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and 2 <= len(k) <= 30:
                yield k
            yield from _walk_strings(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:5000]:
            yield from _walk_strings(v, depth + 1)


# ------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART))
    args = ap.parse_args()
    out = Path(args.out)
    t0 = time.time()

    items = load_items()
    print(f"[1/5] 条目 {len(items)} 条  {time.time()-t0:.1f}s")

    lexicon = build_lexicon(items)
    print(f"[2/5] 词表 {len(lexicon)} 词  {time.time()-t0:.1f}s")

    from .text import Lexicon

    lex = Lexicon(lexicon)
    docs: list[tuple[DocMeta, dict[str, list[str]]]] = []
    for n, it in enumerate(items, 1):
        raw = item_fields(it)
        toks: dict[str, list[str]] = {}
        for f in ("title", "alias", "kw"):
            txt = norm(raw[f])
            g = ["G:" + b for b in bigrams(txt)]
            p = ["P:" + w for w in lex.match(txt) if len(w) >= 3]
            toks[f] = g + p
        bt = norm(raw["body"])
        toks["body"] = ["G:" + b for b in bigrams(bt)]
        mt = norm(raw["meta"])
        toks["meta"] = ["G:" + b for b in bigrams(mt)]
        docs.append((DocMeta(id=str(it.get("id")), kind=item_kind(it),
                             archive=bool(it.get("archive_only")), dept=_s(it.get("dept")),
                             title=_s(it.get("title")),
                             aliases=tuple(a for a in (it.get("aliases") or []) if isinstance(a, str)),
                             keywords=tuple(k for k in (it.get("keywords") or []) if isinstance(k, str))),
                     toks))
        if n % 10000 == 0:
            print(f"    切词 {n} 条 {time.time()-t0:.1f}s", flush=True)
    print(f"[3/5] 切词完成  {time.time()-t0:.1f}s")

    ix = Index.build(docs)
    ix.save(out)
    print(f"[4/5] 索引 {ix.stats}  {time.time()-t0:.1f}s")

    (out / "lexicon.json").write_text(json.dumps(lexicon, ensure_ascii=False), encoding="utf-8")
    print(f"[5/5] 词表落盘  {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
