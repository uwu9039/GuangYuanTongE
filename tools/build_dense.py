"""建 dense 向量索引：`python -m tools.build_dense [--limit N]`

每条条目一段文本：标题。别名。关键词。部门 + 正文摘要（<=220 字）。
39,274 条 ≈ 900 万字 ≈ 6M token，约 3 元、5～8 分钟；按 sha256 缓存，重跑不重复付费。

产物 `retrieval/artifacts/dense/`：vectors.npy(float32 N×1024) + ids.json + manifest.json。
索引是派生数据，不进仓库（见 .gitignore）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from retrieval.build import item_fields, load_items
from retrieval.dense import (DEFAULT_DIM, DEFAULT_MODEL, DENSE_DIR, DenseIndex,
                             VectorCache, embed_texts)

ROOT = Path(__file__).resolve().parents[1]
BODY_CHARS = 220


def doc_text(it: dict) -> str:
    f = item_fields(it)
    title = f["title"].strip()
    aliases = [a for a in (it.get("aliases") or []) if isinstance(a, str)][:8]
    kws = [k for k in (it.get("keywords") or []) if isinstance(k, str)][:6]
    body = " ".join(f["body"].split())[:BODY_CHARS]
    parts = [f"标题：{title}"]
    if aliases:
        parts.append("别名：" + " / ".join(aliases))
    if kws:
        parts.append("关键词：" + "、".join(kws))
    if f["meta"].strip():
        parts.append("部门：" + f["meta"].strip())
    if body:
        parts.append("正文：" + body)
    return "。".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DENSE_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--include-archive", action="store_true",
                    help="把 archive_only 的原文摘录也编进去（默认只编可答池，省一半时间）")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = load_items()
    if not args.include_archive:
        # archive_only 是「只入库不参与问答路由」的原文摘录，不当答案，dense 不编它们
        items = [it for it in items if not it.get("archive_only")]
    if args.limit:
        items = items[: args.limit]
    ids = [str(it.get("id")) for it in items]
    texts = [doc_text(it) for it in items]
    print(f"条目 {len(texts)} 条，平均 {sum(map(len, texts))/max(len(texts),1):.0f} 字")

    t0 = time.time()
    cache = VectorCache(out / "cache.sqlite")
    last = [0.0]

    def progress(done: int, total: int) -> None:
        now = time.time()
        if now - last[0] > 20 or done == total:
            last[0] = now
            print(f"  编码 {done}/{total}  ({now - t0:.0f}s)", flush=True)

    mat = embed_texts(texts, text_type="document", model=args.model, dim=args.dim,
                      workers=args.workers, cache=cache, on_progress=progress)
    np.save(out / "vectors.npy", mat.astype(np.float32))
    (out / "ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    manifest = {"model": args.model, "dim": args.dim, "n": len(ids),
                "text_type": "document", "body_chars": BODY_CHARS,
                "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                "norm": float(np.linalg.norm(mat, axis=1).mean())}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    idx = DenseIndex.load(out)
    print(f"完成 {manifest}  ({time.time()-t0:.0f}s) -> {out}")
    # 冒烟：查一句看看最近的中文条目
    from retrieval.dense import embed_batch

    qv = embed_batch(["饭卡丢了怎么办"], text_type="query", model=args.model, dim=args.dim)[0]
    print("冒烟 top5:", [(i, round(s, 3)) for i, s in idx.search(qv, 5)])


if __name__ == "__main__":
    main()
