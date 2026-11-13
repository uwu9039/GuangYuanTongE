"""倒排索引与 BM25F 打分。

索引两类词条（前缀区分，互不干扰）：

* `G:` 字符二元组 —— 所有字段都建。中文短查询里最稳的一档，改一个字还有一半重合。
* `P:` 领域词（>=3 字）—— 只建在标题 / 别名 / 关键词这些短字段上。词表来自条目库
  自身（见 `build.py`），所以「统一身份认证」「微专业」「推免」这类整词只算一个词条。

字段：

    title   标题       —— 用户说的往往就是条目名
    alias   别名       —— 库里自带的口语说法（「饭卡丢了」「卡消磁了」）
    kw      关键词
    body    正文       —— 答案 / 步骤 / 材料 / 费用 / 地点 / 时间 / 注意事项 / 联系方式
    meta    机构元数据 —— 部门、渠道、适用人群（权重最低，只兜底）
    short   标题∪别名∪关键词（去重）—— 只用于算覆盖度，不参与 BM25

存储是 `.npy` 目录（可 mmap），加载 0.2s 级；构建见 `python -m retrieval.build`。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

FIELDS: tuple[str, ...] = ("title", "alias", "kw", "body", "meta")
COV_FIELDS: tuple[str, ...] = ("short", "title", "alias", "kw", "body")
ALL_FIELDS: tuple[str, ...] = (*FIELDS, "short")
FIELD_W: dict[str, float] = {"title": 3.0, "alias": 2.4, "kw": 1.5, "body": 1.0, "meta": 0.35}
FIELD_B: dict[str, float] = {"title": 0.55, "alias": 0.55, "kw": 0.60, "body": 0.75, "meta": 0.60}
K1 = 1.3


@dataclass(frozen=True)
class DocMeta:
    id: str
    kind: str          # hand / auto / clause / para
    archive: bool
    dept: str
    title: str
    aliases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


class Index:
    """一份不可变索引。查询方法都是只读的，可以多线程共用。"""

    __slots__ = ("docs", "vocab", "post", "lens", "avg", "n", "grams", "phrases",
                 "title_keys", "alias_keys", "uniq", "stats")

    def __init__(self) -> None:
        self.docs: list[DocMeta] = []
        self.vocab: dict[str, int] = {}
        self.post: dict[str, dict[str, np.ndarray]] = {}
        self.lens: dict[str, np.ndarray] = {}
        self.avg: dict[str, float] = {}
        self.n = 0
        self.grams: list[str] = []
        self.phrases: list[str] = []
        self.title_keys: dict[str, list[int]] = {}
        self.alias_keys: dict[str, list[int]] = {}
        self.uniq: dict[str, np.ndarray] = {}
        self.stats: dict[str, Any] = {}

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, docs: Sequence[tuple[DocMeta, Mapping[str, Sequence[str]]]]) -> "Index":
        """docs: [(meta, {字段: 词条列表})]，词条已带类型前缀（G:/P:）。"""
        ix = cls()
        ix.docs = [m for m, _ in docs]
        ix.n = len(ix.docs)
        vocab: dict[str, int] = {}
        rows: dict[str, list[list[int]]] = {f: [[], [], []] for f in FIELDS}
        lens: dict[str, list[int]] = {f: [0] * ix.n for f in FIELDS}
        shorts: list[list[int]] = [[], [], []]

        for di, (_meta, fields) in enumerate(docs):
            for f in FIELDS:
                toks = fields.get(f) or ()
                lens[f][di] = len(toks)
                if not toks:
                    continue
                counts: dict[str, int] = {}
                for t in toks:
                    counts[t] = counts.get(t, 0) + 1
                tid_list, tid_doc, tid_tf = rows[f]
                for key, tf in counts.items():
                    tid = vocab.get(key)
                    if tid is None:
                        tid = len(vocab)
                        vocab[key] = tid
                    tid_list.append(tid)
                    tid_doc.append(di)
                    tid_tf.append(tf)
            merged: set[str] = set()
            for f in ("title", "alias", "kw"):
                merged.update(fields.get(f) or ())
            for key in merged:
                tid = vocab.get(key)
                if tid is None:
                    tid = len(vocab)
                    vocab[key] = tid
                shorts[0].append(tid)
                shorts[1].append(di)
                shorts[2].append(1)

        ix.vocab = vocab
        for f in FIELDS:
            ix._finish_field(f, rows[f], lens[f])
        ix._finish_field("short", shorts, [0] * ix.n)
        ix.lens["short"] = ix.uniq["short"].astype(np.float32)
        ix.avg["short"] = float(ix.lens["short"].mean()) if ix.n else 0.0
        ix.grams = [k for k in vocab if k.startswith("G:")]
        ix.phrases = [k for k in vocab if k.startswith("P:")]
        ix._rebuild_keys()
        ix.stats = {"n_docs": ix.n, "n_terms": len(vocab), "n_grams": len(ix.grams),
                    "n_phrases": len(ix.phrases),
                    "postings": int(sum(len(ix.post[f]["docs"]) for f in ALL_FIELDS))}
        return ix

    def _rebuild_keys(self) -> None:
        """标题 / 别名的精确串表：整串、标题主干、去掉问句套话后的别名都收。

        别名本身也常带问句框架（「户口迁移怎么办」「校园卡怎么补办」），用户问的时候
        框架会被拆掉，所以别名这一侧也要存一份拆过框架的形态，否则「户口迁移怎么办」
        和「户口迁移流程」永远对不上。
        """
        from .text import query_text, strip_how

        self.title_keys = {}
        self.alias_keys = {}
        stripped: dict[str, list[int]] = {}
        for di, meta in enumerate(self.docs):
            full = _norm_key(meta.title)
            core = _core_key(meta.title)
            for key in {full, core}:
                if len(key) >= 2:
                    self.title_keys.setdefault(key, []).append(di)
            for a in meta.aliases:
                ak = _norm_key(a)
                if len(ak) >= 2 and ak not in (full, core):
                    self.alias_keys.setdefault(ak, []).append(di)
                for form in (query_text(a), strip_how(query_text(a))):
                    if 2 <= len(form) < len(ak) and form not in (full, core):
                        stripped.setdefault(form, []).append(di)
        # 拆过套话的别名也进精确表：用户问的时候套话会被拆掉，
        # 「户口迁移怎么办」和「户口迁移流程」只有这一层能对上（presets 里就有这种）。
        # 与完整别名合并而不是覆盖——「户口迁移」既是某条的完整别名，也是另一条剥出来的形态。
        for ac, docs in stripped.items():
            bucket = self.alias_keys.setdefault(ac, [])
            for d in docs:
                if d not in bucket:
                    bucket.append(d)

    def _finish_field(self, f: str, rows: list[list[int]], lens: list[int]) -> None:
        tids, dids, tfs = rows
        tids_a = np.asarray(tids, dtype=np.int32)
        dids_a = np.asarray(dids, dtype=np.int32)
        tfs_a = np.asarray(tfs, dtype=np.int16)
        order = np.argsort(tids_a, kind="stable")
        tids_a, dids_a, tfs_a = tids_a[order], dids_a[order], tfs_a[order]
        indptr = np.zeros(len(self.vocab) + 1, dtype=np.int64)
        np.add.at(indptr, tids_a + 1, 1)
        indptr = np.cumsum(indptr)
        df = np.diff(indptr).astype(np.float32)
        idf = np.log1p((self.n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.post[f] = {"indptr": indptr, "docs": dids_a, "tf": tfs_a, "idf": idf}
        self.lens[f] = np.asarray(lens, dtype=np.float32)
        self.avg[f] = float(self.lens[f].mean()) if self.n else 0.0
        self.uniq[f] = np.bincount(dids_a, minlength=self.n).astype(np.int32)

    # ------------------------------------------------------------------- save

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for f in ALL_FIELDS:
            for k, arr in self.post[f].items():
                np.save(folder / f"post_{f}_{k}.npy", arr)
            np.save(folder / f"lens_{f}.npy", self.lens[f])
            np.save(folder / f"uniq_{f}.npy", self.uniq[f])
        (folder / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")
        (folder / "docs.json").write_text(json.dumps(
            [[d.id, d.kind, d.archive, d.dept, d.title, list(d.aliases), list(d.keywords)]
             for d in self.docs], ensure_ascii=False), encoding="utf-8")
        (folder / "stats.json").write_text(json.dumps(
            {"n": self.n, "avg": self.avg, "stats": self.stats}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> "Index":
        folder = Path(folder)
        ix = cls()
        ix.vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        for f in ALL_FIELDS:
            ix.post[f] = {k: np.load(folder / f"post_{f}_{k}.npy")
                          for k in ("indptr", "docs", "tf", "idf")}
            ix.lens[f] = np.load(folder / f"lens_{f}.npy")
            ix.uniq[f] = np.load(folder / f"uniq_{f}.npy")
        raw = json.loads((folder / "docs.json").read_text(encoding="utf-8"))
        ix.docs = [DocMeta(id=r[0], kind=r[1], archive=bool(r[2]), dept=r[3], title=r[4],
                           aliases=tuple(r[5]), keywords=tuple(r[6])) for r in raw]
        ix._rebuild_keys()
        st = json.loads((folder / "stats.json").read_text(encoding="utf-8"))
        ix.n = int(st["n"])
        ix.avg = st["avg"]
        ix.stats = st["stats"]
        ix.grams = [k for k in ix.vocab if k.startswith("G:")]
        ix.phrases = [k for k in ix.vocab if k.startswith("P:")]
        return ix

    # ------------------------------------------------------------------ score

    def _accumulate(self, terms: Mapping[str, float], field: str) -> np.ndarray:
        """BM25F 单字段累加。terms: {词条: 查询侧权重}。"""
        post = self.post[field]
        indptr, dids, tf, idf = post["indptr"], post["docs"], post["tf"], post["idf"]
        out = np.zeros(self.n, dtype=np.float32)
        lens = self.lens[field]
        b = FIELD_B.get(field, 0.6)
        avg = max(self.avg[field], 1e-6)
        denom_len = ((1.0 - b) + b * (lens / avg)).astype(np.float32)
        vocab = self.vocab
        for term, qw in terms.items():
            tid = vocab.get(term)
            if tid is None:
                continue
            lo, hi = indptr[tid], indptr[tid + 1]
            if hi <= lo:
                continue
            d = dids[lo:hi]
            f = tf[lo:hi].astype(np.float32)
            v = f * (K1 + 1.0) / (f + K1 * denom_len[d])
            out[d] += (qw * idf[tid]) * v
        return out

    def field_scores(self, term_groups: Mapping[str, Mapping[str, float]]
                     ) -> dict[str, dict[str, np.ndarray]]:
        """term_groups: {通道名: {词条: 权重}}；返回 {通道: {字段: 分数}}。"""
        out: dict[str, dict[str, np.ndarray]] = {}
        for chan, terms in term_groups.items():
            if not terms:
                continue
            out[chan] = {f: self._accumulate(terms, f) for f in FIELDS}
        return out

    def cover(self, terms: Iterable[str], field: str) -> tuple[np.ndarray, np.ndarray]:
        """覆盖度：每个文档命中了多少个查询词条 / 命中的 IDF 之和。"""
        post = self.post[field]
        indptr, dids, idf = post["indptr"], post["docs"], post["idf"]
        cnt = np.zeros(self.n, dtype=np.float32)
        isum = np.zeros(self.n, dtype=np.float32)
        vocab = self.vocab
        for term in terms:
            tid = vocab.get(term)
            if tid is None:
                continue
            lo, hi = indptr[tid], indptr[tid + 1]
            if hi <= lo:
                continue
            d = dids[lo:hi]
            cnt[d] += 1.0
            isum[d] += idf[tid]
        return cnt, isum

    def top_k(self, score: np.ndarray, k: int) -> np.ndarray:
        n = min(k, self.n)
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if n >= self.n:
            return np.argsort(-score, kind="stable")
        idx = np.argpartition(-score, n - 1)[:n]
        return idx[np.argsort(-score[idx], kind="stable")]

    def exact_hits(self, q: str, min_len: int = 3) -> dict[int, float]:
        """问句与标题 / 别名互为子串时的直接命中（分值按重合长度给）。"""
        hits: dict[int, float] = {}
        if len(q) < min_len:
            return hits
        for keys, base in ((self.title_keys, 3.0), (self.alias_keys, 2.4)):
            for d in keys.get(q, ()):                      # 整串相等
                hits[d] = max(hits.get(d, 0.0), base + 1.0)
            for i in range(len(q)):                        # 问句里含标题 / 别名
                for L in range(min(len(q) - i, 16), min_len - 1, -1):
                    docs = keys.get(q[i : i + L])
                    if docs:
                        for d in docs:
                            hits[d] = max(hits.get(d, 0.0), base * L / max(len(q), 1))
                        break
        return hits


def _norm_key(s: str) -> str:
    from .text import norm
    return norm(s)


def _core_key(s: str) -> str:
    from .text import core_title
    return core_title(s)
