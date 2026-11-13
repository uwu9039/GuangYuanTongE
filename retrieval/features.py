"""候选对特征。

排序器不看「分数阈值」，看的是每个候选的一组可比特征：两个通道（字符二元组 /
领域词）在 5 个字段上的分、字面覆盖度（IDF 加权）、标题/别名重合、条目种类先验、
是否存档条目，以及「字面命中 × 正文命中」这类交互项（线性模型没有交互就学不出
「标题原样命中时不该被正文更长的分段条目挤掉」）。

覆盖度全部是向量化算的：查询词条顺着 postings 一次累加出「每个文档命中几个词条」，
候选只做数组切片，没有逐条目的字符串比较，单查询毫秒级。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .index import FIELDS, Index
from .text import (Lexicon, bigrams, content_words, core_title, is_hub_title,
                   is_question_title, norm, strip_modal, topic_text)

FEATURE_NAMES: tuple[str, ...] = (
    "log_bm25_total", "log_bm25_short", "log_g_title", "log_g_alias", "log_g_kw",
    "log_g_body", "log_p_title", "log_p_alias", "log_p_kw", "log_p_total",
    "title_eq", "title_in_q", "q_in_title", "alias_eq", "alias_in_q", "q_in_alias",
    "cov_short", "cov_title", "cov_body", "dice_title", "dice_alias", "dice_kw",
    "len_ratio_title", "len_ratio_alias", "kind_hand", "kind_auto", "kind_clause",
    "kind_para", "archive", "phrase_hit_kw", "dept_hit", "log_title_len",
    "log_doc_len", "bm25_rank_recip", "phrase_hit_short",
    "title_cov_rev", "short_exact", "exact_x_total", "exact_x_body", "phrase_in_title",
    "hand_x_exact", "alias_cover", "title_cover",
    # 同名条目：库里同一个事项常有「人工整理 + 自动抽取 + 原文条款」多份，
    # 用户问的是那条服务，不是条文碎片；这两个特征让排序器学会把非人工的同名件压下去。
    "hand_twin", "twin_nonhand",
    # dense 通道（可关；关了这两个特征恒为 0）
    "dense_score", "dense_rank_recip",
    # 主题（剥掉问句框架词后的比对）与「标题本身是问句」的碎片条目。
    # 问句里的「什么时候 / 在哪里 / 怎么」和候选标题里的同样几个字对上，字面分虚高；
    # 剥掉框架词之后还共享的那部分，才是「讲的是同一件事」。
    "topic_cov_title", "topic_dice_title", "topic_cov_short", "topic_in_title",
    "qtitle_doc", "qtitle_x_gtitle",
    # 实词覆盖：问句里的领域实词有几个出现在候选的名字（标题 / 别名 / 关键词）里。
    # 二元组覆盖会被框架词灌水（「什么时候」共享 3 个二元组），实词不会。
    "word_cov_title", "word_cov_name",
    # 大杂烩条目（常见问题 / 汇总 / 一览 / 带网址的整段网页答案）：命中次数很少，
    # 但一旦当上 top1 观感极差，单独给一维让排序器与规则都能用。
    "hub_doc",
)


class TopicIndex:
    """主题二元组的倒排（只建在「条目名分」上：标题 / 别名 / 关键词）。

    和 `Index.cover` 的区别：这里的词条是**剥掉问句框架词之后**的二元组
    （`text.topic_text`）。「什么时候加入班级群」这种候选标题剥完只剩「加入班级群」，
    与「选课什么时候开始」不再共享任何词条，字面虚高在主题这一路就归零了。

    体量很小（每个二元组平均 2 条 posting），一次查询只碰命中那几个词条，
    所以可以直接在 3.9 万条上做数组累加，不需要再走候选切片。
    """

    __slots__ = ("n", "post_title", "post_short", "len_title", "len_short",
                 "title_topic")

    def __init__(self, n: int) -> None:
        self.n = n
        self.post_title: dict[str, list[int]] = {}
        self.post_short: dict[str, list[int]] = {}
        self.len_title = np.zeros(n, dtype=np.float32)
        self.len_short = np.zeros(n, dtype=np.float32)
        self.title_topic: list[str] = [""] * n

    @classmethod
    def build(cls, ix: Index, title_key: list[str], alias_keys: list[tuple[str, ...]],
              kw_keys: list[tuple[str, ...]]) -> "TopicIndex":
        ti = cls(ix.n)
        for di in range(ix.n):
            tt = topic_text(title_key[di])
            if len(tt) >= 2:
                ti.title_topic[di] = tt
                g = set(bigrams(tt))
                ti.len_title[di] = len(g)
                for x in g:
                    ti.post_title.setdefault(x, []).append(di)
                st = tt
            else:
                st = ""
            parts = [tt, *(a for a in alias_keys[di] if len(a) >= 2),
                     *(k for k in kw_keys[di] if len(k) >= 2)]
            sg: set[str] = set()
            for p in parts:
                tp = topic_text(p)
                if len(tp) >= 2:
                    sg.update(bigrams(tp))
            if sg:
                ti.len_short[di] = len(sg)
                for x in sg:
                    ti.post_short.setdefault(x, []).append(di)
        return ti

    def _cover(self, grams, post) -> np.ndarray:
        cnt = np.zeros(self.n, dtype=np.float32)
        for g in grams:
            d = post.get(g)
            if d:
                cnt[d] += 1.0
        return cnt

    def cover_title(self, grams) -> np.ndarray:
        return self._cover(grams, self.post_title)

    def cover_short(self, grams) -> np.ndarray:
        return self._cover(grams, self.post_short)


class IndexCache:
    """索引的只读派生视图（不复制词条集合，只缓存能快速取到的标量/字符串）。"""

    __slots__ = ("ix", "lex", "idf_list", "title_key", "alias_keys", "dept_keys", "kind",
                 "archive", "title_len", "doc_len", "log_title_len", "log_doc_len",
                 "alias0_len", "is_hand", "is_auto", "is_clause", "is_para", "hand_twin",
                 "twin_of", "topic", "qtitle", "title_raw", "name_raw", "hub")

    def __init__(self, ix: Index, lex: Lexicon | None = None) -> None:
        self.ix = ix
        self.lex = lex or Lexicon(())
        self.idf_list = [float(x) for x in ix.post["short"]["idf"]]
        self.title_key: list[str] = [core_title(d.title) for d in ix.docs]
        self.alias_keys: list[tuple[str, ...]] = [tuple(norm(a) for a in d.aliases) for d in ix.docs]
        self.dept_keys: list[str] = [norm(d.dept) for d in ix.docs]
        self.kind: list[str] = [d.kind for d in ix.docs]
        self.archive: np.ndarray = np.array([d.archive for d in ix.docs], dtype=np.float32)
        self.title_len = np.array([len(k) for k in self.title_key], dtype=np.float32)
        self.alias0_len = np.array([len(a[0]) if a else 0 for a in self.alias_keys], dtype=np.float32)
        self.doc_len = (ix.uniq["short"] + ix.uniq["body"]).astype(np.float32)
        self.log_title_len = np.log1p(self.title_len).astype(np.float32)
        self.log_doc_len = np.log1p(self.doc_len).astype(np.float32)
        self.is_hand = np.array([d.kind == "hand" for d in ix.docs], dtype=np.float32)
        self.is_auto = np.array([d.kind == "auto" for d in ix.docs], dtype=np.float32)
        self.is_clause = np.array([d.kind == "clause" for d in ix.docs], dtype=np.float32)
        self.is_para = np.array([d.kind == "para" for d in ix.docs], dtype=np.float32)
        self.hand_twin = self._hand_twin()
        kw_keys = [tuple(norm(k) for k in d.keywords) for d in ix.docs]
        self.topic = TopicIndex.build(ix, self.title_key, self.alias_keys, kw_keys)
        # 标题是问句 / 整句、又不是人工整理条目 —— 这类「原文碎片」不是事项条目，
        # 但它的标题恰好含问句框架词，字面分会虚高（hub 抢答的根因）。
        is_q = np.array([is_question_title(d.title) for d in ix.docs], dtype=np.float32)
        self.qtitle = is_q * (1.0 - self.is_hand)
        # 名字的原始串（不剥括号、不剥文号）：实词覆盖要按「用户真能看到的字」比对
        self.title_raw: list[str] = [norm(d.title) for d in ix.docs]
        self.name_raw: list[str] = [t + "".join(a) + "".join(k) for t, a, k
                                    in zip(self.title_raw, self.alias_keys, kw_keys)]
        self.hub = np.array([is_hub_title(d.title) for d in ix.docs], dtype=np.float32)

    def _hand_twin(self) -> np.ndarray:
        """非人工条目与某条人工条目的标题相似度（二元组重合）。

        「学生电子交通卡申领」这类同名条目在库里同时有 hand / auto / rule / doc 四份，
        调换一下词序就躲过精确匹配；这里用二元组覆盖兜住，让排序器学到「同名优先人工版」。
        """
        hand_idx = [i for i, k in enumerate(self.kind) if k == "hand"]
        postings: dict[str, list[int]] = {}
        for h in hand_idx:
            for g in set(slice_ngrams(self.title_key[h])):
                if len(self.title_key[h]) >= 2:
                    postings.setdefault(g, []).append(h)
        out = np.zeros(len(self.kind), dtype=np.float32)
        twin_of = np.full(len(self.kind), -1, dtype=np.int32)
        for i, core in enumerate(self.title_key):
            if self.kind[i] == "hand" or len(core) < 2:
                continue
            grams = set(slice_ngrams(core))
            if not grams:
                continue
            hits: dict[int, int] = {}
            for g in grams:
                for h in postings.get(g, ()):  # noqa: E501
                    hits[h] = hits.get(h, 0) + 1
            if not hits:
                continue
            best, best_h = 0.0, -1
            for h, c in hits.items():
                # 重叠系数（除以较短一方），词序调换也认：「电子学生交通卡」↔「学生电子交通卡」
                denom = min(len(grams), len(set(slice_ngrams(self.title_key[h]))))
                score = c / denom if denom else 0.0
                if score > best:
                    best, best_h = score, h
            if best >= 0.6:
                out[i] = best
                twin_of[i] = best_h
        self.twin_of = twin_of
        return out


@dataclass
class QueryCtx:
    raw: str
    core: str
    gram_ids: np.ndarray
    gram_w: np.ndarray
    phrase_ids: np.ndarray
    gram_set: frozenset[int]
    phrase_set: frozenset[int]
    dept: str = ""
    core_m: str = ""                    # 去掉情态词（能/可以）后的主题串，只用于字面包含比对
    topic: str = ""                     # 剥掉问句框架词后的主题串（主题比对用）
    topic_grams: frozenset[str] = frozenset()
    words: tuple[str, ...] = ()          # 问句里的领域实词（框架词已剔除）
    _idf_denom: float = field(default=0.0, repr=False)


def slice_ngrams(s: str) -> list[str]:
    return [s[i:i + 2] for i in range(len(s) - 1)]


DEPTS = ("信息化处", "教务", "学生工作部", "图书馆", "后勤", "保卫处", "校医院", "体育部",
         "财务", "学工", "研究生院", "就业", "人力", "招生")


def make_ctx(ix: Index, cache: IndexCache, raw: str, core: str) -> QueryCtx:
    grams = list(dict.fromkeys(bigrams(core)))
    gids = [ix.vocab["G:" + g] for g in grams if "G:" + g in ix.vocab]
    phrases = [w for w in cache.lex.match(core) if len(w) >= 3]
    pids = [ix.vocab["P:" + w] for w in phrases if "P:" + w in ix.vocab]
    nc = norm(core)
    dept = ""
    for d in DEPTS:
        if norm(d) in nc:
            dept = norm(d)
            break
    ctx = QueryCtx(raw=raw, core=core,
                   gram_ids=np.asarray(gids, dtype=np.int32),
                   gram_w=np.ones(len(gids), dtype=np.float32),
                   phrase_ids=np.asarray(pids, dtype=np.int32),
                   gram_set=frozenset(gids), phrase_set=frozenset(pids), dept=dept,
                   core_m=strip_modal(core))
    tp = topic_text(core)
    ctx.topic = tp
    ctx.topic_grams = frozenset(bigrams(tp))
    # 实词：领域词表切一遍，剔掉框架词；一个都没切出来时退回主题二元组
    words = content_words(cache.lex, tp)
    if not words and len(tp) >= 2:
        words = [tp]
    ctx.words = tuple(words)
    ctx._idf_denom = sum(cache.idf_list[t] for t in ctx.gram_set) or 1.0
    return ctx


def coverage(ix: Index, ctx: QueryCtx) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """一次算出各字段的覆盖度（词条数 / IDF 和），供 featurize 取用。"""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    grams = [("G:" + ctx.core[i:i + 2]) for i in range(max(len(ctx.core) - 1, 0))]
    grams = list(dict.fromkeys(grams))
    for f in ("short", "title", "alias", "kw", "body"):
        out[f] = ix.cover(grams, f)
    return out


def _log(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0)).astype(np.float32)


def featurize(cache: IndexCache, ctx: QueryCtx, doc_ids: Sequence[int],
              field_scores: dict[str, dict[str, np.ndarray]] | None = None,
              bm25_order: dict[int, int] | None = None,
              cov: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
              phrase_cov: dict[str, np.ndarray] | None = None,
              dense: dict[int, tuple[float, int]] | None = None) -> np.ndarray:
    """返回 (len(doc_ids), F) 的特征矩阵。"""
    ix = cache.ix
    ids = np.asarray(list(doc_ids), dtype=np.int64)
    m = len(ids)
    x = np.zeros((m, len(FEATURE_NAMES)), dtype=np.float32)
    if m == 0:
        return x
    zero = np.zeros(m, dtype=np.float32)

    def fs(chan: str, f: str) -> np.ndarray:
        if not field_scores or chan not in field_scores:
            return zero
        return field_scores[chan][f][ids]

    g = {f: fs("g", f) for f in FIELDS}
    p = {f: fs("p", f) for f in FIELDS}
    total = sum(g[f] for f in FIELDS) + 1.35 * sum(p[f] for f in FIELDS)
    short = g["title"] + g["alias"] + g["kw"] + 1.35 * (p["title"] + p["alias"] + p["kw"])
    x[:, 0], x[:, 1] = _log(total), _log(short)
    x[:, 2], x[:, 3], x[:, 4], x[:, 5] = _log(g["title"]), _log(g["alias"]), _log(g["kw"]), _log(g["body"])
    x[:, 6], x[:, 7], x[:, 8] = _log(p["title"]), _log(p["alias"]), _log(p["kw"])
    x[:, 9] = _log(sum(p[f] for f in FIELDS))

    nq = max(len(ctx.gram_set), 1)
    idf_denom = max(ctx._idf_denom, 1e-6)
    cov = cov or {}
    c_short, i_short = (cov.get("short") or (zero, zero))
    c_title = (cov.get("title") or (zero, zero))[0]
    c_alias = (cov.get("alias") or (zero, zero))[0]
    c_kw = (cov.get("kw") or (zero, zero))[0]
    c_body = (cov.get("body") or (zero, zero))[0]
    x[:, 16] = i_short[ids] / idf_denom
    x[:, 17] = c_title[ids] / nq
    x[:, 18] = c_body[ids] / nq
    x[:, 19] = 2.0 * c_title[ids] / np.maximum(ix.uniq["title"][ids] + nq, 1)
    x[:, 20] = 2.0 * c_alias[ids] / np.maximum(ix.uniq["alias"][ids] + nq, 1)
    x[:, 21] = 2.0 * c_kw[ids] / np.maximum(ix.uniq["kw"][ids] + nq, 1)
    if phrase_cov:
        x[:, 34] = phrase_cov["short"][ids] / max(len(ctx.phrase_set), 1)
        x[:, 29] = phrase_cov["kw"][ids]
        x[:, 39] = (phrase_cov["title"][ids] > 0).astype(np.float32)

    x[:, 22] = np.minimum(cache.title_len[ids], len(ctx.core)) / np.maximum(cache.title_len[ids], len(ctx.core))
    x[:, 23] = np.minimum(cache.alias0_len[ids], len(ctx.core)) / np.maximum(cache.alias0_len[ids], len(ctx.core))
    x[:, 24] = cache.is_hand[ids]
    x[:, 25] = cache.is_auto[ids]
    x[:, 26] = cache.is_clause[ids]
    x[:, 27] = cache.is_para[ids]
    x[:, 28] = cache.archive[ids]
    if ctx.dept:
        x[:, 30] = np.array([ctx.dept in cache.dept_keys[d] for d in ids], dtype=np.float32)
    x[:, 31] = cache.log_title_len[ids]
    x[:, 32] = cache.log_doc_len[ids]

    core = ctx.core
    core_m = ctx.core_m or core      # 去掉情态词后的同一句，用于字面包含比对
    # 问句里出现的标题 / 别名：先用「问句子串 → 文档」的字典查表，避免逐候选做字符串比较
    title_in_q: set[int] = set()
    alias_in_q: set[int] = set()
    title_eq: set[int] = set()
    alias_eq: set[int] = set()
    tk = cache.ix.title_keys
    ak = cache.ix.alias_keys
    for form in (core, core_m):
        for d in tk.get(form, ()):
            title_eq.add(d)
            title_in_q.add(d)
        for d in ak.get(form, ()):
            alias_eq.add(d)
            alias_in_q.add(d)
        n_form = len(form)
        for i in range(n_form):
            for L in range(3, min(n_form - i, 16) + 1):
                sub = form[i:i + L]
                docs = tk.get(sub)
                if docs:
                    title_in_q.update(docs)
                docs = ak.get(sub)
                if docs:
                    alias_in_q.update(docs)

    for r, di in enumerate(ids):
        di = int(di)
        x[r, 10] = 1.0 if di in title_eq else 0.0
        x[r, 11] = 1.0 if di in title_in_q else 0.0
        # 别名只有人工条目才可信：自动抽取条目的 alias 多半是标题碎片，
        # 一句「图书馆」就能让一堆段落混进「别名命中」，实测会把服务条目挤下去。
        hand = x[r, 24] > 0.5
        x[r, 13] = 1.0 if (hand and di in alias_eq) else 0.0
        x[r, 14] = 1.0 if (hand and di in alias_in_q) else 0.0
        title = cache.title_key[di]
        if len(title) >= 3 and (core in title or core_m in title):
            x[r, 12] = 1.0
        best_alias = 0
        for a in cache.alias_keys[di]:
            if a and (a in core or a in core_m) and len(a) > best_alias:
                best_alias = len(a)
            if hand and not x[r, 15] and len(a) >= 3 and (core in a or core_m in a):
                x[r, 15] = 1.0
        ncore = max(len(core), 1)
        x[r, 41] = best_alias / ncore if hand else 0.0
        x[r, 42] = (len(title) / ncore) if (len(title) >= 3 and title in core) else 0.0
        if bm25_order is not None:
            x[r, 33] = 1.0 / (1.0 + bm25_order.get(di, 99))

    exact = np.maximum.reduce([x[:, 10], x[:, 11], x[:, 13], x[:, 14]])
    x[:, 35] = x[:, 17] * np.minimum(1.0, x[:, 22] * 2.0)
    x[:, 36] = exact
    x[:, 37] = exact * x[:, 0]
    x[:, 38] = exact * x[:, 18]
    x[:, 40] = exact * x[:, 24]
    # 同名条目：非人工件与某条人工件的标题重合度越高，越该让位
    x[:, 43] = cache.hand_twin[ids]
    x[:, 44] = x[:, 43] * (1.0 - x[:, 24])
    # 主题比对（剥掉问句框架词）：问句与候选标题「讲的是不是同一件事」
    tg = ctx.topic_grams
    if tg:
        nt = float(len(tg))
        c_t = cache.topic.cover_title(tg)[ids]
        c_s = cache.topic.cover_short(tg)[ids]
        x[:, 47] = c_t / nt
        x[:, 48] = 2.0 * c_t / np.maximum(cache.topic.len_title[ids] + nt, 1.0)
        x[:, 49] = c_s / nt
        tps = cache.topic.title_topic
        qtp = ctx.topic
        x[:, 50] = np.array(
            [1.0 if (len(tps[d]) >= 2 and tps[d] in qtp) else 0.0 for d in ids],
            dtype=np.float32)
    # 标题是问句 / 整句的原文碎片：靠框架词跟问句对上，不该当事项条目用
    x[:, 51] = cache.qtitle[ids]
    x[:, 52] = x[:, 51] * x[:, 2]
    # 实词覆盖：问句的领域实词有几个出现在候选的名字里
    ws = ctx.words
    if ws:
        nw = float(len(ws))
        tr = cache.title_raw
        nr = cache.name_raw
        x[:, 53] = np.array([sum(1 for w in ws if w in tr[d]) for d in ids],
                            dtype=np.float32) / nw
        x[:, 54] = np.array([sum(1 for w in ws if w in nr[d]) for d in ids],
                            dtype=np.float32) / nw
    x[:, 55] = cache.hub[ids]
    # dense 通道：只有开了才填，关了恒为 0（两套权重共用同一套特征位）
    if dense:
        x[:, 45] = np.array([dense.get(int(d), (0.0, 999))[0] for d in ids], dtype=np.float32)
        x[:, 46] = np.array([1.0 / (1.0 + dense.get(int(d), (0.0, 999))[1]) for d in ids],
                            dtype=np.float32)
    return x
