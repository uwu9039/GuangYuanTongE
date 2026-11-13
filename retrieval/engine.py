"""检索引擎：候选召回 → 特征 → 学习排序 → 拒答判定 → 答案装配。

一次查询的固定流程（没有阈值补丁、没有分支特判）：

1. `query_text` 拆套话，得到「主题串」（饭卡丢了怎么办 → 饭卡丢了）；
2. 两路词条（字符二元组 / 领域词）在 5 个字段上做 BM25F，取三份候选榜的并集；
3. 每个候选算 35 个特征，过线性排序器（离线在 2 万条别名对上训练）；
4. 取 top-k，交给拒答层判断「这条问题库里有没有答案」；
5. 有答案就装配成条目化答复（步骤/材料/费用/地点/时间/时限/注意事项/联系方式/出处）。
"""
from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import compose as _compose
from . import gate as _gate
from . import sections as _sections
from .features import FEATURE_NAMES, IndexCache, make_ctx, featurize
from .index import FIELD_W, FIELDS, Index
from .ranker import Ranker
from .text import Lexicon, bigrams, query_text, strip_how

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "retrieval" / "artifacts"
MODEL_DIR = ROOT / "retrieval" / "model"

CAND_TOTAL = 300
CAND_SHORT = 150
CAND_BODY = 80
DENSE_TOP = 120
# hub 让位时给「标题是问句的原文碎片」扣多少（logit 尺度；冠军 prob 1.0 时仍有 ~5 分余量）
HUB_PENALTY = 2.5
# 标题落位：问句的实词出现在候选标题里就给这么多加分（按覆盖比例）。
# 排序器是线性的，学不出「正文沾边」与「标题就是这件事」的差别；这一档把它显式化，
# 否则「选课什么时候开始」会被「体测时间安排」（正文也提选课）顶掉。
TITLE_ANCHOR = 0.5
# hub 让位时同时把「标题落位最好」的那一档抬一档：光扣 hub 只是把它按下去，
# 下一名可能仍是「体测时间安排」这种只沾框词的条目（「新生选课是什么时候」就会这样）。
HUB_SWAP = 1.0


@dataclass
class Hit:
    doc: Any
    item: dict[str, Any] | None
    score: float                  # 排序器 logit
    prob: float
    feats: np.ndarray | None = None

    @property
    def id(self) -> str:
        return self.doc.id

    @property
    def title(self) -> str:
        return self.doc.title


@dataclass
class Result:
    query: str
    core: str
    hits: list[Hit]
    refused: bool
    reason: str = ""
    confidence: float = 0.0
    answer: _compose.Answer | None = None
    latency_ms: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> Hit | None:
        return self.hits[0] if self.hits else None


class Engine:
    __slots__ = ("ix", "cache", "lex", "ranker", "gate", "items", "meta",
                 "dense", "ranker_dense", "use_dense", "_doc_of", "_qv")

    def __init__(self, ix: Index, lex: Lexicon, ranker: Ranker | None,
                 gate: _gate.Gate | None, items: dict[str, dict[str, Any]],
                 meta: dict[str, Any] | None = None,
                 dense: Any = None, ranker_dense: Ranker | None = None,
                 use_dense: bool = False) -> None:
        self.ix = ix
        self.lex = lex
        self.cache = IndexCache(ix, lex)
        self.ranker = ranker or Ranker(np.zeros(len(FEATURE_NAMES), dtype=np.float32), 0.0,
                                       np.zeros(len(FEATURE_NAMES), dtype=np.float32),
                                       np.ones(len(FEATURE_NAMES), dtype=np.float32))
        self.gate = gate or _gate.Gate.default()
        self.items = items
        self.meta = meta or {}
        self.dense = dense
        self.ranker_dense = ranker_dense
        # dense 通道能不能用只看索引；权重没有时先退回字面权重（训练 dense 模型时就是这种状态）
        self.use_dense = bool(use_dense and dense is not None)
        self._doc_of = {d.id: i for i, d in enumerate(ix.docs)}
        self._qv: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, art: Path | str = ART, items_path: Path | str | None = None,
             with_ranker: bool = True, dense: bool = False) -> "Engine":
        art = Path(art)
        model_dir = MODEL_DIR
        ix = Index.load(art)
        lex = Lexicon(json.loads((art / "lexicon.json").read_text(encoding="utf-8")))
        ranker = None
        # 权重进仓库（几 KB），索引是派生数据（88 MB）由 Docker 现建
        rp = model_dir / "ranker.json"
        if with_ranker and rp.exists():
            ranker = Ranker.load(rp)
        elif with_ranker and (art / "ranker.json").exists():
            ranker = Ranker.load(art / "ranker.json")
        ranker_dense = None
        rdp = model_dir / "ranker_dense.json"
        if with_ranker and rdp.exists():
            try:
                ranker_dense = Ranker.load(rdp)
            except ValueError as e:
                print(f"[检索] dense 权重与特征版本不一致：{e}")
        dense_idx = None
        try:
            from .dense import DenseIndex

            if (art / "dense" / "manifest.json").exists():
                dense_idx = DenseIndex.load(art / "dense")
        except Exception as e:  # noqa: BLE001
            print(f"[检索] dense 索引不可用：{e}")
        gate = None
        gp = model_dir / "gate.json"
        if gp.exists():
            gate = _gate.Gate.load(gp)
        elif (art / "gate.json").exists():
            gate = _gate.Gate.load(art / "gate.json")
        items = load_items(items_path)
        return cls(ix, lex, ranker, gate, items,
                   meta={"artifacts": str(art), "index": ix.stats,
                         "dense": bool(dense_idx)},
                   dense=dense_idx, ranker_dense=ranker_dense, use_dense=dense)

    # --------------------------------------------------------------- retrieve

    def retrieve(self, core: str, *, use_dense: bool = False, raw: str = ""
                 ) -> tuple[Any, dict[str, dict[str, np.ndarray]], list[int], np.ndarray]:
        """候选召回 + 特征。返回 (ctx, field_scores, cand_list, X)。"""
        core = strip_how(core)
        ctx = make_ctx(self.ix, self.cache, raw or core, core)
        gram_terms = list(dict.fromkeys("G:" + g for g in bigrams(core)))
        phrase_terms = [("P:" + w) for w in dict.fromkeys(self.lex.match(core)) if len(w) >= 3]
        grams = {t: 1.0 for t in gram_terms}
        phrases = {t: 1.6 for t in phrase_terms}
        gram_terms = list(grams)
        phrase_terms = list(phrases)

        field_scores = self.ix.field_scores({"g": grams, "p": phrases})
        n = self.ix.n
        total = np.zeros(n, dtype=np.float32)
        short = np.zeros(n, dtype=np.float32)
        body = np.zeros(n, dtype=np.float32)
        for f in FIELDS:
            gt = field_scores.get("g", {}).get(f)
            pt = field_scores.get("p", {}).get(f)
            if gt is not None:
                total += FIELD_W[f] * gt
                if f in ("title", "alias", "kw"):
                    short += gt
                if f == "body":
                    body += gt
            if pt is not None:
                total += 1.35 * FIELD_W[f] * pt
                if f in ("title", "alias", "kw"):
                    short += 1.35 * pt
                if f == "body":
                    body += 1.35 * pt

        cand: set[int] = set(int(i) for i in self.ix.top_k(total, CAND_TOTAL))
        cand |= set(int(i) for i in self.ix.top_k(short, CAND_SHORT))
        cand |= set(int(i) for i in self.ix.top_k(body, CAND_BODY))
        cand |= set(self.ix.exact_hits(core))
        dense_map: dict[int, tuple[float, int]] | None = None
        if use_dense and self.dense is not None:
            qv = self._query_vec(raw or core)
            dense_hits = self.dense.search(qv, DENSE_TOP)
            dense_map = {}
            for rank, (iid, score) in enumerate(dense_hits):
                di = self._doc_of.get(iid)
                if di is not None:
                    dense_map[di] = (score, rank)
            cand |= set(dense_map)
        cand_list = sorted(cand)
        if not cand_list:
            return ctx, field_scores, [], np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        cov = {f: self.ix.cover(gram_terms, f) for f in ("short", "title", "alias", "kw", "body")}
        pcov = {f: (self.ix.cover(phrase_terms, f)[0] if phrase_terms else np.zeros(n, dtype=np.float32))
                for f in ("short", "title", "kw")}
        order = {int(d): r for r, d in enumerate(self.ix.top_k(total, min(len(cand_list), n)))}
        x = featurize(self.cache, ctx, cand_list, field_scores, order, cov, pcov, dense_map)
        return ctx, field_scores, cand_list, x

    # ------------------------------------------------------------- dense 查询侧

    def _query_vec(self, text: str) -> np.ndarray:
        """查询向量：内存 LRU + 落盘缓存，评测重跑不重复调用云端。"""
        v = self._qv.get(text)
        if v is not None:
            return v
        from .dense import VectorCache, embed_texts

        cache = None
        try:
            cache = VectorCache(ART / "dense" / "cache.sqlite")
        except Exception:  # noqa: BLE001
            pass
        v = embed_texts([text], text_type="query", model=self.dense.model,
                        dim=self.dense.dim, workers=1, cache=cache)[0]
        if len(self._qv) > 2000:
            self._qv.clear()
        self._qv[text] = v
        return v


    # ----------------------------------------------------------------- 排序后规则

    def _adjust(self, logits: np.ndarray, ctx: Any, cand_list: Sequence[int],
                x: np.ndarray) -> np.ndarray:
        """两档确定性调整（都不改候选集，只改相对次序）。

        1. **hub 让位**：冠军是「标题本身就是问句的原文碎片」时，如果候选里已经有
           一条像样的具体条目（人工条目 + 主题覆盖够），就不要让碎片抢答——
           `信息化服务常见问题`、`大创相关内容在哪里查看`、`什么时候加入班级群？`
           这些大杂烩条目是靠问句框架词跟用户对上的，不是因为它就是那个事项。
        2. **章条锚定**：问句点了「第X条 / 第X章」时，给对得上的那一节里的条目加分
           （见 `retrieval/sections.py`）。
        """
        if len(logits) == 0:
            return logits
        wcv = x[:, FEATURE_NAMES.index("word_cov_title")]
        if TITLE_ANCHOR and wcv.max() > 0:
            # 只在「冠军标题一个实词都不含」时用落位重排：这时冠军明显是靠正文沾边
            # 上去的（「选课什么时候开始」被「体测时间安排」顶掉就是这种情形）。
            # 冠军本来就在标题里落位时不动，避免把「换了个说法」的问题改坏。
            top0 = int(np.argmax(logits))
            if wcv[top0] <= 0.0 and (wcv >= 0.5).any():
                logits = logits + TITLE_ANCHOR * wcv
        hb = x[:, FEATURE_NAMES.index("hub_doc")]
        if hb.max() > 0.5:
            top = int(np.argmax(logits))
            if hb[top] > 0.5:
                # 只有在「标题落位明显更强」的候选存在时才让位：条目本身若就是
                # 标题含这些实词的那一条，不该被它自己的长标题惩罚。
                better = ((hb < 0.5) & (wcv > wcv[top] + 0.15))
                if better.any():
                    logits = logits - HUB_PENALTY * hb
                    # 落位最好的那一档再抬一档，避免让位后仍是「只沾框词」的条目接管
                    best = wcv >= max(float(wcv.max()) - 1e-6, 0.5)
                    logits = logits + HUB_SWAP * best
        bonus = _sections.item_bonus(self, ctx)
        if bonus:
            add = np.array([bonus.get(self.ix.docs[di].id, 0.0) for di in cand_list],
                           dtype=np.float32)
            if add.any():
                logits = logits + add
        return logits

    # ------------------------------------------------------------------ search

    def search(self, q: str, k: int = 5, explain: bool = False,
               with_answer: bool = False) -> Result:
        t0 = time.perf_counter()
        core = query_text(q)
        if len(core) < 1:
            return Result(query=q, core=core, hits=[], refused=True, reason="空查询",
                          latency_ms=(time.perf_counter() - t0) * 1000)
        dense_on = self.use_dense
        debug: dict[str, Any] = {"dense": dense_on}
        try:
            ctx, _fs, cand_list, x = self.retrieve(core, use_dense=dense_on, raw=q)
        except Exception as e:  # noqa: BLE001  dense 挂了就用字面模型，不混着用
            print(f"[检索] dense 通道失败（{type(e).__name__}: {e}），本轮改用字面模型")
            debug["dense"], debug["dense_error"] = False, str(e)[:120]
            dense_on = False
            ctx, _fs, cand_list, x = self.retrieve(core)
        if not cand_list:
            return Result(query=q, core=core, hits=[], refused=True, reason="没有任何候选条目",
                          latency_ms=(time.perf_counter() - t0) * 1000, debug=debug)

        ranker = self.ranker_dense if (dense_on and self.ranker_dense is not None) else self.ranker
        logits = ranker.score(x)
        exact = self.ix.exact_hits(core)
        if exact:
            for r, di in enumerate(cand_list):
                if di in exact:
                    logits[r] += 0.35 * exact[di]
        logits = self._adjust(logits, ctx, cand_list, x)
        order_idx = np.argsort(-logits)
        kk = max(k, 1)
        pick_idx = order_idx[:kk]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits[pick_idx], -30, 30)))
        hits = [Hit(doc=self.ix.docs[di], item=self.items.get(self.ix.docs[di].id),
                    score=float(logits[i]), prob=float(probs[n]),
                    feats=x[i] if explain else None)
                for n, (i, di) in enumerate(zip(pick_idx, (cand_list[i] for i in pick_idx)))]

        decision = self.gate.decide(self, ctx, hits, probs, cand_list, x)
        res = Result(query=q, core=core, hits=hits, refused=decision.refused,
                     reason=decision.reason, confidence=decision.confidence,
                     latency_ms=(time.perf_counter() - t0) * 1000, debug=debug)
        if with_answer and not res.refused:
            res.answer = _compose.build(res, self.items)
        return res

    # ------------------------------------------------------------------ util

    def explain(self, q: str, k: int = 5) -> list[tuple[str, dict[str, float]]]:
        res = self.search(q, k=k, explain=True)
        out = []
        for h in res.hits:
            if h.feats is None:
                continue
            out.append((h.title, {FEATURE_NAMES[i]: round(float(h.feats[i]), 3)
                                  for i in range(len(FEATURE_NAMES))}))
        return out


def load_items(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    p = Path(path) if path else None
    if p is None:
        plain = ROOT / "items.json"
        p = plain if plain.exists() else ROOT / "items.json.gz"
    raw = p.read_bytes()
    if p.suffix == ".gz":
        raw = gzip.decompress(raw)
    return {it["id"]: it for it in json.loads(raw.decode("utf-8"))["items"]}
