"""上下文 v2 的闸门与仲裁单测：不联网、不起服务、不 import app。

跑法：
    PYTHONPATH=".venv/Lib/site-packages" python -m unittest tests.test_context_v2 -v

这里锁死两件事：
1. 闸门 0 的每一类**硬独立信号**都能确定性拦住（跨事项 / 跨地点 / 无指代长句 /
   自足闲聊 / 不是追问形状），尤其是「怎么申请换宿舍」这种被通用动词「申请」
   带偏的历史坑；
2. 仲裁的每个分支（A 强且别的主题 / A 强且就是锚点 / A 弱 B 落回锚点 /
   A 中间带 B 落回锚点 / B 不落锚点）都按表走。
"""
from __future__ import annotations

import unittest

from retrieval import context as ctx

ITEMS = [
    {"id": "zhuanzhuanye", "title": "转专业",
     "aliases": ["转专业", "换专业", "转专业条件"], "keywords": ["转专业", "专业"]},
    {"id": "xyk-buka", "title": "校园卡补卡", "aliases": ["补卡", "补办校园卡"], "keywords": ["校园卡"]},
    {"id": "shu-tang", "title": "食堂供餐时间", "aliases": ["食堂几点开"], "keywords": ["食堂"]},
    {"id": "tsg-jieyue", "title": "图书馆借阅", "aliases": ["借书"], "keywords": ["图书馆"]},
    {"id": "tice", "title": "体测时间安排", "aliases": ["体测什么时候"], "keywords": ["体测"]},
    {"id": "sushe-huansu", "title": "换宿舍 / 调宿", "aliases": ["换宿舍", "调宿"], "keywords": ["宿舍"]},
    {"id": "baoyan", "title": "推免（保研）资格条件", "aliases": ["保研"], "keywords": ["保研", "推免"]},
    {"id": "jwc-chengjidan", "title": "成绩单办理", "aliases": ["开成绩单"], "keywords": ["成绩单"]},
    {"id": "ruxue-cailiao", "title": "报到材料清单", "aliases": ["报到要带什么"], "keywords": ["报到"]},
]
PLACES = ["图书馆", "食堂", "宿舍楼", "校医院"]
BY = {it["id"]: it for it in ITEMS}

AID = "zhuanzhuanye"


def fake_rank(q: str) -> list[tuple[str, int]]:
    """按关键词给分：句子里点到某个事项就是它 95 分；点了锚点就是锚点 99。"""
    for it in ITEMS:
        terms = [it["title"], *(it.get("aliases") or []), *(it.get("keywords") or [])]
        if any(t and t in q for t in terms):
            if it["id"] == AID and "转专业" in q:
                return [(AID, 99), ("xyk-buka", 40)]
            return [(it["id"], 95), (AID, 30)]
    if "转专业" in q:
        return [(AID, 99)]
    return [("wd-noise", 20)]


def fake_embed(text: str):
    """极简向量：「转专业」相关 -> 方向 A；否则方向 C；坐标固定，便于断言 cos。"""
    if "转专业" in text or "专业" in text:
        return [1.0, 0.0]
    if text in ("那要带什么材料", "要交钱吗", "找谁办"):
        return [0.45, 0.89]          # 与锚点 cos≈0.45（中间带）
    return [0.0, 1.0]                # 与锚点 cos=0（极远）


def make(**kw) -> ctx.Resolver:
    kw.setdefault("embed_fn", None)
    kw.setdefault("anchor_vec_fn", None)
    use_gate1 = kw.pop("use_gate1", False)
    use_gate2 = kw.pop("use_gate2", False)
    return ctx.Resolver(ITEMS, rank_fn=fake_rank, places=PLACES,
                        use_gate1=use_gate1, use_gate2=use_gate2, **kw)


class TestGate0(unittest.TestCase):
    """闸门 0：硬独立信号。命中的一律 new，且理由要对得上具体那一类。"""

    def setUp(self) -> None:
        self.r = make()

    def test_cross_item(self):
        d = self.r.resolve("食堂几点开", BY[AID])
        self.assertEqual(d.label, "new")
        self.assertEqual(d.reason, "gate0.foreign_item")
        self.assertTrue(any("食堂" in t for t in d.foreign), d.foreign)

    def test_cross_item_required(self):
        d = self.r.resolve("保研要什么条件", BY[AID])
        self.assertEqual(d.label, "new")
        self.assertEqual(d.reason, "gate0.foreign_item")

    def test_cross_place(self):
        d = self.r.resolve("图书馆开放时间", BY[AID])
        self.assertEqual(d.label, "new")
        self.assertIn(d.reason, ("gate0.foreign_item", "gate0.opens_with_subject"))

    def test_generic_verb_is_not_a_topic(self):
        """历史坑：`怎么申请换宿舍` 与「转专业」共享「申请」，旧判定算沾边。"""
        d = self.r.resolve("怎么申请换宿舍", BY[AID])
        self.assertEqual(d.label, "new")
        self.assertEqual(d.reason, "gate0.foreign_item")
        self.assertTrue(any("宿舍" in t for t in d.foreign), d.foreign)

    def test_long_sentence_without_anaphora(self):
        d = self.r.resolve("我下周要去外地实习了", BY[AID])
        self.assertEqual(d.label, "new")
        self.assertEqual(d.reason, "gate0.not_followup_shape")

    def test_self_contained_smalltalk(self):
        for q in ("现在几点了", "今天星期几", "明天天气怎么样", "讲个笑话"):
            d = self.r.resolve(q, BY[AID])
            self.assertEqual(d.label, "new", q)
            self.assertIn(d.reason, ("gate0.self_contained", "gate0.not_followup_shape"), q)

    def test_bare_chitchat(self):
        for q in ("饿了", "下雨了", "你好"):
            d = self.r.resolve(q, BY[AID])
            self.assertEqual(d.label, "new", q)

    def test_attr_compound_is_not_a_subject(self):
        """「能不能线上办」「有截止时间吗」只是属性说法，不能当成另有主体。"""
        for q in ("能不能线上办", "有截止时间吗", "需要预约吗"):
            d = self.r.resolve(q, BY[AID])
            self.assertNotIn(d.reason, ("gate0.foreign_item", "gate0.opens_with_subject"), q)

    def test_explicit_anchor_skips_independence(self):
        """原句自己点了锚点 → 不按「另有主体」拦。"""
        d = self.r.resolve("转专业要什么材料", BY[AID])
        self.assertEqual(d.label, "followup")

    def test_followup_templates_pass_gate0(self):
        for q in ("那要带什么材料", "几天能办好", "要交钱吗", "找谁办",
                  "周末能办吗", "多久能下来", "要预约吗", "需要什么材料", "可以代办吗"):
            d = self.r.resolve(q, BY[AID])
            self.assertNotIn(d.reason.split(".")[0], ("gate0",), q)


class TestGate1(unittest.TestCase):
    def test_far_similarity_is_new(self):
        r = make(use_gate1=True, embed_fn=fake_embed,
                 anchor_vec_fn=lambda i: [1.0, 0.0])
        d = r.resolve("饿了", BY[AID])          # 闸门 0 先拦；这里换一句形状成立的
        self.assertEqual(d.label, "new")
        d2 = r.resolve("那要带什么材料", BY[AID])
        self.assertGreaterEqual(d2.sim, 0.4)

    def test_no_embed_falls_back(self):
        r = make(use_gate1=True, embed_fn=None)
        d = r.resolve("那要带什么材料", BY[AID])
        self.assertEqual(d.label, "followup")
        self.assertEqual(d.sim, 0.0)


class TestArbitration(unittest.TestCase):
    """仲裁表逐条锁死（直接调 arbitrate，绕开闸门 0 的形状判断）。"""

    def setUp(self) -> None:
        self.r = make()
        self.a_terms = self.r._anchor_terms(AID)

    def test_a_strong_other_topic(self):
        def rank_other(q):
            if q.startswith("转专业"):        # 补全串
                return [(AID, 99)]
            return [("tsg-jieyue", 92)]      # 原句单轮很强，且是别的事
        r = make()
        r.rank = rank_other
        d = r.arbitrate("图书馆开放时间", "转专业，图书馆开放时间", AID, self.a_terms)
        self.assertEqual(d.label, "new")
        self.assertEqual(d.reason, "arb.a_strong_other_topic")

    def test_a_strong_anchor(self):
        def rank_both(q):
            return [(AID, 96)]
        r = make()
        r.rank = rank_both
        d = r.arbitrate("转专业要什么材料", "转专业，转专业要什么材料", AID, self.a_terms)
        self.assertEqual(d.label, "followup")
        self.assertEqual(d.reason, "arb.a_strong_anchor")

    def test_a_weak_b_anchor(self):
        def rank_fn(q):
            if q.startswith("转专业"):
                return [(AID, 99)]
            return [("wd-noise", 20)]
        r = make()
        r.rank = rank_fn
        d = r.arbitrate("几天能办好", "转专业，几天能办好", AID, self.a_terms)
        self.assertEqual(d.label, "followup")
        self.assertEqual(d.reason, "arb.a_weak_b_anchor")

    def test_a_mid_b_anchor(self):
        def rank_fn(q):
            if q.startswith("转专业"):
                return [(AID, 98)]
            return [("tice", 70)]
        r = make()
        r.rank = rank_fn
        d = r.arbitrate("那要带什么材料", "转专业，那要带什么材料", AID, self.a_terms)
        self.assertEqual(d.label, "followup")
        self.assertEqual(d.reason, "arb.a_mid_b_anchor")

    def test_b_not_anchor(self):
        def rank_fn(q):
            if q.startswith("转专业"):
                return [("tsg-jieyue", 60)]
            return [("wd-noise", 20)]
        r = make()
        r.rank = rank_fn
        d = r.arbitrate("几天能办好", "转专业，几天能办好", AID, self.a_terms)
        self.assertEqual(d.label, "new")
        self.assertEqual(d.reason, "arb.no_b_anchor")

    def test_same_topic_by_title_substring(self):
        """「校外 VPN 使用说明」不是子条目，但标题含锚点基名，算同一件事。"""
        items = ITEMS + [{"id": "wd-x", "title": "校外访问 VPN 使用说明"}]
        r = ctx.Resolver(items, rank_fn=fake_rank)
        self.assertTrue(r._same_topic("wd-x", "it-vpn" if False else "zhuanzhuanye",
                                      r._anchor_terms("zhuanzhuanye")) is False)
        r2 = ctx.Resolver(ITEMS + [{"id": "wd-x", "title": "转专业学分认定"}], rank_fn=fake_rank)
        self.assertTrue(r2._same_topic("wd-x", AID, r2._anchor_terms(AID)))


class TestGate2(unittest.TestCase):
    """受约束改写：三条硬约束任一不过就丢弃改写。"""

    def _resolver(self, reply: dict) -> ctx.Resolver:
        return make(use_gate2=True, llm_fn=lambda prompt: reply)

    def test_accepts_prefixed_rewrite(self):
        r = self._resolver({"need": True, "query": "转专业要带什么材料"})
        d = r.resolve("要带什么材料", BY[AID])
        self.assertEqual(d.label, "followup")
        self.assertEqual(d.rewrite, "转专业要带什么材料")

    def test_rejects_new_words(self):
        # 原句仍是子串（开头补锚点），但多带了「身份证」这种锚点没提过的词
        r = self._resolver({"need": True, "query": "转专业要带什么材料身份证"})
        d = r.resolve("要带什么材料", BY[AID])
        self.assertEqual(d.rewrite, "")
        self.assertEqual(d.gate2, "fail_new_words")
        self.assertEqual(d.label, "followup")      # 丢弃改写，回落到仲裁

    def test_rejects_non_substring(self):
        r = self._resolver({"need": True, "query": "学分绩点怎么算"})
        d = r.resolve("要带什么材料", BY[AID])
        self.assertEqual(d.rewrite, "")
        self.assertEqual(d.gate2, "fail_substring")

    def test_veto_on_need_false(self):
        r = self._resolver({"need": False, "query": ""})
        r.gate2_veto = True
        d = r.resolve("要带什么材料", BY[AID])
        self.assertEqual(d.label, "new")
        self.assertTrue(d.reason.startswith("gate2.veto"))

    def test_no_veto_falls_back(self):
        r = self._resolver({"need": False, "query": ""})
        d = r.resolve("要带什么材料", BY[AID])
        self.assertEqual(d.label, "followup")

    def test_no_llm_skips_gate2(self):
        r = make(use_gate2=True, llm_fn=None)
        d = r.resolve("那要带什么材料", BY[AID])
        self.assertEqual(d.label, "followup")
        self.assertEqual(r.calls["gate2"], 0)


class TestDegraded(unittest.TestCase):
    def test_noun_overlap(self):
        lex = ctx.Lexicon(ITEMS, PLACES)
        self.assertGreaterEqual(ctx.noun_overlap("转专业要什么材料", {"转专业"}, lex), 1)
        self.assertEqual(ctx.noun_overlap("饿了", {"转专业"}, lex), 0)

    def test_no_anchor(self):
        r = make()
        d = r.resolve("那要带什么材料", None)
        self.assertEqual(d.label, "new")
        self.assertEqual(d.reason, "no_anchor")

    def test_normalize_and_base_title(self):
        self.assertEqual(ctx.norm("转专业，要带什么材料？"), "转专业要带什么材料")
        self.assertEqual(ctx.base_title("换宿舍 / 调宿"), "换宿舍")
        self.assertEqual(ctx.base_title("学生公费医疗报销（享受范围、就诊）"), "学生公费医疗报销")


if __name__ == "__main__":
    unittest.main()
