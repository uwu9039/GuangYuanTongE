"""答案层三个缺陷的回归测试（W1 联系方式错配 / W2 问句式锚点 / W3 强命中答非所问）。

跑法：
    PYTHONPATH=".venv/Lib/site-packages" python -m unittest tests.test_answer_quality -v

每个 W 至少两条：一条现场样例（父会话复现过的原句/原条目），一条泛化
（同机制在另一条上也得成立），防止「只把转专业那条写死」。
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_APP = None


def app():
    """整个模块只加载一次（ITEMS 39k 条，加载一次就够）。"""
    global _APP
    if _APP is None:
        import app as _a  # noqa: PLC0415
        _APP = _a
    return _APP


def part_text(part: dict | None) -> str:
    return str((part or {}).get("text") or "")


# --------------------------------------------------------------------- W1
class TestW1Contact(unittest.TestCase):
    """「找谁办」不能把别的事的联系方式端出来。"""

    def test_zhuanzhuanye_site_case(self):
        """现场：转专业找谁办曾返回电视学院短训班宋老师（65779262）。"""
        a = app()
        part = a._contact_part(a.ITEM_BY_ID["zhuanzhuanye"], "转专业 找谁办")
        text = part_text(part)
        self.assertTrue(part and part.get("found"), "转专业应当有联系方式")
        self.assertIn("65783223", text, f"应当给本科生院 65783223，实际：{text}")
        self.assertIn("本科生院", text)
        self.assertNotIn("65779262", text, "不许再把电视学院短训班的号码端出来")
        self.assertNotIn("电视学院", text)
        # 逐字校验：引文里必须含号码本身
        quotes = part.get("quotes") or []
        self.assertTrue(any("65783223" in str(q.get("quote") or "") for q in quotes),
                        "引文必须包含号码本身")

    def test_chengjidan_generalize(self):
        """泛化：成绩单办理的自己正文里就有本科生院邮箱，别再抓「不予承认」那句话。"""
        a = app()
        part = a._contact_part(a.ITEM_BY_ID["jwc-chengjidan"], "成绩单 找谁办")
        text = part_text(part)
        self.assertTrue(part and part.get("found"), "成绩单应当有联系方式")
        self.assertTrue("jiaowuchu@cuc.edu.cn" in text or "65783878" in text, text)
        self.assertNotIn("不予承认", text, "那句话不是联系方式")
        quotes = part.get("quotes") or []
        self.assertTrue(any(re.search(r"@|电话", str(q.get("quote") or "")) for q in quotes))

    def test_manual_contact_untouched(self):
        """回归：已有人工 contact 的条目必须原样输出（不许改坏）。"""
        a = app()
        items = [it for it in a.ITEMS if isinstance(it.get("contact"), str)
                 and it["contact"].strip()]
        self.assertGreaterEqual(len(items), 10, f"人工 contact 条目只剩 {len(items)} 条")
        for it in items[:14]:
            # 主答面（_answer_one）本来就原样端 contact；补的那一条不许把号码改掉
            main = a._answer_one(it, "contact", {}, "")
            self.assertIn(it["contact"], str(main.get("text") or ""), it["id"])
            part = a._contact_part(it, "找谁办")
            if part is None:
                continue
            phones = set(re.findall(r"\d{7,}", it["contact"]))
            got = set(re.findall(r"\d{7,}", part_text(part)))
            self.assertTrue(part_text(part) == it["contact"] or not phones or got & phones,
                            f"{it['id']} 补的那条跟人工联系方式打架：{part_text(part)}")

    def test_known_mismatch_not_reproduced(self):
        """5 条已知错配反例：不许再返回错的那一支。"""
        a = app()
        bad = {
            "zhuanzhuanye": "65779262",
            "jwc-zhuanhuan": "65779262",
            "jwc-chengjidan": "不予承认",
            "tuimian-tiaojian": "校园卡服务中心",
            "it-ganzhi": "13911312873",
            "xueji-biye": "虚心接受教师指导",
        }
        for iid, wrong in bad.items():
            part = a._contact_part(a.ITEM_BY_ID[iid], "找谁办")
            self.assertNotIn(wrong, part_text(part), f"{iid} 又命中了旧错配：{wrong}")
        # 「监督申诉」「宿舍网络报修」的手机号也曾经因为「正文里有个号码」被当成联系方式
        for iid, wrong in (("tuimian-tiaojian", "17610957295"),
                           ("it-ganzhi", "13911312873")):
            part = a._contact_part(a.ITEM_BY_ID[iid], "找谁办")
            self.assertNotIn(wrong, part_text(part), iid)


# --------------------------------------------------------------------- W2
class TestW2QuestionAnchor(unittest.TestCase):
    """问句式标题当锚点时，追问要接得上；无关问句仍判新话题。"""

    ITEMS = [
        {"id": "qa-qingjia", "title": "学生如何请假", "aliases": ["学生如何请假？"],
         "keywords": []},
        {"id": "xueji-qingjia", "title": "请假（病假 / 事假、审批与销假）",
         "aliases": ["请假"], "keywords": ["请假", "病假", "事假"]},
        {"id": "yjs-baodao", "title": "研究生新生报到（要带什么、请假）",
         "aliases": ["研究生新生报到"], "keywords": ["报到"]},
        {"id": "campus-food", "title": "食堂供餐时间", "aliases": ["食堂几点开"],
         "keywords": ["食堂"]},
        {"id": "sushe-huansu", "title": "换宿舍 / 调宿", "aliases": ["换宿舍"],
         "keywords": ["宿舍"]},
    ]

    def _resolver(self):
        from retrieval import context as ctx

        table = {
            # A 串：单轮自己答不动（落到别的主题）
            "那要带什么材料": [("yjs-baodao", 73), ("xueji-qingjia", 61)],
            "几天能办好": [("yjs-baodao", 66), ("xueji-qingjia", 60)],
            "找谁办": [("campus-food", 58)],
            "要预约吗": [("campus-food", 57)],
            # B 串：拼上锚点标题（或主题名）后落回同一族
            "学生如何请假，那要带什么材料": [("xueji-qingjia", 70), ("qa-qingjia", 63)],
            "请假，那要带什么材料": [("xueji-qingjia", 75), ("qa-qingjia", 66)],
            "学生如何请假，几天能办好": [("xueji-qingjia", 68), ("qa-qingjia", 62)],
            "请假，几天能办好": [("xueji-qingjia", 72), ("qa-qingjia", 65)],
            "学生如何请假，找谁办": [("xueji-qingjia", 67)],
            "请假，找谁办": [("xueji-qingjia", 70)],
            "学生如何请假，要预约吗": [("xueji-qingjia", 66)],
            "请假，要预约吗": [("xueji-qingjia", 69)],
            # 新话题：拼上锚点也不该落回请假族
            "食堂几点开，那要带什么材料": [("campus-food", 90)],
            "换宿舍，那要带什么材料": [("sushe-huansu", 88)],
        }

        def rank(q: str):
            return table.get(q) or []

        return ctx.build_resolver(self.ITEMS, rank_fn=rank, places=[], depts=[],
                                  use_gate1=False, use_gate2=False)

    def test_topic_name(self):
        from retrieval import context as ctx
        self.assertEqual(ctx.topic_name("学生如何请假"), "请假")
        self.assertEqual(ctx.topic_name("学生如何请假？"), "请假")
        self.assertEqual(ctx.topic_name("怎么申请换宿舍"), "换宿舍")
        # 本来就不是问句的标题不动
        self.assertEqual(ctx.topic_name("换宿舍 / 调宿"), "换宿舍 / 调宿")

    def test_four_followups_linked(self):
        """现场：锚点「学生如何请假」时 4 条追问全漏（arb.no_b_anchor）。"""
        r = self._resolver()
        anchor = self.ITEMS[0]
        for q in ("那要带什么材料", "几天能办好", "找谁办", "要预约吗"):
            d = r.resolve(q, anchor)
            self.assertEqual(d.label, "followup", f"{q} 漏接：{d.reason}")

    def test_family_needs_b_same_topic(self):
        """泛化：B 串不落回同一族时仍然不接。"""
        r = self._resolver()
        anchor = self.ITEMS[0]
        d = r.resolve("食堂几点开", anchor)
        self.assertEqual(d.label, "new", d.reason)
        d = r.resolve("换宿舍", anchor)
        self.assertEqual(d.label, "new", d.reason)


# --------------------------------------------------------------------- W3
class TestW3Facet(unittest.TestCase):
    """强命中免检也不能拿整体流程回答一个条目里没写的那一面。"""

    def test_zhuanzhuanye_fee_site_case(self):
        """现场：转专业要交钱吗 → 曾答「转专业办理程序」整体流程。"""
        a = app()
        it = a.ITEM_BY_ID["zhuanzhuanye"]
        got = a._facet_gate("转专业要交钱吗", it)
        self.assertIsNotNone(got, "问费用、条目没有费用，维度校验应当报出来")
        self.assertEqual(got.get("focus"), "fee")
        self.assertFalse(got.get("found"))
        self.assertIn("费用", part_text(got))

    def test_generalize_fee_absent(self):
        """泛化：换一条没有费用字段、正文也没写费用的条目，同样要报「没写」。"""
        a = app()
        it = a.ITEM_BY_ID["tuimian-tiaojian"]
        self.assertIsNone(it.get("fee"))
        got = a._facet_gate("推免要交钱吗", it)
        self.assertIsNotNone(got)
        self.assertEqual(got.get("focus"), "fee")

    def test_fee_present_not_gated(self):
        """条目真有费用字段时不报「没写」，照常答金额。"""
        a = app()
        paid = [it for it in a.ITEMS if str(it.get("fee") or "").strip()]
        self.assertGreater(len(paid), 20)
        for it in paid[:6]:
            self.assertIsNone(a._facet_gate(f"{it['title']} 要交钱吗", it), it["id"])
            part = a._answer_one(it, "fee", {}, "")
            self.assertTrue(part["found"], it["id"])

    def test_other_facets_not_false_alarm(self):
        """不带维度词 / 维度真有内容时不许误报。"""
        a = app()
        for q in ("转专业怎么申请", "转专业需要什么条件", "转专业什么时候开始"):
            self.assertIsNone(a._facet_gate(q, a.ITEM_BY_ID["zhuanzhuanye"]), q)

    def test_answer_does_not_fall_back_to_overview(self):
        """端到端：规则模式下「转专业要交钱吗」不许再端整体流程。"""
        a = app()
        r = a.answer_question("转专业要交钱吗", router_llm=False)
        self.assertTrue(r.get("matched"))
        texts = [str(p.get("text") or "") for p in (r.get("answers") or [])]
        joined = " ".join(texts)
        self.assertNotIn("整体流程", [p.get("label") for p in (r.get("answers") or [])])
        self.assertNotIn("秋季学期", joined, f"整体流程又被端上来了：{joined[:120]}")
        self.assertIn("费用", joined + str(r.get("note") or ""))


if __name__ == "__main__":
    unittest.main()
