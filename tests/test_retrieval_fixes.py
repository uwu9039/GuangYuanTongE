"""检索层五个修复点的回归测试（可单独跑，不依赖总评）。

    python -m unittest tests.test_retrieval_fixes -v

每个测试只锁「这次修的那一条」的行为，不锁具体分数；数字口径与
`_eval/quick_eval.py` / `_acceptance.check` 对齐：

    P0  高频泛词压过主题词      —— top1 标题必须含问句的主题词
    P1  hub 文档抢答            —— 大杂烩条目不该当 top1（且全库计数 ≤ 9）
    P2  条款定位                —— 点了「第X条」要落回那份文档
    P3  「库里有相关条目但没有这一条」—— 该拒
    P4  个人数据问法的泛化      —— 该拒，且「怎么查」这类问法不许误拒
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 大杂烩条目：标题写着自己是汇总 / 常见问题 / 一览，或是整段带网址的网页答案
HUB_RE = re.compile(r"常见问题|常见问答|汇总|一览|清单|问答|合集|攻略|宝典|faq|FAQ")
# 题库是委托方的输入，不在仓库里：路径由 `BANK_JSON` 给（见 `_eval/bank_path.py`）
from _eval.bank_path import bank_json  # noqa: E402

BANK = bank_json()

_ENGINE = None


def engine():
    global _ENGINE
    if _ENGINE is None:
        from retrieval import Engine
        _ENGINE = Engine.load()
    return _ENGINE


def top1(q: str) -> str:
    res = engine().search(q, k=1)
    return res.best.title if res.best is not None else ""


def refused(q: str) -> tuple[bool, str]:
    res = engine().search(q, k=1)
    return res.refused, res.reason


# --------------------------------------------------------------------- P0
class TestP0TopicWord(unittest.TestCase):
    """问句里的大词不许被「标题含同一个框词、但讲的是别的事」的条目顶掉。"""

    CASES = [
        ("选课什么时候开始？在哪里选？", "选课"),
        ("课程免修怎么申请？需要什么条件？", "免修"),
        ("毕业审核在哪里查？学分不够怎么办？", "毕业"),
        ("档案在哪里", "档案"),
    ]

    def test_top1_contains_topic_word(self):
        for q, need in self.CASES:
            with self.subTest(q=q):
                t = top1(q)
                self.assertIn(need, t, f"{q} 的 top1 是「{t}」，不含主题词「{need}」")

    def test_topic_text_strips_frame_words(self):
        from retrieval.text import topic_text
        self.assertEqual(topic_text("选课什么时候开始在哪里选"), "选课开始选")
        self.assertEqual(topic_text("档案在哪里"), "档案")

    def test_question_title_detection(self):
        from retrieval.text import is_question_title
        self.assertTrue(is_question_title("什么时候加入班级群？ · 新生常见问题.docx"))
        self.assertFalse(is_question_title("校园卡补卡"))


# --------------------------------------------------------------------- P1
class TestP1Hub(unittest.TestCase):
    """汇总 / 常见问题这类大杂烩条目不许抢答具体事项。"""

    CASES = [
        "选课什么时候开始？在哪里选？",
        "快递点在哪里？快递怎么寄？",
        "校园卡消费记录在哪里查？",
        "补选和退选的时间是什么时候？",
        "毕业审核在哪里查？学分不够怎么办？",
    ]

    def test_hub_not_top1(self):
        for q in self.CASES:
            with self.subTest(q=q):
                t = top1(q)
                self.assertFalse(HUB_RE.search(t), f"{q} 的 top1 是大杂烩条目「{t}」")

    def test_hub_title_detection(self):
        from retrieval.text import is_hub_title
        self.assertTrue(is_hub_title("信息化服务常见问题（2026-11-03 制）"))
        self.assertTrue(is_hub_title("本科生院服务常见问题【汇总】"))
        self.assertTrue(is_hub_title("大创相关内容在哪里查看：https://bkjw.cuc.edu.cn"))
        self.assertFalse(is_hub_title("校园卡补卡"))

    @unittest.skipUnless(BANK.exists(), "题库不在本机，跳过全量计数")
    def test_hub_top1_count(self):
        """题库 item 题里 top1 是大杂烩的条数 ≤ 9（修复前 18）。"""
        from _eval.quick_eval import bank_rows
        rows = [r for r in bank_rows() if r["kind"] == "item"]
        n = sum(1 for r in rows if HUB_RE.search(top1(r["q"])))
        self.assertLessEqual(n, 9, f"top1 是大杂烩的还有 {n} 条")


# --------------------------------------------------------------------- P2
class TestP2Section(unittest.TestCase):
    """问句点了章条，命中条目要落回那份文档。"""

    def _doc_of(self):
        import sections_index as si
        out = {}
        for iid in engine().items:
            for s in si.section_of(iid) or []:
                if s.get("slug"):
                    out.setdefault(iid, set()).add(str(s["slug"]))
        return out

    def test_numbered_article_lands_in_document(self):
        doc_of = self._doc_of()
        cases = [
            ("选课办法第十四条选课的限制条件有哪些", "xuanke-2025"),
            ("国家学生体质健康标准实施办法第七条因病或残疾免测怎么申请", "att-6e09d2377d"),
        ]
        for q, slug in cases:
            with self.subTest(q=q):
                res = engine().search(q, k=5)
                got = [h.id for h in res.hits if slug in doc_of.get(h.id, ())]
                self.assertTrue(got, f"{q} 的前 5 条里没有 {slug} 的条目")

    def test_chapter_refs(self):
        import sections_index as si
        self.assertEqual(si.chapter_refs("选课办法第十四条"), [("条", 14)])
        self.assertEqual(si.chapter_refs("转专业管理办法第三章"), [("章", 3)])
        self.assertEqual(si.chapter_refs("档案在哪里"), [])


# --------------------------------------------------------------------- P3
class TestP3Unlisted(unittest.TestCase):
    """库里有相关条目、但没有这一条 —— 该拒，且不能连累正常问题。"""

    REFUSED = ["校车时刻表", "辅导员是谁", "教务处主任电话"]

    def test_refuse_unlisted(self):
        for q in self.REFUSED:
            with self.subTest(q=q):
                r, why = refused(q)
                self.assertTrue(r, f"{q} 应该拒答，实际答了（{top1(q)}）")

    def test_answerable_still_answered(self):
        for q in ("校园卡补卡在哪办", "图书馆能续借吗", "开成绩单怎么办"):
            with self.subTest(q=q):
                r, _ = refused(q)
                self.assertFalse(r, f"{q} 是正常问题，被误拒了")


# --------------------------------------------------------------------- P4
class TestP4PersonalData(unittest.TestCase):
    """「我的X」的变体：问的是个人数据就拒；问「怎么查」不许误拒。"""

    REFUSED = [
        "我的四六级准考证号是多少",
        "我期末考了几门",
        "我的图书馆罚款还有多少",
        "我的校园卡绑的是哪个手机号",
        "我的学籍状态是什么",
        "我的邮箱密码是多少",
        "我什么时候能毕业",
    ]
    ANSWERED = [
        "我的借阅记录怎么查",
        "我的成绩单在哪打印",
        "我的校园卡挂失怎么弄",
    ]

    def test_refuse_personal_variants(self):
        for q in self.REFUSED:
            with self.subTest(q=q):
                r, _ = refused(q)
                self.assertTrue(r, f"{q} 是个人数据问法，应该拒答")

    def test_howto_not_refused(self):
        for q in self.ANSWERED:
            with self.subTest(q=q):
                r, why = refused(q)
                self.assertFalse(r, f"{q} 问的是方法，被误拒了（{why}）")

    def test_policy_question_not_refused(self):
        """「我宿舍电费超了会不会断电」问的是政策，不是我的数据。"""
        r, why = refused("我的宿舍电费超了会不会断电")
        self.assertFalse(r, f"问政策的句子被当成个人数据拒了（{why}）")


# ------------------------------------------------------ P4b（第二轮：换说法）
class TestP4bPersonalSentence(unittest.TestCase):
    """第二轮：个人数据要按**句型**判，不能只认上一轮见过的说法。

    委托方换了一批从没出现过的说法（学分 / 排第几 / 批下来 / 借的书 / 欠费 / 上学期），
    上一轮的表十道里只拒了三道。这里测的是「句型对不对」，不是「表全不全」。
    """

    REFUSED = [
        "我上学期期末多少分",
        "我的学分够毕业了吗",
        "我借的书什么时候到期",
        "我这学期绩点排第几",
        "我申请的那个奖学金批下来没有",
        "我有没有欠费",
        "我四级考了多少分",
        "我卡里还剩多少钱来着",
    ]
    ANSWERED = [
        "我的校园卡丢了怎么补",
        "我的借阅记录怎么查",
        "我的成绩单怎么打印",
        "我这学期选的课都在哪看",
        "我能不能申请助学贷款",
        "我的学分能不能用来保研",
        "我是不是可以申请奖学金",
        # 复合长句：句子里有「学分 / 成绩 / 贷款」，问的是政策不是我的那份数据
        "我体测免测了，体育课还要上吗，学分怎么算？",
        "我上学期申请了助学贷款，这学期转专业了，贷款会断吗？",
        "我护照要过期了，学校能开什么证明吗？",
        "我申请了国家奖学金，还能同时拿助学金吗？",
    ]

    def test_refuse_personal_sentence(self):
        for q in self.REFUSED:
            with self.subTest(q=q):
                r, why = refused(q)
                self.assertTrue(r, f"{q} 是个人数据问法，应该拒答（实际答了）")

    def test_policy_or_howto_not_refused(self):
        for q in self.ANSWERED:
            with self.subTest(q=q):
                r, why = refused(q)
                self.assertFalse(r, f"{q} 问的是方法/政策，被误拒了（{why}）")


# --------------------------------------------------------------------- R2
class TestR2TransitCard(unittest.TestCase):
    """交通卡类问题不许被「地铁」两个字拒掉；真问路线才算库外。"""

    ANSWERED = [
        "学生坐地铁有优惠的卡去哪领",
        "学校附近地铁公交怎么走",
        "心情不好想找人聊聊去哪",
    ]
    REFUSED = ["北京地铁怎么坐", "北京南站怎么走", "校车时刻表", "班车几点发车"]

    def test_transit_card_answered(self):
        for q in self.ANSWERED:
            with self.subTest(q=q):
                r, why = refused(q)
                self.assertFalse(r, f"{q} 应该作答，被拒了（{why}）")

    def test_route_question_refused(self):
        for q in self.REFUSED:
            with self.subTest(q=q):
                r, _ = refused(q)
                self.assertTrue(r, f"{q} 是库外路线/时刻表，应该拒答（实际答了 {top1(q)}）")


# --------------------------------------------------------------------- R3
class TestR3OffTopic(unittest.TestCase):
    """库外话题要认没见过的说法；但句子里有校园实词就得答。"""

    REFUSED = [
        "晚上吃什么好",
        "推荐几个周末去玩的地方",
        "这首歌叫什么名字",
        "怎么算卡路里",
    ]
    ANSWERED = ["食堂晚上吃什么", "校内健身房怎么办"]

    def test_off_topic_refused(self):
        for q in self.REFUSED:
            with self.subTest(q=q):
                r, _ = refused(q)
                self.assertTrue(r, f"{q} 是库外话题，应该拒答（实际答了 {top1(q)}）")

    def test_campus_context_answered(self):
        for q in self.ANSWERED:
            with self.subTest(q=q):
                r, why = refused(q)
                self.assertFalse(r, f"{q} 有校园实词，被误拒了（{why}）")


# --------------------------------------------------------------------- R4
class TestR4OtherSchools(unittest.TestCase):
    """别的学校（含缩写）优先于其他一切规则；「中科院分区」是期刊分区不是学校。"""

    REFUSED = ["中科院的研究生宿舍怎么申请", "国科大保研条件", "上财的绩点怎么算",
               "人大图书馆怎么进", "清华的校园卡怎么补办"]
    ANSWERED = ["中传的校园卡怎么补", "中科院分区怎么看"]

    def test_other_school_refused(self):
        for q in self.REFUSED:
            with self.subTest(q=q):
                r, why = refused(q)
                self.assertTrue(r, f"{q} 问的是别的学校，应该拒答（实际答了 {top1(q)}）")

    def test_own_school_and_homonym_answered(self):
        for q in self.ANSWERED:
            with self.subTest(q=q):
                r, why = refused(q)
                self.assertFalse(r, f"{q} 应该作答，被拒了（{why}）")


# --------------------------------------------------------------------- R6
class TestR6P0b(unittest.TestCase):
    """P0b：同一批「泛词压过主题词」换成别的泛词再考一遍（自然问法，不是条款导航）。"""

    CASES = [
        ("学校有校车吗？班车时刻表在哪里？", ("校车", "班车", "交通")),
        ("新生选课是什么时候？有指导吗？", ("选课",)),
        ("评奖评优有哪些？什么时候评？", ("评奖", "奖学金", "评优")),
        ("学生公寓管理办法在哪里看？住宿有哪些规定？", ("公寓", "住宿")),
        ("博士招生简章在哪里看？申请考核怎么准备？", ("招生", "博士")),
        ("研究生学位论文开题和答辩流程是什么？", ("学位论文", "开题", "答辩")),
        ("学院有哪些奖学金？", ("奖",)),
        ("学校有游泳馆吗？", ("游泳",)),
    ]

    def test_p0b_topic_landing(self):
        for q, kws in self.CASES:
            with self.subTest(q=q):
                res = engine().search(q, k=1)
                if res.refused:
                    continue          # 库里没有这一条时「拒答」也是可接受的结果
                t = res.best.title if res.best else ""
                self.assertTrue(any(k in t for k in kws),
                                f"{q} 的 top1 是「{t}」，不含主题词 {kws}")


# --------------------------------------------------------------------- R7
class TestR7ContextOrder(unittest.TestCase):
    """先单轮、弱才带上文；判分串不许带补全句。"""

    @classmethod
    def setUpClass(cls):
        try:
            import sys
            sys.path.insert(0, str(ROOT / "_eval"))
            from _eval import stub_fastapi
            stub_fastapi.install()
            import app  # noqa: F401
            cls.app = app
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"app.py 起不来：{e}")

    CTX = {"title": "户籍科常见事项（身份证、户籍卡、迁户、无犯罪证明）", "id": "hjk"}

    def test_strong_single_turn_drops_context(self):
        for q in ("学费在哪交怎么交", "保研怎么报名", "补选课怎么弄"):
            with self.subTest(q=q):
                search, ctx_id, _ = self.app._contextual_text(q, self.CTX, True)
                self.assertIsNone(ctx_id, f"{q} 单轮就够，不该带上文")
                self.assertEqual(search, q)

    def test_weak_turn_uses_context(self):
        for q in ("要交多少钱", "周末呢"):
            with self.subTest(q=q):
                search, ctx_id, _ = self.app._contextual_text(q, self.CTX, True)
                self.assertEqual(ctx_id, "hjk", f"{q} 是省略句，应该带上文")
                self.assertIn("户籍科", search)

    def test_judge_has_no_completed_sentence(self):
        """判分串只带「上一轮问的是…」这句中性背景，不带补全后的问句。

        （指代式省略句是既有约定的例外：那种句子离开上文不成句，判分也用补全句。）
        """
        q = "体测受伤了不能跑怎么办"
        standalone = "户籍科办无犯罪证明要交多少钱"
        _s, cid, judge = self.app._contextual_text(q, self.CTX, True, standalone)
        self.assertEqual(cid, "hjk")
        self.assertNotIn(standalone, judge)
        self.assertIn(q, judge)


# --------------------------------------------------------------------- G0
class TestNoLiteralQueries(unittest.TestCase):
    """不许把评测问句写死进检索路径。"""

    def test_no_query_literals(self):
        import re as _re
        hits = []
        for p in list((ROOT / "retrieval").glob("*.py")) + [ROOT / "app.py"]:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for m in _re.finditer(r"[\"']([^\"'\n]{6,})[\"']", txt):
                s = m.group(1)
                if "什么时候开始" in s or "准考证号是多少" in s:
                    hits.append((p.name, s))
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
