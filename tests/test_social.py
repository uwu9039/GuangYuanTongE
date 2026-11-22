"""校园经验页的离线测试：检索、引用校验、歧义、缺口、追问与降级。

跑法（本机沙箱里没有 pytest，统一用 stdlib unittest）：
    PYTHONPATH=".venv/Lib/site-packages" python -m unittest tests.test_social -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import social


POSTS = [
    {
        "post_id": "bili:cv14026308", "platform": "bilibili",
        "url": "https://www.bilibili.com/read/cv14026308",
        "title": "美食测评——中传食堂篇～", "author": "从前有只皮皮落",
        "summary": "星光四楼馋嘴鸡，味道7分，鸡肉嫩，配菜有豆芽小白菜金针菇，馋嘴鸡16元，馋嘴鱼17元。",
        "topics": ["星光"], "entities": ["星光食堂"], "published_at": "2021-11-16",
        "collected_at": "2026-11-22T01:00:00+08:00", "school_evidence": "中国传媒大学",
        "title_only": False,
    },
    {
        "post_id": "bili:cv14052189", "platform": "bilibili",
        "url": "https://www.bilibili.com/read/cv14052189",
        "title": "美食测评——中传食堂篇④", "author": "从前有只皮皮落",
        "summary": "星光四楼麻辣烫，藤椒味和猪肚鸡汤味，人均20左右，饭点很火爆，来晚要排15分钟以上的队。",
        "topics": ["星光"], "entities": ["星光食堂"], "published_at": "2021-11-18",
        "collected_at": "2026-11-22T01:00:00+08:00", "school_evidence": "中国传媒大学",
        "title_only": False,
    },
    {
        "post_id": "bili:cv14026017", "platform": "bilibili",
        "url": "https://www.bilibili.com/read/cv14026017",
        "title": "美食测评——中传食堂篇～", "author": "从前有只皮皮落",
        "summary": "北苑二楼崔永元真面，麻辣牛肉面9分，汤浓肉味足，面条弹牙。",
        "topics": ["北苑食堂"], "entities": ["北苑餐厅"], "published_at": "2021-11-16",
        "collected_at": "2026-11-22T01:00:00+08:00", "school_evidence": "中国传媒大学",
        "title_only": False,
    },
    {
        "post_id": "bili:BV13t411Y7yz", "platform": "bilibili",
        "url": "https://www.bilibili.com/video/BV13t411Y7yz/",
        "title": "中国传媒大学食堂探秘，南苑食堂、星光食堂都尝尝！", "author": "逛吃小猪猪",
        "summary": "打卡南苑食堂、星光食堂，星光的锅盔真的好好吃。",
        "topics": ["星光", "南苑"], "entities": ["星光食堂", "南苑餐厅"],
        "published_at": "2018-12-21", "collected_at": "2026-11-22T01:00:00+08:00",
        "school_evidence": "中国传媒大学", "title_only": False,
    },
    {
        "post_id": "bili:BV1Z8Ev65EwB", "platform": "bilibili",
        "url": "https://www.bilibili.com/video/BV1Z8Ev65EwB/",
        "title": "第一视角带你逛逛中传（超长一镜到底）", "author": "西柚西er",
        "summary": "毕业前完整记录中传校园，每个假期都在装修，2026年夏天的校园。",
        "topics": ["校园日常"], "entities": ["校园"], "published_at": "2026-06-11",
        "collected_at": "2026-11-22T01:00:00+08:00", "school_evidence": "中传",
        "title_only": False,
    },
    {
        "post_id": "bili:BV1ATJUzrEwo", "platform": "bilibili",
        "url": "https://www.bilibili.com/video/BV1ATJUzrEwo/",
        "title": "中国传媒大学北苑餐厅早餐", "author": "传媒大学杨老头",
        "summary": "记录北苑餐厅早餐。",
        "topics": ["北苑食堂"], "entities": ["北苑餐厅"], "published_at": "2025-09-24",
        "collected_at": "2026-11-22T01:00:00+08:00", "school_evidence": "中国传媒大学",
        "title_only": False,
    },
]


class SocialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 系统临时目录在当前沙箱里不可写/不可清理，测试产物统一放工作区的 .tmp。
        cls._tmp = Path.cwd() / ".tmp" / "test-social-data"
        cls._tmp.mkdir(parents=True, exist_ok=True)
        cls.path = cls._tmp / "posts.jsonl"
        cls.path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in POSTS), encoding="utf-8")
        cls._old_path = social.LOG_PATH
        social.LOG_PATH = cls._tmp / "collection-log.json"
        social.STORE = social.Store(cls.path)
        social.STORE.reload(force=True)

    @classmethod
    def tearDownClass(cls):
        social.LOG_PATH = cls._old_path

    def make_req(self, **kwargs):
        return social.SocialAskReq(**{"text": "有人分享过星光的哪些吃食？", **kwargs})

    def fake_chat(self, payload):
        def _call(messages, **kwargs):
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("校园经验助手", messages[0]["content"])
            return json.dumps(payload, ensure_ascii=False)
        return _call

    def test_posts_list_and_topic_filter(self):
        self.assertEqual(len(social.STORE.posts), len(POSTS))
        star = social._filter_posts("星光")
        self.assertEqual(len(star), 3)
        self.assertTrue(all("星光" in p["topics"] for p in star))
        public = social.STORE.public_post(star[0], 1)
        self.assertEqual(public["date_label"], public["published_at"])
        self.assertEqual(social.STORE.topics()[0]["name"], "北苑食堂")

    def test_llm_claims_are_validated_and_links_stripped(self):
        payload = {
            "status": "answered",
            "claims": [
                {"text": "有作者提到星光四楼的馋嘴鸡，价格 16 元。", "post_ids": ["bili:cv14026308"]},
                {"text": "编造的引用", "post_ids": ["bili:not-exist"]},
                {"text": "带链接 https://evil.example/x", "post_ids": ["bili:cv14052189"]},
            ],
            "note": "只代表个别作者体验。",
        }
        out = social._build_answer(self.make_req(), self.fake_chat(payload), lambda: True)
        self.assertEqual(out["mode"], "llm")
        self.assertEqual(out["status"], "answered")
        self.assertEqual([c["text"] for c in out["claims"]], [
            "有作者提到星光四楼的馋嘴鸡，价格 16 元。",
            "带链接",
        ])
        self.assertTrue(all("http" not in c["text"] for c in out["claims"]))
        self.assertEqual(out["sources"][0]["post_id"], "bili:cv14026308")
        self.assertTrue(all(s["url"].startswith("https://") for s in out["sources"]))

    def test_model_failure_falls_back_to_excerpt(self):
        def boom(*args, **kwargs):
            raise RuntimeError("model down")

        out = social._build_answer(self.make_req(), boom, lambda: True)
        self.assertEqual(out["mode"], "excerpt")
        self.assertEqual(out["status"], "partial")
        self.assertTrue(out["claims"])
        self.assertIn("直接摘录", out["note"])
        self.assertTrue(out["sources"])

    def test_no_key_uses_excerpt_without_calling_model(self):
        called = {"n": 0}

        def must_not_call(*args, **kwargs):
            called["n"] += 1
            return "{}"

        out = social._build_answer(self.make_req(), must_not_call, lambda: False)
        self.assertEqual(called["n"], 0)
        self.assertEqual(out["mode"], "excerpt")
        self.assertIn("没有可用的模型凭据", out["note"])

    def test_empty_result_is_insufficient_and_sources_empty(self):
        out = social._build_answer(social.SocialAskReq(text="量子力学怎么复习"), None, lambda: False)
        self.assertEqual(out["status"], "insufficient")
        self.assertEqual(out["mode"], "empty")
        self.assertEqual(out["sources"], [])
        self.assertIn("暂时没有", out["note"])

    def test_bare_star_asks_which_place(self):
        out = social._build_answer(social.SocialAskReq(text="星光"), None, lambda: False)
        self.assertEqual(out["mode"], "ambiguous")
        self.assertEqual([o["label"] for o in out["ambiguous"]["options"]], ["星光食堂", "星光超市"])

    def test_other_star_place_has_no_hit(self):
        out = social._build_answer(social.SocialAskReq(text="星光影视园有什么好吃的"), None, lambda: False)
        self.assertEqual(out["status"], "insufficient")
        self.assertEqual(out["mode"], "empty")
        self.assertIn("同名地点", out["note"])

    def test_official_question_redirects(self):
        out = social._build_answer(social.SocialAskReq(text="校园卡怎么补办"), None, lambda: False)
        self.assertEqual(out["status"], "official_redirect")
        self.assertTrue(out["redirect"]["url"].startswith("/field?q="))
        self.assertIn("办事向导", out["note"])

    def test_followup_uses_context_topic_and_post_ids(self):
        req = social.SocialAskReq(
            text="那哪篇提过排队",
            context_topic="有人分享过星光的哪些吃食？",
            context_post_ids=["bili:cv14052189"],
        )
        out = social._build_answer(req, None, lambda: False)
        self.assertEqual(out["mode"], "excerpt")
        self.assertTrue(out["sources"])
        self.assertEqual(out["sources"][0]["post_id"], "bili:cv14052189")
        self.assertTrue(out["followup"]["used_context"])

    def test_old_posts_cannot_confirm_today(self):
        out = social._build_answer(social.SocialAskReq(text="星光四楼的麻辣烫今天还开吗"), None, lambda: False)
        self.assertEqual(out["status"], "partial")
        self.assertIn("不能确认今天", out["note"])
        self.assertTrue(out["sources"])

    def test_place_filter_does_not_mix_beiyuan_and_nanyuan(self):
        out = social._build_answer(social.SocialAskReq(text="北苑食堂有什么值得吃的"), None, lambda: False)
        ids = [s["post_id"] for s in out["sources"]]
        self.assertTrue(ids)
        self.assertNotIn("bili:BV13t411Y7yz", ids)

    def test_reload_skips_bad_lines(self):
        bad = self._tmp / "bad.jsonl"
        bad.write_text('{"post_id":"x"}\nnot json\n' + json.dumps(POSTS[0], ensure_ascii=False), encoding="utf-8")
        s = social.Store(bad)
        self.assertEqual(len(s.posts), 1)
        self.assertEqual(s.skipped, 2)


if __name__ == "__main__":
    unittest.main()
