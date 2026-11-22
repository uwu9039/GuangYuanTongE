"""普通问答最终答复 + 匿名赞踩的离线联调测试。

跑法：
    PYTHONPATH=".venv/Lib/site-packages" python -m unittest tests.test_feedback_api -v

不启动 HTTP：直接调用 app.py 的最终答复出口 `_finalize_answer()`，覆盖
普通/综合/未命中/澄清四种 payload；HTTP 四分支另由第四节实跑覆盖。
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TEST_DB = ROOT / ".tmp" / "test-feedback-api.sqlite3"


class FinalAnswerFeedbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _TEST_DB.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(_TEST_DB) + suffix).unlink()
            except OSError:
                pass
        os.environ["FEEDBACK_DB_PATH"] = str(_TEST_DB)
        import app  # noqa: PLC0415  只在测试里加载一次 39k 条事项

        cls.app = app
        import feedback  # noqa: PLC0415

        cls.fb = feedback
        feedback.init_db()

    def setUp(self):
        # 每个用例用不同的匿名标记，互不影响
        self.voter = self.fb.hash_voter("anon-test-" + self._testMethodName)

    def test_normal_and_synth_and_not_found_get_answer_id(self):
        normal = self.app._finalize_answer({"matched": True, "answers": [], "quotes": []})
        synth = self.app._finalize_answer({"matched": True, "synth": {"claims": []}}, "synth")
        missing = self.app._finalize_answer({"matched": False, "answers": []})
        self.assertTrue(normal.get("answer_id", "").startswith("ans_"))
        self.assertTrue(synth.get("answer_id", "").startswith("ans_"))
        self.assertTrue(missing.get("answer_id", "").startswith("ans_"))
        self.assertNotEqual(normal["answer_id"], synth["answer_id"])

    def test_clarify_menus_have_no_feedback(self):
        hold = self.app._finalize_answer({"matched": True, "hold": True, "followups": []})
        dis = self.app._finalize_answer({"matched": True, "disambiguate": [{"id": "x"}]})
        self.assertNotIn("answer_id", hold)
        self.assertNotIn("answer_id", dis)

    def test_same_question_twice_gets_two_ids_and_keeps_old_vote(self):
        first = self.app._finalize_answer({"matched": True, "answers": []})
        second = self.app._finalize_answer({"matched": True, "answers": []})
        self.assertNotEqual(first["answer_id"], second["answer_id"])
        self.fb.set_vote(first["answer_id"], self.voter, "up")
        self.assertEqual(self.fb.get_vote(first["answer_id"], self.voter)["vote"], "up")
        self.assertEqual(self.fb.get_vote(second["answer_id"], self.voter)["vote"], None)

    def test_up_change_revoke_and_counts(self):
        out = self.app._finalize_answer({"matched": True, "answers": []})
        aid = out["answer_id"]
        self.assertEqual(self.fb.set_vote(aid, self.voter, "up")["counts"], {"up": 1, "down": 0})
        self.assertEqual(self.fb.set_vote(aid, self.voter, "up")["counts"], {"up": 1, "down": 0})
        self.assertEqual(self.fb.set_vote(aid, self.voter, "down")["counts"], {"up": 0, "down": 1})
        self.assertEqual(self.fb.set_vote(aid, self.voter, None)["counts"], {"up": 0, "down": 0})
        self.assertEqual(self.fb.get_vote(aid, self.voter)["vote"], None)

    def test_channels_are_counted_separately(self):
        official = self.fb.attach({"matched": True}, "official", kind="normal")
        social = self.fb.attach({"mode": "llm"}, "social", kind="llm")
        self.fb.set_vote(official["answer_id"], self.voter, "up")
        self.fb.set_vote(social["answer_id"], self.voter, "down")
        stats = self.fb.stats()
        self.assertGreaterEqual(stats["official"]["up"], 1)
        self.assertGreaterEqual(stats["social"]["down"], 1)

    def test_tice_tool_link_only_when_relevant(self):
        self.assertTrue(self.app._tool_links_for("体测怎么算分"))
        self.assertTrue(self.app._tool_links_for("国家奖学金评选条件"))
        self.assertTrue(self.app._tool_links_for("体测成绩影响奖学金评选吗"))
        self.assertEqual(self.app._tool_links_for("体测免测怎么办"), [])
        self.assertEqual(self.app._tool_links_for("体育课免修怎么申请"), [])
        self.assertEqual(self.app._tool_links_for("奖学金多少钱"), [])
        self.assertEqual(self.app._tool_links_for("校园卡怎么补办"), [])

    def test_tice_tool_link_attached_to_final_answer(self):
        out = self.app._finalize_answer({"matched": True, "answers": []}, question="体测怎么算分")
        self.assertEqual(out["tool_links"][0]["url"], "/tice.html")
        plain = self.app._finalize_answer({"matched": True, "answers": []}, question="体测免测怎么办")
        self.assertNotIn("tool_links", plain)

    def test_unknown_answer_and_bad_vote(self):
        with self.assertRaises(KeyError):
            self.fb.set_vote("ans_nope", self.voter, "up")
        out = self.app._finalize_answer({"matched": True, "answers": []})
        with self.assertRaises(ValueError):
            self.fb.set_vote(out["answer_id"], self.voter, "sideways")


if __name__ == "__main__":
    unittest.main()
