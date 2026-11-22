"""反馈存储层单测：不 import app、不联网、不写仓库里的 runtime/。

跑法（本机全局 python + venv 依赖，按仓库既有约定）：

    $env:PYTHONPATH="$PWD\\.venv\\Lib\\site-packages"; python -m unittest tests.test_feedback_store -v

覆盖的是「一票就是一行」这件事的各个角度：重复投、改票、撤票、多人聚合、
重开库读回、按渠道统计、只读库写失败、meta 脱敏、attach 不拖垮问答。
"""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
import threading
import unittest
import uuid
from pathlib import Path

import feedback


def _temp_root() -> Path:
    """临时目录：用 mkdir 而不是 tempfile.mkdtemp。

    mkdtemp 会建 0700 的目录，本机沙箱下子进程反而写不进去（ACL），
    所以这里建普通目录，实在不行退到仓库的 .tmp/。
    """
    name = f"gyt-feedback-{uuid.uuid4().hex}"
    for base in (Path(tempfile.gettempdir()), feedback.ROOT / ".tmp"):
        try:
            path = base / name
            path.mkdir(parents=True, exist_ok=False)
            return path
        except OSError:
            continue
    raise unittest.SkipTest("没有可写的临时目录")


class FeedbackStoreTest(unittest.TestCase):
    """每个用例一个独立库：FEEDBACK_DB_PATH 指到临时目录。"""

    def setUp(self) -> None:
        self.root = _temp_root()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.db = self.root / "runtime" / "feedback.sqlite3"
        self._old_env = os.environ.get(feedback.DB_PATH_ENV)
        os.environ[feedback.DB_PATH_ENV] = str(self.db)
        self.addCleanup(self._restore_env)
        # 有几个用例故意制造登记失败，日志里的 traceback 别混进测试输出
        logger = logging.getLogger("feedback")
        self._old_level = logger.level
        logger.setLevel(logging.CRITICAL)
        self.addCleanup(logger.setLevel, self._old_level)
        feedback.init_db()

    def _restore_env(self) -> None:
        if self._old_env is None:
            os.environ.pop(feedback.DB_PATH_ENV, None)
        else:
            os.environ[feedback.DB_PATH_ENV] = self._old_env

    # ---------------------------------------------------------- helpers

    def _rows(self, sql: str, args: tuple = ()) -> list[tuple]:
        con = sqlite3.connect(str(self.db))
        try:
            return con.execute(sql, args).fetchall()
        finally:
            con.close()

    def _meta(self, answer_id: str) -> dict:
        import json

        rows = self._rows("SELECT meta_json FROM answers WHERE answer_id = ?", (answer_id,))
        self.assertEqual(len(rows), 1)
        raw = rows[0][0]
        return json.loads(raw) if raw else {}

    # ---------------------------------------------------------- 登记

    def test_register_both_channels(self) -> None:
        official = feedback.register_answer("official", meta={"item_id": "zhuanzhuanye"})
        social = feedback.register_answer("social", kind="llm", meta={"mode": "llm"})
        self.assertTrue(official.startswith("ans_"))
        self.assertTrue(social.startswith("ans_"))
        self.assertNotEqual(official, social)
        self.assertGreaterEqual(len(official), len("ans_") + 12)
        rows = self._rows("SELECT answer_id, channel, kind FROM answers ORDER BY channel")
        self.assertEqual(
            [(r[0], r[1], r[2]) for r in rows],
            [(official, "official", "normal"), (social, "social", "llm")],
        )
        self.assertEqual(feedback.vote_counts(official), {"up": 0, "down": 0})

    def test_unknown_channel_rejected(self) -> None:
        with self.assertRaises(ValueError):
            feedback.register_answer("blog")
        with self.assertRaises(ValueError):
            feedback.register_answer("")
        # attach 走「不让登记失败毁了答复」的路子：不抛异常，也不挂 answer_id
        payload = {"answer": "x"}
        self.assertIs(feedback.attach(payload, "weibo"), payload)
        self.assertNotIn("answer_id", payload)
        # 拒绝之后库里不该留下半条记录
        self.assertEqual(self._rows("SELECT COUNT(*) FROM answers")[0][0], 0)

    def test_init_db_idempotent(self) -> None:
        answer_id = feedback.register_answer("official")
        feedback.init_db()
        feedback.init_db()
        self.assertEqual(feedback.get_vote(answer_id, None)["counts"], {"up": 0, "down": 0})

    # ---------------------------------------------------------- 投票

    def test_same_voter_up_twice_counts_once(self) -> None:
        answer_id = feedback.register_answer("official")
        voter = feedback.hash_voter("voter-1")
        first = feedback.set_vote(answer_id, voter, "up")
        second = feedback.set_vote(answer_id, voter, "up")
        self.assertEqual(first["counts"], {"up": 1, "down": 0})
        self.assertEqual(second["counts"], {"up": 1, "down": 0})
        self.assertEqual(second["vote"], "up")
        self.assertEqual(
            self._rows("SELECT COUNT(*) FROM votes WHERE answer_id = ?", (answer_id,))[0][0], 1
        )

    def test_change_vote_overwrites(self) -> None:
        answer_id = feedback.register_answer("official")
        voter = feedback.hash_voter("voter-1")
        feedback.set_vote(answer_id, voter, "up")
        changed = feedback.set_vote(answer_id, voter, "down")
        self.assertEqual(changed["vote"], "down")
        self.assertEqual(changed["counts"], {"up": 0, "down": 1})
        self.assertEqual(
            self._rows("SELECT vote FROM votes WHERE answer_id = ?", (answer_id,)),
            [("down",)],
        )

    def test_revoke_deletes_row(self) -> None:
        answer_id = feedback.register_answer("official")
        voter = feedback.hash_voter("voter-1")
        feedback.set_vote(answer_id, voter, "up")
        revoked = feedback.set_vote(answer_id, voter, None)
        self.assertIsNone(revoked["vote"])
        self.assertEqual(revoked["counts"], {"up": 0, "down": 0})
        self.assertEqual(
            self._rows("SELECT COUNT(*) FROM votes WHERE answer_id = ?", (answer_id,))[0][0], 0
        )
        # 没投过票的人撤票也不该报错
        self.assertEqual(
            feedback.set_vote(answer_id, feedback.hash_voter("never"), None)["counts"],
            {"up": 0, "down": 0},
        )

    def test_two_voters_aggregate(self) -> None:
        answer_id = feedback.register_answer("social")
        alice = feedback.hash_voter("alice")
        bob = feedback.hash_voter("bob")
        feedback.set_vote(answer_id, alice, "up")
        self.assertEqual(feedback.get_vote(answer_id, bob), {
            "answer_id": answer_id, "vote": None, "counts": {"up": 1, "down": 0},
        })
        feedback.set_vote(answer_id, bob, "down")
        self.assertEqual(feedback.vote_counts(answer_id), {"up": 1, "down": 1})
        # 两个人各看各的票，互不影响
        self.assertEqual(feedback.get_vote(answer_id, alice)["vote"], "up")
        self.assertEqual(feedback.get_vote(answer_id, bob)["vote"], "down")
        feedback.set_vote(answer_id, bob, "up")
        self.assertEqual(feedback.vote_counts(answer_id), {"up": 2, "down": 0})
        self.assertEqual(
            self._rows("SELECT COUNT(*) FROM votes WHERE answer_id = ?", (answer_id,))[0][0], 2
        )

    def test_rapid_repeated_writes_single_row(self) -> None:
        answer_id = feedback.register_answer("official")
        for _ in range(30):
            feedback.set_vote(answer_id, "same-voter", "up")
        self.assertEqual(
            self._rows("SELECT COUNT(*) FROM votes WHERE answer_id = ?", (answer_id,))[0][0], 1
        )
        self.assertEqual(feedback.vote_counts(answer_id), {"up": 1, "down": 0})

    def test_concurrent_voters(self) -> None:
        answer_id = feedback.register_answer("official")
        errors: list[BaseException] = []

        def vote(i: int) -> None:
            try:
                feedback.set_vote(answer_id, feedback.hash_voter(f"thread-{i}"), "up")
            except BaseException as exc:  # noqa: BLE001 - 线程里的异常要带回主线程
                errors.append(exc)

        threads = [threading.Thread(target=vote, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(feedback.vote_counts(answer_id), {"up": 5, "down": 0})

    # ---------------------------------------------------------- 错误分支

    def test_unknown_answer_id_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            feedback.set_vote("ans_missing", "voter", "up")
        with self.assertRaises(KeyError):
            feedback.get_vote("ans_missing", "voter")
        with self.assertRaises(KeyError):
            feedback.get_vote("ans_missing", None)
        with self.assertRaises(KeyError):
            feedback.vote_counts("ans_missing")

    def test_bad_vote_rejected(self) -> None:
        answer_id = feedback.register_answer("official")
        for bad in ("yes", "UP", 1, []):
            with self.assertRaises(ValueError):
                feedback.set_vote(answer_id, "voter", bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            feedback.set_vote(answer_id, "", "up")
        self.assertEqual(feedback.vote_counts(answer_id), {"up": 0, "down": 0})

    # ---------------------------------------------------------- 持久化

    def test_reopen_reads_back_vote(self) -> None:
        answer_id = feedback.register_answer("official")
        voter = feedback.hash_voter("voter-1")
        feedback.set_vote(answer_id, voter, "down")
        # 换一个全新连接直接读文件，证明真的落盘了（而不是留在进程内存里）
        rows = self._rows(
            "SELECT vote, voter_hash FROM votes WHERE answer_id = ?", (answer_id,)
        )
        self.assertEqual(rows, [("down", voter)])
        self.assertEqual(
            feedback.get_vote(answer_id, "voter-1"),
            {"answer_id": answer_id, "vote": "down", "counts": {"up": 0, "down": 1}},
        )
        # 重新 init（模拟服务重启）后答案与票都还在
        feedback.init_db()
        self.assertEqual(feedback.vote_counts(answer_id), {"up": 0, "down": 1})

    # ---------------------------------------------------------- 统计

    def test_stats_per_channel_and_updates(self) -> None:
        off_1 = feedback.register_answer("official")
        off_2 = feedback.register_answer("official")
        soc_1 = feedback.register_answer("social")
        feedback.set_vote(off_1, "a", "up")
        feedback.set_vote(off_1, "b", "up")
        feedback.set_vote(off_2, "a", "down")
        feedback.set_vote(soc_1, "a", "up")
        self.assertEqual(feedback.stats(), {
            "official": {"up": 2, "down": 1, "answers": 2},
            "social": {"up": 1, "down": 0, "answers": 1},
            "total": {"up": 3, "down": 1, "answers": 3},
        })
        # 改票 + 撤票之后统计要跟着变
        feedback.set_vote(off_2, "a", "up")
        feedback.set_vote(soc_1, "a", None)
        self.assertEqual(feedback.stats(), {
            "official": {"up": 3, "down": 0, "answers": 2},
            "social": {"up": 0, "down": 0, "answers": 1},
            "total": {"up": 3, "down": 0, "answers": 3},
        })

    def test_stats_empty_db(self) -> None:
        empty = {"up": 0, "down": 0, "answers": 0}
        self.assertEqual(
            feedback.stats(),
            {"official": dict(empty), "social": dict(empty), "total": dict(empty)},
        )
        self.assertEqual(feedback.recent_answers(None, 5), [])

    def test_recent_answers_returns_votes(self) -> None:
        first = feedback.register_answer("official", meta={"item_id": "a"})
        second = feedback.register_answer("social", kind="llm")
        feedback.set_vote(first, "v1", "up")
        recent = feedback.recent_answers(None, 5)
        self.assertEqual([r["answer_id"] for r in recent], [second, first])
        self.assertEqual(recent[1]["counts"], {"up": 1, "down": 0})
        self.assertEqual([r["answer_id"] for r in feedback.recent_answers("social", 5)], [second])

    # ---------------------------------------------------------- 配置 / 只读

    def test_db_path_env_and_default(self) -> None:
        self.assertEqual(feedback.DB_PATH_ENV, "FEEDBACK_DB_PATH")
        self.assertEqual(feedback.db_path(), self.db)
        rel = "runtime/from-env.sqlite3"
        os.environ[feedback.DB_PATH_ENV] = rel
        self.assertEqual(feedback.db_path(), feedback.ROOT / rel)
        os.environ.pop(feedback.DB_PATH_ENV, None)
        self.assertEqual(feedback.db_path(), feedback.DEFAULT_DB_PATH)

    def test_readonly_db_write_raises_and_keeps_data(self) -> None:
        answer_id = feedback.register_answer("official")
        feedback.set_vote(answer_id, "voter-a", "up")
        os.chmod(self.db, 0o444)
        self.addCleanup(os.chmod, self.db, 0o644)
        with self.assertRaises((sqlite3.Error, OSError)):
            feedback.set_vote(answer_id, "voter-b", "up")
        # 原来的数据不能被破坏：撤掉只读位再读一遍
        os.chmod(self.db, 0o644)
        self.assertEqual(feedback.get_vote(answer_id, "voter-a")["counts"], {"up": 1, "down": 0})
        self.assertEqual(feedback.get_vote(answer_id, "voter-a")["vote"], "up")
        self.assertEqual(feedback.stats()["official"], {"up": 1, "down": 0, "answers": 1})

    # ---------------------------------------------------------- attach

    def test_attach_registers_and_mutates(self) -> None:
        payload = {"channel": "official", "status": "answered", "claims": []}
        out = feedback.attach(payload, "official", meta={"item_id": "zhuanzhuanye"})
        self.assertIs(out, payload)
        answer_id = payload["answer_id"]
        self.assertTrue(answer_id.startswith("ans_"))
        self.assertEqual(feedback.vote_counts(answer_id), {"up": 0, "down": 0})
        self.assertEqual(self._meta(answer_id), {"item_id": "zhuanzhuanye"})

    def test_attach_swallows_storage_failure(self) -> None:
        blocker = self.root / "blocked"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")
        os.environ[feedback.DB_PATH_ENV] = str(blocker / "fb.sqlite3")
        payload = {"channel": "official", "status": "answered"}
        out = feedback.attach(payload, "official")
        self.assertIs(out, payload)
        self.assertNotIn("answer_id", payload)
        with self.assertRaises((sqlite3.Error, OSError)):
            feedback.register_answer("official")

    def test_meta_whitelist_drops_sensitive_keys(self) -> None:
        answer_id = feedback.register_answer("official", meta={
            "item_id": "zhuanzhuanye",
            "post_ids": ["p1", "p2"],
            "query": "转专业 什么时候  申请\n还有 换宿舍",
            "name": "张三",
            "student_id": "20230001",
            "phone": "13800000000",
            "ip": "10.0.0.1",
            "question": "我是张三，学号 20230001，想转专业怎么办",
            "content": "这是一段回答正文",
        })
        meta = self._meta(answer_id)
        self.assertEqual(meta["item_id"], "zhuanzhuanye")
        self.assertEqual(meta["post_ids"], ["p1", "p2"])
        self.assertLessEqual(len(meta["query"]), feedback.META_QUERY_LIMIT)
        self.assertNotIn("\n", meta["query"])
        for key in ("name", "student_id", "phone", "ip", "question", "content"):
            self.assertNotIn(key, meta)
        raw = self._rows("SELECT meta_json FROM answers WHERE answer_id = ?", (answer_id,))[0][0]
        self.assertNotIn("张三", raw)
        self.assertNotIn("20230001", raw)

    # ---------------------------------------------------------- 摘要

    def test_hash_voter_is_digest_only(self) -> None:
        raw = "203.0.113.7/校园网"
        digest = feedback.hash_voter(raw)
        self.assertEqual(len(digest), 64)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertNotIn(raw, digest)
        self.assertNotIn("203.0.113.7", digest)
        self.assertEqual(digest, feedback.hash_voter(raw))
        self.assertNotEqual(digest, feedback.hash_voter(raw + "x"))
        # 库里存的也只有摘要
        answer_id = feedback.register_answer("official")
        feedback.set_vote(answer_id, raw, "up")
        stored = self._rows("SELECT voter_hash FROM votes WHERE answer_id = ?", (answer_id,))
        self.assertEqual(stored, [(digest,)])

    def test_get_vote_without_missing_reference(self) -> None:
        """未知 citation/引用不影响票数读取：只认 answer_id。"""
        answer_id = feedback.register_answer("social")
        self.assertEqual(feedback.get_vote(answer_id, None), {
            "answer_id": answer_id, "vote": None, "counts": {"up": 0, "down": 0},
        })


if __name__ == "__main__":
    unittest.main()
