"""匿名回答反馈的 SQLite 存储层。

前端在答复下面挂「有用 / 没用」，这里只负责两件事：
1. 每条最终答复登记一个随机 ``answer_id``（回答正文、问题、姓名一概不落库）；
2. 按 ``(answer_id, voter_hash)`` 记一票，同一投票人重复提交只占一行。

设计约束（详见 docs/FEEDBACK_STORE.md）：

- **独立库**：默认 ``runtime/feedback.sqlite3``，与 ``kb.sqlite`` 无关；
  可用环境变量 ``FEEDBACK_DB_PATH`` 覆盖（相对路径按仓库根解析）。
- **短连接**：每次操作开一个连接、用完即关。FastAPI 的同步路由跑在线程池里，
  共享一个 ``sqlite3.Connection`` 会踩线程亲和性；短连接最省心。
- **写事务**：一律 ``BEGIN IMMEDIATE``，配 WAL + ``busy_timeout=5000``，
  避免「先读后写」升级事务时的死锁；遇到 ``database is locked`` 再退避重试几次。
- **隐私**：投票人只存 ``sha256("gyt-feedback:" + raw)`` 摘要；meta 走键白名单，
  姓名 / 学号 / 手机号 / IP / 正文这类键直接丢弃。
- **失败语义**：写不进去时抛 ``OSError`` / ``sqlite3.Error``，由上层映射成 503；
  但 ``attach()`` 会吞掉登记失败——答复本身不能因为记不上票就发不出去。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parent

DB_PATH_ENV = "FEEDBACK_DB_PATH"
DEFAULT_DB_PATH = ROOT / "runtime" / "feedback.sqlite3"

CHANNELS: tuple[str, ...] = ("official", "social")
VOTES: tuple[str, ...] = ("up", "down")

BUSY_TIMEOUT_MS = 5000
CONNECT_TIMEOUT_S = 5.0
LOCK_RETRIES = 3
LOCK_BACKOFF_S = 0.05

ANSWER_ID_PREFIX = "ans_"
VOTER_SALT = "gyt-feedback:"

_LOG = logging.getLogger("feedback")

# 只允许落库的 meta 键：够统计「哪类问题」用，但不含任何正文。
SAFE_META_KEYS: tuple[str, ...] = (
    "kind", "query", "item_id", "item_ids", "post_ids",
    "mode", "status", "topic", "channel", "model", "phase",
)
# 键名命中这些片段就丢：姓名、学号、IP、正文……宁可少存，不可多存。
_SENSITIVE_HINTS: tuple[str, ...] = (
    "name", "student", "学号", "姓名", "phone", "mobile", "tel", "email", "mail",
    "ip", "addr", "地址", "token", "cookie", "auth", "secret", "session",
    "openid", "unionid", "wechat", "weixin", "qq", "idcard", "id_card", "身份证",
    "question", "answer", "content", "text", "payload", "raw", "user", "voter",
    "题", "正文",
)
META_TEXT_LIMIT = 120      # 普通文本值上限
META_QUERY_LIMIT = 60      # query 只留短的检索线索
META_LIST_LIMIT = 20       # 列表元素上限

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_READY: set[str] = set()

_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS answers (
        answer_id  TEXT PRIMARY KEY,
        channel    TEXT NOT NULL,
        kind       TEXT NOT NULL,
        created_at TEXT NOT NULL,
        meta_json  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS votes (
        answer_id  TEXT NOT NULL,
        voter_hash TEXT NOT NULL,
        vote       TEXT NOT NULL CHECK (vote IN ('up', 'down')),
        updated_at TEXT NOT NULL,
        PRIMARY KEY (answer_id, voter_hash),
        FOREIGN KEY (answer_id) REFERENCES answers (answer_id) ON DELETE CASCADE
    )
    """,
    # 统计面板按 channel + 时间取「最近 N 条」
    "CREATE INDEX IF NOT EXISTS idx_answers_channel_created ON answers (channel, created_at DESC)",
    # 聚合某条答复的赞踩
    "CREATE INDEX IF NOT EXISTS idx_votes_answer_vote ON votes (answer_id, vote)",
)

__all__ = [
    "DB_PATH_ENV", "DEFAULT_DB_PATH", "CHANNELS", "VOTES",
    "db_path", "init_db", "register_answer", "attach",
    "set_vote", "get_vote", "vote_counts", "stats", "hash_voter",
    "recent_answers",
]


# ------------------------------------------------------------------ 路径 / 时间


def db_path() -> Path:
    """反馈库路径：``FEEDBACK_DB_PATH`` 优先，否则 ``runtime/feedback.sqlite3``。

    环境变量给相对路径时按**仓库根**解析（而不是当前工作目录），
    这样从别的目录启动 uvicorn 也不会把库建到意料之外的地方。
    每次调用都重新读环境变量：测试和现场演示可以随时改。
    """
    raw = os.environ.get(DB_PATH_ENV, "").strip()
    if not raw:
        return DEFAULT_DB_PATH
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    return p


def _now() -> str:
    """UTC 秒级 ISO8601（``2026-11-22T01:30:00Z``）：字典序即时间序。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ 连接 / 建表


def _connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    """开一个短连接：WAL + busy_timeout + 外键。只读时走 URI ``mode=ro``。"""
    if readonly:
        con = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro", uri=True,
            timeout=CONNECT_TIMEOUT_S, isolation_level=None,
        )
    else:
        con = sqlite3.connect(
            str(path), timeout=CONNECT_TIMEOUT_S, isolation_level=None,
        )
    con.row_factory = sqlite3.Row
    con.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        # 只读介质上 WAL 设不了：读不受影响；真要写的时候会在 BEGIN/INSERT 上报错，
        # 那正是上层要映射成 503 的信号。
        _LOG.debug("feedback: PRAGMA 初始化失败（只读介质？）: %s", path)
    return con


@contextmanager
def _read_conn(path: Path | None = None) -> Iterator[sqlite3.Connection | None]:
    """只读连接；库不存在时 yield None，调用方按「空库」处理。"""
    p = Path(path) if path is not None else db_path()
    con: sqlite3.Connection | None = None
    if p.exists():
        try:
            con = _connect(p, readonly=True)
        except sqlite3.Error:
            con = _connect(p)
    try:
        yield con
    finally:
        if con is not None:
            con.close()


@contextmanager
def _write_tx(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """写事务：``BEGIN IMMEDIATE`` + 遇锁退避重试，异常必回滚。"""
    p = Path(path) if path is not None else db_path()
    _prepare(p)
    last: sqlite3.Error | None = None
    for attempt in range(LOCK_RETRIES + 1):
        con = _connect(p)
        try:
            con.execute("BEGIN IMMEDIATE")
            yield con
        except sqlite3.OperationalError as exc:
            _rollback(con)
            con.close()
            if _is_locked(exc) and attempt < LOCK_RETRIES:
                last = exc
                time.sleep(LOCK_BACKOFF_S * (attempt + 1))
                continue
            raise
        except BaseException:
            _rollback(con)
            con.close()
            raise
        try:
            con.commit()
        except BaseException:  # 提交阶段也可能失败（磁盘满等），一样回滚
            _rollback(con)
            raise
        finally:
            con.close()
        return
    assert last is not None  # pragma: no cover - 循环必然 return 或 raise
    raise last


def _rollback(con: sqlite3.Connection) -> None:
    try:
        con.execute("ROLLBACK")
    except sqlite3.Error:
        pass


def _is_locked(exc: sqlite3.Error) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def _prepare(path: Path) -> None:
    """确保目录和表都在（幂等）；进程内只做一次。"""
    key = str(path)
    if key in _READY and path.exists():
        return
    _READY.discard(key)
    init_db(path)


def init_db(path: Path | None = None) -> None:
    """建目录、建表、建索引、开 WAL。幂等，可反复调用。

    路由启动时调一次即可；写操作内部也会兜底调（防止有人忘了初始化）。
    """
    p = Path(path) if path is not None else db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = _connect(p)
    try:
        con.execute("BEGIN IMMEDIATE")
        for stmt in _SCHEMA:
            con.execute(stmt)
        con.commit()
    except BaseException:
        _rollback(con)
        raise
    finally:
        con.close()
    _READY.add(str(p))


# ------------------------------------------------------------------ 输入清洗


def _clean_channel(channel: str) -> str:
    value = str(channel if channel is not None else "").strip().lower()
    if value not in CHANNELS:
        raise ValueError(f"channel 只支持 {'/'.join(CHANNELS)}，收到 {channel!r}")
    return value


def _clean_answer_id(answer_id: str) -> str:
    value = str(answer_id if answer_id is not None else "").strip()
    if not value:
        raise KeyError(answer_id)
    return value


def _normalize_voter(voter_hash: str | None) -> str:
    """只存摘要：已经是 64 位十六进制的按原样用，其余（原始 IP/随机 token）当场哈希。"""
    if voter_hash is None:
        raise ValueError("voter_hash 不能为空")
    raw = voter_hash.decode("utf-8", "ignore") if isinstance(voter_hash, bytes) else str(voter_hash)
    raw = raw.strip()
    if not raw:
        raise ValueError("voter_hash 不能为空")
    if _HEX64.match(raw):
        return raw.lower()
    return hash_voter(raw)


def _is_sensitive(key: str) -> bool:
    low = key.strip().lower()
    return any(hint in low for hint in _SENSITIVE_HINTS)


def _clean_value(key: str, value: Any) -> Any:
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = " ".join(value.split())  # 去换行/连续空白，顺手压掉正文形态
        limit = META_QUERY_LIMIT if key == "query" else META_TEXT_LIMIT
        return text[:limit]
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            if isinstance(item, bool) or isinstance(item, (int, float)):
                out.append(item)
            elif isinstance(item, str):
                out.append(" ".join(item.split())[:META_TEXT_LIMIT])
            if len(out) >= META_LIST_LIMIT:
                break
        return out
    return None


def _meta_json(meta: Mapping[str, Any] | None) -> str | None:
    """白名单过滤 → JSON。非白名单键、敏感键、认不出的值类型一律丢弃。"""
    if not meta:
        return None
    if not isinstance(meta, Mapping):
        raise TypeError(f"meta 必须是 dict，收到 {type(meta).__name__}")
    clean: dict[str, Any] = {}
    for raw_key, value in meta.items():
        key = str(raw_key).strip()
        if key not in SAFE_META_KEYS or _is_sensitive(key):
            continue
        cleaned = _clean_value(key, value)
        if cleaned is None:
            continue
        if isinstance(cleaned, str) and not cleaned:
            continue
        clean[key] = cleaned
    if not clean:
        return None
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


# ------------------------------------------------------------------ 登记 / 投票


def register_answer(channel: str, kind: str = "normal", meta: dict | None = None) -> str:
    """登记一条答复，返回随机 ``answer_id``（``ans_`` + token_urlsafe(12)）。"""
    ch = _clean_channel(channel)
    kind_text = " ".join(str(kind or "normal").split())[:40] or "normal"
    meta_json = _meta_json(meta)
    for _ in range(3):  # id 撞车的概率约 2^-96，重试只是兜底
        answer_id = f"{ANSWER_ID_PREFIX}{secrets.token_urlsafe(12)}"
        try:
            with _write_tx() as con:
                con.execute(
                    "INSERT INTO answers (answer_id, channel, kind, created_at, meta_json)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (answer_id, ch, kind_text, _now(), meta_json),
                )
            return answer_id
        except sqlite3.IntegrityError:
            _LOG.warning("feedback: answer_id 冲突，换一个重试")
    raise sqlite3.IntegrityError("无法生成唯一的 answer_id")  # pragma: no cover


def attach(payload: dict, channel: str, kind: str = "normal",
           meta: dict | None = None) -> dict:
    """在最终答复上挂 ``answer_id`` 并返回同一个 dict。

    登记失败**不抛给问答**：记日志、原样返回，答复照常发给前端
    （代价是这条答复没有赞踩，比整条答复 500 好）。
    """
    if not isinstance(payload, dict):
        _LOG.warning("feedback: attach 收到非 dict（%s），原样返回", type(payload).__name__)
        return payload
    try:
        answer_id = register_answer(channel, kind=kind, meta=meta)
    except Exception:  # noqa: BLE001 - 存储层任何异常都不该毁掉答复
        _LOG.warning("feedback: 登记失败，本次答复不带 answer_id（channel=%r）",
                     channel, exc_info=True)
        payload["feedback_unavailable"] = True
        return payload
    payload["answer_id"] = answer_id
    return payload


def _answer_exists(con: sqlite3.Connection | None, answer_id: str) -> bool:
    if con is None:
        return False
    try:
        row = con.execute(
            "SELECT 1 FROM answers WHERE answer_id = ?", (answer_id,)
        ).fetchone()
    except sqlite3.OperationalError:  # 表还没建
        return False
    return row is not None


def _counts(con: sqlite3.Connection | None, answer_id: str) -> dict[str, int]:
    counts = {"up": 0, "down": 0}
    if con is None:
        return counts
    try:
        rows = con.execute(
            "SELECT vote, COUNT(*) FROM votes WHERE answer_id = ? GROUP BY vote",
            (answer_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return counts
    for vote, n in rows:
        if vote in counts:
            counts[vote] = int(n)
    return counts


def _my_vote(con: sqlite3.Connection | None, answer_id: str,
             voter_hash: str | None) -> str | None:
    if con is None or not voter_hash:
        return None
    try:
        row = con.execute(
            "SELECT vote FROM votes WHERE answer_id = ? AND voter_hash = ?",
            (answer_id, voter_hash),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return str(row["vote"]) if row is not None else None


def set_vote(answer_id: str, voter_hash: str, vote: str | None) -> dict:
    """写/改/撤一票，返回该答复的最新票数。

    ``vote='up'/'down'`` 是 UPSERT（同一投票人只占一行，改票即覆盖）；
    ``vote=None`` 删掉这一行（撤销）。答复不存在抛 ``KeyError``。
    """
    aid = _clean_answer_id(answer_id)
    if vote not in VOTES and vote is not None:
        raise ValueError(f"vote 只支持 {'/'.join(VOTES)} 或 None，收到 {vote!r}")
    voter = _normalize_voter(voter_hash)
    with _write_tx() as con:
        if not _answer_exists(con, aid):
            raise KeyError(aid)
        if vote is None:
            con.execute(
                "DELETE FROM votes WHERE answer_id = ? AND voter_hash = ?",
                (aid, voter),
            )
        else:
            con.execute(
                "INSERT INTO votes (answer_id, voter_hash, vote, updated_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(answer_id, voter_hash)"
                " DO UPDATE SET vote = excluded.vote, updated_at = excluded.updated_at",
                (aid, voter, vote, _now()),
            )
        counts = _counts(con, aid)
        mine = _my_vote(con, aid, voter)
    return {"answer_id": aid, "vote": mine, "counts": counts}


def get_vote(answer_id: str, voter_hash: str | None) -> dict:
    """读某条答复的票数 + 该投票人自己的票（没投过是 None）。未知答复抛 KeyError。"""
    aid = _clean_answer_id(answer_id)
    voter = _normalize_voter(voter_hash) if voter_hash else None
    with _read_conn() as con:
        if not _answer_exists(con, aid):
            raise KeyError(aid)
        counts = _counts(con, aid)
        mine = _my_vote(con, aid, voter)
    return {"answer_id": aid, "vote": mine, "counts": counts}


def vote_counts(answer_id: str) -> dict:
    """只要票数：``{"up": n, "down": n}``。未知答复抛 KeyError。"""
    aid = _clean_answer_id(answer_id)
    with _read_conn() as con:
        if not _answer_exists(con, aid):
            raise KeyError(aid)
        return _counts(con, aid)


# ------------------------------------------------------------------ 统计


def stats(path: Path | None = None) -> dict:
    """按渠道汇总：``{"official": {...}, "social": {...}, "total": {...}}``。

    每个桶都是 ``{"up", "down", "answers"}``；没有数据时给 0，不让前端判空。
    """
    out: dict[str, dict[str, int]] = {
        ch: {"up": 0, "down": 0, "answers": 0} for ch in CHANNELS
    }
    out["total"] = {"up": 0, "down": 0, "answers": 0}
    with _read_conn(path) as con:
        if con is None:
            return out
        try:
            for row in con.execute(
                "SELECT channel, COUNT(*) AS n FROM answers GROUP BY channel"
            ):
                bucket = out.get(str(row["channel"]))
                if bucket is not None:
                    bucket["answers"] = int(row["n"])
            for row in con.execute(
                "SELECT a.channel AS channel, v.vote AS vote, COUNT(*) AS n"
                " FROM votes v JOIN answers a ON a.answer_id = v.answer_id"
                " GROUP BY a.channel, v.vote"
            ):
                bucket = out.get(str(row["channel"]))
                if bucket is not None and str(row["vote"]) in ("up", "down"):
                    bucket[str(row["vote"])] = int(row["n"])
        except sqlite3.OperationalError:
            return {ch: {"up": 0, "down": 0, "answers": 0} for ch in (*CHANNELS, "total")}
    for ch in CHANNELS:
        out["total"]["up"] += out[ch]["up"]
        out["total"]["down"] += out[ch]["down"]
        out["total"]["answers"] += out[ch]["answers"]
    return out


def recent_answers(channel: str | None = None, limit: int = 5, *,
                   path: Path | None = None) -> list[dict]:
    """最近登记的回答（含票数），给统计命令核对用；``channel=None`` 不过滤。"""
    ch = _clean_channel(channel) if channel else None
    limit = max(0, min(int(limit), 200))
    with _read_conn(path) as con:
        if con is None or limit == 0:
            return []
        try:
            rows = con.execute(
                "SELECT a.answer_id AS answer_id, a.channel AS channel,"
                " a.kind AS kind, a.created_at AS created_at,"
                " COALESCE(SUM(CASE WHEN v.vote = 'up' THEN 1 ELSE 0 END), 0) AS up,"
                " COALESCE(SUM(CASE WHEN v.vote = 'down' THEN 1 ELSE 0 END), 0) AS down"
                " FROM answers a LEFT JOIN votes v ON v.answer_id = a.answer_id"
                " WHERE (? IS NULL OR a.channel = ?)"
                " GROUP BY a.answer_id, a.channel, a.kind, a.created_at"
                " ORDER BY a.created_at DESC, a.rowid DESC LIMIT ?",
                (ch, ch, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [
        {
            "answer_id": str(r["answer_id"]),
            "channel": str(r["channel"]),
            "kind": str(r["kind"]),
            "created_at": str(r["created_at"]),
            "counts": {"up": int(r["up"]), "down": int(r["down"])},
        }
        for r in rows
    ]


# ------------------------------------------------------------------ 摘要


def hash_voter(raw: str) -> str:
    """投票人标识的摘要：``sha256("gyt-feedback:" + raw).hexdigest()``。

    库里只有这 64 位十六进制，任何原始标识都无法反查。
    """
    if not isinstance(raw, str):
        raw = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
    return hashlib.sha256((VOTER_SALT + raw).encode("utf-8")).hexdigest()
