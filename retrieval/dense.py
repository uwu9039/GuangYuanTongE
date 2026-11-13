"""百炼（DashScope）embedding 客户端 + 向量索引（可选通道）。

只用标准库 urllib + numpy，不装 requests/openai。三条硬约束：

1. 令牌只从环境变量 / `.env` 读，不写进代码；`.env` 已 gitignore。
2. 批量上限按模型查表（`qwen3.7-text-embedding` 是 20 条/请求），超限云端直接 400。
3. **失败不静默降级**：建库时抛 `CloudError`；查询侧由调用方显式决定
   `dense=False` 才走字面通道，不允许「云端挂了悄悄退回 BM25」还当作 dense 结果。

向量缓存按 `sha256(model|dim|text_type|text)` 落 sqlite，断点续跑不重复付费。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DENSE_DIR = ROOT / "retrieval" / "artifacts" / "dense"

EMBED_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
DEFAULT_MODEL = "qwen3.7-text-embedding"
DEFAULT_DIM = 1024
# 批量上限按模型查表；实际用一半——20 条的响应约 430KB，链路抖动时会 IncompleteRead
BATCH = {"qwen3.7-text-embedding": 10, "qwen3.7-text-embedding-flash": 10,
         "text-embedding-v4": 10, "text-embedding-v3": 10}
RETRY = (0.8, 1.6, 3.2, 6.4, 12.0)


class CloudError(RuntimeError):
    """云端没给向量。调用方必须让它冒上去，不许当成空结果。"""


def load_env(path: Path | None = None) -> dict[str, str]:
    p = path or (ROOT / ".env")
    out: dict[str, str] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    key = load_env().get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise CloudError("没有 DASHSCOPE_API_KEY（环境变量或 .env）")
    return key


def embed_batch(texts: Sequence[str], *, text_type: str = "document",
                model: str = DEFAULT_MODEL, dim: int = DEFAULT_DIM,
                key: str | None = None, timeout: float = 120.0) -> np.ndarray:
    """一次请求（<= 批量上限），返回 (n, dim) float32。"""
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    key = key or api_key()
    body = {"model": model, "input": {"texts": list(texts)},
            "parameters": {"dimension": dim, "text_type": text_type}}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last: Exception | None = None
    for wait in (0.0, *RETRY):
        if wait:
            time.sleep(wait)
        req = urllib.request.Request(
            EMBED_URL, data=data, method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
            embs = payload["output"]["embeddings"]
            embs.sort(key=lambda e: e["text_index"])
            return np.asarray([e["embedding"] for e in embs], dtype=np.float32)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            last = CloudError(f"http={e.code} {msg}")
            if e.code in (400, 401, 403):   # 参数/密钥/欠费：重试没意义
                break
        except Exception as e:  # noqa: BLE001  (IncompleteRead / 超时 / 连接重置都重试)
            last = e
    raise CloudError(f"embedding 失败（{len(texts)} 条）：{last}")


class VectorCache:
    """sha256 -> 向量，落 sqlite；断点续跑不重复付费。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._local = threading.local()
        with self._con() as c:
            c.execute("CREATE TABLE IF NOT EXISTS vec (h TEXT PRIMARY KEY, v BLOB)")

    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            con.execute("PRAGMA journal_mode=WAL")
            self._local.con = con
        return con

    @staticmethod
    def digest(model: str, dim: int, text_type: str, text: str) -> str:
        return hashlib.sha256(f"{model}|{dim}|{text_type}|{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        con = self._con()
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            q = "SELECT h, v FROM vec WHERE h IN (%s)" % ",".join("?" * len(chunk))
            for h, blob in con.execute(q, chunk):
                out[h] = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
        return out

    def put_many(self, rows: Iterable[tuple[str, np.ndarray]]) -> None:
        con = self._con()
        con.executemany("INSERT OR REPLACE INTO vec (h, v) VALUES (?, ?)",
                        [(h, v.astype(np.float16).tobytes()) for h, v in rows])
        con.commit()


def embed_texts(texts: Sequence[str], *, text_type: str = "document",
                model: str = DEFAULT_MODEL, dim: int = DEFAULT_DIM,
                workers: int = 6, cache: VectorCache | None = None,
                on_progress=None) -> np.ndarray:
    """带缓存与并发的批量编码；返回与 texts 同序的 float32 矩阵。"""
    keys = [VectorCache.digest(model, dim, text_type, t) for t in texts]
    got = cache.get_many(keys) if cache else {}
    todo: list[tuple[int, str]] = []
    todo_keys: list[str] = []
    for i, k in enumerate(keys):
        if k not in got:
            todo.append((i, texts[i]))
            todo_keys.append(k)
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, k in enumerate(keys):
        if k in got:
            out[i] = got[k]
    if todo:
        bs = BATCH.get(model, 10)
        pending = [todo[i:i + bs] for i in range(0, len(todo), bs)]
        lock = threading.Lock()
        done = [0]
        failed: list[list[tuple[int, str]]] = []

        def work(batch: list[tuple[int, str]]) -> None:
            try:
                vecs = embed_batch([t for _, t in batch], text_type=text_type,
                                   model=model, dim=dim)
            except CloudError:
                with lock:
                    failed.append(batch)
                return
            with lock:
                for (i, _t), v in zip(batch, vecs):
                    out[i] = v
                done[0] += len(batch)
                if cache:
                    cache.put_many([(keys[i], out[i]) for i, _ in batch])
                if on_progress:
                    on_progress(done[0], len(todo))

        for round_no in range(3):
            if not pending:
                break
            failed = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(work, pending))
            pending = [b for b in failed]
            if pending and round_no < 2:
                print(f"[dense] {sum(len(b) for b in pending)} 条失败，第 {round_no + 2} 轮重试",
                      flush=True)
        if pending:
            raise CloudError(f"{sum(len(b) for b in pending)} 条编码失败（重试 3 轮仍失败）")
    return out


class DenseIndex:
    """条目向量矩阵（float32，行 = 条目）。查询就是一次内积。"""

    __slots__ = ("ids", "mat", "id_of", "model", "dim", "meta")

    def __init__(self, ids: list[str], mat: np.ndarray, model: str, dim: int,
                 meta: dict | None = None) -> None:
        self.ids = ids
        self.mat = mat
        self.id_of = {i: n for n, i in enumerate(ids)}
        self.model = model
        self.dim = dim
        self.meta = meta or {}

    @classmethod
    def load(cls, folder: Path = DENSE_DIR) -> "DenseIndex":
        folder = Path(folder)
        meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        ids = json.loads((folder / "ids.json").read_text(encoding="utf-8"))
        mat = np.load(folder / "vectors.npy", mmap_mode="r")
        if tuple(mat.shape) != (len(ids), meta["dim"]):
            raise ValueError("dense 索引与 manifest 不一致")
        return cls(ids=list(ids), mat=mat, model=meta["model"], dim=int(meta["dim"]), meta=meta)

    def search(self, qvec: np.ndarray, k: int = 60) -> list[tuple[str, float]]:
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        n = self.mat.shape[0]
        scores = np.empty(n, dtype=np.float32)
        step = 16384
        for i in range(0, n, step):
            scores[i:i + step] = self.mat[i:i + step] @ q
        k = min(k, n)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.ids[i], float(scores[i])) for i in idx]
