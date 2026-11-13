"""测一下百炼 embedding 的并发甜点：不同并发下 8 个请求要多久。"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from retrieval.dense import embed_batch

TEXTS = [f"校园卡补办流程第{i}条测试文本，用于测量并发吞吐。" for i in range(200)]


def run(workers: int, batch: int = 10, reqs: int = 8) -> float:
    batches = [TEXTS[i * batch:(i + 1) * batch] for i in range(reqs)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda b: embed_batch(b, text_type="document"), batches))
    dt = time.perf_counter() - t0
    return reqs * batch / dt


def main() -> None:
    for w in (1, 4, 8, 16, 32):
        try:
            tps = run(w)
            print(f"workers={w:2d}  {tps:6.1f} 条/秒")
        except Exception as e:  # noqa: BLE001
            print(f"workers={w:2d}  失败 {type(e).__name__} {str(e)[:80]}")


if __name__ == "__main__":
    main()
