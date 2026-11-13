"""测单请求延迟：是不是被 sqlite 缓存写或全局串行拖慢了。"""
from __future__ import annotations

import time

from retrieval.dense import embed_batch

T = ["校园卡补办流程测试文本，用于测量单请求延迟。"] * 10


def one(tag: str) -> None:
    t0 = time.perf_counter()
    embed_batch(T, text_type="document")
    print(f"{tag}: {time.perf_counter() - t0:.2f}s", flush=True)


def main() -> None:
    for i in range(3):
        one(f"serial#{i}")
    import concurrent.futures as cf

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: one("par#8"), range(8)))
    print(f"8 并发总耗时 {time.perf_counter() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
