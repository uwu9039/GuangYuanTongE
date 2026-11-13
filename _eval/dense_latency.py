"""量 dense 查询的真实延迟：冷（要调云端编码）与热（命中本地缓存）。"""
from __future__ import annotations

import time

from retrieval import Engine

QS = ["学生证丢了在哪补办呀", "研究生能不能换宿舍呢", "校医院的牙科几点开门啊"]


def main() -> None:
    eng = Engine.load(dense=True)
    print("dense on:", eng.use_dense, "| index", eng.dense.mat.shape if eng.dense else None)
    for q in QS:
        t = time.perf_counter()
        r = eng.search(q, k=3)
        dt = (time.perf_counter() - t) * 1000
        top = r.hits[0].title[:20] if r.hits else "-"
        print(f"{dt:7.0f}ms  {q[:16]:18s} refused={r.refused} top={top}")
    t = time.perf_counter()
    eng.search(QS[0], k=3)
    print(f"{(time.perf_counter()-t)*1000:7.0f}ms  同一句第二次（内存缓存）")


if __name__ == "__main__":
    main()
