"""对比：每条请求新建 TLS 连接 vs 复用 keep-alive 连接。"""
from __future__ import annotations

import http.client
import json
import time
import urllib.request

from retrieval.dense import api_key

PATH = "/api/v1/services/embeddings/text-embedding/text-embedding"
BODY = json.dumps({"model": "qwen3.7-text-embedding",
                   "input": {"texts": ["校园卡补办流程测试文本。"] * 5},
                   "parameters": {"dimension": 512, "text_type": "document"}}).encode()
KEY = api_key()


def fresh(n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        req = urllib.request.Request(
            "https://dashscope.aliyuncs.com" + PATH, data=BODY, method="POST",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()
    return time.perf_counter() - t0


def keepalive(n: int) -> float:
    con = http.client.HTTPSConnection("dashscope.aliyuncs.com", timeout=60)
    t0 = time.perf_counter()
    for _ in range(n):
        con.request("POST", PATH, body=BODY,
                    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        r = con.getresponse()
        r.read()
        if r.status != 200:
            raise RuntimeError(f"http={r.status}")
    con.close()
    return time.perf_counter() - t0


def main() -> None:
    print(f"新建连接 5 次: {fresh(5):.2f}s")
    print(f"keep-alive 5 次: {keepalive(5):.2f}s")


if __name__ == "__main__":
    main()
