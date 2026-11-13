"""对比测试：dense 开关对检索的影响 + 复合问题在「综合层开/关」下的表现。

    python -m _eval.local_compare                 # dense 对比（引擎内，快）
    python -m _eval.local_compare --ask           # 复合问题（走本机 uvicorn 的 /api/ask，会调模型）
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

BASE = "http://127.0.0.1:7860"

# 手写改写里「说法差得远」的几类：dense 的意义就在这些句子上
DENSE_CASES = [
    "学生卡丟了去哪能做一张新的",
    "钥匙落在屋里了找谁开门",
    "心情不好想找人聊聊去哪",
    "教材什么时候发",
    "在家怎么查学校买的数据库",
    "往卡里打钱从哪个入口进去",
]

# 复合题：两个子问题串在一句里（题库 compound30/compound9 的口径）
COMPOUND = [
    "推免要求计算机等级证书吗？没过会影响报名吗？",
    "国家奖学金要求体测 80 分吗？体测不到 80 还能申请吗？",
    "记过处分到期了，能评奖学金吗？能保研吗？",
    "我选了微专业，主修课冲突了，能退吗，钱退不退？",
    "转专业后，我之前修的课还算数吗？奖学金还能申请吗？",
]


def _post(path: str, payload: dict[str, Any], timeout: int = 240) -> dict[str, Any]:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def dense_compare() -> None:
    from retrieval import Engine

    print("== dense 关 / 开：同一句的 top1 与置信度")
    engs = {"关": Engine.load(), "开": Engine.load(dense=True)}
    for q in DENSE_CASES:
        row = []
        for tag, eng in engs.items():
            t0 = time.perf_counter()
            res = eng.search(q, k=1)
            ms = (time.perf_counter() - t0) * 1000
            title = res.best.title[:24] if res.best else "-"
            row.append(f"{tag}: {title} ({res.best.prob:.2f}, {ms:.0f}ms)" if res.best
                       else f"{tag}: 拒答")
        print(f"  {q[:22]:24s} | " + " || ".join(row))


def ask_compare() -> None:
    print("\n== 复合问题：综合层关 / 开（/api/ask，真实模型）")
    for q in COMPOUND:
        cells = []
        for tag, synth in (("关", False), ("开", True)):
            t0 = time.perf_counter()
            try:
                r = _post("/api/ask", {"text": q, "router_llm": True, "synth": synth})
            except urllib.error.HTTPError as e:
                cells.append(f"{tag}: HTTP {e.code}")
                continue
            except Exception as e:  # noqa: BLE001
                cells.append(f"{tag}: {type(e).__name__}")
                continue
            ms = (time.perf_counter() - t0) * 1000
            syn = r.get("synth") or {}
            if synth and syn:
                layers = syn.get("layers") or []
                tally: dict[str, int] = {}
                for L in layers:
                    tally[L.get("verdict") or "?"] = tally.get(L.get("verdict") or "?", 0) + 1
                body = (syn.get("conclusion") or "").replace("\n", " ")
                cells.append(f"{tag}: {ms/1000:.1f}s premises={len(syn.get('premises') or [])} "
                             f"verdict={tally} | {body[:120]}")
            else:
                body = " ".join((p.get("text") or "") for p in (r.get("answers") or []))
                cells.append(f"{tag}: matched={r.get('matched')} {ms/1000:.1f}s "
                             f"| {body.replace(chr(10), ' ')[:110]}")
        print(f"\n  Q: {q}")
        for c in cells:
            print(f"    {c}")


def main() -> None:
    if "--ask" in sys.argv:
        ask_compare()
    else:
        dense_compare()


if __name__ == "__main__":
    main()
