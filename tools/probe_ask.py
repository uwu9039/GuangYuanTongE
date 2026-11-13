"""第二轮前端/链路验收：无计划学院、示例问法、复合题、转专业计划年份。

    python tools/probe_ask.py
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:7860"


def ask(text: str, synth: bool = True) -> dict:
    req = urllib.request.Request(
        BASE + "/api/ask",
        data=json.dumps({"text": text, "router_llm": False, "synth": synth}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def line(tag: str, q: str, d: dict) -> None:
    plan = (d.get("plan") or {}).get("college") or "-"
    print(f"  {tag:10s} {q[:26]:28s} plan={plan[:12]:13s} no_plan={str(d.get('plan_no_file')):5s} "
          f"synth={'有' if d.get('synth') else '无'} matched={str(d.get('matched')):5s} "
          f"| {(d.get('normalized') or '')[:34]}")


def main() -> None:
    print("== 有计划的学院（应出整篇计划，不走语言组织）")
    for q in ("转专业到信息与通信工程学院", "新闻学院转专业接收计划"):
        line("plan", q, ask(q))

    print("\n== 没有计划的学院（应给「暂无接收计划」的说明，不编）")
    for q in ("转专业到马克思主义学院", "转专业到体育部", "转专业到国际传媒教育学院"):
        d = ask(q)
        line("no_plan", q, d)
        if d.get("note"):
            print(f"             说明：{str(d['note'])[:80]}")

    print("\n== 示例问法（应直接答，不走语言组织）")
    for q in ("补选课要带什么", "校卡怎么补办", "48教在哪", "奖学金怎么申请"):
        line("preset", q, ask(q))

    print("\n== 非示例的复合题（应走语言组织）")
    d = ask("记过处分到期了，能评奖学金吗？能保研吗？")
    s = d.get("synth") or {}
    line("compound", "记过处分到期了，能评奖学金吗？能保研吗？", d)
    print(f"    premises={len(s.get('premises') or [])} layers={len(s.get('layers') or [])}")

    print("\n== 计划年份进了「理解成」那一行")
    d = ask("转专业到信息与通信工程学院")
    print("   ", d.get("normalized"))


if __name__ == "__main__":
    main()
