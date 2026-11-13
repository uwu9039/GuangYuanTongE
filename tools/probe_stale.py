"""直接跑 field.js 里那段「时效判断」，用真实响应核对它会不会触发。

    python tools/probe_stale.py

做法：把 `STALE_MIN_YEARS` / `sourceYears` / `staleHint` 三个定义从 field.js 原文里抠出来，
用 node 执行（其余全局都不需要），再喂真实 /api/ask 响应。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# node：环境变量 NODE 优先，其次 PATH 里的 node
import os  # noqa: E402

_cand = os.environ.get("NODE") or shutil.which("node") or ""
NODE = Path(_cand) if _cand else Path("node")
BASE = "http://127.0.0.1:7860"

CASES = ["公费医疗怎么报销", "户口迁移怎么办", "图书馆能续借吗", "档案在哪里",
         "转专业到信息与通信工程学院", "新闻学院转专业接收计划"]


def ask(text: str) -> dict:
    req = urllib.request.Request(
        BASE + "/api/ask",
        data=json.dumps({"text": text, "router_llm": False, "synth": False}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode("utf-8"))


def extract(src: str) -> str:
    start = src.index("const STALE_MIN_YEARS")
    end = src.index("function cardHead")
    return src[start:end]


def main() -> int:
    js = (ROOT / "web-next" / "field.js").read_text(encoding="utf-8")
    block = extract(js)
    payloads = {q: ask(q) for q in CASES}
    payload_json = json.dumps(payloads, ensure_ascii=False).replace("\\", "\\\\").replace("`", "\\`")
    payload_file = ROOT / ".tmp" / "probe" / "_payloads.json"
    payload_file.parent.mkdir(parents=True, exist_ok=True)
    payload_file.write_text(json.dumps(payloads, ensure_ascii=False), encoding="utf-8")
    script = f"""
import fs from "node:fs";
const PAYLOADS = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }}[c]));
{block}
for (const [q, d] of Object.entries(PAYLOADS)) {{
  const items = sourceYears(d);
  const html = staleHint(d);
  const years = items.map((x) => x.y);
  const cyc = items.filter((x) => x.cycle).map((x) => x.y);
  console.log(JSON.stringify({{ q, newest: years.length ? Math.max(...years) : null,
    newestCycle: cyc.length ? Math.max(...cyc) : null, n: years.length, fired: !!html,
    text: (html.match(/>([^<]+)</) || [])[1] || "", hasItem: !!(d.item && d.item.title),
    plan: !!d.plan }}));
}}
"""
    tmp = ROOT / ".tmp" / "probe" / "_stale_check.mjs"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(script, encoding="utf-8")
    out = subprocess.run([str(NODE), str(tmp), str(payload_file)],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        print("node 失败：", out.stderr[:800])
        return 1
    for line in out.stdout.strip().splitlines():
        d = json.loads(line)
        mark = "触发" if d["fired"] else "不触发"
        print(f"  {mark:4s} 最新依据年={d['newest']} 候选年数={d['n']:2d} 有条目={d['hasItem']}  {d['q']}")
        if d["fired"]:
            print(f"       文案：{d['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
