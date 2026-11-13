"""题库文件的位置：不进仓库（它是委托方的评测输入），用环境变量给。

    set BANK_JSON=D:\\path\\to\\题库_20261113.json      # Windows
    export BANK_JSON=/path/to/题库_20261113.json       # bash

没设时按仓库内几个常见位置找一遍，找不到就是「没有」，调用方自己跳过。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    ROOT / "_review_items" / "题库_20261113.json",
    ROOT / "题库_20261113.json",
    ROOT.parent / "题库_20261113.json",
)


def bank_json() -> Path:
    """题库路径（可能不存在；用 `.exists()` 判）。"""
    env = (os.environ.get("BANK_JSON") or "").strip().strip('"')
    if env:
        return Path(env)
    for p in CANDIDATES:
        if p.exists():
            return p
    return CANDIDATES[0]
