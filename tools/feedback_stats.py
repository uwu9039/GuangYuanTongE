"""匿名回答反馈的只读统计命令。

    python -m tools.feedback_stats                  # 人类可读表格
    python -m tools.feedback_stats --json           # 机器可读
    python -m tools.feedback_stats --db other.sqlite3 --recent 0

库不存在时打印「暂无反馈库」并以 0 退出——现场演示不该因为还没人投票就报错。
全程只读：用 sqlite3 的 URI ``mode=ro`` 打开，不会顺手建库、也不会写 WAL。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import feedback  # noqa: E402

HEADERS = ("渠道", "赞", "踩", "净分", "回答数")
CHANNEL_LABEL = {"official": "official", "social": "social", "total": "total"}


def _width(text: str) -> int:
    """终端显示宽度：中日韩全角字符按 2 格算。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, *, right: bool = False) -> str:
    gap = max(0, width - _width(text))
    return (" " * gap + text) if right else (text + " " * gap)


def _table(rows: list[tuple[str, ...] | None], numeric_from: int = 1) -> tuple[str, int]:
    """把行渲染成等宽文本；``None`` 行画分隔线。数字列右对齐。"""
    body = [r for r in rows if r is not None]
    widths = [max(_width(r[i]) for r in body) for i in range(len(body[0]))]
    total = sum(widths) + 2 * (len(widths) - 1)
    lines: list[str] = []
    for row in rows:
        if row is None:
            lines.append("-" * total)
            continue
        cells = [_pad(cell, widths[i], right=(i >= numeric_from)) for i, cell in enumerate(row)]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines), total


def _row(name: str, bucket: dict[str, int]) -> tuple[str, ...]:
    up, down = int(bucket.get("up", 0)), int(bucket.get("down", 0))
    return (
        CHANNEL_LABEL.get(name, name),
        str(up),
        str(down),
        f"{up - down:+d}",
        str(int(bucket.get("answers", 0))),
    )


def _print_human(path: Path, data: dict[str, Any], recent: dict[str, list[dict]],
                 recent_limit: int) -> None:
    table_rows: list[tuple[str, ...] | None] = [
        HEADERS,
        _row("official", data.get("official") or {}),
        _row("social", data.get("social") or {}),
        None,
        _row("total", data.get("total") or {}),
    ]
    table, width = _table(table_rows)
    print(f"反馈统计 · {path}")
    print("-" * width)
    print(table)
    if recent_limit > 0:
        for name in ("official", "social"):
            items = recent.get(name) or []
            print()
            print(f"最近回答 · {name}（{len(items)}/{recent_limit}）")
            if not items:
                print("  暂无登记")
                continue
            for item in items:
                aid = item["answer_id"]
                short = aid if len(aid) <= 22 else aid[:21] + "…"
                counts = item.get("counts") or {}
                print(
                    f"  {short:<24} {item['created_at']}  "
                    f"赞 {int(counts.get('up', 0))}  踩 {int(counts.get('down', 0))}"
                    f"  （{item.get('kind') or '-'}）"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.feedback_stats",
        description="查看匿名回答反馈（只读；库不存在时直接退出 0）",
    )
    parser.add_argument("--db", default=None, metavar="PATH",
                        help="反馈库路径；默认取 FEEDBACK_DB_PATH 或 runtime/feedback.sqlite3")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    parser.add_argument("--recent", type=int, default=5, metavar="N",
                        help="每个渠道附带最近 N 条回答（默认 5，0 关闭）")
    args = parser.parse_args(argv)

    path = Path(args.db).expanduser() if args.db else feedback.db_path()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    recent_limit = max(0, min(int(args.recent), 50))

    if not path.exists():
        if args.json:
            empty = {ch: {"up": 0, "down": 0, "answers": 0}
                     for ch in (*feedback.CHANNELS, "total")}
            print(json.dumps({"db": str(path), "available": False, "stats": empty,
                              "recent": {"official": [], "social": []}},
                             ensure_ascii=False, indent=2))
        else:
            print(f"暂无反馈库：{path}")
        return 0

    try:
        data = feedback.stats(path)
        recent = {
            ch: (feedback.recent_answers(ch, recent_limit, path=path) if recent_limit else [])
            for ch in feedback.CHANNELS
        }
    except (sqlite3.Error, OSError) as exc:  # 只读打开失败也要说人话
        print(f"读取反馈库失败：{path}（{type(exc).__name__}: {exc}）", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"db": str(path), "available": True, "stats": data,
                          "recent": recent}, ensure_ascii=False, indent=2))
    else:
        _print_human(path, data, recent, recent_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
