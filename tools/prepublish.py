"""发布前体检 + 打包清单：找出项目里的密钥、绝对路径、个人信息，并给出该上传哪些文件。

    python tools/prepublish.py            # 体检 + 打印清单
    python tools/prepublish.py --copy <暂存目录>   # 顺带把清单里的文件拷到一个目录

只读工程文件；不读 .env 的内容（只报「文件存在」），不打印任何密钥。
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 不上传：派生数据、依赖、临时产物、密钥，以及**委托方的验收脚本与任务书**
# （`_acceptance/` 里是留存负例，`*_PROMPT.md` 是任务书，都不该进公开仓库）
EXCLUDE_DIRS = {
    ".git", ".venv", ".tmp", "__pycache__", "tmpbuild",
    "retrieval/artifacts", "_eval/out", "sources/vector", "_acceptance",
}
EXCLUDE_FILES = {".env", ".env.local", "kb.sqlite",
                 "PROMPT_FOR_AGENT.md", "ROUND2_PROMPT.md",
                 # 这两个探针要读本机的密钥文件，路径也是本机绝对路径，不随仓库走
                 "probe_dashscope.py", "probe_key_diff.py",
                 # 它 import `_acceptance.check`，而那个目录不发布
                 "run_check_base.py"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".log"}

# 文本类型才扫密钥/路径
TEXT_SUFFIX = {".py", ".js", ".css", ".html", ".md", ".json", ".jsonl", ".tsv", ".txt",
               ".ps1", ".sh", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".example"}

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dashscope/deepseek key", re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}")),
    ("modelscope token", re.compile(r"\bms-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")),
    ("oauth2 url", re.compile("oauth" + "2:[^@" + r"\s]+@")),
    ("bearer/basic header", re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9+/=_\-]{16,}")),
    # 值必须跟在同一行（`\s*` 会跨行，把下一行的变量名当成值的误报踩过）
    ("key assignment", re.compile(r"(?i)\b(DASHSCOPE_API_KEY|DEEPSEEK_API_KEY|OPENAI_API_KEY|"
                                  r"AMAP_KEY|SECRET_KEY)[ \t]*[:=][ \t]*[\"']?"
                                  r"(?![A-Z0-9_]{8,}\b)[A-Za-z0-9_\-]{12,}")),
)
PATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # 盘符后面必须紧跟 反斜杠 或 「/目录/」，避免把 `Env:\VAR`、`（G:/P:）` 这类误判
    ("windows abs", re.compile(r"(?<![\w])[A-Za-z]:\\[A-Za-z0-9_.\-]+")),
    ("windows abs fwd", re.compile(r"(?<![\w])[A-Za-z]:/[A-Za-z0-9_.\-]+/")),
    ("unix home", re.compile(r"/Users/[A-Za-z0-9_.\-]+|/home/[A-Za-z0-9_.\-]+")),
)


def is_excluded(rel: str) -> bool:
    for d in EXCLUDE_DIRS:
        if rel == d or rel.startswith(d + "/"):
            return True
    name = rel.rsplit("/", 1)[-1]
    if name in EXCLUDE_FILES:
        return True
    return any(name.endswith(s) for s in EXCLUDE_SUFFIX)


def walk() -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = Path(dirpath).relative_to(ROOT).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = [d for d in dirnames if not is_excluded(f"{rel_dir}/{d}".lstrip("/"))]
        for fn in filenames:
            rel = f"{rel_dir}/{fn}".lstrip("/")
            if not is_excluded(rel):
                out.append(ROOT / rel)
    return sorted(out)


def main() -> int:
    copy_to = None
    if "--copy" in sys.argv:
        copy_to = Path(sys.argv[sys.argv.index("--copy") + 1])

    files = walk()
    total = sum(p.stat().st_size for p in files)
    print(f"待发布文件 {len(files)} 个，共 {total/1024/1024:.1f} MB\n")

    problems = 0
    print("== 密钥扫描")
    secret_hits = 0
    for p in files:
        if p.suffix.lower() not in TEXT_SUFFIX:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pat in SECRET_PATTERNS:
            for m in pat.finditer(txt):
                secret_hits += 1
                line = txt.count("\n", 0, m.start()) + 1
                shown = m.group(0)[:6] + "…" + m.group(0)[-4:]
                print(f"  ✗ {label}: {p.relative_to(ROOT)}:{line}  {shown}")
    if not secret_hits:
        print("  ✓ 没发现密钥")
    problems += secret_hits

    print("\n== 绝对路径扫描")
    path_hits = 0
    for p in files:
        if p.suffix.lower() not in TEXT_SUFFIX:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pat in PATH_PATTERNS:
            for m in pat.finditer(txt):
                path_hits += 1
                line = txt.count("\n", 0, m.start()) + 1
                seg = txt[max(0, m.start() - 20):m.start() + 60].replace("\n", " ")
                print(f"  ✗ {label}: {p.relative_to(ROOT)}:{line}  …{seg}…")
    if not path_hits:
        print("  ✓ 没有绝对路径")
    problems += path_hits

    print("\n== 不该上传但仍在树里的文件")
    for name in (".env", "kb.sqlite"):
        p = ROOT / name
        if p.exists():
            print(f"  · {name} 存在（{p.stat().st_size/1024:.0f} KB）—— 不上传，仅提示")

    if copy_to:
        copy_to.mkdir(parents=True, exist_ok=True)
        for p in files:
            dst = copy_to / p.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        print(f"\n已拷贝到 {copy_to}")

    print(f"\n结论：{'有 ' + str(problems) + ' 处需要处理' if problems else '干净，可以发布'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
