"""检查：代码运行时读的文件，是否都在「发布清单」里。

    python tools/audit_runtime_files.py

做法：从 app.py / retrieval / tools 里抠出 `ROOT / "xxx"` 与 `open("xxx")` 这类字面量，
再去发布清单里核对。派生文件（构建时才生成）单独列出来，不算缺。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepublish import walk  # noqa: E402

# 运行时生成、仓库里不该有的
DERIVED = {
    "kb.sqlite", "retrieval/artifacts", "retrieval/artifacts/", ".tmp",
    "sources/vector/vectors.npy", "tmpbuild",
}

PATTERNS = (
    re.compile(r'ROOT\s*/\s*"([^"]+)"'),
    re.compile(r"ROOT\s*/\s*'([^']+)'"),
    re.compile(r'Path\(ROOT\)\s*/\s*"([^"]+)"'),
    re.compile(r'(?<![\w.])open\(\s*"([^"]+)"'),
)

# 前端静态资源：按目录整体核对
WEB_DIRS = ("web-next",)


def published() -> tuple[set[str], set[str]]:
    files = {p.relative_to(ROOT).as_posix() for p in walk()}
    dirs: set[str] = set()
    for f in files:
        parts = f.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    return files, dirs


def main() -> int:
    files, dirs = published()
    refs: dict[str, set[str]] = {}
    for src in list((ROOT / "app.py",)) + sorted((ROOT / "retrieval").glob("*.py")) + sorted((ROOT / "tools").glob("*.py")):
        try:
            txt = src.read_text(encoding="utf-8")
        except OSError:
            continue
        for pat in PATTERNS:
            for m in pat.finditer(txt):
                ref = m.group(1).strip()
                if not ref or ref.startswith(("http", "data:", "#")) or "%" in ref:
                    continue
                refs.setdefault(ref, set()).add(src.name)

    missing, derived, ok = [], [], []
    for ref, where in sorted(refs.items()):
        clean = ref.rstrip("/")
        if ref in DERIVED or clean in DERIVED or clean.startswith("retrieval/artifacts"):
            derived.append(ref)
            continue
        hit = ref in files or ref in dirs or clean in files or clean in dirs
        if not hit:
            # 「web-next/xxx」这类前缀匹配
            if any(f.startswith(clean + "/") for f in files):
                hit = True
        (ok if hit else missing).append(ref)

    print(f"发布清单：{len(files)} 个文件 / {len(dirs)} 个目录")
    print(f"代码里点名的路径：{len(refs)} 个 → 在清单里 {len(ok)}，派生/构建时生成 {len(derived)}，缺 {len(missing)}")
    if derived:
        print("\n[派生，构建时生成，仓库里没有是对的]")
        for r in derived:
            print("  ·", r)
    if missing:
        print("\n[!! 清单里没有，但代码会去读]")
        for r in missing:
            print("  ✗", r, "  ←", ", ".join(sorted(refs[r])))
    else:
        print("\n✓ 没有缺失的运行时文件")

    # 前端里引用的本地资源
    html = sorted((ROOT / "web-next").glob("*.html"))
    bad = []
    for h in html:
        t = h.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r'(?:src|href)="([^"#:]+)"', t):
            ref = m.group(1)
            if ref.startswith(("http", "data:", "mailto:", "//")):
                continue
            target = (h.parent / ref).resolve()
            if not target.exists():
                bad.append(f"{h.name} → {ref}")
    print(f"\n前端页面 {len(html)} 个，引用本地资源缺失 {len(bad)} 处")
    for b in bad[:10]:
        print("  ✗", b)
    return 1 if missing or bad else 0


if __name__ == "__main__":
    sys.exit(main())
