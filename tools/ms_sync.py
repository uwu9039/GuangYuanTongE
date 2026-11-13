"""把「该发布的那部分」同步到一个已 clone 好的仓库目录（创空间 / 其它 Git 远端）。

    git clone <仓库> /tmp/space
    python tools/ms_sync.py --dest /tmp/space [--with-archive]

只覆盖同名文件，不删除目标里已有的东西（远端可能有 vectors.npy.part* 之类
我们本地不上传的大文件）。清单与 GitHub 那边同一份（`tools/prepublish.py`）。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepublish import walk  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--with-archive", action="store_true",
                    help="顺带拷 sources-archive.tar.gz（离线重建正文树用，7 MB）")
    ap.add_argument("--with-vector", action="store_true",
                    help="顺带拷 sources/vector/vectors.npy.part*（204 MB，原文语义检索用）")
    args = ap.parse_args()

    dest = Path(args.dest).resolve()
    if not (dest / ".git").exists():
        raise SystemExit(f"{dest} 不是一个 git 仓库（先 clone）")

    n = 0
    for p in walk():
        rel = p.relative_to(ROOT)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1

    extra: list[Path] = []
    if args.with_archive and (ROOT / "sources-archive.tar.gz").exists():
        extra.append(ROOT / "sources-archive.tar.gz")
    if args.with_vector:
        extra += sorted((ROOT / "sources" / "vector").glob("vectors.npy.part*"))
    for p in extra:
        out = dest / p.relative_to(ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1

    mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"同步 {n} 个文件到 {dest}（目标共 {mb:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
