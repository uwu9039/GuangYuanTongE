"""把 fastapi / uvicorn 及其纯 Python 依赖装到 .venv/Lib/site-packages。

机器上的 pip 在本机沙箱里装不了（unpack 阶段写 .whl.metadata 报 EACCES），
所以这里只做三件事：查 PyPI 的 wheel 链接 → 下载 → 用 zipfile 解开。
没有编译、没有构建隔离，全部是 py3-none-any 或 py3-none-win 的纯 Python 轮子。

    python tools/local_install.py            # 装到 .venv/Lib/site-packages
"""
from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".venv" / "Lib" / "site-packages"
WHEELS = ROOT / ".venv" / "wheels"

# （包名, 版本或约束, 允许的 wheel 标签前缀）
PLAN: tuple[tuple[str, str, str], ...] = (
    ("fastapi", "0.151.0", "py3-none-any"),
    ("starlette", "latest", "py3-none-any"),
    ("uvicorn", "0.54.0", "py3-none-any"),
    ("click", "latest", "py3-none-any"),
    ("annotated-doc", "latest", "py3-none-any"),
    ("opentelemetry-api", "latest", "py3-none-any"),
    ("importlib-metadata", "latest", "py3-none-any"),
    ("zipp", "latest", "py3-none-any"),
)


def _meta(name: str, version: str) -> dict:
    url = f"https://pypi.org/pypi/{name}/json" if version == "latest" else \
        f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _pick(data: dict, want: str) -> tuple[str, str]:
    """从 PyPI 的 urls 里挑一个纯 Python wheel（name, url）。"""
    cands = []
    for u in data["urls"]:
        fn = u["filename"]
        if not fn.endswith(".whl"):
            continue
        if want != "py3-none-any" and want not in fn:
            continue
        if want == "py3-none-any" and "py3-none-any" not in fn:
            continue
        cands.append((fn, u["url"]))
    if not cands:
        raise SystemExit(f"{data['info']['name']} {data['info']['version']}: 找不到纯 Python wheel")
    return cands[0]


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    WHEELS.mkdir(parents=True, exist_ok=True)
    for name, ver, want in PLAN:
        data = _meta(name, ver)
        info = data["info"]
        fn, url = _pick(data, want)
        dest = WHEELS / fn
        if not dest.exists():
            print(f"↓ {fn}")
            urllib.request.urlretrieve(url, dest)
        else:
            print(f"= {fn}（已下载）")
        with zipfile.ZipFile(dest) as z:
            z.extractall(TARGET)
        print(f"  → {info['name']} {info['version']} 装好（{name} 要求的版本：{ver}）")
    print(f"\n完成，目标目录 {TARGET}")
    print("依赖核查：", end="")
    sys.path.insert(0, str(TARGET))
    for m in ("fastapi", "starlette", "uvicorn"):
        try:
            mod = __import__(m)
            print(f"{m}={getattr(mod, '__version__', '?')}", end="  ")
        except Exception as e:  # noqa: BLE001
            print(f"{m}=FAILED({e})", end="  ")
    print()


if __name__ == "__main__":
    main()
