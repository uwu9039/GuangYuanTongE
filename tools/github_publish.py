"""把仓库里「该发布的那部分」传成 GitHub 仓库的一个 commit，并打 tag。

本机 `github.com` 连不通（只有 api.github.com 通），所以不能用 git push：
这里走 GitHub 的 Git Data API（blob → tree → commit → ref），一次传完再打 tag。

    set GITHUB_TOKEN=...            # 只从环境变量读，绝不写进文件
    python tools/github_publish.py --repo GuangYuanTongE --tag v0.67 [--private]
    python tools/github_publish.py --repo X --dry-run     # 只看会传哪些文件

注意：上传内容用 `tools/prepublish.py` 的清单（已经排除密钥、绝对路径、派生数据）。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prepublish import walk  # noqa: E402  同一份清单

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"


def _token() -> str:
    t = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if not t:
        raise SystemExit("缺 GITHUB_TOKEN（只从环境变量读，不要写进文件）")
    return t


def api(path: str, method: str = "GET", body: dict | None = None,
        token: str | None = None, tries: int = 3) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={"Authorization": f"token {token or _token()}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "dsh-prepublish",
                 "Content-Type": "application/json"})
    for n in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read().decode()
                return r.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:400]
            if e.code in (422, 404) or n == tries - 1:
                return e.code, {"error": detail}
            time.sleep(1.5 * (n + 1))
        except Exception as e:  # noqa: BLE001
            if n == tries - 1:
                return 0, {"error": f"{type(e).__name__}: {e}"}
            time.sleep(1.5 * (n + 1))
    return 0, {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = walk()
    total = sum(p.stat().st_size for p in files)
    print(f"清单：{len(files)} 个文件 / {total/1024/1024:.1f} MB")
    if args.dry_run:
        for p in files[:20]:
            print("  ", p.relative_to(ROOT))
        print(f"  …（共 {len(files)} 个）")
        return 0

    token = _token()
    status, me = api("/user", token=token)
    if status != 200:
        raise SystemExit(f"token 不可用：{status} {me}")
    owner = me["login"]
    print(f"账号：{owner}")

    # 1) 建仓库（已存在就复用）
    status, repo = api("/user/repos", "POST",
                       {"name": args.repo, "private": bool(args.private),
                        "description": "校园办事向导 · 中国传媒大学（广院通）v0.67",
                        "has_issues": False, "has_wiki": False}, token=token)
    if status not in (200, 201):
        status2, repo = api(f"/repos/{owner}/{args.repo}", token=token)
        if status2 != 200:
            raise SystemExit(f"建仓库失败 / 读仓库失败：{status} {repo}")
        print("仓库已存在，复用：", repo["html_url"])
    else:
        print("仓库已创建：", repo["html_url"])

    base = f"/repos/{owner}/{args.repo}"

    # 0) 空仓库直接建 blob 会被拒（409 Git Repository is empty）：
    #    先用 Contents API 落一个 README，拿到初始 commit 当父提交。
    st, refinfo = api(f"{base}/git/ref/heads/{args.branch}", token=token)
    parent = ""
    if st == 200:
        parent = (refinfo.get("object") or {}).get("sha", "")
        print("已有初始提交：", parent[:10])
    else:
        readme = ROOT / "README.md"
        raw = readme.read_bytes() if readme.exists() else b"# GuangYuanTongE\n"
        st2, out = api(f"{base}/contents/README.md", "PUT",
                       {"message": "chore: 初始化仓库", "branch": args.branch,
                        "content": base64.b64encode(raw).decode()}, token=token)
        if st2 in (200, 201):
            parent = out["commit"]["sha"]
            print("已用 README 建初始提交：", parent[:10])
        else:
            raise SystemExit(f"初始化空仓库失败：{st2} {out}")

    # 2) 一次传一个目录树，文本内容内联在 tree 里（`content`，默认 utf-8）。
    #    逐文件建 blob 会撞 GitHub 的 secondary rate limit（内容生成 ~80 次/分钟）；
    #    但 tree 内联只对**文本**可靠：把二进制也塞进 `content` 会被当成字面量存成
    #    base64 文本（踩过：整库文件变成 base64 乱码）。所以二进制（.gz/.pdf/.png…）
    #    仍然单独建 blob —— 只有二三十个，离限流很远。
    CHUNK_FILES = 120
    CHUNK_BYTES = 4 * 1024 * 1024     # 每请求原始字节；GitHub 对超大 body 会回 401
                                      # （踩过：34 MB 的 chunks.jsonl 混在别的文件里被拒）

    def entry(p: Path) -> dict:
        rel = p.relative_to(ROOT).as_posix()
        data = p.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is not None:
            return {"path": rel, "mode": "100644", "type": "blob", "content": text}
        st, out = api(f"{base}/git/blobs", "POST",
                      {"content": base64.b64encode(data).decode(), "encoding": "base64"},
                      token=token)
        if st != 201:
            raise SystemExit(f"二进制 blob 失败 {rel}: {st} {out}")
        return {"path": rel, "mode": "100644", "type": "blob", "sha": out["sha"]}

    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_bytes = 0
    # 大文件排到最后：每加一个 base_tree 都要基于上一棵树复制，34 MB 的
    # chunks.jsonl 先进树会让后面几次 tree 调用超时（422 input too large）。
    small = [p for p in files if p.stat().st_size < CHUNK_BYTES]
    big = [p for p in files if p.stat().st_size >= CHUNK_BYTES]
    for p in small + big:
        size = p.stat().st_size
        # 先按「加进去会不会超」判断，保证一个大文件独占一块
        if cur and (len(cur) >= CHUNK_FILES or cur_bytes + size > CHUNK_BYTES):
            chunks.append(cur)
            cur, cur_bytes = [], 0
        e = entry(p)
        cur.append(e)
        cur_bytes += size
    if cur:
        chunks.append(cur)
    print(f"  分 {len(chunks)} 棵树上传（每棵 ≤{CHUNK_FILES} 个文件 / ≤4 MB，大文件在后）")

    tree_sha = ""
    for n, chunk in enumerate(chunks, 1):
        body: dict = {"tree": chunk}
        if tree_sha:
            body["base_tree"] = tree_sha
        for attempt in range(6):
            st, tree = api(f"{base}/git/trees", "POST", body, token=token)
            if st in (200, 201):
                tree_sha = tree["sha"]
                break
            if st in (403, 422):
                wait = 15 * (attempt + 1)
                print(f"    第 {n} 棵被拒（{st}），等 {wait}s 重试…")
                time.sleep(wait)
                continue
            raise SystemExit(f"tree 失败（第 {n} 棵）：{st} {tree}")
        else:
            raise SystemExit(f"tree 一直失败（第 {n} 棵）")
        print(f"    第 {n}/{len(chunks)} 棵 OK：{len(chunk)} 个文件")

    # 3) commit → ref
    msg = args.message or f"广院通 v{args.tag.lstrip('v') if args.tag else ''}：检索层两轮修复 + 前端三处调整"
    st, commit = api(f"{base}/git/commits", "POST",
                     {"message": msg.strip(), "tree": tree_sha,
                      "parents": [parent] if parent else []}, token=token)
    if st not in (200, 201):
        raise SystemExit(f"commit 失败：{st} {commit}")
    sha = commit["sha"]
    print("commit:", sha)

    st, ref = api(f"{base}/git/refs", "POST",
                  {"ref": f"refs/heads/{args.branch}", "sha": sha}, token=token)
    if st not in (200, 201):
        st2, ref = api(f"{base}/git/refs/heads/{args.branch}", "PATCH",
                       {"sha": sha, "force": True}, token=token)
        if st2 not in (200, 201):
            raise SystemExit(f"建分支失败：{st} {ref} / {st2}")
    print(f"分支 {args.branch} -> {sha[:10]}")

    if args.tag:
        st, tag = api(f"{base}/git/refs", "POST",
                      {"ref": f"refs/tags/{args.tag}", "sha": sha}, token=token)
        if st not in (200, 201):
            print("  （tag 已存在或失败：", st, str(tag)[:120], "）")
        else:
            print("tag:", args.tag)
        st, rel = api(f"{base}/releases", "POST",
                      {"tag_name": args.tag, "target_commitish": args.branch,
                       "name": args.tag,
                       "body": "校园办事向导（广院通）· 中国传媒大学\n\n"
                               "- 检索层：P0 泛词压主题词、hub 抢答、条款定位、"
                               "有词没条拒答、个人数据问法泛化\n"
                               "- 前端：综合回答默认 3 条 + 折叠长条、资料过期小红字、"
                               "提问框基线对齐\n"
                               "- 综合层闸门：转专业接收计划 / 示例问法 / 单个短问直接答\n\n"
                               "运行：`pip install -r requirements.txt && python -m retrieval.build"
                               " && uvicorn app:app --port 7860`"},
                      token=token)
        if st in (200, 201):
            print("release:", rel["html_url"])
        else:
            print("  （release 失败：", st, str(rel)[:160], "）")

    print("仓库地址：", repo.get("html_url") or f"https://github.com/{owner}/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
