"""用无头 Edge 给前端改动拍图（探针页 + 截图 + 裁切 + 基线测量），只用于本地验证。

思路：`web-next/` 是只读的，所以把 field.html 复制到 .tmp，静态资源指回原目录；
再把 /api/ask 的**真实响应**预先取好内联进页面（替换 fetch），
这样渲染是同步的，无头浏览器不用等真网络，截图就一定能拍到卡片。

    python tools/probe_ui.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".tmp" / "probe"
# 浏览器：环境变量 BROWSER 优先，其次 PATH 里的，再退到各平台的常规安装位置
# （都用环境变量拼，不写死本机绝对路径）
def _browser_candidates() -> list[Path]:
    out: list[Path] = []
    for name in ("msedge", "chrome", "chromium", "chromium-browser", "google-chrome"):
        hit = shutil.which(name)
        if hit:
            out.append(Path(hit))
    roots = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
             os.environ.get("LOCALAPPDATA")]
    rels = (r"Microsoft\Edge\Application\msedge.exe",
            r"Google\Chrome\Application\chrome.exe")
    for root in roots:
        if not root:
            continue
        for rel in rels:
            out.append(Path(root) / rel)
    out.append(Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"))
    return out


def _find_browser() -> Path:
    env = os.environ.get("BROWSER") or ""
    if env:
        return Path(env)
    for p in _browser_candidates():
        if p.exists():
            return p
    return Path("msedge")


EDGE = _find_browser()
BASE = "http://127.0.0.1:7860"

CASES = {
    "synth": ("记过处分到期了，能评奖学金吗？能保研吗？", True),
    "stale": ("公费医疗怎么报销", False),
    "plan": ("转专业到信息与通信工程学院", False),
    "lib": ("图书馆能续借吗", False),
}


def ask(text: str, synth: bool) -> dict:
    req = urllib.request.Request(
        BASE + "/api/ask",
        data=json.dumps({"text": text, "router_llm": False, "synth": synth}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode("utf-8"))


FETCH_PATCH = """
<script>
/* 探针页从 file:// 打开，/api/ask 的响应在这里内联返回（真实响应，事先取好）。 */
(function () {
  const FAKE = window.__FAKE__ || {};
  const orig = window.fetch;
  window.fetch = function (u, o) {
    const url = typeof u === "string" ? u : (u && u.url) || "";
    if (url.indexOf("/api/ask") >= 0) {
      const body = JSON.stringify(FAKE[new URLSearchParams(location.search).get("case")] || {});
      return Promise.resolve(new Response(body, { status: 200, statusText: "OK",
        headers: { "Content-Type": "application/json" } }));
    }
    if (url.startsWith("/")) u = "http://127.0.0.1:7860" + u;
    return orig.call(this, u, o);
  };
})();
</script>
"""

DRIVER = """
<script>
(function () {
  const P = new URLSearchParams(location.search);
  const mode = P.get("align");
  const q = document.getElementById("q");
  const wrap = document.querySelector(".qwrap .typer-text");
  const showTyper = () => { const t = document.getElementById("typer"); if (t) t.style.opacity = "1"; };
  const hideTyper = () => { const t = document.getElementById("typer"); if (t) t.style.display = "none"; };
  // 探针页自己报错：把异常写在页面顶上，截图就能看见（产品页不加这个）
  const report = (msg) => {
    const d = document.createElement("pre");
    d.style.cssText = "position:fixed;left:0;top:0;right:0;z-index:9999;margin:0;padding:8px;"
      + "background:#fee;color:#900;font:12px/1.4 monospace;white-space:pre-wrap";
    d.textContent = "PROBE-ERR " + msg;
    document.body.appendChild(d);
  };
  window.addEventListener("error", (e) => report((e.message || "") + " @" + (e.filename || "").split("/").pop() + ":" + e.lineno));
  window.addEventListener("unhandledrejection", (e) => report("reject: " + (e.reason && (e.reason.stack || e.reason.message || e.reason))));
  const start = () => {
    try {
      document.body.setAttribute("data-mode", "work");
      if (mode) {
        document.body.setAttribute("data-probe", "align");
        if (mode === "A") { q.value = "校园办事指南"; hideTyper(); }
        else if (mode === "B") { if (wrap) wrap.textContent = "校园办事指南"; showTyper(); }
        else { q.value = "校园办事指南"; if (wrap) wrap.textContent = "校园办事指南"; showTyper(); }
        return;
      }
      const text = P.get("q");
      if (!text) return;
      const sw = document.getElementById("synthsw");
      if (sw) sw.checked = P.get("synth") !== "0";
      q.value = text;
      q.dispatchEvent(new Event("input", { bubbles: true }));
      document.getElementById("composer")
        .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
      document.body.setAttribute("data-probe", "ask");
    } catch (e) { report("start: " + (e && e.stack || e)); }
  };
  if (document.readyState === "complete") setTimeout(start, 400);
  else window.addEventListener("load", () => setTimeout(start, 400));
})();
</script>
"""

ALIGN_CSS = """
<style>
  body[data-probe="align"] .hero,
  body[data-probe="align"] .field-layer,
  body[data-probe="align"] .page-footer,
  body[data-probe="align"] .presets { display: none !important; }
</style>
"""


def build_probe(payloads: dict[str, dict]) -> Path:
    html = (ROOT / "web-next" / "field.html").read_text(encoding="utf-8")
    for name in ("field.css", "field.js", "vendor/leaflet/leaflet.css", "vendor/leaflet/leaflet.js",
                 "assets/"):
        html = html.replace(f'"{name}', f'"../../web-next/{name}')
    data = "<script>window.__FAKE__ = " + json.dumps(payloads, ensure_ascii=False) + ";</script>"
    html = html.replace("</head>", ALIGN_CSS + data + "</head>")
    html = html.replace('<script src="../../web-next/field.js" defer></script>',
                        FETCH_PATCH + '<script src="../../web-next/field.js" defer></script>')
    html = html.replace("</body>", DRIVER + "</body>")
    dest = OUT / "_probe.html"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    return dest


def shot(name: str, query: str, *, width: int, height: int, budget: int = 8000) -> Path:
    url = (OUT / "_probe.html").as_uri() + f"?{query}"
    png = OUT / f"{name}.png"
    cmd = [str(EDGE), "--headless=new", "--disable-gpu", "--hide-scrollbars",
           "--force-prefers-reduced-motion", "--disable-web-security",
           "--allow-file-access-from-files", f"--user-data-dir={OUT / ('profile_' + name)}",
           f"--window-size={width},{height}",
           f"--virtual-time-budget={budget}", f"--screenshot={png}", url]
    subprocess.run(cmd, timeout=240, capture_output=True)
    print("  →", png.name, png.exists() and f"{png.stat().st_size//1024} KB" or "FAILED")
    return png


def ink_rows(png: Path, box: tuple[int, int, int, int]) -> tuple[int, int]:
    """在给定区域里找「有墨」的行范围（判定基线用）。"""
    from PIL import Image
    im = Image.open(png).convert("L").crop(box)
    w, h = im.size
    px = im.load()
    rows = []
    for y in range(h):
        dark = sum(1 for x in range(w) if px[x, y] < 170)
        if dark >= 2:
            rows.append(y)
    return (rows[0] + box[1], rows[-1] + box[1]) if rows else (-1, -1)


def main() -> None:
    if not EDGE.exists():
        raise SystemExit(f"没找到 Edge：{EDGE}（换成本机任一 Chromium 内核浏览器的路径即可）")
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {k: ask(q, s) for k, (q, s) in CASES.items()}
    build_probe(payloads)

    shot("alignA", "align=A", width=900, height=380, budget=4000)
    shot("alignB", "align=B", width=900, height=380, budget=4000)
    for name in ("synth", "stale", "plan", "lib"):
        text, synth = CASES[name]
        shot(name, f"case={name}&q={urllib.parse.quote(text)}&synth={1 if synth else 0}",
             width=1180, height=1900)

    try:
        from PIL import Image  # noqa: F401
    except Exception:  # noqa: BLE001
        print("（没有 Pillow，跳过裁切与测量）")
        return

    band = (40, 120, 860, 240)      # 提问框里那一行字
    a, b = ink_rows(OUT / "alignA.png", band), ink_rows(OUT / "alignB.png", band)
    print(f"\n基线测量：输入框文字行 {a}  灰色提示行 {b}  偏移 {b[0]-a[0]} px")
    print("  （0 = 完全同高；A/B 各自起止差 1px 以内算对齐）")

    for name, box in (("alignA", (0, 90, 900, 260)), ("alignB", (0, 90, 900, 260)),
                      ("synth", (0, 40, 1180, 1800)), ("stale", (0, 40, 1180, 1000)),
                      ("plan", (0, 40, 1180, 1000)), ("lib", (0, 40, 1180, 900))):
        src = OUT / f"{name}.png"
        if src.exists():
            Image.open(src).crop(box).save(OUT / f"{name}_crop.png")
    print("  ✂ 裁切完成")


if __name__ == "__main__":
    sys.exit(main())
