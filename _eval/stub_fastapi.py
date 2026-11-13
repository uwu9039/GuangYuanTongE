"""把 fastapi / fastapi.responses / fastapi.staticfiles 替换成占位模块。

用法（其它脚本开头调用一次）：
    from _eval import stub_fastapi
    stub_fastapi.install()
    import app
"""
from __future__ import annotations

import sys
import types


def install() -> None:
    if "fastapi" in sys.modules and not hasattr(sys.modules["fastapi"], "__stub__"):
        raise RuntimeError("真的 fastapi 已经装上了，不需要假模块")

    from _eval import fastapi_stub as stub

    root = types.ModuleType("fastapi")
    root.__stub__ = True
    root.FastAPI = stub.FastAPI
    root.HTTPException = stub.HTTPException
    root.Request = stub.Request

    responses = types.ModuleType("fastapi.responses")

    class _Resp:
        def __init__(self, *a, **kw):
            self.args = a
            self.kw = kw

    responses.FileResponse = _Resp
    responses.JSONResponse = _Resp
    responses.RedirectResponse = _Resp
    responses.StreamingResponse = _Resp
    responses.HTMLResponse = _Resp
    responses.PlainTextResponse = _Resp
    root.responses = responses

    staticfiles = types.ModuleType("fastapi.staticfiles")

    class StaticFiles:
        def __init__(self, *a, **kw):
            pass

    staticfiles.StaticFiles = StaticFiles
    root.staticfiles = staticfiles

    sys.modules.setdefault("fastapi", root)
    sys.modules.setdefault("fastapi.responses", responses)
    sys.modules.setdefault("fastapi.staticfiles", staticfiles)
