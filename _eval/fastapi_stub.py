"""导入 app.py 用的 fastapi 占位实现（只为了在裸 numpy 环境里调用纯检索函数）。

不是 FastAPI 的替代品：只实现 app.py 在 import 期用到的那几个符号，
`@app.get/.post/.middleware/.exception_handler` 一律返回原函数，`mount` 空转。
"""
from __future__ import annotations

from typing import Any, Callable


class HTTPException(Exception):
    def __init__(self, status_code: int = 500, detail: Any = ""):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class Request:  # noqa: D101 - 仅作类型占位
    headers: dict = {}
    url: Any = None


class _Route:
    def __init__(self, *a: Any, **kw: Any) -> None:
        self.args = a
        self.kw = kw

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn


class FastAPI:
    def __init__(self, *a: Any, **kw: Any) -> None:
        self.title = kw.get("title", "")
        self.version = kw.get("version", "")
        self.router = _Route()
        self.routes: list[Any] = []

    # 装饰器一律返回原函数，路由表不建
    def get(self, *a: Any, **kw: Any) -> _Route:
        return _Route()

    def post(self, *a: Any, **kw: Any) -> _Route:
        return _Route()

    def put(self, *a: Any, **kw: Any) -> _Route:
        return _Route()

    def delete(self, *a: Any, **kw: Any) -> _Route:
        return _Route()

    def middleware(self, *a: Any, **kw: Any) -> _Route:
        return _Route()

    def exception_handler(self, *a: Any, **kw: Any) -> _Route:
        return _Route()

    def mount(self, *a: Any, **kw: Any) -> None:
        return None

    def include_router(self, *a: Any, **kw: Any) -> None:
        return None

    def add_middleware(self, *a: Any, **kw: Any) -> None:
        return None
