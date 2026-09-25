"""API route modules."""

from importlib import import_module

from .query import query_router

# `import` 是 Python 关键字，不能用于常规的 from ... import 语句。
import_router = import_module(".import", __name__).import_router

__all__ = ["import_router", "query_router"]
