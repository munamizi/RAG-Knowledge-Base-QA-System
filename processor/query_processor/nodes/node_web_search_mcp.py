import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient

from processor.configs.bailian_mcp_config import mcp_config
from processor.query_processor.BaseNode import BaseNode
from processor.query_processor.state import QueryGraphState
from utils.logger import logger


class NodeWebSearchMcp(BaseNode):

    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行网络搜索节点（百炼 MCP）。"""

        query = (state.get("rewritten_query", "")).strip()

        if not query:
            return {
                "web_search_docs": []
            }

        try:
            # 同步节点中运行异步MCP调用
            result = asyncio.run(self._mcp_call(query))

            # 解析并清洗搜索结果
            docs = self._parse_search_result(result)

            logger.info("MCP 搜索结果: %s", docs)

            return {
                "web_search_docs": docs
            }

        except Exception:
            logger.exception("MCP 网络搜索失败")
            return {
                "web_search_docs": []
            }




    async def _mcp_call(self, query: str):
        """
        使用 MultiServerMCPClient 配置单个 MCP Server，
        并调用 bailian_web_search 工具。
        """

        client = MultiServerMCPClient(
            {
                "bailian-web-search": {
                    "transport": "http",
                    "url": mcp_config.mcp_base_url,
                    "headers": {
                        "Authorization": f"Bearer {mcp_config.api_key}"
                    },
                }
            }
        )

        # 创建指定服务的会话
        # 离开上下文后自动关闭连接并释放资源
        async with client.session("bailian-web-search") as session:
            result = await session.call_tool(
                name="bailian_web_search",
                arguments={
                    "query": query,
                    "count": 3,
                },
            )

            return result



    def _parse_search_result(self, result) -> list[dict[str, str]]:
        """
        解析 MCP web search 的结果

        Agrs:
            result - MCP web search服务执行结果

        Reutrns:
            docs - 解析后的字典
            [
                {
                    "title": "...",
                    "url": "...",
                    "snippet": "..."
                }
            ]
        """

        if result is None:
            return []

        if getattr(result, "isError", False):
            error_text = "；".join(
                getattr(block, "text", "")
                for block in (getattr(result, "content", None) or [])
                if getattr(block, "text", "")
            )
            raise RuntimeError(
                f"MCP 工具执行失败：{error_text or '未知错误'}"
            )

        pages = []

        def collect_pages(payload):
            """从不同包装结构中提取 pages。"""
            if not isinstance(payload, dict):
                return

            page_list = payload.get("pages")

            # 兼容 {"data": {"pages": [...]}}
            if page_list is None and isinstance(payload.get("data"), dict):
                page_list = payload["data"].get("pages")

            # 兼容 {"result": {"pages": [...]}}
            if page_list is None and isinstance(payload.get("result"), dict):
                page_list = payload["result"].get("pages")

            if isinstance(page_list, list):
                pages.extend(page_list)

        # 优先读取结构化结果
        structured_content = (
                getattr(result, "structuredContent", None)
                or getattr(result, "structured_content", None)
        )

        if structured_content:
            collect_pages(structured_content)

        # 结构化结果中没提取到数据，再解析文本内容
        if not pages:
            for content_block in getattr(result, "content", None) or []:
                text = getattr(content_block, "text", None)

                if not text:
                    continue

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(
                        "忽略非 JSON 的 MCP 返回内容: %s",
                        text,
                    )
                    continue

                collect_pages(payload)

        docs = []

        for item in pages:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title") or "").strip()
            url = str(
                item.get("url")
                or item.get("link")
                or ""
            ).strip()
            snippet = str(
                item.get("snippet")
                or item.get("content")
                or item.get("summary")
                or ""
            ).strip()

            if not snippet:
                continue

            docs.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

        return docs


