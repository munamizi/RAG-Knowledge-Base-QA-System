"""离线回归测试：运行真实引用、工作流与接口代码，外部服务使用测试替身。

运行命令：.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from utils.citation_util import build_reference_context, citation_ids, resolve_citations


LOCAL = {
    "chunk_id": 9223372036854775800,
    "content": "连接前请先关闭设备电源。",
    "item_name": "测试设备",
    "file_title": "测试设备使用手册",
    "title": "接线说明-1",
    "parent_title": "接线说明",
    "part": 1,
    "source": "local",
}
WEB = {"source": "web", "title": "厂家说明", "content": "网络摘要。", "url": "https://example.com/help"}


class CitationTests(unittest.TestCase):
    def test_metadata_and_large_chunk_id_are_preserved(self):
        context, refs, remaining = build_reference_context([LOCAL, WEB], 1000)
        self.assertIn("[1]", context)
        self.assertEqual(refs[0]["chunk_id"], str(LOCAL["chunk_id"]))
        self.assertEqual(refs[0]["section"], "接线说明")
        self.assertEqual(refs[1]["url"], WEB["url"])
        self.assertEqual(remaining, 1000 - len(context))
        self.assertNotIn("page", refs[0])

    def test_context_budget_does_not_create_phantom_references(self):
        context, refs, remaining = build_reference_context(
            [{**LOCAL, "content": "长" * 1000}, {"content": ""}, WEB], 200
        )
        self.assertEqual([r["id"] for r in refs], [1])
        self.assertEqual(refs[0]["source"], "web")
        self.assertNotIn("测试设备", context)
        self.assertEqual(len(context) + remaining, 200)
        self.assertEqual(build_reference_context([LOCAL], 0), ("", [], 0))

    def test_exact_budget_and_missing_metadata(self):
        context, _, _ = build_reference_context([{"content": "正文"}], 100)
        _, refs, remaining = build_reference_context([{"content": "正文"}], len(context))
        self.assertEqual(remaining, 0)
        self.assertIsNone(refs[0]["file_title"])
        self.assertNotIn("None", context)

    def test_only_used_current_sources_are_returned_in_order(self):
        _, refs, _ = build_reference_context([LOCAL, WEB], 1000)
        citations, warnings = resolve_citations("网络说法。[2] 操作说明。[1][2]", refs)
        self.assertEqual([c["id"] for c in citations], [2, 1])
        self.assertEqual(warnings, [])
        citations, _ = resolve_citations("只引用本地。[1]", refs)
        self.assertEqual(len(citations), 1)

    def test_unknown_and_missing_citations_are_reported_not_fabricated(self):
        _, refs, _ = build_reference_context([LOCAL], 1000)
        citations, warnings = resolve_citations("答案。[99]", refs)
        self.assertEqual(citations, [])
        self.assertEqual(warnings[0], {"code": "unknown_citation", "citation_ids": [99]})
        self.assertEqual(warnings[1]["code"], "no_valid_citations")
        self.assertEqual(resolve_citations("请补充型号。", []), ([], []))
        self.assertEqual(resolve_citations("未标来源。", refs)[1], [{"code": "no_valid_citations"}])

    def test_code_and_markdown_links_are_not_citations(self):
        text = "`array[8]`\n```python\nx[9]\n```\n[7](https://example.com) ![6](img)\n[5]: /url\n依据。[1][2]"
        self.assertEqual(citation_ids(text), [1, 2])

    def test_unsafe_web_url_is_not_exposed_as_a_link(self):
        for url in ("javascript:alert(1)", "file:///etc/passwd", "https://user:secret@example.com"):
            with self.subTest(url=url):
                context, refs, _ = build_reference_context([{**WEB, "url": url}], 1000)
                self.assertIsNone(refs[0]["url"])
                self.assertNotIn(url, context)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 导入现有历史模块会初始化 MongoDB，因此替换外部服务调用，
        # 使用测试配置，避免读取项目 .env。
        from processor.configs.base import AppBaseSettings

        embedding = ModuleType("utils.embedding_util")
        embedding.generate_embeddings = MagicMock(return_value={"dense": [[0.1, 0.2]], "sparse": [{1: 0.3}]})
        previous_embedding = sys.modules.get("utils.embedding_util")
        sys.modules["utils.embedding_util"] = embedding

        def restore_embedding():
            if previous_embedding is None:
                sys.modules.pop("utils.embedding_util", None)
            else:
                sys.modules["utils.embedding_util"] = previous_embedding

        cls.addClassCleanup(restore_embedding)
        settings = {
            "MONGO_URL": "mongodb://127.0.0.1:1", "MONGO_DB_NAME": "citation-tests",
            "BASE_URL": "http://127.0.0.1:1", "API_KEY": "test-only",
            "LLM_DEFAULT_MODEL": "test-model", "LLM_TEMPERATURE": "0",
            "MILVUS_URL": "http://127.0.0.1:1", "CHUNKS_COLLECTION": "test_chunks",
            "ITEM_NAME_COLLECTION": "test_items", "TEXT_RERANK_MODEL": "test-reranker",
            "RERANK_MAX_TOPK": "10", "RERANK_MIN_TOPK": "3",
            "RERANK_GAP_ABS": "0.5", "RERANK_GAP_RATIO": "0.25",
            "MCP_DASHSCOPE_BASE_URL": "http://127.0.0.1:1",
        }
        with (
            patch.object(AppBaseSettings, "model_config", {**AppBaseSettings.model_config, "env_file": None}),
            patch.dict(os.environ, settings),
            patch("pymongo.MongoClient"),
            patch("socket.socket.connect", side_effect=AssertionError("Tests must not connect to services")),
        ):
            cls.answer_module = importlib.import_module("processor.query_processor.nodes.node_answer_output")
            cls.rerank_module = importlib.import_module("processor.query_processor.nodes.node_rerank")
            cls.search_module = importlib.import_module("processor.query_processor.nodes.node_search_embedding")
            cls.hyde_module = importlib.import_module("processor.query_processor.nodes.node_search_embedding_hyde")
            cls.graph_module = importlib.import_module("processor.query_processor.main_graph")
            cls.history_module = importlib.import_module("utils.mongo_history_utils")
            # 只加载问答路由，避免导入其他路由时触发初始化副作用，
            # 例如现有 MinIO 模块会修改存储桶策略。
            path = Path(__file__).resolve().parents[1] / "backend/api/routers/query.py"
            spec = importlib.util.spec_from_file_location("citation_test_query_router", path)
            cls.route = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.route)

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        cls.app = FastAPI()
        cls.app.include_router(cls.route.query_router)
        cls.client = TestClient(cls.app)
        # Windows 的异步事件循环在启动时会创建内部套接字对，
        # 因此先启动事件循环，再在各项测试中拦截外部服务连接。
        cls.client.__enter__()
        cls.previous_log_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        logging.disable(cls.previous_log_level)

    def setUp(self):
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("Unexpected network call")))
        self.enterContext(patch("builtins.print"))
        self.collection = MagicMock()
        self.collection.insert_one.return_value.inserted_id = "saved-message"
        self.enterContext(patch.object(
            self.history_module, "get_history_mongo_tool",
            return_value=SimpleNamespace(chat_message=self.collection),
        ))
        self.llm = MagicMock()
        self.llm.invoke.return_value = SimpleNamespace(content="请先关闭电源。[1]")
        self.llm.stream.side_effect = lambda prompt: iter([
            SimpleNamespace(content="请先关闭"), SimpleNamespace(content="电源。["), SimpleNamespace(content="1]")
        ])
        self.enterContext(patch.object(self.answer_module, "get_llm_client", return_value=self.llm))

    def state(self, **overrides):
        return {"session_id": self.id(), "original_query": "如何连接？", "item_names": ["测试设备"],
                "is_stream": False, "reranked_docs": [dict(LOCAL)], **overrides}

    def test_answer_and_history_share_source_snapshot(self):
        from utils.task_util import get_task_result
        state = self.answer_module.NodeAnswerOutput().process(self.state())
        saved = self.collection.insert_one.call_args.args[0]
        self.assertEqual(saved["citations"], state["citations"])
        self.assertEqual(saved["text"], state["answer"])
        self.assertEqual(get_task_result(self.id(), "citations"), state["citations"])
        state["reranked_docs"][0]["content"] = "更新后的手册"
        self.assertEqual(saved["citations"][0]["content"], LOCAL["content"])

    def test_stream_final_matches_nonstream_and_history(self):
        from utils.sse_util import create_sse_queue, remove_sse_queue
        plain = self.answer_module.NodeAnswerOutput().process(self.state())
        q = create_sse_queue(self.id())
        self.addCleanup(remove_sse_queue, self.id())
        self.answer_module.NodeAnswerOutput().process(self.state(is_stream=True))
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        final = next(e["data"] for e in events if e["event"] == "final")
        self.assertEqual(final["answer"], plain["answer"])
        self.assertEqual(final["citations"], plain["citations"])
        self.assertEqual(final["citation_warnings"], [])
        self.assertEqual("".join(e["data"]["delta"] for e in events if e["event"] == "delta"), final["answer"])

    def test_no_evidence_and_clarification_skip_llm(self):
        for overrides in ({"reranked_docs": []}, {"answer": "请提供具体型号。"}):
            with self.subTest(overrides=overrides):
                result = self.answer_module.NodeAnswerOutput().process(self.state(**overrides))
                self.assertEqual(result["citations"], [])
                self.assertEqual(result["citation_warnings"], [])
        self.llm.invoke.assert_not_called()

    def test_empty_generation_is_not_saved_as_success(self):
        self.llm.invoke.return_value = SimpleNamespace(content=" ")
        self.llm.stream.side_effect = lambda prompt: iter([])
        for stream in (False, True):
            with self.subTest(stream=stream), self.assertRaises(ValueError):
                self.answer_module.NodeAnswerOutput().process(self.state(is_stream=stream))
        self.collection.insert_one.assert_not_called()

    def test_prompt_truncation_and_invalid_id_cannot_cite_excluded_doc(self):
        self.llm.invoke.return_value = SimpleNamespace(content="回答。[1][2]")
        result = self.answer_module.NodeAnswerOutput().process(self.state(reranked_docs=[
            {**LOCAL, "content": "长" * 13000}, WEB,
        ]))
        self.assertEqual(result["citations"][0]["url"], WEB["url"])
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citation_warnings"], [{"code": "unknown_citation", "citation_ids": [2]}])

    def prepare_graph(self):
        from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
        from processor.query_processor.nodes.node_web_search_mcp import NodeWebSearchMcp

        self.enterContext(patch.object(NodeItemNameConfirm, "process", side_effect=lambda state: {
            "item_names": ["测试设备"], "rewritten_query": state["original_query"], "history": [],
        }))
        self.enterContext(patch.object(NodeWebSearchMcp, "process", return_value={"web_search_docs": []}))
        self.enterContext(patch.object(self.hyde_module.NodeSearchEmbeddingHyde, "_step_1_create_hyde_doc", return_value="假设答案"))
        client = MagicMock()
        # 模拟向量库只返回请求指定的字段，以便检查普通检索或 HyDE 检索
        # 是否遗漏了来源元数据字段。
        client.hybrid_search.side_effect = lambda **kw: [[{
            "entity": {field: LOCAL[field] for field in kw["output_fields"]}, "distance": 0.9,
        }]]
        self.enterContext(patch.object(self.search_module, "get_milvus_client", return_value=client))
        self.enterContext(patch.object(self.hyde_module, "get_milvus_client", return_value=client))
        self.enterContext(patch.object(self.rerank_module, "rerank_documents", side_effect=lambda q, docs: [0.9] * len(docs)))
        return client

    def test_actual_graph_preserves_metadata_through_both_retrieval_paths(self):
        client = self.prepare_graph()
        result = self.graph_module.KBQueryWorkflow().run({
            "session_id": self.id(), "original_query": "如何连接？", "is_stream": False,
        })
        self.assertEqual(client.hybrid_search.call_count, 2)
        self.assertEqual(result["embedding_chunks"][0]["entity"]["file_title"], LOCAL["file_title"])
        self.assertEqual(result["hyde_embedding_chunks"][0]["entity"]["parent_title"], LOCAL["parent_title"])
        self.assertEqual(result["citations"][0]["file_title"], LOCAL["file_title"])
        self.assertEqual(result["citations"][0]["section"], LOCAL["parent_title"])
        self.assertEqual(len(result["reference_sources"]), 1)

    def test_http_query_sse_and_history_contract(self):
        self.prepare_graph()
        response = self.client.post("/query", json={"query": "如何连接？", "session_id": self.id()})
        self.assertEqual(response.status_code, 200)
        plain = response.json()
        self.assertEqual(plain["citations"][0]["file_title"], LOCAL["file_title"])
        saved = self.collection.insert_one.call_args.args[0]
        self.assertEqual(self.client.post("/query", json={
            "query": "如何连接？", "session_id": self.id(), "is_stream": True,
        }).status_code, 200)
        stream = self.client.get(f"/stream/{self.id()}").text
        final_event = next(block for block in stream.split("\n\n") if block.startswith("event: final"))
        final = json.loads(final_event.split("data: ", 1)[1])
        self.assertEqual(final["citations"], plain["citations"])
        with patch.object(self.route, "get_recent_messages", return_value=[saved, {"role": "assistant", "text": "旧答案"}]):
            history = self.client.get(f"/history/{self.id()}").json()["items"]
        self.assertEqual(history[0]["citations"], plain["citations"])
        self.assertEqual(history[1]["citations"], [])
        self.assertEqual(history[1]["citation_warnings"], [])

    def test_failed_generation_returns_error_not_success_with_citations(self):
        self.prepare_graph()
        self.llm.invoke.side_effect = RuntimeError("test generation failure")
        response = self.client.post("/query", json={"query": "如何连接？", "session_id": self.id()})
        self.assertEqual(response.status_code, 502)

        def broken_stream(prompt):
            yield SimpleNamespace(content="尚未生成完的答案。[1]")
            raise RuntimeError("test stream failure")

        self.llm.stream.side_effect = broken_stream
        self.client.post("/query", json={"query": "如何连接？", "session_id": self.id(), "is_stream": True})
        stream = self.client.get(f"/stream/{self.id()}").text
        self.assertIn("event: error", stream)
        self.assertNotIn("event: final", stream)
        self.collection.insert_one.assert_not_called()


if __name__ == "__main__":
    unittest.main()
