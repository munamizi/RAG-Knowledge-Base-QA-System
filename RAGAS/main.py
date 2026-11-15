"""
HAK180 文档的 RAGAS 评估程序。

执行流程：
1. 读取 documents 目录中的 QA CSV。
2. 逐题调用项目现有的 LangGraph RAG 工作流，采集最终回答和实际生成上下文。
3. 使用项目现有的 llm_util 与 embedding_util 初始化 RAGAS 五个指标。
4. 执行评估，并把结果按固定 9 列保存到 result 目录下的同名 CSV。

说明：导入本模块不会调用 RAG、LLM、Embedding 或创建结果文件；只有直接运行
`python RAGAS/main.py` 时才会开始评估。
"""

from __future__ import annotations

import asyncio
import csv
import importlib
import math
import sys
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# =============================================================================
# 第 0 步：路径和运行参数
# =============================================================================

RAGAS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAGAS_DIR.parent

# 直接执行 RAGAS/main.py 时，Python 默认只把 RAGAS 目录加入模块搜索路径。
# 这里加入项目根目录，确保可以正常导入 processor 和 utils。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 明确加载项目根目录的配置，但不覆盖调用者已经设置的环境变量。
load_dotenv(PROJECT_ROOT / ".env", override=False)

INPUT_CSV = RAGAS_DIR / "documents" / "H3C_LA2608室内无线网关_ragas_qa.csv"
OUTPUT_DIR = RAGAS_DIR / "result"
OUTPUT_CSV = OUTPUT_DIR / INPUT_CSV.name

# 每道测试题使用独立会话，避免不同测试题之间的历史消息互相影响。
# 当前 RAG 工作流会把会话记录写入 MongoDB；统一前缀便于识别评估会话。
EVALUATION_SESSION_PREFIX = "ragas-eval"

# 默认严格使用 CSV 中的原始问题。
# 如果将来要模拟“用户已经明确正在咨询 HAK180”的固定会话背景，可在这里设置前缀，
# 例如："关于 Brother HAK180 烫金机，"。修改后，结果中会同时保留原问题和 rag_query。
QUESTION_PREFIX = ""

# None 表示评估 CSV 中全部数据。首次联调时可临时改成较小整数，例如 3。
MAX_ROWS: int | None = None

# RAGAS 评审模型需要稳定的结构化输出，因此使用低温度。
EVALUATOR_TEMPERATURE = 0.01
EVALUATOR_MAX_TOKENS = 2048


# =============================================================================
# 第 1 步：兼容项目当前锁定的 RAGAS 0.4.3 依赖组合
# =============================================================================

def _apply_ragas_043_vertexai_compatibility() -> None:
    """处理 RAGAS 0.4.3 与新版 langchain-community 的导入兼容问题。

    项目当前的 RAGAS 0.4.3 会在导入时引用已经从新版
    `langchain_community.chat_models.vertexai` 中移除的 ChatVertexAI。
    本项目并不使用 VertexAI，因此仅在该旧模块确实不存在时提供一个最小占位类型，
    使 RAGAS 能够正常导入；这不会替换或改变项目实际使用的 qwen/OpenAI 兼容客户端。

    后续升级到已修复该问题的 RAGAS 版本后，可删除此函数及其调用。
    """

    module_name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    compatibility_module = types.ModuleType(module_name)
    compatibility_module.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[module_name] = compatibility_module


_apply_ragas_043_vertexai_compatibility()

# RAGAS 核心组件：使用 0.4 系列推荐的 Collections API，而不是旧版 evaluate API。
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

# 当前项目的 RAG、LLM 和 Embedding 入口。
from processor.query_processor.main_graph import KBQueryWorkflow
from processor.query_processor.nodes.node_answer_output import MAX_CONTEXT_CHARS
from utils.embedding_util import generate_embeddings
from utils.llm_util import get_llm_client


# =============================================================================
# 第 2 步：测试数据和运行结果的数据结构
# =============================================================================

QUESTION_COLUMN_CANDIDATES = ("user_input", "question", "Question")
REFERENCE_COLUMN_CANDIDATES = (
    "reference",
    "ground_truth",
    "Ground_truth",
    "ground_truths",
)

RAGAS_SCORE_COLUMNS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
)

OUTPUT_COLUMNS = (
    "Question",
    "Ground_truth",
    "Context",
    "Answer",
    "Context_Precision",
    "Context_Recall",
    "Faithfulness",
    "Answer_Relevancy",
    "Answer_Correctness",
)


@dataclass
class EvaluationRecord:
    """一行测试数据在采集和评估过程中的完整状态。"""

    user_input: str
    reference: str
    rag_query: str
    response: str = ""
    retrieved_contexts: list[str] = field(default_factory=list)
    retrieved_context_details: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, float] = field(
        default_factory=lambda: {
            metric_name: math.nan for metric_name in RAGAS_SCORE_COLUMNS
        }
    )
    rag_error: str = ""
    evaluation_error: str = ""


def _find_required_column(
    fieldnames: list[str], candidates: tuple[str, ...], semantic_name: str
) -> str:
    """从兼容字段名中找到 CSV 实际使用的列名。"""

    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        f"CSV 缺少{semantic_name}列；支持的列名为：{', '.join(candidates)}"
    )


def load_test_dataset() -> list[EvaluationRecord]:
    """读取并校验 QA CSV，同时兼容两套常见字段命名。"""

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"找不到测试集：{INPUT_CSV}")

    # utf-8-sig 同时兼容带 BOM 和不带 BOM 的 UTF-8 CSV。
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"测试集没有表头：{INPUT_CSV}")

        question_column = _find_required_column(
            fieldnames, QUESTION_COLUMN_CANDIDATES, "问题"
        )
        reference_column = _find_required_column(
            fieldnames, REFERENCE_COLUMN_CANDIDATES, "标准答案"
        )

        source_rows = list(reader)

    if MAX_ROWS is not None:
        source_rows = source_rows[:MAX_ROWS]

    records: list[EvaluationRecord] = []
    for row_number, source_row in enumerate(source_rows, start=2):
        user_input = (source_row.get(question_column) or "").strip()
        reference = (source_row.get(reference_column) or "").strip()
        if not user_input or not reference:
            raise ValueError(
                f"CSV 第 {row_number} 行的问题或标准答案为空，请先修正测试集。"
            )

        rag_query = f"{QUESTION_PREFIX}{user_input}".strip()
        records.append(
            EvaluationRecord(
                user_input=user_input,
                reference=reference,
                rag_query=rag_query,
            )
        )

    if not records:
        raise ValueError(f"测试集没有可评估的数据：{INPUT_CSV}")

    return records


# =============================================================================
# 第 3 步：调用当前 RAG 系统并采集 Context + Answer
# =============================================================================

def _collect_contexts_seen_by_generator(
    final_state: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """优先使用生成节点实际采用的来源快照，旧结果兼容旧的预算计算。"""

    if "reference_sources" in final_state:
        sources = final_state["reference_sources"] or []
        return (
            [source["content"] for source in sources],
            [{**source, "rank": source["id"]} for source in sources],
        )

    contexts: list[str] = []
    context_details: list[dict[str, Any]] = []
    used_chars = 0

    for index, document in enumerate(final_state.get("reranked_docs") or [], start=1):
        content = str(document.get("content") or "").strip()
        if not content:
            continue

        # 兼容加入 reference_sources 之前的工作流结果。
        metadata_tags = [f"[{index}]"]
        for field_name, template in (
            ("source", "[source={}]"),
            ("chunk_id", "[chunk_id={}]"),
            ("url", "[url={}]"),
            ("title", "[title={}]"),
        ):
            field_value = str(document.get(field_name)).strip()
            if field_value:
                metadata_tags.append(template.format(field_value))

        score = document.get("score")
        if score is not None:
            metadata_tags.append(f"[score={float(score):.4f}]")

        prompt_entry = " ".join(metadata_tags) + "\n" + content
        if used_chars + len(prompt_entry) > MAX_CONTEXT_CHARS:
            break

        used_chars += len(prompt_entry) + 2
        contexts.append(content)
        context_details.append(
            {
                "rank": index,
                "source": document.get("source"),
                "chunk_id": document.get("chunk_id"),
                "url": document.get("url"),
                "title": document.get("title"),
                "score": score,
                "content": content,
            }
        )

    return contexts, context_details


def collect_rag_outputs(records: list[EvaluationRecord]) -> None:
    """逐题运行项目现有 RAG，并把 Answer/Context 写回记录对象。"""

    # 只构建和编译一次图；每道题仍使用全新的 state 与 session_id。
    compiled_graph = KBQueryWorkflow().compile()
    total = len(records)

    for index, record in enumerate(records, start=1):
        session_id = f"{EVALUATION_SESSION_PREFIX}-{uuid.uuid4()}"
        initial_state = {
            "original_query": record.rag_query,
            "session_id": session_id,
            "is_stream": False,
        }

        print(f"[RAG {index}/{total}] {record.rag_query}")
        try:
            # 直接调用项目当前编译后的 LangGraph，检索、Rerank、生成逻辑均与业务一致。
            final_state = compiled_graph.invoke(initial_state)
            record.response = str(final_state.get("answer") or "").strip()
            (
                record.retrieved_contexts,
                record.retrieved_context_details,
            ) = _collect_contexts_seen_by_generator(final_state)

            if not record.response:
                record.rag_error = "RAG 工作流结束后未返回 answer。"
        except Exception as exc:  # 单题失败不应中断其余测试题。
            record.rag_error = f"{type(exc).__name__}: {exc}"
            print(f"  -> RAG 失败：{record.rag_error}")


# =============================================================================
# 第 4 步：把项目 Embedding 和 LLM 适配为 RAGAS 核心组件
# =============================================================================

class ProjectRagasEmbedding(BaseRagasEmbedding):
    """RAGAS Embedding 适配器：直接复用 utils.embedding_util。

    RAGAS 的 AnswerRelevancy 只需要稠密向量，因此这里取 BGE-M3 返回值中的 dense；
    稀疏向量仍由业务 RAG 检索链使用，不参与余弦相似度指标计算。
    """

    def __init__(self) -> None:
        super().__init__()

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_texts([text], **kwargs)[0]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        # BGE-M3 是同步本地模型，放入工作线程以免阻塞 RAGAS 的异步评估流程。
        return await asyncio.to_thread(self.embed_text, text, **kwargs)

    def embed_texts(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        if not texts:
            return []
        return generate_embeddings(texts)["dense"]

    async def aembed_texts(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        # 一次批量编码比逐条加载/调用模型更高效，也与 embedding_util 的接口一致。
        return await asyncio.to_thread(self.embed_texts, texts, **kwargs)


@dataclass
class RagasComponents:
    """集中保存五个 RAGAS 指标，便于理解和维护。"""

    context_precision: ContextPrecision
    context_recall: ContextRecall
    faithfulness: Faithfulness
    answer_relevancy: AnswerRelevancy
    answer_correctness: AnswerCorrectness


def build_ragas_components() -> RagasComponents:
    """用项目当前 LLM/Embedding 初始化 RAGAS 五个指标。"""

    # RAGAS 核心组件：先通过项目 llm_util 获取同一个模型、Base URL 和密钥配置。
    project_llm = get_llm_client()
    async_openai_client = getattr(project_llm, "root_async_client", None)
    if async_openai_client is None:
        raise RuntimeError("llm_util 返回的客户端没有 root_async_client，无法供 RAGAS 使用。")

    evaluator_model = (
        getattr(project_llm, "model_name", None)
        or getattr(project_llm, "model", None)
    )
    if not evaluator_model:
        raise RuntimeError("无法从 llm_util 返回的客户端确定模型名称。")

    # RAGAS 0.4 Collections 指标要求支持 Pydantic 结构化输出的 Instructor LLM。
    # llm_factory 仅做接口适配，底层仍然是 llm_util 创建的同一个 AsyncOpenAI 客户端。
    evaluator_llm_kwargs: dict[str, Any] = {
        "temperature": EVALUATOR_TEMPERATURE,
        "max_tokens": EVALUATOR_MAX_TOKENS,
    }
    project_extra_body = getattr(project_llm, "extra_body", None)
    if project_extra_body:
        evaluator_llm_kwargs["extra_body"] = project_extra_body

    evaluator_llm = llm_factory(
        model=evaluator_model,
        provider="openai",
        client=async_openai_client,
        **evaluator_llm_kwargs,
    )
    evaluator_embeddings = ProjectRagasEmbedding()

    # RAGAS 核心指标 1：Context Precision，检查召回片段是否相关且排序靠前。
    context_precision = ContextPrecision(llm=evaluator_llm)

    # RAGAS 核心指标 2：Context Recall，检查标准答案所需信息是否被召回完整。
    context_recall = ContextRecall(llm=evaluator_llm)

    # RAGAS 核心指标 3：Faithfulness，检查生成答案是否都能由召回上下文支持。
    faithfulness = Faithfulness(llm=evaluator_llm)

    # RAGAS 核心指标 4：Answer Relevancy，检查回答是否切题。
    # 该指标同时使用评审 LLM 和项目 embedding_util 的 BGE-M3 稠密向量。
    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        strictness=3,
    )

    # RAGAS 核心指标 5：Answer Correctness，对照 Ground_truth 检查回答正确性。
    # 默认按 75% 事实正确性 + 25% 语义相似度计算，并复用项目 embedding_util。
    answer_correctness = AnswerCorrectness(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return RagasComponents(
        context_precision=context_precision,
        context_recall=context_recall,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        answer_correctness=answer_correctness,
    )


# =============================================================================
# 第 5 步：调用 RAGAS 五个指标并记录失败信息
# =============================================================================

async def _run_metric(
    metric_name: str,
    metric_call,
    errors: list[str],
) -> float:
    """统一执行单个 RAGAS 指标；指标失败时保留错误并返回 NaN。"""

    try:
        metric_result = await metric_call
        return float(metric_result.value)
    except Exception as exc:
        errors.append(f"{metric_name}: {type(exc).__name__}: {exc}")
        return math.nan


async def evaluate_with_ragas(
    records: list[EvaluationRecord],
) -> None:
    """逐条执行五个 RAGAS 指标，并在每条完成后保存一次进度。"""

    components = build_ragas_components()
    total = len(records)

    # 先写入已采集的 RAG 数据。若后续人工中断，仍可查看已完成的采集结果。
    write_result_csv(records)

    for index, record in enumerate(records, start=1):
        print(f"[RAGAS {index}/{total}] {record.user_input}")
        errors: list[str] = []

        # 三个上下文指标必须有实际召回片段；没有上下文时用 NaN 明确表示不可评分。
        if record.retrieved_contexts:
            record.scores["context_precision"] = await _run_metric(
                "context_precision",
                components.context_precision.ascore(
                    user_input=record.user_input,
                    reference=record.reference,
                    retrieved_contexts=record.retrieved_contexts,
                ),
                errors,
            )
            record.scores["context_recall"] = await _run_metric(
                "context_recall",
                components.context_recall.ascore(
                    user_input=record.user_input,
                    reference=record.reference,
                    retrieved_contexts=record.retrieved_contexts,
                ),
                errors,
            )
            record.scores["faithfulness"] = await _run_metric(
                "faithfulness",
                components.faithfulness.ascore(
                    user_input=record.user_input,
                    response=record.response,
                    retrieved_contexts=record.retrieved_contexts,
                ),
                errors,
            )
        else:
            errors.append("context metrics: retrieved_contexts 为空，三个上下文指标记为 NaN。")

        # Answer Relevancy 不依赖召回上下文，只要有最终回答就可以单独评分。
        if record.response:
            record.scores["answer_relevancy"] = await _run_metric(
                "answer_relevancy",
                components.answer_relevancy.ascore(
                    user_input=record.user_input,
                    response=record.response,
                ),
                errors,
            )

            # Answer Correctness 同时使用 Question、Answer 和 Ground_truth。
            record.scores["answer_correctness"] = await _run_metric(
                "answer_correctness",
                components.answer_correctness.ascore(
                    user_input=record.user_input,
                    response=record.response,
                    reference=record.reference,
                ),
                errors,
            )
        else:
            errors.append("answer_relevancy: response 为空，指标记为 NaN。")
            errors.append("answer_correctness: response 为空，指标记为 NaN。")

        record.evaluation_error = " | ".join(errors)
        if record.evaluation_error:
            print(f"  -> 指标异常：{record.evaluation_error}")

        # 每完成一条就覆盖保存同一个结果文件，支持长任务中途查看和断点排查。
        write_result_csv(records)


# =============================================================================
# 第 6 步：输出同名结果 CSV
# =============================================================================

def _score_to_csv_value(score: float) -> str | float:
    """CSV 中用空字符串表示 NaN，避免把无法评分误解为 0 分。"""

    return "" if math.isnan(score) else score


def _contexts_to_csv_value(record: EvaluationRecord) -> str:
    """把多个召回片段整理成单个、可读的 Context 单元格。"""

    context_blocks: list[str] = []
    for index, context in enumerate(record.retrieved_contexts, start=1):
        detail = (
            record.retrieved_context_details[index - 1]
            if index <= len(record.retrieved_context_details)
            else {}
        )
        metadata = [f"Context {index}"]
        for field_name in ("source", "chunk_id", "url", "title", "score"):
            field_value = detail.get(field_name)
            if field_value not in (None, ""):
                metadata.append(f"{field_name}={field_value}")

        context_blocks.append(f"[{' | '.join(metadata)}]\n{context}")

    return "\n\n".join(context_blocks)


def write_result_csv(records: list[EvaluationRecord]) -> None:
    """按固定 9 列输出 Question、Ground_truth、Context、Answer 和五项分数。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # utf-8-sig 便于 Windows Excel 直接打开中文 CSV。
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "Question": record.user_input,
                    "Ground_truth": record.reference,
                    "Context": _contexts_to_csv_value(record),
                    "Answer": record.response,
                    "Context_Precision": _score_to_csv_value(
                        record.scores["context_precision"]
                    ),
                    "Context_Recall": _score_to_csv_value(
                        record.scores["context_recall"]
                    ),
                    "Faithfulness": _score_to_csv_value(
                        record.scores["faithfulness"]
                    ),
                    "Answer_Relevancy": _score_to_csv_value(
                        record.scores["answer_relevancy"]
                    ),
                    "Answer_Correctness": _score_to_csv_value(
                        record.scores["answer_correctness"]
                    ),
                }
            )


# =============================================================================
# 主入口：只有用户手动运行本文件时才执行
# =============================================================================

def main() -> None:
    """串联“读取数据 -> 调用 RAG -> RAGAS 评估 -> 保存结果”的完整流程。"""

    print(f"测试集：{INPUT_CSV}")
    print(f"结果文件：{OUTPUT_CSV}")

    records = load_test_dataset()
    print(f"共读取 {len(records)} 条测试数据。")

    collect_rag_outputs(records)
    asyncio.run(evaluate_with_ragas(records))

    print(f"评估完成，结果已保存：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
