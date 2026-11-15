"""答案溯源模块：给参考片段编号，再根据答案中的编号找到出处。"""
import re
from urllib.parse import urlsplit


# 检索时返回的正文和来源字段
RETRIEVAL_OUTPUT_FIELDS = [
    "chunk_id",
    "content",
    "item_name",
    "file_title",
    "title",
    "parent_title",
    "part",
]

# 答案中的引用编号
_CITATION = re.compile(r"(?<!!)\[([0-9]{1,6})\](?!\()")

# 不参与引用编号识别的代码和链接
_MARKDOWN_LITERAL = re.compile(
    r"```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`"
    r"|!?\[[^\]\n]*\]\([^\n]*?\)"
    r"|^\s*\[[^\]\n]+\]:[^\n]*",
    re.MULTILINE,
)



def build_reference_context(docs: list[dict], char_budget: int) -> tuple[str, list[dict], int]:
    """
    构造本轮 LLM 使用的文本块上下文和编号与文本块对应表。

    内容：
        按 docs 顺序组装文本块与来源标签，跳过空内容和超出剩余预算的文本块
        为纳入上下文的文本块连续编号，生成对应表 references
        按正文、来源标签和块间分隔符的字符数扣减预算，计算 remaining

    Args:
        docs - 待组装的文本块及来源元数据，按列表顺序处理
        char_budget - 上下文字符预算，小于 0 时按 0 处理

    Returns:
        entries - 纳入预算的文本块与来源标签组成的大字符串，块间以空行分隔
        references - 已纳入上下文的编号与文本块对应表，包含来源元数据
        remaining - 扣除已纳入内容后的剩余字符预算
    """
    entries = []
    references = []
    remaining = max(0, char_budget)
    for doc in docs:
        content = _text(doc.get("content"))
        if not content:
            continue
        source = "web" if doc.get("source") == "web" else "local"
        reference = {
            "id": len(references) + 1,
            "source": source,
            "chunk_id": _text(doc.get("chunk_id")) or None if source == "local" else None,
            "file_title": _text(doc.get("file_title")) or None if source == "local" else None,
            "title": _text(doc.get("title")) or None,
            "section": (_text(doc.get("parent_title")) or _text(doc.get("title")) or None) if source == "local" else None,
            "item_name": _text(doc.get("item_name")) or None if source == "local" else None,
            "url": _web_url(doc.get("url")) if source == "web" else None,
            "content": content,
            "content_kind": "web_snippet" if source == "web" else "processed_chunk",
        }
        tags = [f"[{reference['id']}]", f"[source={source}]"]
        for field in ("file_title", "section", "title", "item_name", "url"):
            if reference[field]:
                tags.append(f"[{field}={reference[field]}]")
        entry = " ".join(tags) + "\n" + content
        cost = len(entry) + (2 if entries else 0)
        if cost > remaining:
            continue

        entries.append(entry)
        references.append(reference)
        remaining -= cost

    return "\n\n".join(entries), references, remaining


def resolve_citations(answer: str, references: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    按答案中的引用编号匹配来源记录，并返回编号检查结果。

    步骤：
        1. 构建编号与来源记录映射，提取答案正文中的引用编号并按首次出现顺序去重
        2. 匹配已有编号对应的来源记录，按引用顺序生成 citations
        3. 检查未知编号及无有效引用的情况，生成 warnings，不判断来源内容是否支持答案

    Args:
        answer - 待解析引用编号的答案文本，匹配到的代码片段、Markdown 链接等内容不参与编号识别
        references - 本轮参考来源记录列表，每条记录包含引用编号 id

    Returns:
        citations - 匹配成功的来源记录副本列表，按编号在答案中首次出现的顺序排列
        warnings - 编号检查警告列表，无警告时为空
            unknown_citation 表示答案中出现了来源列表没有的编号
            no_valid_citations 表示已有来源且答案去除首尾空白后非空，但没有有效引用
    """
    by_id = {reference["id"]: reference for reference in references}
    ids = _citation_ids(answer)
    citations = [dict(by_id[ref_id]) for ref_id in ids if ref_id in by_id]
    # 只检查编号是否存在，不判断片段是否支持答案
    warnings = []
    unknown = [ref_id for ref_id in ids if ref_id not in by_id]
    if unknown:
        warnings.append({"code": "unknown_citation", "citation_ids": unknown})
    if references and answer.strip() and not citations:
        warnings.append({"code": "no_valid_citations"})
    return citations, warnings


def _citation_ids(answer: str) -> list[int]:
    """排除匹配到的 Markdown 代码片段、链接及链接定义，提取引用编号并按首次出现顺序去重。"""
    prose = _MARKDOWN_LITERAL.sub("", answer)
    return list(dict.fromkeys(int(match.group(1)) for match in _CITATION.finditer(prose)))

def _text(value) -> str:
    """转换字段文本并去除首尾空白，None 返回空字符串。"""
    return str(value).strip() if value is not None else ""

def _web_url(value) -> str | None:
    """返回含主机名且不含非空用户名的 HTTP/HTTPS 地址，解析失败或条件不符时返回 None。"""
    url = _text(value)
    try:
        parsed = urlsplit(url)
        if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username:
            return url
    except ValueError:
        pass
    return None
