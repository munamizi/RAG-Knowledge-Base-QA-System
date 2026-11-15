import re
import json
from pathlib import Path
from typing import Tuple, List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from processor.configs.document_split_config import document_split_config
from processor.import_processor.BaseNode import BaseNode
from processor.import_processor.state import ImportGraphState
from utils.logger import logger

class NodeDocumentSplit(BaseNode):

    name: str = "node_document_split"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        文档切分节点：对 md 先按标题粗切,再根据阈值长切短合为合适的chunks

        内容：
            1. 拿输入数据
            2. 按标题粗切。没标题就兜底处理
            3. 长切过滤、短合过滤
            4. 打印统计信息
            5. 备份到本地 JSON

        Args:
            state[md_path] - 新 md文件路径
            state[md_content] - 新 md文件内容
            state[file_title] - 上传资料文件名
        Returns:
             state[chunks]: list[dict] - 切分后的chunks

        """

        # 1. 拿到md内容和文件标题校验、换行标准化，返回md内容和上传文件名
        content, file_title = self._step_1_get_inputs(state)

        # 2. 按 1~6 # 进行标题粗切，返回一个元组
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)
        # 过滤：如果粗切后的chunks都没有标题,就把全文当成一个section返回
        sections = self._step_3_handle_chunks_no_title(sections, title_count, content,  file_title)

        # 3. 太长的section切小, 太短的section合并
        sections = self._step_4_refine_chunks(sections)

        # 4. 打印切分结果,方便调试
        self._step_5_print_stats(lines_count, sections)

        # 5. 把切分结果存一份到本地,排查问题用
        self._step_6_backup(state, sections)

        # 更新state[chunks]
        state["chunks"] = sections
        return state




    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        """从 state 里取出 md 内容和文件名,做基础清理。"""

        file_title = state.get("file_title", "")
        if not file_title:
            raise ValueError("核心参数 file_title 缺失")

        md_content = state.get("md_content", "")
        if not md_content:
            raise ValueError("核心参数 md_content 缺失")

        # 统一换行符：把 Windows 的 \r\n 和老 Mac 的 \r 都换成 \n
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        logger.info(f"步骤1:输入数据加载完成,文件标题:{file_title}")
        return md_content, file_title



    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, str]], int, int]:
        """
        按标题切大块:
            按 md 的标题(# ## ### ...)切分,每个标题下的内容算一个章节；
            注意：代码块里的 # 不算标题,要跳过。

        Args:
            content - md内容
            file_title - 上传资料文件名

        Returns:
            Tuple(list[dict]、int、int)
            (
                [
                    {
                      'title': 当前块所在标题
                      'content': 当前块内容
                      'file_title': 当前资料文件名
                    },
                    ......
                ],
                标题总数,
                md总行数,
            )

        """
        # 把md按换行分割成一个行列表，等待后续被检索
        lines = content.split("\n")
        # 标题正则：匹配 1~6 级标题:行首可有空格,然后 1~6 个#, #后必须有空格和文字
        title_pattern = r'\s*#{1,6}\s+.+'
        # sections列表
        sections: List[Dict[str, str]] = []
        title_count = 0                 # 标题数量
        current_title = ""              # 当前section的标题
        current_lines: List[str] = []   # 当前section累积的行
        # 是否正在代码块里
        in_code_block = False
        # 记一下进入代码块时用的是 ``` 还是 ~~~, 要用同样的才算结束
        code_block_start_marker = None

        def _flush_section():
            """把当前累积的内容存成一个section。"""
            if not current_lines:
                return
            sections.append({
                "title": current_title,
                "content": "\n".join(current_lines),
                "file_title": file_title,
            })

        # 逐行分析：
        for line in lines:
            # 去掉该行的首尾空格符
            stripped_line = line.strip()
            # 检查是不是代码块的开始/结束(``` 或 ~~~,3 个或更多)
            code_block_marker_match = re.match(r'^(`{3,}|~{3,})$', stripped_line)

            # 情况一：如果在代码块中
            if code_block_marker_match:
                marker = code_block_marker_match.group(1)
                if not in_code_block:
                    # 进入代码块
                    in_code_block = True
                    code_block_start_marker = marker
                elif stripped_line == code_block_start_marker:
                    # 遇到一样的标记,代码块结束
                    in_code_block = False
                    code_block_start_marker = None
                current_lines.append(line)
                continue

            # 情况二：如果不在代码块中,才把 # 开头的行当成标题
            is_valid_title = (not in_code_block) and re.match(title_pattern, line)
            if is_valid_title:
                # 遇到新标题:先把上一个section存起来,再开始新的
                _flush_section()
                current_title = stripped_line
                current_lines = [current_title]
                title_count += 1
                logger.info(f"识别标题:{current_title}")
            else:
                # 普通行,追加到当前章节
                current_lines.append(line)

        # 最后一个章节
        _flush_section()

        logger.info(f"按标题切分完成,共 {len(sections)} 个章节,标题数 {title_count},原始行数 {len(lines)}")

        return sections, title_count, len(lines)



    def _step_3_handle_chunks_no_title(
        self,
        sections: List[Dict[str, str]],
        title_count: int,
        content: str,
        file_title: str,
    ) -> List[Dict[str, str]]:
        """
        无标题整合：如果md粗切后的chunks都没有标题,就把全文当成一个section

        Args:
            content - md内容
            sections - 上步骤粗切后的所有块
            title_count - 粗切后统计的标题数量
            file_title - 上传资料名

        Returns:
            section: ditc
            {
                "title": "无标题",
                "content": md内容,
                "file_title": 上传资料名
            }

        """
        if title_count == 0:
            logger.warning(f"没识别到任何 MD 标题,把全文当成一个章节处理:{file_title}")
            return [
                {
                    "title": "无标题",
                    "content": content,
                    "file_title": file_title
                 }
            ]

        return sections



    def _step_4_refine_chunks(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        长切、短合:
        1. 太长的章节切成小块
        2. 太短的章节合并起来
        3. 给每个 Chunk 补上 parent_title 字段(Milvus 入库要用,不能是 None)

        Args:
            sections - 上哥步骤粗切和过滤后的chunks

        Returns:
            final_sections - 长切短合后的最终chunks

        """

        refined_split: List[Dict[str, str]] = []

        for sec in sections:
            refined_split.extend(self._split_long_section(sec))

        logger.info(f"长切完成：{len(refined_split)} 个 Chunk")

        for sec in refined_split:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""

        final_sections = self._merge_short_sections(refined_split)

        logger.info(f"短合完成：{len(final_sections)} 个 Chunk")

        return final_sections


    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        """
        把超过阈值上限的section切小（切后的每一小块都带上parent_title和part, 这样知道它属于哪个父section）

        Args:
            section - 一个section

        Returns:
            sections - 长切后的secitions

        """

        content = section.get("content", "")

        if len(content) <= document_split_config.max_content_length:
            return [section]

        title = section.get("title", "")
        # 每个小块都加上 "标题\n\n" 作为前缀
        prefix = f"{title}\n\n" if title else ""

        # 减去前缀长度,正文部分最多能有多少字
        available_len = document_split_config.max_content_length - len(prefix)
        if available_len <= 0:
            logger.warning(f"标题太长无法切分:{title[:20]}...")
            return [section]

        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title):].lstrip()

        # 用 LangChain 递归分割器切
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=0,  # 已经按标题切过了,不需要重叠
            separators=["\n\n", "\n", "。", "!", "?", ";", ".", "!", "?", ";", " "],
        )

        sub_sections: List[Dict[str, str]] = []
        # enumerate 同时拿到序号和内容,start=1 让序号从 1 开始
        for idx, chunk in enumerate(splitter.split_text(body), start=1):
            text = chunk.strip()
            if not text:
                continue
            sub_sections.append({
                "title": f"{title}-{idx}" if title else f"chunk-{idx}",
                "content": (prefix + text).strip(),  # 前缀+正文
                "parent_title": title,  # 记下父标题,后面合并时要用
                "part": idx,
                "file_title": section.get("file_title"),
            })
        logger.debug(f"切长:{title} → {len(sub_sections)} 个小块")
        return sub_sections


    def _merge_short_sections(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        把太短的相邻 sections 合并。
        合并必须同时满足条件:
          1. 当前块长度小于配置的最小长度
          2. 当前块和下一块属于同一个父标题(parent_title相同)

        Args:
            sections - 长切过滤后的sections

        Returns:
            sections - 短合过滤后的sections

        """
        if not sections:
            return []
        merged_sections: List[Dict[str, str]] = []
        current_chunk = None  # 暂存正在合并的块
        for sec in sections:
        # 第一个块直接拿来当起点
            if current_chunk is None:
                current_chunk = sec
                continue

            # 判断要不要合并
            is_current_short = (
                len(current_chunk["content"])
                < document_split_config.min_content_length
            )
            is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

            if is_current_short and is_same_parent:
                # 合并前先把下一块开头重复的父标题去掉,免得内容里出现两遍标题
                parent_title = sec.get("parent_title", "")
                next_content = sec["content"]
                if parent_title and next_content.startswith(parent_title):
                    next_content = next_content[len(parent_title):].lstrip()

                # 用空行拼接,保持格式整洁
                current_chunk["content"] += "\n\n" + next_content

                # 更新一下序号,记成最新的
                if "part" in sec:
                    current_chunk["part"] = sec["part"]

                logger.debug(
                    f"合并短块:{current_chunk.get('parent_title')} → 当前长度 {len(current_chunk['content'])}"
                )
            else:
                # 不能合并:把当前块存起来,开始处理下一块
                merged_sections.append(current_chunk)
                current_chunk = sec

        # 循环结束后,最后一块也要存进去
        if current_chunk is not None:
            merged_sections.append(current_chunk)

        logger.debug(f"合并完成:原 {len(sections)} 个 → 合并后 {len(merged_sections)} 个")
        return merged_sections



    def _step_5_print_stats(self, lines_count: int, sections: List[Dict[str, str]]) -> None:
        """打印统计：打印切分结果,方便看效果和排查问题。"""
        chunk_num = len(sections)
        logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        logger.info(f"MD 原始行数:{lines_count}")
        logger.info(f"最终 Chunk 数量:{chunk_num}")



    def _step_6_backup(self, state: ImportGraphState, sections: List[Dict[str, str]]) -> None:
        """本地备份：把切分结果存成 chunks.json,放在 md 文件同目录下,方面排查问题。"""

        try:
            # 备份文件路径:和 MD 文件同目录,固定叫 chunks.json
            backup_path = Path(state["md_path"]).parent / "chunks.json"

            with open(backup_path, "w", encoding="utf-8") as f:
                # ensure_ascii=False:中文正常显示,不会变成 \u 编码
                # indent=2:格式化缩进,人工看 JSON 时更舒服
                json.dump(sections, f, ensure_ascii=False, indent=2)

            logger.info(f"步骤6:Chunk 结果备份成功,文件路径:{backup_path}")
        except Exception as e:
            # 备份失败不影响主流程,只记日志
            logger.error(f"步骤6:Chunk 结果备份失败:{str(e)}", exc_info=False)
