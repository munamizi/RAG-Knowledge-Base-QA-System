from processor.configs.embedding_config import embedding_config
from processor.import_processor.BaseNode import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from typing import List, Dict
from utils.embedding_util import generate_embeddings
from utils.logger import logger


class NodeBGEEmbedding(BaseNode):

    name: str = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        BGE-M3向量化chunks
        内容：
            1. 输入校验：验证 chunks 有效性，数据缺失则终止当前节点
            2. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
            3. 状态更新：将带向量字段的 chunks 回写state

        Args:
            state[chunks] - 更新前chunks

        Returns:
            state[chunks更新后] - 更新后chunks
            新chunck结构:
                {
                 "title": "安全注意事项",
                 "parent_title": "第一章"
                 "content": "很长很长的正文...",
                 "file_title": "hak180产品安全手册",

                  "dense_vector"：稠密向量,
                  "sparse_vector"：稀疏向量
                }

        """

        # 校验chunks有效性，返回校验后chunks
        chunks = self._step_1_validate_input(state)

        # 批量生成双向量，为chunks添加向量字段
        output_data = self._step_2_generate_embeddings(chunks)

        # 更新state[chunks]
        state['chunks'] = output_data

        return state




    def _step_1_validate_input(self, state: ImportGraphState) -> List[Dict]:
        """
        输入数据有效性校验

        Args:
            state[chunks]

        Returns:
            List[Dict] - 校验通过的文本切片列表

        Raise:
            StateFieldError

        """

        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return chunks



    def _step_2_generate_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        批量生成双向量，为 chunks添加向量字段

        内容:
            1. 分批处理：减少云端API调用 / 防止本地爆内存
            2. 文本构造：item_name 前置拼接 chunk的content
            3. 向量生成：批量产出稠密、稀疏向量。

        Args:
            chunks: List[Dict] 待向量化的文本切片列表

        Returns:
            List[Dict] 包含 dense_vector / sparse_vector 字段的文本切片列表

        """

        # 初始化结果列表[dict]
        output_data = []
        # 批次大小
        batch_size = embedding_config.batch_size

        # 按批次遍历（本地BGE-M3防止爆内存）
        for i in range(0, len(chunks), batch_size):

            batch_texts = chunks[i:i + batch_size]
            # 对一批中的每个chunk处理，构造为[str]
            input_texts = []
            for doc in batch_texts:
                item_name = doc.get("item_name")
                content = doc.get("content", "")
                input_texts.append(f"{item_name}\n{content}" if item_name else content)
            # 向量化
            results = generate_embeddings(input_texts)

            # 配置新的chunk结构
            for j, doc in enumerate(batch_texts):
                item = doc.copy()
                item["dense_vector"] = results["dense"][j]
                item["sparse_vector"] = results["sparse"][j]
                output_data.append(item)
            logger.info(f"成功获取第 {i + 1}-{min(i + len(batch_texts), len(chunks))} 项的嵌入。")

        # 返回添加向量字段新chunks
        return output_data
