from typing import Tuple, List, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pymilvus import DataType

from processor.config_plus.milvus_config import milvus_config
from processor.import_processor.BaseNode import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_util import generate_embeddings
from utils.escape_milvus_string import escape_milvus_string
from utils.logger import logger
from utils.milvus_client_util import get_milvus_client


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点（业务强要求：要求输入的一个md文档讲的是一个商品）
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph 核心节点：商品主体名称识别
        流程：
            1. 提取输入
            2. 构建大模型上下文
            3. 调用大模型识别商品名称
            4. 回填商品名称到状态和切片
            5. 生成商品名称的稠密/稀疏向量
            6. 将数据存入Milvus向量数据库
        注意：
            必要参数：task_id、 file_title、 chunks
            更新参数：item_name

        """

        # 步骤1：提取并校验输入
        file_title, chunks = self._step_1_get_inputs(state)

        # 步骤2：构建大模型识别的上下文
        context = self._step_2_build_context(chunks)

        # 步骤3：调用大模型识别商品名称
        item_name = self._step_3_call_llm(file_title, context)

        # 步骤4：回填商品名称到状态和切片
        self._step_4_update_chunks(state, chunks, item_name)

        # 步骤5：为商品名称生成稠密、稀疏向量
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)

        # 步骤6：将数据存入Milvus向量数据库
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        # 打印识别结果
        logger.info(f"--- 识别完成: {item_name} ---")

        return state




    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, List[Dict]]:
        """
        步骤 1: 接收并校验流程输入

        Agrs：
            file_title(str)：待被检验的文件标题
            chunks(list[dict[str,str]])：待被检验的chunks
        Returns：
            file_title(str)：被检验后的文件标题
            chunks(list[dict[str,str]])：被检验后的chunks

        raise:

        """

        # 1、参数校验
        file_title = state["file_title"]
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state["chunks"]
        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确，不为list[dict[str,str]]",expected_type=list)
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)



        return file_title, chunks




    def _step_2_build_context(self, chunks: List[Dict]) -> str:
        """
        步骤 2: 构造要喂给llm商品名称识别的提示词
        注意事项：
            1. 限制切片数量：仅取前k个切片，避免上下文过长
            2. 限制字符长度：总上下文字符限制，适配大模型输入上限
            3. 格式化内容：带序号的结构化格式，提升大模型识别精度

        Args：
            chunks(list[dict[str,str]]): 文本切片列表
        Returns：
            context(str): 格式化后的要喂给llm主体识别的字符串

        """

        parts: List[str] = []
        total_chars = 0
        for idx, chunk in enumerate(chunks[:self.config.item_name_chunk_k], start=1):

            # 1. 提取前k的切片，避免上下文过长
            chunk_title = chunk["title"].strip()
            chunk_content = chunk["content"].strip()

            # 2. 格式化切片
            piece = f"【切片{idx}】\n标题{chunk_title}\n内容：{chunk_content}"
            parts.append(piece)

            # 3. 计算累计的字符数，判断是否要继续切分
            total_chars += len(piece)
            if total_chars > self.config.item_name_chunk_size:
                logger.warning(f"累计字符数{total_chars}已超过限制{self.config.item_name_chunk_size}，停止切分")
                break

        # 4. 使用换行符对切分后的片段进行连接
        context = "\n\n".join(parts).strip()

        # 5. 对返回结果进行二次截断
        final_context = context[:self.config.item_name_chunk_size]

        return final_context




    def _step_3_call_llm(self, file_title: str, context: str) -> str:
        """
        步骤 3: 调用 LLM 识别商品名称/型号
        兜底处理：context 空 / LLM 返回空 / 调用异常 → 均返回 file_title作为商品名称

        Agrs:
            file_title: 文件标题，兼作兜底值
            context:    步骤2构建的chunk提示词
        Returns:
             item_name: llm识别后的商品名称
        """

        if not context:
            return file_title

        try:
            # 构建 Prompt：约束输出为纯商品名字符串
            prompt = f"""
                       请从以下信息中识别出商品名称与型号：
                       “文件名：{file_title}
                       正文片段（用于辅助识别）：
                       {context}”
    
                       约束要求：
                       1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉
                       2. 返回结果只包含商品名称，不要添加任何解释或其他内容
                       3. 如果无法识别商品名称,请返回空字符串
            """

            # llm主体识别时，无需开thinking
            model = ChatOpenAI(
                model=self.config.item_model,
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                temperature=self.config.item_model_template,
                extra_body={"enable_thinking": False}
            )


            messages = [
                SystemMessage(content="你是商品识别专家，请根据提供的信息，识别商品名称。"),
                HumanMessage(content=prompt)
            ]

            resp = model.invoke(messages)

            # 去空白：商品名当 Milvus 主键去重依据，不能带脏字符
            item_name = resp.content.strip().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

            return item_name if item_name else file_title

        except Exception as e:
            # 超时/网络/格式错均兜底，不中断主流程，直接返回file_title作为主体识别的结果
            logger.exception(f"步骤3：大模型调用失败，原因：{str(e)}")
            return file_title




    def _step_4_update_chunks(self, state: ImportGraphState, chunks: List[Dict[str, str]], item_name: str):
        """
        步骤 4: 回填商品名称到流程状态和所有文本切片
        核心作用：
            1. 全局状态更新：将item_name存入state，供下游所有节点直接使用
            2. 切片数据补全：为每个切片添加item_name字段，保证数据一致性
            3. 状态同步：更新state中的chunks，确保切片修改全局生效


        :param    state
        :param    chunks:上一节点切分得到的chunks
        :param    item_name: step3大模型识别得到的商品名
        """
        # 1、将商品名称存入state状态，供下游节点调用
        state["item_name"] = item_name

        # 2、遍历所有切片，为每个切片添加商品名字段
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["chunks"] = chunks




    def _step_5_generate_vectors(self, item_name: str) -> Tuple[List|None, Dict|None]:
        """
        步骤 5: 通过BGE-M3对识别出的商品名item_name做向量嵌入(稀疏+稠密)

        Agrs：
            item_name: step3 识别的商品名称（非空，空值时直接返回空向量）
        Returns：
            Tuple[List, Dict]: (稠密向量列表，稀疏向量字典)

        """

        # 1. 商品名称为空 → 直接返回空向量
        if not item_name:
            return None, None

        # 2、调用向量嵌入工具
        vectors = generate_embeddings([item_name])
        return vectors["dense"][0], vectors["sparse"][0]




    def _step_6_save_to_milvus(self, state: ImportGraphState, file_title: str, item_name: str,
                               dense_vector, sparse_vector):
        """
        步骤 6: 将商品名 + 双向量持久化到 Milvus
            集合不存在则建（Schema + 索引），存在则复用；
            插入前删同名旧数据，实现幂等。
            全程 try/except 兜底，入库失败只告警，不中断主流程。

        Args：
            file_title:    文件标题
            item_name:     主体识别出的商品名
            dense_vector:  1024 维稠密向量
            sparse_vector: {索引: 权重} 稀疏向量
        """


        logger.info(f"将商品名称：'{item_name}' 保存到 milvus 集合: {milvus_config.item_name_collection}")

        try:
            client = get_milvus_client()
            if not client:
                logger.warning("Milvus客户端创建失败")
                return

            # 集合不存在才创建
            if not client.has_collection(collection_name=milvus_config.item_name_collection):
                logger.info(f"集合 {milvus_config.item_name_collection} 不存在，开始创建")
                # 配置schema
                schema = (
                    client.create_schema(auto_id=True, enable_dynamic_field=True)
                    .add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
                    .add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
                    .add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
                    .add_field(field_name="dense_vector", datatype=DataType .FLOAT_VECTOR, dim=1024)
                    .add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
                )

                # 配置索引
                index_params = client.prepare_index_params()
                index_params.add_index(
                    field_name="dense_vector",
                    index_name="dense_vector_index",
                    index_type="IVF_FLAT",
                    metric_type="COSINE",
                    params={
                        "nlist": 128
                    }
                )
                index_params.add_index(
                    field_name="sparse_vector",
                    index_name="sparse_vector_index",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="IP",
                    params={
                        "inverted_index_algo": "DAAT_MAXSCORE",
                        "normalize": True,
                        "quantization": "none"
                    }
                )
                # 创建集合
                client.create_collection(
                    collection_name=milvus_config.item_name_collection,
                    schema=schema,
                    index_params=index_params
                )

            # 幂等性处理
            clean_item_name = (item_name or "").strip()
            if clean_item_name:
                client.load_collection(collection_name=milvus_config.item_name_collection)
                safe_item_name = escape_milvus_string(clean_item_name)
                client.delete(collection_name=milvus_config.item_name_collection, filter=f'item_name=="{safe_item_name}"')

            # 组装插入的list[dict](向量非空才加，避免 None 触发类型错)
            data = {
                "file_title": file_title,
                "item_name": item_name
            }
            if dense_vector is not None:
                data["dense_vector"] = dense_vector
            if sparse_vector is not None:
                data["sparse_vector"] = sparse_vector

            client.insert(collection_name=milvus_config.item_name_collection, data=[data])

            # 插入后 load，让新数据立即可被Attu查
            client.load_collection(collection_name=milvus_config.item_name_collection)

            state["item_name"] = item_name

        except Exception as e:
            logger.warning(f"步骤6：数据存入 Milvus 失败，原因：{str(e)}", exc_info=True)




