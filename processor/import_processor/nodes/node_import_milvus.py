from typing import Dict, Any, List

from pymilvus import DataType

from processor.configs.milvus_config import milvus_config
from processor.import_processor.BaseNode import BaseNode
from processor.import_processor.exceptions import StateFieldError, MilvusError
from processor.import_processor.state import ImportGraphState
from utils.escape_milvus_string import escape_milvus_string
from utils.logger import logger
from utils.milvus_client_util import get_milvus_client


class NodeImportMilvus(BaseNode):

    name: str = "node_import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        chunks存入milvus的kb_chunks集合

        步骤：
            1. 输入校验，返回稠密向量维度(方便后续创建集合)
            2. 创建milvus集合kb_chunks，保证幂等性
            3. 批量插入：chunck批量入库，回填Milvus自增chunk_id到state[chunks]

        Args:
            state[chunks]
        Returns:
            chunks字段回填chunk_id
        """

        # chunks有效性校验，返回校验后chunks和稠密向量维度
        checked_chunks, vector_dimension = self._step_1_check_input(state)

        # 获取milvus客户端、创建集合kb_chuncks
        client = self._step_2_prepare_collection(vector_dimension)
        # 幂等性处理（清理file_title同名旧数据）
        self._step_3_clean_old_data(client, checked_chunks)

        # 批量插入数据，主键chunk_id回填; 更新state[chunks]
        updated_chunks = self._step_4_insert_data(client, checked_chunks)
        state["chunks"] = updated_chunks

        return state



    def _step_1_check_input(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        """
         chunks有效性校验

        内容：
            1. 校验chunks非空且类型正确[dict]
            2. 校验chunks包含dense_vector和sparse_vector字段
            3. 提取向量维度（为集合创建/索引构建提供依据）

        Args:
            state[chunks]
        Returns:
            (chunks, 稠密向量维度)
        """

        # 校验chunks非空
        chunks = state.get("chunks")
        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        # 校验切片包含dense_vector字段, 校验切片包含 sparse_vector 字段
        first_chunk = chunks[0]
        if 'dense_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失dense_vector字段")
        if 'sparse_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失sparse_vector字段")

        # 提取向量维度
        vector_dimension = len(first_chunk['dense_vector'])

        return chunks, vector_dimension



    def _step_2_prepare_collection(self, vector_dimension: int):
        """
        创建向量库集合

        Args：
            vector_dimension - 稠密向量维度
        Reutrns：
            MilvusClient - 已连接、集合准备完成的milvus客户端实例
        Raise：
            客户端获取失败/集合名称未配置，抛出异常终止流程
        """

        # 获取milvus客户端实例
        milvus_client = get_milvus_client()
        if not milvus_client:
            logger.error("Milvus 连接失败")
            raise MilvusError("Milvus 连接失败")

        # 集合不存在则创建，有则复用
        collections_name = milvus_config.chunks_collection
        if not milvus_client.has_collection(collections_name):
            self._create_chunks_collection(collections_name, milvus_client, vector_dimension)

        return milvus_client

    def _create_chunks_collection(self, collections_name, milvus_client, vector_dimension):

        # 配置schema和向量索引
        schema = (
            milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)   # 允许添加schema外的额外字段
            .add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)   # 主键pk，且自动生成
            .add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)               # 切片内容
            .add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)                   # 切片标题
            .add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)            # 切片父标题
            .add_field(field_name="part", datatype=DataType.INT8)                                       # 分片父标题编号
            .add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)              # 导入资料名
            .add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)               # 商品名称(幂等清理依据)
            .add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)               # 稀疏向量
            .add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension) # 稠密向量
        )

        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",  # 自动匹配最优索引类型
            metric_type="COSINE"     # 余弦相似度
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX", # 倒排索引
            metric_type="IP",                   # 点积
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )


        # 创建集合
        milvus_client.create_collection(
            collection_name=collections_name,
            schema=schema,
            index_params=index_params
        )



    def _step_3_clean_old_data(self, client, checked_chunks):

        """
        清理集合中与fifle_title同名的旧数据

        Args:
            client - milvus客户端
            chunks - 检验后的chunks

        """

        # 1. 获取查询条件
        file_title = checked_chunks[0].get("file_title")

        # 2. 执行幂等清理
        self._clear_chunks_by_file_title(client, file_title)

    def _clear_chunks_by_file_title(self, client, file_title):

        try:
            file_title = escape_milvus_string(file_title)
            client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=f"file_title=='{file_title}'")
        except Exception as e:
            logger.error(f"Milvus 数据删除失败: {str(e)}")
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}")



    def _step_4_insert_data(self, client, checked_chunks):
        """
        批量插入切片数据到Milvus+主键回填
        内容:
            1. 批量插入数据
            2. 回填chunk_id：将Milvus生成的自增主键回填到state[chunks]

        Args:
            client - milvus客户端实例
            checked_chunks - 校验后的chunks

        Returns:
            state[chunks] - 回填chunks_id字段的chunks
            {
                "title": "注意事项-1",
                "parent_title": "注意事项"
                "part":"1"
                "content": "正文内容",
                "file_title": "hak180产品安全手册",

                "dense_vector"：content稠密向量,
                "sparse_vector"：content稀疏向量

                "chunk_id":主键
            }
        """

        # 1. 填充part字段
        for item in  checked_chunks:
            if item.get("part") is None:
                item["part"] = 0

        # 2. 批量插入数据
        result = client.insert(
            collection_name=milvus_config.chunks_collection,
            data= checked_chunks
        )

        # 3. 回填chunk_id
        inserted_ids = result.get("ids")
        for idx, item in enumerate(checked_chunks):
            item["chunk_id"] = inserted_ids[idx]

        return checked_chunks
