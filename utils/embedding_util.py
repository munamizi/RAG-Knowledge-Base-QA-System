from typing import Tuple, List, Dict

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from processor.configs.embedding_config import embedding_config

embedding_model = None

def get_embedding_model():
    """
        生成所需要的embedding模型
    """
    global embedding_model
    if embedding_model is not None:
        return embedding_model

    embedding_model = BGEM3EmbeddingFunction(
        model_name=embedding_config.bge_m3_path,
        device=embedding_config.bge_device,
        use_fp16=embedding_config.bge_fp16
    )
    return embedding_model



def generate_embeddings(texts:list[str]):
    """向量嵌入"""
    model = get_embedding_model()
    result = model.encode_documents(texts)

    dense_list = [dense.tolist() for dense in result['dense']]

    sparse_list = []
    for i in range(len(texts)):
        sparse_obj = result['sparse']

        sparse_indices = sparse_obj.indices[sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]].tolist()
        sparse_data = sparse_obj.data[sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]].tolist()
        sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}

        sparse_list.append(sparse_dict)

    return{
        'dense': dense_list,
        'sparse': sparse_list
    }

