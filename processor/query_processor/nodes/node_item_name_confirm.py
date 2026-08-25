from typing import Tuple, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from processor.configs.llm_config import llm_config
from processor.configs.milvus_config import milvus_config
from processor.query_processor.BaseNode import BaseNode
from processor.query_processor.prompt.node_item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from utils.embedding_util import generate_embeddings
from utils.logger import logger
from utils.milvus_client_util import get_milvus_client
from utils.milvus_hybrid_search_util import create_hybrid_search_requests, hybrid_search
from utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names

class NodeItemNameConfirm(BaseNode):
    """
        主体识别 + 指代消除、混合检索、评分对齐、条件分支
    """


    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        # 1.校验参数
        session_id, original_query = self._step_1_validate_param(state)
        logger.info(f"步骤1：参数校验通过")

        # 2.获取历史对话记录
        history = get_recent_messages(session_id)
        logger.info(f"步骤2：获取到 {len(history)} 条历史消息")
        state["history"] = history

        # 3.保存用户提问到MongoDB历史对话
        message_id = save_chat_message(session_id, "user", original_query)
        logger.info(f"步骤3：用户原始提问已保存, ID: {message_id}")

        # 4.主体识别和指代消除
        extract_res = self._step_4_extract_info(original_query, history)
        item_names = extract_res.get("item_names")
        rewritten_query = extract_res.get("rewritten_query")
        # 更新state对应字段
        state["item_names"] = item_names
        state["rewritten_query"] = rewritten_query

        # 5&6.如果有提取到商品名，进行评分对齐
        align_result = {}
        if len(item_names) > 0:
            query_results = self._step_5_vectorize_and_query(item_names)
            align_result = self._step_6_align_item_names(query_results)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")

        # 检查确认状态
        state = self._step_7_check_confirmation(state, align_result, history)

        # 写入最终历史
        self._step_8_write_history(state, session_id, rewritten_query, message_id)

        return state




    def _step_1_validate_param(self, state: QueryGraphState) -> Tuple[str, str]:

        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心参数original_query缺失")

        return session_id, original_query




    def _step_4_extract_info(self, query, history) -> Dict:
        """
        内容：
            1.LLM从当前用户问题、历史对话中提取出主要询问的商品名称item_names（可多个，JSON列表形式）若商品名不够明确则返回空列表，
            2.同时根据上下文重新改写问题，保证问题独立完整

        Args:
            query: str - 用户提问
            history: list[dict] - 近期会话历史，dict格式：{"role": "user/assistant", "text": "消息内容", "_id": "消息ID"}

        Returns:
            字典：
             {
                 "item_names": ["商品名1", "商品名2", ...],  # 提取的商品名列表，无则空列表
                 "rewritten_query": "改写后的用户提问"       # 包含商品名的独立问题，无则返回原始query
             }

        """
        try:
            logger.info("步骤4：正在初始化 LLM 客户端...")

            # 创建model
            model = ChatOpenAI(
                model=llm_config.model,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                temperature=llm_config.llm_temperature,
                # 开启JSON标准输出模式，强制模型返回可解析的json_object
                model_kwargs={
                    "response_format": {"type": "json_object"}
                }
            )

            # 构造历史对话文本为字符串
            history_text = ""
            for msg in history:
                role = msg.get("role")
                content = msg.get("text")
                history_text += f"{role}: {content}\n"

            # 处理拼接提示词
            user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text=history_text,
                query=query
            )

            # 构造LLM调用的消息列表
            messages = [
                SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]

            # 调用LLM客户端，发起请求获取结果
            response = model.invoke(messages)
            result = JsonOutputParser().invoke(response)


            # 字段检查：确保返回结果包含item_names字段，无则设为空列表、确保返回结果包含rewritten_query字段，无则复用原始查询
            if "item_names" not in result:
                result["item_names"] = []
            if "rewritten_query" not in result:
                result["rewritten_query"] = query

            # 给item_names 去除空格
            result["item_names"] = [
                name.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
                for name in result["item_names"]
            ]

            # 返回解析后的结果
            return result

        except Exception as e:
            # 捕获所有异常（如LLM调用失败、JSON解析失败等），记录错误日志
            logger.error(f"大模型调用异常：{e}")
            # 异常时返回默认结果：空商品名列表+原始查询
            return {"item_names": [], "rewritten_query": query}




    def _step_5_vectorize_and_query(self, item_names: list[str]) -> List[Dict]:
        """
           把提取到的所有item_names逐个做milvus向量混合检索，获取匹配评分

           Args:
               item_names - （如["商品A", "商品B"]）

           Returns:
               list[dict]
                [
                    {
                        "extracted_name": 提取商品名
                        "matches": [
                            {"item_name": "milvus库中的真实商品名", "score": 0.98},
                            ......
                        ]
                    },
                    ......
                ]
        """
        # 初始化最终返回结果列表，存储每个商品名的向量化查询结果
        results = []

        # 获取Milvus向量数据库的客户端连接对象
        client = get_milvus_client()
        if not client:
            logger.error("连接 Milvus 失败")
            return results

        # 获取Milvus中存储商品集合名
        collection_name = milvus_config.item_name_collection

        # 对list[商品名]向量嵌入（稠密+稀疏）
        embeddings = generate_embeddings(item_names)     # 返回结果：{"dense": [稠密向量1,...], "sparse": [稀疏向量1...]}

        # 6、遍历每个商品名称，逐个执行向量搜索
        for i in range(len(item_names)):
            try:
                # 从批量生成的向量结果中，取出当前商品名对应的稠密向量（列表形式）和稀疏向量（字典形式）
                dense_vector = embeddings.get("dense")[i]
                sparse_vector = embeddings.get("sparse")[i]

                # 混合检索
                reqs = create_hybrid_search_requests(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5
                )
                search_res = hybrid_search(
                    client=client,
                    collection_name=collection_name,
                    output_fields=["item_name"],    # 指定返回Milvus中存储的商品名字段
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2),      # 稠/稀疏向量评分权重配比
                    limit=5,                        # 最终返回Top5匹配结果
                    norm_score=True                 # 开启评分归一化，统一评分量级为0-1
                )

                # 将检索结果封装
                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        matches.append(
                            {
                                "item_name": hit.get("entity", {}).get("item_name"),
                                "score": hit.get("distance"),
                            }
                        )

                results.append({
                    "extracted_name": item_names[i],  # step4提取的原始商品名称
                    "matches": matches  # 该商品名的Top5匹配结果（含评分）
                })

            # 捕获单个商品名处理的异常（不中断其他商品名执行），仅记录错误日志
            except Exception as e:
                logger.error(f"查询商品名 '{item_names[i]}' 时出错: {e}")

        # 返回所有结果
        return results




    def _step_6_align_item_names(self, query_results) -> dict:
        """
        对齐step4提取的item_names，做评分对齐，生成「确认商品名」和「候选商品名」
        处理规则：
                评分＞0.8：全部取入确定项
                评分0.6~0.8：取分数最高前5个作为候选
                评分＜0.6：不返回任何商品名（确认+候选均为空）

        Args:
            query_results: list[dict]

        Returns:
            字典 - 商品名对齐结果，包含确认列表和候选列表，格式：
                 {
                     "confirmed_item_names": ["高分真实商品名1", "高分真实商品名2"],  # 无则空列表
                     "options": ["中分真实商品名1", "中分真实商品名2", ...]          # 无则空列表
                 }
        """

        # 初始化确认商品名列表和候选商品名列表
        confirmed_item_names: List[str] = []
        all_confirmed_item_names = True
        options: List[str] = []
        logger.info(f"步骤6：获得待处理的数据源：{query_results}")

        for res in query_results:
            # 提取匹配结果
            matches = res.get("matches", [])
            # 若无匹配结果，直接跳过当前商品名的对齐
            if not matches:
                all_confirmed_item_names = False
                continue

            # 筛选 评分>0.8 和 0.6≤评分≤0.8 的匹配结果：
            high = [m for m in matches if m.get("score", 0) > 0.8]
            mid = [m for m in matches if m.get("score", 0) >= 0.6]

            # 评分>0.8 的全部放入确定项后，直接检索下一个，不存options；如果没有则标记all_confirmed_item_names，保证后续不走分支A
            if len(high)>0:
                for i in high:
                    confirmed_item_names.append(i.get("item_name"))
                continue
            else:
                all_confirmed_item_names = False

            # 0.6 ≤ 评分 ≤ 0.8 的取最高前3个作为候选
            if len(mid) > 0:
                for m in mid[:3]:
                    options.append(m.get("item_name"))

            # c：评分＜0.6 不做任何操作，此时确认、候选列表均为空

        # 返回最终结果：确认列表、全进标志和候选列表
        return {
            "confirmed_item_names": list(dict.fromkeys(confirmed_item_names)),
            "all_confirmed_item_names": all_confirmed_item_names,
            "options": list(dict.fromkeys(options))
        }




    def _step_7_check_confirmation(self, state, align_result, history):
        """
        检查step6评分对齐后的结果做条件分支

        Args:
            state
            align_result: dict - step6结果
            history: list[dict] - 近期会话历史

        Returns:
            字典 - 更新后的会话状态，item_names、answer

        """
        # 从对齐结果中提取高分真实商品名列表和中分真实商品名列表，无则空列表, 批量更新历史消息中的item_name字段
        confirmed = align_result.get("confirmed_item_names", [])
        options = align_result.get("options", [])
        all_confirmed_item_names = align_result.get("all_confirmed_item_names",False)

        # 分支A：所有涉及的商品都有高分商品名，收集mongodb历史消息中未关联商品名的消息ID，并更新item_names
        if confirmed and all_confirmed_item_names:
            ids_to_update = []
            for msg in history:
                if not msg.get("item_names"):
                    m_id = msg.get("_id")
                    if m_id:
                        ids_to_update.append(str(m_id))
            if ids_to_update:
                update_message_item_names(ids_to_update, confirmed)

            state["item_names"] = confirmed
            state["answer"] = ""
            return state


        # 分支B：有候选商品名，构造向用户确认的回复
        elif options:
            options_str = "、".join(options)

            state["answer"] = f"您是想问以下哪个产品：{options_str}？"
            state["item_names"] = []
            return state


        # 分支C：无确认商品名和候选商品名，让用户重新提问
        else:
            state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"
            state["item_names"] = []
            return state




    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        """
         把本次处理的信息（用户问题、商品名、改写查询）写入MongoDB的会话历史
         核心操作：更新用户原始问题的关联信息

         Args:
            state
            session_id: str
            rewritten_query: str - 改写后的完整问题
            message_id: str - 将用户问题保存到MongoDB数据库中生成的主键_id

         Returns:
             state
         """


        save_chat_message(
            session_id=session_id,
            role="user",
            text=state["original_query"],
            rewritten_query=rewritten_query,
            item_names=state.get("item_names", []),
            message_id=message_id
        )

        return state




if __name__ == "__main__":
    node = NodeItemNameConfirm()
    node(
        {
            'session_id':'2026-7-11',
            'original_query':'180烫金机怎么样'
         }
    )
