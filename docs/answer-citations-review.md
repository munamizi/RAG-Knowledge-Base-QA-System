# 答案溯源源码与改动 review 清单

核对日期：2026-10-08。分支：`improve`，当前 HEAD：`723f9f7`。
以下行号对应核对时的工作区文件；后续编辑可能改变行号。

先看这三个位置，就能理解实现主线：

1. [建立编号与来源表：citation_util.py 第 36 行](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:36)
2. [要求模型标记引用：答案提示词第 39 行](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/prompt/node_answer_output.py:39)
3. [生成后筛选实际引用：citation_util.py 第 86 行](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:86)

## 1. 按运行顺序标记源码

| 顺序 | 源码入口 | 状态 | 作用及本次改动 |
|---|---|---|---|
| 1 | [node_document_split.py:78](E:/Projects/Pycharm/BrainTrustSystem/processor/import_processor/nodes/node_document_split.py:78)、[长章节切分:243](E:/Projects/Pycharm/BrainTrustSystem/processor/import_processor/nodes/node_document_split.py:243) | 原有实现，未修改 | 分块生成 `file_title/title/parent_title/part` 等元数据；本次复用这些字段 |
| 2 | [node_import_milvus.py:114](E:/Projects/Pycharm/BrainTrustSystem/processor/import_processor/nodes/node_import_milvus.py:114) | 原有实现，未修改 | Milvus 集合已有正文、手册、标题、父标题、型号、分片编号等字段；没有新增页码或改数据库结构 |
| 3 | [node_search_embedding.py:75](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_search_embedding.py:75) | 修改 | 普通检索原来只读 3 个字段，现在统一读取来源元数据 |
| 4 | [node_search_embedding_hyde.py:144](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_search_embedding_hyde.py:144) | 修改 | HyDE 检索采用相同字段集合，避免某一路缺失出处 |
| 5 | [node_rrf.py:51](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_rrf.py:51) | 原有实现，未修改 | RRF 按 `chunk_id` 融合并保留原实体字典，新增读取的元数据自然跟随 |
| 6 | [node_rerank.py:74](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_rerank.py:74) | 修改 | 本地合并改为保留完整实体字典；网络结果新增保留网页标题。重排打分及截断算法未修改 |
| 7 | [node_answer_output.py:121](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_answer_output.py:121) | 修改 | 组装提示词时，同时拿到编号来源表，写入本轮 `reference_sources` |
| 8 | [citation_util.py:36](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:36) | 新增 | 对实际纳入上下文预算的片段编号，构建元数据和正文快照 |
| 9 | [prompt/node_answer_output.py:39](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/prompt/node_answer_output.py:39) | 修改 | 提示模型使用 `[1]`、`[1][2]`，禁止编造编号和单独追加参考来源列表 |
| 10 | [node_answer_output.py:62](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_answer_output.py:62)、[citation_util.py:86](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:86) | 修改／新增 | 生成后解析标号，从本轮表中挑出实际有效引用，生成 `citations` 和警告 |
| 11 | [node_answer_output.py:66](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_answer_output.py:66)、[task_util.py:137](E:/Projects/Pycharm/BrainTrustSystem/utils/task_util.py:137) | 修改 | 将答案、引用和警告存入任务结果，供普通响应读取 |
| 12 | [query.py:81](E:/Projects/Pycharm/BrainTrustSystem/backend/api/routers/query.py:81) | 修改 | 普通 `/query` 返回新增 `citations`、`citation_warnings` |
| 13 | [node_answer_output.py:79](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_answer_output.py:79) | 修改 | SSE `final` 返回相同的引用数据；`delta` 仍然是增量文本 |
| 14 | [node_answer_output.py:321](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_answer_output.py:321)、[mongo_history_utils.py:82](E:/Projects/Pycharm/BrainTrustSystem/utils/mongo_history_utils.py:82) | 修改 | 保存引用正文与元数据快照，历史回答不再依赖当前向量库内容 |
| 15 | [query.py:149](E:/Projects/Pycharm/BrainTrustSystem/backend/api/routers/query.py:149) | 修改 | `/history` 返回引用和警告；没有这些字段的旧消息返回空数组 |
| 16 | [RAGAS/main.py:232](E:/Projects/Pycharm/BrainTrustSystem/RAGAS/main.py:232) | 修改 | 评测优先读取实际进入生成上下文的 `reference_sources`，避免自行重算预算后错报片段 |

流程图路由本身仍在 [main_graph.py:68](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/main_graph.py:68)，本次未修改。
队列与 SSE 打包沿用 [sse_util.py:38](E:/Projects/Pycharm/BrainTrustSystem/utils/sse_util.py:38) 和 [sse_generator:56](E:/Projects/Pycharm/BrainTrustSystem/utils/sse_util.py:56)。

## 2. 新增核心文件：每个函数的 review 位置

[utils/citation_util.py](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py) 是本次唯一新增的生产源码文件。

| 位置 | 内容 | 重点看什么 |
|---|---|---|
| [第 8 行](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:8) | `RETRIEVAL_OUTPUT_FIELDS` | 两路检索都需要读出哪些字段 |
| [第 12 行](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:12) | 引用及 Markdown 正则 | 支持 `[1][2]`，排除代码、普通链接和链接定义 |
| [第 21 行](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:21) | `_text()` | 缺失值不会变成字符串 `None` |
| [第 25 行](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:25) | `_web_url()` | 网络引用只允许 HTTP(S)，本地来源不猜测文件地址 |
| [第 36 行](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:36) | `build_reference_context()` | 上下文与对应表同时建立；预算排除的片段没有引用条目 |
| [第 51 行](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:51) | 来源对象字段 | `id/source/chunk_id/file_title/title/section/item_name/url/content/content_kind` |
| [第 70 行](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:70) | 字符预算与纳入判断 | 只给实际纳入提示词的片段编号；超大片段跳过后继续尝试较小片段 |
| [第 80 行](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:80) | `citation_ids()` | 提取答案中的编号，按首次出现顺序去重 |
| [第 86 行](E:/Projects/Pycharm/BrainTrustSystem/utils/citation_util.py:86) | `resolve_citations()` | 只返回本轮存在且实际被引用的来源；不存在的编号产生警告 |

`chunk_id` 输出为字符串，避免浏览器处理 Milvus INT64 时丢失精度。
`section` 优先父标题，再用切片标题；未保存可靠 PDF 页码时不生成页码。

## 3. 三个状态字段的区别

定义位置：[QueryGraphState 第 25 行](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/state.py:25)。

| 字段 | 何时生成 | 含义 |
|---|---|---|
| `reference_sources` | 重排后、调用生成模型前 | 本轮实际送入提示词的所有编号片段，后端内部使用 |
| `citations` | 答案生成后 | 从上表挑出的、答案实际有效引用的记录，返回前端 |
| `citation_warnings` | 答案生成后 | 不存在的编号，或有参考内容但没有可解析的有效引用 |

编号检查不判断“某句话是否确实受到片段支持”。模型引用由提示词要求，可能漏标或标错。
非法编号保留在原回答中便于排查，但不会被赋予虚构出处。

## 4. 必须一起 review 的附带行为变化

这些是同一次实现中的额外行为调整，不能仅看 `citations` 字段而漏掉。

| 位置 | 修改后的行为 |
|---|---|
| [node_answer_output.py:45](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_answer_output.py:45) | 每轮重新初始化引用数据，避免沿用旧请求来源 |
| [node_answer_output.py:56](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_answer_output.py:56) | 没有实际进入上下文的片段时，直接说明资料不足，不调用生成模型；已有澄清答案继续走原来的直接输出分支 |
| [node_answer_output.py:71](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_answer_output.py:71) | 图片改从实际进入生成上下文的片段提取，避免展示被预算排除的资料图片 |
| [node_answer_output.py:160](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_answer_output.py:160) | `_format_reranked_docs()` 的返回值从两个变为三个：上下文、来源表、剩余预算 |
| [node_answer_output.py:222](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_answer_output.py:222)、[非流式:237](E:/Projects/Pycharm/BrainTrustSystem/processor/query_processor/nodes/node_answer_output.py:237) | 生成异常及空答案向上抛出；失败的部分答案不存为完整回答，不发送成功 `final` |
| [query.py:73](E:/Projects/Pycharm/BrainTrustSystem/backend/api/routers/query.py:73) | 非流式同步工作流改在线程执行，避免直接阻塞异步接口；任务失败返回 HTTP 502 |
| [query.py:106](E:/Projects/Pycharm/BrainTrustSystem/backend/api/routers/query.py:106) | 流式工作流结束后发送队列关闭信号，SSE 可结束读取 |
| [task_util.py:137](E:/Projects/Pycharm/BrainTrustSystem/utils/task_util.py:137) | 任务结果读写函数的类型标注从 `str` 改为 `Any`，以容纳引用对象列表 |
| [RAGAS/main.py:237](E:/Projects/Pycharm/BrainTrustSystem/RAGAS/main.py:237) | 新状态直接使用真实来源快照；没有新字段的旧工作流结果保留原兼容分支 |

## 5. 前端接入位置：Gemini 负责

前端集中在 [frontend/chat.html](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html)。
当前文件包含 Gemini 的前端改动；这里仅给源码定位，不评审其布局、样式和交互设计。

| 入口 | 用于定位的内容 |
|---|---|
| [第 583 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:583) | 引用编号、弹层相关样式区域 |
| [第 1661 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:1661) | `cleanCitationSection()` |
| [第 1678 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:1678) | `renderCitationMarkdown()` |
| [第 1792 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:1792) | `ensureCitationPopover()` |
| [第 1911 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:1911) | `positionCitationPopover()` |
| [第 1959 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:1959) | `closeCitationPopover()` |
| [第 1974 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:1974) | `toggleCitationPopover()` |
| [第 2083 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:2083) | `createCitationBadge()` 及点击事件入口 |
| [第 2104 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:2104) | `renderTextWithCitations()` |
| [第 2158 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:2158) | `renderAnswerWithImages(..., citations)` |
| [第 2330 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:2330) | `addBotMsgWithTime(..., citations)` |
| [第 2822 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:2822) | `finalizeBotAnswer(..., citations)` |
| [第 2864 行](E:/Projects/Pycharm/BrainTrustSystem/frontend/chat.html:2864) | `loadHistory()` |

前端行号仅是当前快照；Gemini 若仍在修改该文件，按函数名搜索更可靠。
后端与前端共有的数据契约见 [answer-citations.md](E:/Projects/Pycharm/BrainTrustSystem/docs/answer-citations.md)。

## 6. 测试与验证代码

[tests/test_citations.py](E:/Projects/Pycharm/BrainTrustSystem/tests/test_citations.py) 为本次新增测试文件。

| 入口 | 覆盖内容 |
|---|---|
| [CitationTests:33](E:/Projects/Pycharm/BrainTrustSystem/tests/test_citations.py:33) | 元数据、INT64、预算、编号去重、非法编号、代码/链接排除、URL 过滤 |
| [PipelineTests:89](E:/Projects/Pycharm/BrainTrustSystem/tests/test_citations.py:89) | 真实节点、真实 LangGraph 和 API，外部服务用替身 |
| [第 173 行](E:/Projects/Pycharm/BrainTrustSystem/tests/test_citations.py:173) | 来源内容快照和 MongoDB 保存契约 |
| [第 183 行](E:/Projects/Pycharm/BrainTrustSystem/tests/test_citations.py:183) | 普通输出与流式最终事件的引用一致 |
| [第 198 行](E:/Projects/Pycharm/BrainTrustSystem/tests/test_citations.py:198) | 空证据、已有澄清答案不调用模型 |
| [第 206 行](E:/Projects/Pycharm/BrainTrustSystem/tests/test_citations.py:206) | 空模型结果不能保存为成功 |
| [第 214 行](E:/Projects/Pycharm/BrainTrustSystem/tests/test_citations.py:214) | 被预算排除的文档不能通过引用号重新出现 |
| [第 243 行](E:/Projects/Pycharm/BrainTrustSystem/tests/test_citations.py:243) | 两路检索 → RRF → 重排 → 生成保留出处 |
| [第 255 行](E:/Projects/Pycharm/BrainTrustSystem/tests/test_citations.py:255) | `/query`、SSE、历史消息、旧历史兼容 |
| [第 275 行](E:/Projects/Pycharm/BrainTrustSystem/tests/test_citations.py:275) | 模型失败返回错误、没有成功 final 或完整历史回答 |

已有验证：15 项离线测试、Python 编译和 diff 格式检查通过；服务启动后使用真实
Milvus/MongoDB 与本地固定生成函数，核对了普通和流式出处及历史快照。真实外部模型
联调因审批等待具体授权，未计入已完成验证。本次整理清单没有重新跑模型或数据库。

## 7. 完整改动快照及范围

[后端完整 diff 快照](E:/Projects/Pycharm/BrainTrustSystem/docs/answer-citations-backend.diff)
收录上述 11 个生产/评测源码文件、测试文件和接口文档的改动。
它用于阅读，不是要求在已有改动上再次应用补丁。新文件的完整源码也收录其中。

[前端完整 diff 快照](E:/Projects/Pycharm/BrainTrustSystem/docs/answer-citations-frontend.diff)
收录 `frontend/chat.html` 当前全部未提交改动，供你查看；它可能包含 Gemini 的其他页面改动，
不应将每一行都认定为溯源专属改动。前端实现由 Gemini 负责解释。

本次开始前就存在的两处改动不属于溯源实现：

- `backend/api/main.py` 删除“创建 FastAPI 实例”的注释。
- `utils/mongo_history_utils.py` 的 `get_recent_messages()` 默认条数从 10 改为 20。

后端 diff 快照排除了这两处已有改动与 Gemini 的前端改动。
Git 文件结尾换行等纯格式变化不代表新增功能。
