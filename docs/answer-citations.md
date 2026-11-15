# 答案溯源：第一版接口契约

本版实现“答案编号 → 手册/章节/检索片段”，供 Gemini 对接前端。
沿用已有 Milvus 字段，不需要重建索引或重新生成向量。没有新增 PDF 页码、
坐标映射、原文件下载接口；本地来源的 `url` 为 `null`，不可拼接猜测文件地址。

## 数据如何产生

1. 普通检索、HyDE 都读取 `chunk_id/content/item_name/file_title/title/parent_title/part`。
2. RRF、重排保留来源元数据。
3. 生成节点在 12000 字符资料预算内选取片段，并为**实际进入提示词**的片段从 1 开始编号。
   空片段和超出剩余预算的片段跳过，不为其创建引用。较小的后续片段仍可纳入。
4. 模型使用 `[1]`、`[2]` 或 `[1][2]` 标注依据。后端从本轮来源表解析编号，
   不接受模型生成的文件名、页码、URL 等作为来源元数据。
5. `citations` 仅返回答案实际引用、且存在于本轮上下文的来源，按首次引用顺序去重。
6. 历史消息保存来源与片段内容快照。后续向量库更新或删掉切片，不会抹去旧答案当时的依据。

## 普通问答

请求方式不变：`POST /query`，`is_stream: false`。
返回新增 `citations`、`citation_warnings`，其余原有字段保持兼容。

```json
{
  "message": "处理完成！",
  "session_id": "example-session",
  "answer": "连接前请先关闭设备电源。[1]",
  "done_list": [],
  "citations": [
    {
      "id": 1,
      "source": "local",
      "chunk_id": "9223372036854775800",
      "file_title": "测试设备使用手册",
      "title": "接线说明-1",
      "section": "接线说明",
      "item_name": "测试设备",
      "url": null,
      "content": "连接前请先关闭设备电源。",
      "content_kind": "processed_chunk"
    }
  ],
  "citation_warnings": []
}
```

以上为虚构接口样例，不是实际设备操作说明。

| 字段 | 含义 |
|---|---|
| `id` | 本条答案内的整数引用编号，对应 `[id]`；不能跨消息复用 |
| `source` | `local` 为知识库，`web` 为联网搜索 |
| `chunk_id` | 本地切片 ID，以字符串返回，避免 JavaScript 丢失 INT64 精度；缺失为 `null` |
| `file_title` | 本地手册名称；缺失为 `null` |
| `title` | 切片标题或网页标题；缺失为 `null` |
| `section` | 本地父章节，缺失时使用切片标题；均缺失为 `null` |
| `item_name` | 本地资料所属产品；缺失为 `null` |
| `url` | 网络来源的 HTTP(S) URL；本地或不合法地址为 `null` |
| `content` | 该次提供给模型的完整片段快照，不是模型重新撰写的摘要 |
| `content_kind` | 本地为 `processed_chunk`，网络为 `web_snippet` |

`processed_chunk` 经过文档解析，可能包含自动生成的图片描述，**不等同于 PDF 逐字原文**。
`web_snippet` 是搜索摘要，未抓取或核验网页全文。缺失元数据保留 `null`，不猜测补齐。

## 流式问答与历史记录

`POST /query` 设置 `is_stream: true` 后，继续用 `GET /stream/{session_id}` 读取 SSE。
`delta` 协议不变；引用标记可能跨多个 delta，例如 `"["`、`"1]"`。
`final` 新增与普通响应相同的 `citations`、`citation_warnings`：

```text
event: final
data: {"answer":"连接前请先关闭设备电源。[1]","status":"completed","image_urls":[],"citations":[...],"citation_warnings":[]}
```

上面的 `...` 仅表示省略来源对象。以 `final.answer` 和同一事件内的来源表为准。
普通响应、最终事件及 `GET /history/{session_id}` 的消息均采用相同的引用对象结构。
旧历史和用户消息没有来源时返回空数组，不从当前知识库反向补造历史引用。

模型或工作流失败时，普通请求返回 HTTP 502；流式发送 `error` 并结束，
不发送成功 `final`，不把部分生成的答案存为完整消息。
资料为空或全部被预算排除时，不调用生成模型，直接说明现有资料不足。

## 引用检查及前端对接要求

页面内容要求：显示答案正文和图片，保留正文中引用编号的点击溯源；
不在答案下方额外展开“参考来源”等独立来源列表。来源片段及元数据仅在点击编号时展示。
后端仍返回 `citations`，用于点击查询和历史恢复，不应为隐藏来源列表而删除该字段。

`citation_warnings` 的可能项：

```json
[
  {"code": "unknown_citation", "citation_ids": [99]},
  {"code": "no_valid_citations"}
]
```

- `unknown_citation`：答案出现本轮不存在的编号。该编号不加入 `citations`，不赋予出处；
  原始回答保留此标记以便排查。仅让 `citations` 中存在的编号关联来源。
- `no_valid_citations`：有参考内容，但答案没有任何可解析的有效引用。
  这可能是未按要求标注，也可能是正常拒答，不能仅凭这个警告判定回答错误。
- 编号检查排除代码和普通 Markdown 链接。提示词明确要求单独的 `[1][2]` 格式，
  不支持 `[1,2]`、`【1】` 或模型自建脚注。
- 引用列表不是“所有检索结果”；未被引用的候选资料不会冒充答案依据。
- 标题和片段来自文档或网络，属于不可信文本；展示时不要直接执行其中的 HTML、脚本或命令。
- 当前没有 PDF 跳转坐标。Gemini 可直接使用手册、章节、片段字段完成出处展示；
  精确页码及原文位置映射留待下一版实现。

检查仅确认**编号能对应到真实进入上下文的资料**，不自动判断每句话是否得到原文支持。
本版不增加额外模型调用来做语义审核，也不在模型漏标时自动给整段答案配上所有来源。

## 本地验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖元数据在普通检索/HyDE/RRF/重排/真实 LangGraph 执行过程中的保留、
上下文预算与编号一致性、非法编号、Markdown 排除、普通/SSE/历史接口、
来源快照、旧历史兼容、空证据及模型失败。外部模型和数据库使用替身，测试禁止网络连接。
它验证后端契约，不代表已经验证真实模型的引用准确性或外部服务可用性。

2026-10-08 验证结果：15 项离线回归通过，Python 编译与 `git diff --check` 通过。
Windows 沙箱会阻塞 asyncio 内部 socket pair，接口测试在沙箱外通过。
服务启动后，Milvus 读取、MongoDB ping、MinIO 桶检查均通过。使用真实 Milvus 片段、
真实 MongoDB 和固定本地生成函数，分别验证普通/流式引用与历史快照一致；
本次创建的两条临时测试消息已清理。

真实模型端到端验证尚未执行：自动审批因本地手册片段将发送至外部模型服务而拒绝调用。
已核实配置目的地为阿里云百炼 `dashscope.aliyuncs.com`，生成模型为 `qwen3-max`；
待用户明确授权测试问题及相关手册片段发送至该服务后再联调。上述实库验证未调用外部模型。
