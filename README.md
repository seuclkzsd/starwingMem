# starwingMem

starwingMem is a long-term memory service for the Agent Memory Challenge 2026 (Cycle 2, Open-source Methods division), entered in the Textual, Coding and Multimodal tracks. It implements the AML Add / Search contract: Add stores what the platform sends, and Search returns verbatim history to the platform's fixed answer model, with dates resolved. **gpt-4o-mini: N/A.** No generative model is called at Add or Search time in any version (v1.0–v1.3). The only model is `text-embedding-v4`, on the textual and multimodal paths; the v1.1 coding path uses BM25 and rules only. Images are stored and returned as they are and are never interpreted.

starwingMem 是为第二届 Agent 记忆挑战赛（开源方法榜）实现的记忆服务，参加文本、代码、多模态三个赛道。它只负责 Add / Search：原样保存平台写入的内容，检索时返回带日期标注的原始历史片段，最终答案由平台统一生成。

本文描述 v1.3。各版本的差异见[版本记录](#版本记录)。

## 接口

| 接口 | 说明 |
|---|---|
| `POST /add` | 同步写入，返回 `{"success": true, "request_id", "user_id", "session_id"}`，返回后即可检索。文本和多模态路径写入 SQLite 并完成向量化后才返回；代码路径写入 SQLite 并完成分词后返回，不向量化。同一 `user_id` 下 `request_id` 重复时，按幂等处理。 |
| `POST /search` | 只在请求的 `user_id` 范围内检索。返回 `{"data": [{"id", "content", "score", "created_at"}]}`，最多 `top_k` 条，按相关性从高到低排序。`content` 是字符串或 ContentPart[]，规则见下一节。内容是平台写入的原文或逐字摘录，加上规则生成的结构文字（日期头、相对日期标注、摘要标题、卡片标题、图片标签），不包含生成的答案。 |
| `GET /health` | 无需鉴权，返回 `{"status": "ok"}`。 |

鉴权支持 `Authorization: Token <key>`、`Authorization: Bearer <key>` 或 `X-Api-Key: <key>`，key 来自环境变量 `AML_TOKEN`。

## 赛道识别与各赛道配置

三个赛道共用 `/add` 和 `/search`，因为平台规定同一个 Key 的接口地址不可更改。平台发来的 id 都是匿名哈希，所以按 Add 的内容形态区分。规则按下面的顺序判断，都写在 `service/server.py`、`service/mmmem.py` 和 `service/codemem.py` 里。识别只看内容的结构和标记，不看数据集名称或题目。

1. **多模态路径**：某次 Add 的 `content` 数组里出现 `type=image_url` 的 part。从这次 Add 起，这个用户走 `mmmem.MMStore`，之前写入的消息一起带过去，以后不再改变。
2. **代码路径**：用户第一次带消息的 Add 被判定为 agent 轨迹（`codemem.is_trajectory_add`），这个用户以后走 `codemem.CodeStore`。判定要同时满足三条：
   - 消息数 ≥ 2，并且消息数 > 20 或总词数 > 2,000。普通文本 Add 按 20 条消息或 2,000 词切段，到不了这个规模。
   - 这次 Add 不带图片。
   - 含轨迹标记的消息数 ≥ max(5, 消息数的 20%)。标记包括工具调用、diff、traceback、shell 命令、测试汇总、`"file_path":` 这类字段（`codemem.TRAJ_RE`）。

   第一次 Add 没有判为轨迹的用户，以后不再判定。代码用户如果后来发了图片，转到多模态路径，已写入的内容一起迁移。
3. **文本路径**：其余所有用户，走 `memsys.MemoryStore`，和 v1.0.3 是同一条路径。其中 Add 里出现过 `content` 数组但没有图片的用户，Search 的 `content` 是只含 text 的 ContentPart[]；其他用户是字符串。

| | 文本 | 多模态 | 代码 |
|---|---|---|---|
| 代码 | `memsys.MemoryStore` | `mmmem.MMStore` | `codemem.CodeStore` |
| 配置 | `server.DEFAULT_CFG` | `server.MM_CFG`、`mmmem.MM_DEFAULTS` | `codemem.CodeCfg` |
| 模型 | `text-embedding-v4` | `text-embedding-v4` | 无 |
| 检索 | dense + sparse + BM25 三路 RRF，加会话级 RRF | 同文本；BM25 对 CJK 用字二元组；query 图片精确匹配 | 代码感知 BM25，片段和关键文档两路 RRF |
| 返回预算 | 16k token | 16k token 文本，加最多 12 张原图 | 最多 5 张卡片，共 8k token |
| 日期头 | 问题未写当前时间时用自然写法 | 保留到分钟 | 只写日期 |
| 内存上限（LRU） | 索引 ≤ 20 万 passage | ≤ 10 万 passage | ≤ 128 个用户，≤ 6,000 万 BM25 token |
| 持久化 | `mem.sqlite` | `mem.sqlite`、`blobs/` | `mem_coding.sqlite` |
| 重启 | 从 SQLite 回放消息，索引按需重建 | 不回放，Search 时懒加载 | 不回放，Search 时懒加载 |

部署配置全部写在代码默认值里。`AML_CFG`、`AML_CFG_MM`、`AML_MM`、`AML_CFG_CODING`、`AML_CODE_DETECT` 这些环境变量只用于测试，部署时不设置。

## 方法

### 文本路径

1. **写入**：每条消息原样保存。超过 350 个词的消息切成段，每段 300 词、相邻两段重叠 50 词。每段向量化时同时取 `text-embedding-v4` 的 dense 和 sparse 两种向量，并建立 BM25 索引（bm25s，带英文词干化）。向量缓存在本地 SQLite 中，重启时回放已写入的数据，不会重复计费。多个请求同时写入相同文本时，只有第一个调用接口，其余的等它的结果；它失败时，这些请求一起返回 500，由平台重试。
2. **检索排序**：
   - dense、sparse、BM25 三路各自给全部消息排名（一条消息的得分取它各段的最高分），用加权 RRF 融合：`fused(m) = Σ w / (60 + rank)`，权重 dense 1、sparse 0.5、BM25 1，`rank` 从 1 开始。
   - 会话级 RRF 项：会话得分 = 会话内融合分最高的 5 条消息之和。按会话得分给会话排名，会话内每条消息再加 `0.5 / (60 + 会话排名)`。
3. **时间处理**：
   - 如果问题里写明了"当前时间"（如 "Now is …"），检索前先去掉这句；时间晚于这个日期的消息降权（`future_penalty=3`，分数乘 1/4）。
   - 问题里提到时间范围的，范围内的消息加权（`time_boost=1`，分数乘 2）。
   - 返回的文本里，"yesterday""last Tuesday"等相对时间表达后面会追加按消息时间戳换算出的绝对日期，例如 `yesterday [i.e. 21 February 2023]`。这一步是确定性规则，见 `service/timeannot.py`。
4. **打包**：按排序取出命中的消息，相邻的合并成窗口，最多 100 个窗口。预算没用完时，再向同一会话前后逐轮扩展相邻消息，直到装满 16k token（`tiktoken` o200k_base）。每个窗口以消息日期开头。问题没有写明当前时间时，日期用自然写法，如 "16 June 2023 (Friday)"。
5. **按问题类型路由**：建议类、偏好类问题，先返回一条"最相关的 20 条原话"摘要，再接完整窗口。分类用规则完成（`memsys.qtype`）。
6. **Streaming**：平台回写的 `Streaming online QA feedback:` 消息只存储，不向量化，也不进入检索索引。
7. **选择题**：`options` 只作为额外的 BM25 检索词，不当作事实写入记忆。
8. **索引淘汰（v1.1）**：全部用户的索引合计超过 20 万 passage 时，淘汰最久未使用的空闲用户的索引，消息保留。下次 Search 从向量缓存重建，不花钱，结果不变。

### 多模态路径（v1.1，v1.2 加窗口标签）

1. **写入**：Add 解码 base64 后，把原图字节按 sha256 存成文件（`blobs/<sha 前 2 位>/<sha>`），相同图片只存一份。SQLite 里只存引用（sha256、MIME、字节数、宽高），不存 base64。只接受 JPEG、PNG、WebP，单张解码后不超过 10 MiB，格式看文件头，不看声明的 MIME；其他图片整个 Add 返回 422。
2. **索引文本**：text part 按顺序拼接，每张图原位替换成占位符 `[Image <sha 前 8 位>]`。所以检索看到的是平台给的 caption 和图片周围的对话。
3. **检索**：和文本路径相同的三路 RRF 加会话级 RRF，参数相同。有三处不同：日期头保留到分钟；BM25 把连续的中日韩字符切成字二元组（只在这条路径生效）；query 带图片时做精确匹配，见[多模态处理声明](#多模态处理声明)。
4. **返回**：ContentPart[]。文本窗口在前，每个窗口包在 `<past_conversation index="k">…</past_conversation>` 标签里（v1.2），标签由规则生成，k 从 1 开始，窗口内文逐字不变。然后是这些窗口里的原图，每张图一条，排在全部文本之后，这样平台截断 Answer 输入时先丢图片。每条图片前有一个规则生成的标签，如 `[Image 2658faf8] sent by the user on 2023-05-08 (Mon) 13:56`。上限：12 张；base64 合计 24 MiB；文本加图片按 gpt-4o-mini 的图片计价规则估算不超过 108k token；文本窗口最多 88 个，图片和文本合计不超过 `top_k`，所以 `top_k=100` 时文本不会挤掉图片。`top_k` 不超过 88 时，文本可能占满全部名额，这次就不返回图片。
5. **内存**：图片用户启动时不回放，第一次 Search 时从 SQLite 加载，按 passage 数做 LRU，超过 10 万时卸载最久未搜索的用户。卸载只清内存，SQLite、图片文件和向量缓存都保留。

### 代码路径（v1.1）

只用 BM25 和规则，Add 和 Search 都不调用 embedding 或生成模型。

1. **写入**：Add 原样存入独立的 SQLite 文件，同时分词。用户的 Add 停下 0.5 秒后，后台线程预建索引；重启不回放，Search 时懒加载。
2. **分词**（`codemem.code_tokens`）：路径生成 `path:`、`file:` 精确 token；标识符保留原形，同时按 snake_case 和 camelCase 拆词；限定名（`a.b`、`a::b`）保留整体和各段；英文词干化；中文用字二元组。
3. **排序**：检索单位是一条历史轨迹（`session_id`；一个用户只有一个 `session_id` 时，每次 Add 算一条）。两路 BM25 用 RRF 融合（权重 1 / 1，k=60）：
   - 片段路：消息按行切成不超过 2,000 字符的片段，轨迹得分 = 得分最高的 5 个片段之和。
   - 关键文档路：每条轨迹一份关键文档，由首条 user 消息、最后的 assistant 总结和最多 8 段编辑摘录组成。
4. **过滤**：第一张卡片总是保留，其余卡片的关键文档 BM25 分要达到最高分的 0.5（`rel_cut`）。
5. **返回**：每条轨迹一张"历史任务卡片"，最多 5 张，共 8k token。卡片依次是任务描述（≤ 600 token）、最新的编辑和 diff（共 ≤ 800 token）、最匹配的片段（≤ 300 token）、最后一次测试输出（≤ 120 token）、最后的 assistant 总结（≤ 400 token）。内容是逐字摘录，只有卡片标题和小标题由规则生成。标题写明这是历史轨迹的摘录，做法不一定适用于当前任务。
6. **容错**：Search 内部出错时，这次返回空结果（HTTP 200），不让记忆故障打断 agent 解题。`query` 缺失或为空时也一样，因为平台对 4xx 不重试，这道任务会直接失败。这和文本、多模态路径的 422 不同，见 `tests/test_coding_v11.py`。解析不了的 Add 也返回 200，原始内容保留在 SQLite 里。

### 主要取舍（本地评测）

除非表中注明，本地评测用 Qwen3-14B 代替平台的回答模型和评分模型，数据都是公开数据，见[离线用过的数据和模型](#离线用过的数据和模型)。

| 改动 | 结果 | 是否采用 |
|---|---|---|
| 未写明当前时间的问题，日期用自然写法 | LoCoMo 500 题 +2.4（显著），LongMemEval 不变 | 采用 |
| 建议、偏好类问题先给 top-20 原话摘要 | BEAM +1.5（显著），LongMemEval +2.2，LoCoMo +0.2 | 采用 |
| Streaming 反馈消息存入检索 | 原样存入：LoCoMo −3.8（显著）；只存答对的：−2.8 | 不存入索引 |
| 加重排模型 | 证据覆盖率 +1.6，端到端持平 | 不用 |
| 返回结果按时间排序 | LongMemEval −6.1 | 不用 |
| 多模态文本预算 8k / 16k / 24k（MobileMem + MemLens 300 题） | 24k 比 16k 低 0.045（95% CI [−0.090, −0.001]），8k 与 16k 无显著差异 | 用 16k |
| 多模态 BM25 用 CJK 字二元组（只用 BM25 时的代理指标） | 中文用户"答案出现在返回文本里"的比例 0.057 → 0.272，英文文本不受影响 | 采用 |
| 代码路径用卡片和代码感知分词（SWEContextBench Lite，99 个查询、300 条轨迹，只评检索） | 第 1 张卡片命中金标准 0.626，v1.0 的 BM25 通道 0.566 | 采用 |
| 多模态文本窗口加 `<past_conversation>` 标签（回答模型 gpt-4o-mini，只看文本） | MemLens 36 题用其官方提示词答对 6 → 9，MobileMem 80 题 8 → 11（5 题变好、2 题变差）。样本小，都不显著，平台实际提示词未知 | 采用（v1.2） |
| Search 时用 gpt-4o-mini 筛选证据（文本 239 题，多模态 80 题） | 只重排时文本 +3.3 点，95% CI [+0.1, +6.5]，多重比较校正后不显著；多模态不显著 | 不用 |
| Search 时用 gpt-4o-mini 判断候选里有没有答案，没有就返回空 | MobileMem 拒答题 0/60 → 22/60，但 MemLens、MemEye 72%、64% 的题被判成没有 | 不用 |
| gpt-4o-mini 重排代码卡片（99 个查询） | 第 1 张命中 0.626 → 0.646，2 题变好、0 题变差 | 不用 |
| 文本返回 32k、64k（回答模型 gpt-4o-mini，LoCoMo、LME、BEAM 共 239 / 118 题） | 相对 16k：32k +0.1 点，64k +0.3 点，都不显著 | 用 16k |
| 文本窗口也加 `<past_conversation>` 标签（同上 239 题） | +3.8 点，95% CI [0.0, +7.4]，BEAM +7.5、LME +3.7、LoCoMo 0；校正 3 组比较后下界 −0.7，Streaming 和另外 4 个数据集没有测 | 采用（v1.3） |
| 多模态带图和不带图（MemLens 36 题，官方提示词加标签） | 答对 9 → 12，5 题变好、2 题变差，不显著 | 保留返回原图 |

## 模型使用（学术榜要求）

本节适用于 v1.0–v1.3。

| 组件 | 使用情况 |
|---|---|
| LLM（gpt-4o-mini） | **N/A**。Add/Search 全链路不调用生成式模型，适用 v1.0–v1.3。 |
| Reranker | **N/A**。不使用。 |
| Embedding | `text-embedding-v4`，用于文本和多模态路径。v1.0–v1.0.3 没有代码路径，代码赛道的请求走文本路径，也用它。 |
| 词法检索 | BM25：`bm25s` + `PyStemmer`。代码路径用同一个库，分词规则是自己写的。 |
| 规则模块 | `timeannot`（相对时间标注）、`qtype`（问题类型）；v1.1 另有 `is_trajectory_add`（赛道识别）。 |
| 图片 | 不调用视觉、OCR 或 caption 模型。 |
| v1.1 代码赛道 | 只用 BM25 和规则，不调用任何 embedding 或生成模型。 |

`text-embedding-v4` 的调用方式：

- 走 DashScope 北京地域的原生接口：`https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding`。复现时需要北京地域的 Key。
- 参数：`dimension=1024`，`output_type=dense&sparse`。同一次调用同时取 dense 和 sparse 向量。
- `text_type` 分 `query` 和 `document`。query 端的 `instruct` 原文是 `Given a question about a past conversation, retrieve the messages that help answer it`，document 端不带 instruct。
- 结果缓存在本地 SQLite，缓存键由模型、维度、`text_type` 和文本决定。

`service/dashscope.py` 里的 `chat`、`rerank` 和 `memsys.Config` 里的 `llm_model`、`rerank_top`、`w_hyde`、`w_plan` 是离线消融留下的开关。部署配置不设置它们，运行时走不到这些代码，多模态和代码路径也一样。

## 多模态处理声明

| 项目 | v1.0、v1.0.1 | v1.0.3 | v1.1–v1.3 |
|---|---|---|---|
| 方式 | caption-only | caption-only | 检索只用文本和 caption，返回原图 |
| 是否处理原图 | 否。含 base64 的原始请求随 Add 存入 SQLite，只为幂等和重启回放，不解码、不分析 | 同左 | 不解析内容。存原图，只读文件头取 MIME 和宽高，用来估算图片 token 和控制返回量，不解码像素 |
| 是否使用 OCR | 否 | 否 | 否 |
| 是否使用视觉或 caption 模型 | 否 | 否 | 否 |
| 是否只使用 caption | 是。只用 `type=text` 的 part（含平台提供的 caption）建索引和检索 | 同左 | 是。索引文本是 text part 加图片占位符，见上文 |
| 多模态 embedding | 不使用 | 不使用 | 不使用 |
| query 里的图片 | 忽略 | 忽略 | 不做视觉检索。图片用户的 query 图片只算 sha256，如果和该用户发过的图片完全相同，就把对应消息排到最前；不保存。无图片用户走文本路径，query 图片忽略 |
| 是否返回图片证据 | 否。`content` 是字符串 | 否。`content` 是只含 text 的 ContentPart[] | 是。命中窗口里的原图随结果原样返回（base64 data URI，不缩放、不重新编码），上限见上文 |

Multimodal declaration in English. v1.0 and v1.0.1 are caption-only and return strings. v1.0.3 is caption-only and returns ContentPart[] with text parts only. In these versions images are not processed: no OCR, no vision or caption model; only text parts, including the platform's captions, are indexed and searched; query images are ignored; no image evidence is returned; the raw request, base64 included, stays in SQLite for idempotency and replay only. v1.1 and v1.2 store the original images and returns the images of the retrieved windows unchanged with the Search results. It does not interpret images: no OCR, no vision or caption model, no multimodal embedding, and pixels are never decoded (only file headers are read, for MIME type and size). Retrieval uses text and captions only. A query image is used only for an exact SHA-256 match against that user's own images and is not stored.

## 原始工作与改动披露

本节适用于 v1.0–v1.3。已绑定的 commit 里的 README 没有这些内容，以本节为准。

下面每个来源一张表，字段相同。所有来源都没有复制代码，方法由本团队用自己的代码实现。文本路径的会话级 RRF 和 v1.1 代码路径的两个参考作品实际部署；其余作品只以默认关闭的开关存在于 `memsys.Config`，部署关闭，未参与评测。

### ReFind（文本和多模态路径实际部署）

| 字段 | 内容 |
|---|---|
| 原作者 | Ruizhe Li, Licheng Zhang, Benfeng Xu, Mingxuan Du, Zheren Fu, Weidong Chen（按 arXiv:2608.12888 的 BibTeX）。ReFind 仓库的 `CITATION.cff` 和 `docs/METHOD_CARD.md` 列的是另一组作者：Ruizhe Li, Benfeng Xu, Mingxuan Du, Licheng Zhang, Zhendong Mao，对应标题 *When Your Agent Opens the Chat App: A Human-Inspired Search Interface for Conversational Memory*。 |
| 技术报告 | *When Your Agent Opens the Chat App: Agent-Controlled Search over Raw Chat Logs Rivals Structured Memory*, arXiv:2608.12888 (cs.CL), 2026。https://arxiv.org/abs/2608.12888 |
| 原仓库与许可证 | https://github.com/imlrz/ReFind ，MIT。我们读的版本是 `a80175c`。 |
| 保留的部分 | 把会话级排名作为一项 RRF 分数加给消息，k=60（`w_session`）。 |
| 改动 | 1. 没有检索 Agent：不调用 gpt-4o-mini，Search 一次完成，没有多轮搜索，没有记笔记动作，没有已读过滤，也没有由模型决定的日期过滤。<br>2. 用三路融合分，不是只用 BM25：会话排名的输入是 dense + sparse + BM25 的 RRF 分。ReFind 的输入是 BM25 分。<br>3. 计分方式：会话得分是融合分最高的 5 条消息之和。ReFind 是全部命中块（BM25 分 > 0）的 BM25 分之和。<br>4. 权重：会话项是 `0.5 / (60 + 排名)`，加在三路融合分上。ReFind 是两项等权的 `1 / (60 + 排名)`，一项块排名、一项会话排名。<br>5. 排名范围：全部消息和全部会话都参与。ReFind 只排 BM25 命中的块。<br>6. 检索单位：消息（长消息切段）。ReFind 是相邻轮配对成的对话记录。<br>7. 上下文：按 16k token 预算填充窗口。ReFind 固定扩展前后各 2 块，内部每次搜索看 5 条结果。 |
| 部署中是否启用 | 是，`w_session=0.5`，文本和多模态路径。代码路径不用这段代码，但轨迹级聚合用了相近的结构：轨迹得分是最高 5 个片段的 BM25 分之和，再和关键文档路做 RRF。 |
| 是否复制代码 | 否。 |

### deja-vu（代码路径实际部署）

| 字段 | 内容 |
|---|---|
| 原作者 | Vladislav Shulcz（GitHub `vshulcz`）。 |
| 技术报告 | 无。 |
| 原仓库与许可证 | https://github.com/vshulcz/deja-vu （`scripts/amlserve`），MIT。 |
| 保留的部分 | 思路：按会话打分，给命中的会话返回任务、结尾、代码改动这样的紧凑摘要，不堆原文。 |
| 改动 | 1. 排序是片段和关键文档两路 BM25 的 RRF，分词规则是自己写的。<br>2. 卡片内容改成任务描述、最新编辑、最匹配片段、最后测试输出、最后总结，全部是逐字摘录。<br>3. 预算是 5 张卡片、8k token。原实现是前 2 个会话的详情加其余任务的简介，总上限 412,160 字符。<br>4. 增加 `rel_cut` 过滤。<br>5. 没有采用同一文件多次改动的折叠。 |
| 部署中是否启用 | 是，v1.1 代码路径。 |
| 是否复制代码 | 否。原实现是 Go，我们是 Python。 |

### Memoria（代码路径实际部署）

| 字段 | 内容 |
|---|---|
| 原作者 | GitHub 账号 `lin-guan`。 |
| 技术报告 | 无。 |
| 原仓库与许可证 | https://github.com/lin-guan/aml-rag-v2 ，无 LICENSE 文件（我们读的版本是 `1aa0e2c`）。 |
| 保留的部分 | 思路：把 `::`、点号限定名、路径、snake_case、camelCase 当作代码 token。 |
| 改动 | 1. 规则是自己写的。<br>2. 代码 token 直接进 BM25 词表。原作品是在分数上加分（每个共享 token +0.08，上限 0.24）。<br>3. 路径生成 `path:`、`file:` 精确 token。 |
| 部署中是否启用 | 是，v1.1 代码路径。 |
| 是否复制代码 | 否。仓库没有许可证，只借思路。 |

### InvMem（部署关闭，未参与评测）

| 字段 | 内容 |
|---|---|
| 原作者 | 忆构智能团队（房文晓、武志妍、李星璇、刘昱浩，按 AML 官方解读文章），GitHub 账号 `wenxiaof345-ctrl`。 |
| 技术报告 | 无。 |
| 原仓库与许可证 | https://github.com/wenxiaof345-ctrl/vanilla-rag-memory ，无 LICENSE 文件（本地克隆 `31ab7bf` 和 2026-10-31 的 GitHub 页面都没有）。 |
| 保留的部分 | 思路：只给排名靠前的种子扩展邻居窗口。 |
| 改动 | 1. 实现成开关 `seed_k`：前 `seed_k` 个种子用 ±`win` 窗口，其余种子不扩展。<br>2. 种子来自我们的三路融合排序。原作品是 Dense + BM25 的 RRF。<br>3. 窗口进入我们的打包流程，按 token 预算合并。 |
| 部署中是否启用 | 否，部署关闭，未参与评测。本地 `seed_k=20, win=1` 不如预算填充，未采用。 |
| 是否复制代码 | 否。仓库没有许可证，只借思路。 |

### Chronicle（部署关闭，未参与评测）

| 字段 | 内容 |
|---|---|
| 原作者 | GitHub 账号 `simple-boy`（LICENSE 署名 Chronicle-Memory contributors）。 |
| 技术报告 | 无。 |
| 原仓库与许可证 | https://github.com/simple-boy/Chronicle-Memory ，MIT。上游在 2026-10-09 加入 LICENSE（commit `8b0d660`）；我们当时读的本地克隆 `26bd3ac`（10-01）没有 LICENSE 文件。 |
| 保留的部分 | Chronicle V1 的思路：不让一个会话占满全部结果（`memory_core.py` 的 "Mild diversity"，每会话 5 条）。V2 已去掉这一配额（README："without a per-session quota"）。 |
| 改动 | 1. 实现成开关 `session_cap`，上限可配。<br>2. 作用在排序靠前的 400 个种子上，超额的种子排到后面。原实现作用在最终 top-k 上。 |
| 部署中是否启用 | 否，部署关闭，未参与评测。本地 `session_cap=3` 明显变差，未采用。 |
| 是否复制代码 | 否。 |

### Hybrid Episodic Memory（HEM，部署关闭，未参与评测）

| 字段 | 内容 |
|---|---|
| 原作者 | GitHub 账号 `tlysanhuo`（LICENSE 署名 Agent Memory Challenge submission authors）。 |
| 技术报告 | 无。 |
| 原仓库与许可证 | https://github.com/tlysanhuo/agent-memory-challenge ，MIT。这个地址在 2026-10-31 已返回 404，以我们的本地克隆 `9db3399` 为准。 |
| 保留的部分 | 思路：按新近度温和调整融合分（`apply_recency`）。 |
| 改动 | 1. 实现成开关 `recency`。<br>2. 改成乘法：`fused *= 1 + recency × 相对时间`。原实现是加法：`fused += weight × best × 相对时间`。<br>3. 时间范围取全部消息。原实现取候选里带时间戳的消息。 |
| 部署中是否启用 | 否，部署关闭，未参与评测。本地 `recency=0.15` 没有作用，未采用。 |
| 是否复制代码 | 否。 |

### ActiveMemoryIndex（含 HyDE，部署关闭，未参与评测）

| 字段 | 内容 |
|---|---|
| 原作者 | ActiveMemoryIndex：Xuhao Lin。HyDE：Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan。 |
| 技术报告 | Xuhao Lin, *An Index, Not a Store: The Model Does Remember — It Just Needs Its Notebook*, 2026, doi:10.5281/zenodo.21405963。Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, 2022, arXiv:2212.10496。 |
| 原仓库与许可证 | https://github.com/linxuhao/ActiveMemoryIndex ，MIT。 |
| 保留的部分 | 思路：用 LLM 改写问题，多加一路 dense 检索。 |
| 改动 | 1. 实现成开关 `w_hyde`，需要同时设置 `llm_model`。<br>2. 提示词生成第一人称陈述句。ActiveMemoryIndex 生成第一人称回忆问句，HyDE 原文生成假设答案。<br>3. 改写结果作为独立的一路进 RRF。ActiveMemoryIndex 是相似度加权：`(1-w)·sim(query) + w·sim(recall)`。 |
| 部署中是否启用 | 否，部署关闭，未参与评测。 |
| 是否复制代码 | 否。 |

### ChronoHybridMem（部署关闭，未参与评测）

| 字段 | 内容 |
|---|---|
| 原作者 | Haoxuan Meng（GitHub `Tin11Mn`）。 |
| 技术报告 | 无（其 README 写明未宣称存在正式论文）。 |
| 原仓库与许可证 | https://github.com/Tin11Mn/chrono-hybrid-mem ，MIT。 |
| 保留的部分 | 思路：用一次 LLM 规划调用生成多个检索词。 |
| 改动 | 1. 实现成开关 `w_plan`，需要同时设置 `llm_model`。<br>2. 一次调用返回 2–4 条子查询和最多 8 个关键词。子查询走 dense，关键词走 BM25，都作为额外的 RRF 项。<br>3. 没有采用它的结构化字段（意图、实体、时间线索、证据需求）和证据图。 |
| 部署中是否启用 | 否，部署关闭，未参与评测。 |
| 是否复制代码 | 否。 |

### 本团队实现的部分

下面的代码和具体规则由本团队编写。这里说的是实现，不是说这些思路是首创；已知的同类做法写在对应条目里。

- 证据窗口打包：按排名放入命中消息、合并相邻窗口、按 16k token 预算填充的规则是我们写的。向同一会话前后扩展相邻轮次是常见做法，ReFind（前后各 2 块）和 InvMem（`result_window`）也这样做。
- 时间标注、日期格式选择、未来消息降权：规则和参数是我们写的。按时间戳解析相对日期也是常见做法，Chronicle 在写入时做。
- 按问题类型路由、top-20 原话摘要。
- Streaming 反馈过滤。
- 同步写入和幂等处理：接口语义来自平台规范。
- v1.1：按内容识别赛道的规则；代码路径的卡片组成和关键文档；多模态路径的图片存储、返回顺序和上限；索引 LRU。

通用方法：RRF（Cormack, Clarke, Büttcher, SIGIR 2009，k=60 取自该文）、Okapi BM25（Robertson 等）。第一期其他作品（FlowGrid、aml-memory-mvp 等）读过，没有采用其方法。

依赖的库：bm25s（MIT）、PyStemmer（MIT、BSD）、tiktoken（MIT）、FastAPI（MIT）、Pydantic（MIT）、Starlette（BSD-3-Clause）、Uvicorn（BSD-3-Clause）、HTTPX（BSD-3-Clause）、NumPy（BSD-3-Clause 等）。版本见 `requirements.txt`。

### 离线用过的数据和模型

下面的数据只在本地使用，不随本仓库发布。我们只在公开数据上做开关级消融，未训练或微调任何模型。

| 数据 | 来源 | 许可证 | 本地用途 |
|---|---|---|---|
| LongMemEval（含 Refined、S） | Wu et al., arXiv:2410.10813。Refined 取自第一期 hybrid_search 仓库附带的数据 | MIT（原始仓库）；Refined 副本来源仓库无许可证 | 文本消融；S 版只用于容量估算和赛道识别的误判检查 |
| LoCoMo-Refined | https://github.com/mem-eval-suite/LoCoMo_refined ，基于 Maharana et al., arXiv:2402.17753。取自 hybrid_search 仓库附带的数据 | CC BY-NC 4.0 | 文本消融 |
| BEAM | Tavakoli et al., arXiv:2510.27246 | CC-BY-SA-4.0 | 文本消融 |
| ScriptMem | https://github.com/memorax-ai/ScriptMem （无论文）。该仓库不发布剧本原文，我们用的剧本文本和题目来自 https://github.com/cydd-1972/hybrid_search 的 `data/all_qa` | ScriptMem：CC BY-NC 4.0；hybrid_search：无 LICENSE 文件 | 文本消融 |
| CL-bench | Dou et al., arXiv:2602.03587，https://github.com/Tencent-Hunyuan/CL-bench | CL-Bench 许可条款（禁止用于训练） | 文本消融 |
| PersonaMem v1、v2 | Jiang et al., arXiv:2504.14225、arXiv:2512.06688 | v1 数据卡 MIT；v2 数据卡 CC-BY-4.0 | v2 文本消融；v1 只用于规模估算和赛道识别的误判检查 |
| SWEContextBench | arXiv:2602.08316 | 数据卡 MIT | 代码路径的检索指标和压测 |
| MemEye | Guo et al., arXiv:2605.15128 | Apache-2.0 | 多模态离线验证。`tests/test_multimodal_parts.py` 里的一句示例文本改写自该数据 |
| MemLens | arXiv:2605.14906，https://github.com/xrenaf/MEMLENS | 代码 MIT，数据 CC-BY-4.0 | 多模态消融 |
| MobileMem | Deng et al., arXiv:2608.13606 | MIT | 多模态消融 |

此外用到 AML 官方公开的评测流程（https://github.com/AML-memory/agent-memory-leaderboard ）。

离线消融里的回答和评分替身：本地 Qwen3-14B；早期用过 `qwen-turbo`，重排对照用过 `qwen3-rerank`；2026-10-31 的几项对照用 gpt-4o-mini（经 OpenRouter）做回答模型和筛选器。这些模型只在离线实验中使用，只处理公开数据，不在 Add/Search 路径上。

文本路径的检索和打包配置在 v1.0（早于第一次 Smoke）定下，之后的版本没有改动；v1.3 只给返回的窗口加标签。

## 运行

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
cd service
export DASHSCOPE_API_KEY=...   # text-embedding-v4，需要北京地域的 Key
export AML_TOKEN=...           # memory-system key that the platform sends
AML_DB=./data/mem.sqlite AML_CACHE=./cache/emb.sqlite uvicorn server:app --host 127.0.0.1 --port 8000
```

服务必须单进程运行，因为索引保存在进程内存里。服务器上的 Python 是 3.10。

`scripts/smoke_local.sh` 会在本机启动服务，依次检查写入、重复写入、检索、跨用户隔离和鉴权。这个脚本会调用真实的向量接口，需要 Key。

离线测试不联网，也不花钱：

```bash
# 自带 dashscope 替身。和 v1.0.3 的逐字节比较用 git show 取基线，所以要在克隆里运行
python tests/test_lru.py
python tests/test_logs_v11.py
python tests/test_multimodal_v11.py
python tests/test_coding_v11.py
python tests/test_fixes_v11.py
# 这两个脚本导入真实的 dashscope 模块再替换 embed，只需要一个占位 Key
DASHSCOPE_API_KEY=dummy python tests/test_search_ids.py
DASHSCOPE_API_KEY=dummy python tests/test_multimodal_parts.py
```

## 部署与容量

- **systemd 服务**：`deploy/starwingmem.service`。服务只监听 127.0.0.1:8000。
- **HTTPS 前置**：`deploy/Caddyfile`。用 Caddy 申请 Let's Encrypt 的 IP 证书，只把 `/add`、`/search`、`/health` 转发给服务，其他路径一律返回 404。请求体上限 64 MB。客户端访问 IP 地址时不发送 SNI，所以要设置 `default_sni`。
- **服务器**：阿里云 ECS 杭州，8 vCPU / 14 GB 内存 / 50 GB SSD，Ubuntu 22.04。
- **费用保护**：`AML_EMB_BUDGET_YUAN` 是向量调用的费用上限。`service/server.py` 的默认值：v1.1 及之前是 65 元，v1.2 起是 150 元。花费累计到上限后，文本和多模态路径上未命中缓存的 Add 和 Search 返回 500；全部命中缓存的请求和代码路径不受影响。
- **向量接口限流**：DashScope 1M TPM，大批量写入时速度受这个限制。429 退避重试 12 次，仍失败时 Add 返回 500。
- **内存上限**：文本索引 ≤ 20 万 passage，多模态 ≤ 10 万 passage，代码 ≤ 128 个用户或 6,000 万 BM25 token。都按 LRU 淘汰，淘汰不改变 Search 结果，也不产生向量费用。
- **内存估算**（未按 Full 规模实测）：文本索引 3.6–5.2 GB（每个 passage 18–26 KB），消息原文 1–1.5 GB（相同的文本在用户之间共用一份，平台按题回放同一段历史时不会按题重复）；多模态索引约 1.8 GB，并发 16 时 Add 瞬时峰值最多约 3 GB；代码 0.6–1.5 GB；进程基线约 0.5 GB。全部叠加接近 14 GB，所以三个赛道的 Full 要错开运行。
- **磁盘估算**：多模态约 9 GB（SQLite 约 2 GB，图片文件约 5.4 GB，其余是向量缓存）；代码 0.13–0.31 GB；文本向量缓存约增加 2 GB。开始 Full 前留出 15 GB 以上空闲。
- **建议并发**：Add 16–32（多模态用 16），Search 16–32。
- **实测**：
  - v1.0 在服务器上：Add 32 并发、0 错误、p95 0.09 秒；Search 64 并发、p50 2.5 秒、p95 4.0 秒。
  - v1.1 代码路径，本地 64 用户 × 9 条轨迹、64 线程：首次 Search p50 0.28 秒、p95 1.07 秒，第二次 p50 0.017 秒，最大 RSS 529 MB。
  - v1.1 多模态路径，本地 MemLens-256k 抽样：重启 0.7 秒，每个 passage 约 17.7 KB。
  - v1.1 尚未在服务器上按 Full 规模实测。
- **可用性**：服务保持公网可访问，直到本期全部 Full 结束并完成必要的后续复核，其间 API 契约、鉴权方式和容量设置保持不变。第一个 Full 受理后到全部 Full 结束，不改服务器代码和配置，只做不改变行为的重启。文本用户重启后从 SQLite 回放，不重复计费。

## 数据处理

评测数据及其派生副本只用于完成当前任务：不用于训练或微调，不做产品分析，不用来重建数据集，不对外传播。

**删除**：每个任务完成后 30 天内，删除该任务的数据及派生副本，并保留内部删除记录（任务 ID、删除时间、执行结果）。如果确需延长，比如平台复核还没有结束，会事先取得平台的书面许可。

**隔离**：记忆和索引按 `user_id` 隔离，Search 只读请求的 `user_id`。向量缓存和图片文件按内容去重，相同文本或图片跨用户共用一份，只用来省调用和省磁盘，不会让一个用户检索到另一个用户的记忆。

删除范围：

| 位置 | 内容 |
|---|---|
| `mem.sqlite*` | 文本和多模态的 Add 请求体。v1.0–v1.0.3 写入的行含 base64 图片，v1.1 的图片用户不含 |
| `mem_coding.sqlite*` | 代码赛道的 Add 请求体 |
| `blobs/` | 图片文件 |
| `emb.sqlite*` | 向量缓存（键是文本的哈希，不存原文） |
| `search_log.jsonl`、`add_log.jsonl`、`coding_shape.jsonl` | 诊断日志 |
| `emb_usage_server.log` | 每次计费调用的时间和 token 数 |
| journald | uvicorn 的错误日志和 Caddy 的访问日志（时间、来源地址、路径、状态码、耗时）。上游接口报错时，错误信息可能带上上游响应的前 300 字符 |
| 其他 | 排查用的统计结果、服务器快照和备份、任何本地副本 |

诊断日志全部写在服务器本地。v1.1 的字段如下，不含问题、选项或消息的文本：

| 日志 | 字段 |
|---|---|
| `search_log.jsonl`（文本路径） | `ts`、`user_id`、`mm`、`top_k`、`q_sha`、`q_chars`、`n_options`、`query_images`、`n`、`chars`、`sec` |
| `search_log.jsonl`（多模态路径） | `ts`、`user_id`、`mm`、`top_k`、`q_sha`、`q_chars`、`n_options`、`query_images`、`n`、`n_img`、`img_b64`、`chars`、`resident`、`sec` |
| `add_log.jsonl`（多模态路径） | `ts`、`user_id`、`shape`（每条消息的 part 类型排列，如 `T\|TI\|I`）、`ts_set`、`img_bytes` |
| `coding_shape.jsonl`（代码 Add） | `op`、`ts`、`u`、`sid`、`n_msgs`、`roles`、`chars`、`max_chars`、`list_content`、`has_ts`、`extra_keys`、`sec` |
| `coding_shape.jsonl`（代码 Search） | `op`、`ts`、`u`、`q_type`、`top_k`、`q_sha`、`q_chars`、`n_options`、`n`、`chars`、`extra_keys`、`error`、`sec` |

- `q_sha` 是问题文本 sha256 的前 12 位，未加盐。`u`、`sid` 是平台 id 的同样哈希。它们不含明文，但属于派生数据，一并删除。
- `roles`、`extra_keys` 里，形如标识符、不超过 32 字符的名字原样记录，其他只记哈希。
- 旧版本的日志不一样：v1.0.2 和 v1.0.3 的 `search_log.jsonl` 记录了问题文本（截断到 500 字）和 `options` 全文，这些日志列入上面的删除范围。v1.0、v1.0.1 没有 Search 诊断日志。

## 版本记录

时间为北京时间（UTC+8），2026 年。

| 版本 | 完整 commit | tag | 绑定情况 | 部署时间窗 | 改动 |
|---|---|---|---|---|---|
| v1.0 | `85841682c3332b947dec6949a663c21a8f3ffd3c` | 无 | 已绑定，`version_d97988855583`，随申请绑定 | 10-30 约 09:21 – 10-31 10:55 | 首个版本 |
| v1.0.1 | `26a2a37c1676062ad78bf89af1e51e2bcf66644f` | `v1.0.1` | 已绑定，`version_eba62d0d181d`，10-31 11:03 | 10-31 10:55 – 12:22 | 修复：建议、偏好类问题的摘要条目与只有一条消息的会话窗口 id 相同，一次 Search 返回了重复 id，平台按接口契约拒收（文本赛道 Smoke 中 scriptmem 1/18 题）。现在摘要条目用 `digest:` 前缀，并保证同一次返回中 id 唯一。检索和打包逻辑不变。回归测试：`tests/test_search_ids.py`。 |
| v1.0.2 | `b933073d2a79fc24e1c71685aac04b51020ab98d` | `v1.0.2` | 未绑定 | 10-31 12:22 – 12:33 | 按赛道分接口。平台不允许，未绑定、未被评测，见下方说明。 |
| v1.0.3 | `a128e98be8bd57408439800099757a19b0d8c688` | `v1.0.3` | 已绑定，`version_51b3b5e98008`，10-31 12:33 | 10-31 12:33 – 18:28 | 多模态用户返回 ContentPart[]；Search 诊断日志。见下方说明。 |
| v1.1 | `cf0e917cc1945d43404ac558fc1be651deea9064` | `v1.1` | 已绑定，`version_4aefa3e7a8df`，10-31 18:28 | 10-31 18:28 – 20:05 | 多模态路径、代码路径、文本索引 LRU、日志不含文本。见下方说明。 |
| v1.2 | `280748c97018cba2ac17c688d8b01973a5677595` | `v1.2` | 已绑定，`version_5b119584e2d5`，10-31 20:05 | 10-31 20:05 – v1.3 部署 | 多模态窗口标签，向量费用上限 150 元。见下方说明。 |
| v1.3 | TBD | `v1.3` | TBD | TBD | 文本窗口标签。见下方说明。 |

已有 tag 不移动。v1.0 没有 tag，以 commit 为准。

所有版本共用同一组接口，同一时刻服务器上只运行一个版本。现在线上只有 v1.3，其他版本都已下线，这些版本名下的 Smoke 只是历史记录。三个赛道的 Full 只用 v1.3 的版本名。

**v1.0.2（未绑定）**：曾尝试让三个赛道各用一组接口（`/mm/*`、`/coding/*`）。平台不允许同一个 Key 的不同版本使用不同接口（HTTP 409 "endpoint is immutable for this leaderboard key"），所以这个版本没有绑定。它在服务器上部署过约 11 分钟，这段时间没有任何 Smoke 在运行（此前的代码赛道 Smoke 在 12:20:51 结束，下一次 Smoke 在 v1.0.3 部署后的 12:33:38 开始），没有被评测。

**v1.0.3**：
- 三个赛道共用 `/add`、`/search`。平台发来的 id 都是匿名哈希，所以按内容形态区分：某个用户的 Add 里只要出现过 content 数组（多模态赛道的图文混排），这个用户的 Search 结果就按接口规范返回有序的 ContentPart[]，其中只有 text，不含图片；其他用户仍返回字符串。v1.0.1 在多模态赛道也返回字符串，那次多模态 Smoke 13 道题全部 0 分。
- 检索和打包逻辑与 v1.0.1 完全相同：离线测试中，文本用户 40 次 Search 的返回逐字节一致。
- Search 诊断日志字段：`ts`、`user_id`、`mm`、`top_k`、`query`（截断到 500 字）、`query_images`、`options`（不截断）、`n`、`chars`、`sec`，写在服务器本地。日志属于评测数据，按上文删除。
- 测试：`tests/test_multimodal_parts.py`。

**v1.1**（相对 v1.0.3）：
- 新增多模态路径（`service/mmmem.py`）：存原图，Search 随结果返回命中的原图；SQLite 不再保存 base64；图片用户懒加载，LRU；中日韩 BM25 字二元组。
- 新增代码路径（`service/codemem.py`）：BM25 加历史任务卡片，不调用任何模型，独立 SQLite。
- 文本索引 LRU（`memsys.Config.max_indexed`，部署值 20 万 passage）。
- 诊断日志不再记录问题和选项的文本。
- 费用：并发写入相同文本时只计费一次（`dashscope.embed`）；Streaming 反馈消息不再向量化。
- 内存：相同的消息文本在用户之间共用一份（`memsys.share`）。
- 多模态 Add 按接口规范校验图片格式和大小；多模态文本窗口最多 88 个，给图片留出名额。
- `/health` 不占工作线程，向量接口限流时也能响应。
- 文本零回归：纯文本用户的每次 Search 响应与 v1.0.3 逐字节一致。测试覆盖默认上限和强制小上限、并发、重启，以及穿插代码用户和图片用户的场景。已知例外两处：`options` 里带孤立代理字符时，v1.0.3 因写日志失败返回 500，v1.1 不再把这段文本写进日志，返回 200；Search 建索引时向量调用失败、平台重试时，v1.0.3 会把新消息在索引里写两遍，v1.1 不会，这次重试的排序可能不同。
- 测试：`tests/test_lru.py`、`tests/test_logs_v11.py`、`tests/test_multimodal_v11.py`、`tests/test_coding_v11.py`、`tests/test_fixes_v11.py`。

**v1.3**（相对 v1.2）：
- 文本路径的窗口也加 `<past_conversation index="k">` 标签，窗口内文逐字不变。id、score、顺序不变，Add 不变。代码路径不变，不调用生成式模型。
- `AML_TEXT_TAG=0` 时不加标签，只用于和旧版本逐字节比较的回归测试。测试：`tests/test_text_tag.py`。

**v1.2**（相对 v1.1）：
- 多模态文本窗口加 `<past_conversation index="k">` 标签，窗口内文逐字不变。原因：回答模型可能把返回的原始对话当成进行中的聊天，转而回复其中的问题。
- 向量费用上限的默认值从 65 元改为 150 元。
- 文本路径和代码路径的响应不变，Add 不变，不调用生成式模型。

## 许可证

MIT，见 `LICENSE`。
