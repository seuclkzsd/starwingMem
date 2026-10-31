# starwingMem

starwingMem is a long-term memory service for the Agent Memory Challenge 2026 (Cycle 2, Open-source Methods division). It implements the AML Add / Search contract. Add stores conversation messages verbatim. Search returns the most relevant verbatim history to the platform's fixed answer model, with dates resolved. v1.0 uses `text-embedding-v4` for embeddings and **no generative LLM** at Add or Search time.

starwingMem 是为第二届 Agent 记忆挑战赛（开源方法榜）实现的记忆服务。它只负责 Add / Search：原样保存对话，检索时返回带日期标注的原始历史片段，最终答案由平台统一生成。

## 接口

| 接口 | 说明 |
|---|---|
| `POST /add` | 同步写入。消息先存入 SQLite、完成向量化，然后才返回 `{"success": true, "request_id", "user_id", "session_id"}`，返回后即可检索。同一 `user_id` 下 `request_id` 重复时，按幂等处理。 |
| `POST /search` | 只在请求的 `user_id` 范围内检索。返回 `{"data": [{"id", "content", "score", "created_at"}]}`，最多 `top_k` 条，按相关性从高到低排序。`content` 是原始消息文本，不包含生成的答案。 |
| `GET /health` | 无需鉴权，返回 `{"status": "ok"}`。 |

鉴权支持 `Authorization: Token <key>`、`Authorization: Bearer <key>` 或 `X-Api-Key: <key>`，key 来自环境变量 `AML_TOKEN`。

## 方法

1. **写入**：每条消息原样保存。超过 350 个词的消息切成段，每段 300 词、相邻两段重叠 50 词。每段向量化时同时取 `text-embedding-v4` 的 dense 和 sparse 两种向量，并建立 BM25 索引（bm25s，带英文词干化）。向量缓存在本地 SQLite 中，重启服务时回放已写入的数据，不会重复计费。
2. **检索排序**：dense、sparse、BM25 三路结果用 RRF 融合（k=60，权重 1 / 0.5 / 1）。另加一路会话级排序：先按会话内命中消息的得分之和给会话排名，再把这个排名作为一项 RRF 分数加给会话内的消息（权重 0.5）。
3. **时间处理**：
   - 如果问题里写明了"当前时间"（如 "Now is …"），检索前先去掉这句；时间晚于这个日期的消息降权（`future_penalty=3`）。
   - 问题里提到时间范围的，范围内的消息加权（`time_boost=1`）。
   - 返回的文本里，"yesterday""last Tuesday"等相对时间表达后面会追加按消息时间戳换算出的绝对日期，例如 `yesterday [i.e. 21 February 2023]`。这一步是确定性规则，见 `service/timeannot.py`。
4. **打包**：按排序取出命中的消息，向同一会话前后扩展相邻轮次，合并成窗口，直到装满 16k token 的预算。每个窗口以消息日期开头。问题没有写明当前时间时，日期用自然写法，如 "16 June 2023 (Friday)"。
5. **按问题类型路由**：建议类、偏好类问题，先返回一条"最相关的 20 条原话"摘要，再接完整窗口。分类用规则完成（`memsys.qtype`）。
6. **Streaming**：平台回写的 `Streaming online QA feedback:` 消息只存储，不进入检索索引。
7. **选择题**：`options` 只作为额外的 BM25 检索词，不当作事实写入记忆。

部署配置见 `service/server.py` 里的 `DEFAULT_CFG`。`memsys.Config` 中其他开关在部署配置里都是关闭的。

### 主要取舍（本地评测）

本地评测用 Qwen3-14B 代替平台的回答模型和评分模型，数据来自 LongMemEval、LoCoMo-refined、BEAM 的公开子集。

| 改动 | 结果 | 是否采用 |
|---|---|---|
| 未写明当前时间的问题，日期用自然写法 | LoCoMo 500 题 +2.4（显著），LongMemEval 不变 | 采用 |
| 建议、偏好类问题先给 top-20 原话摘要 | BEAM +1.5（显著），LongMemEval +2.2，LoCoMo +0.2 | 采用 |
| Streaming 反馈消息存入检索 | 原样存入：LoCoMo −3.8（显著）；只存答对的：−2.8 | 不存入索引 |
| 加重排模型 | 证据覆盖率 +1.6，端到端持平 | 不用 |
| 返回结果按时间排序 | LongMemEval −6.1 | 不用 |

## 原始工作与改动披露

- **ReFind**（第一期 AML 开源参赛作品，https://github.com/imlrz/ReFind）：参考了它的会话级排序通道思路（`w_session`），代码为我们独立实现。部署配置中启用。
- 以下第一期开源作品的思路也在 `memsys.Config` 中实现成了可选开关，**部署配置中都关闭**：
  - InvMem（https://github.com/wenxiaof345-ctrl/vanilla-rag-memory，`seed_k`）
  - Chronicle（https://github.com/simple-boy/Chronicle-Memory，`session_cap`）
  - Hybrid Episodic Memory（https://github.com/tlysanhuo/agent-memory-challenge，`recency`）
  - ActiveMemoryIndex（https://github.com/linxuhao/ActiveMemoryIndex，`w_hyde`）
  - ChronoHybridMem（https://github.com/Tin11Mn/chrono-hybrid-mem，`w_plan`）
- 依赖的库：bm25s（MIT）、PyStemmer、tiktoken、FastAPI、NumPy。
- 本地评测用到 AML 官方公开的评测流程（https://github.com/AML-memory/agent-memory-leaderboard）以及 LongMemEval、LoCoMo-refined、BEAM 的公开数据。这些只用于离线评测，不随本仓库发布。
- 其余部分为本团队原创，包括：
  - 时间标注、日期格式选择、未来消息降权；
  - 证据窗口打包、按问题类型路由；
  - Streaming 反馈过滤、同步写入、幂等处理。

## 模型使用（学术榜要求）

- Embedding：阿里云 DashScope `text-embedding-v4`，1024 维，同时取 dense 和 sparse 向量。
- 生成式 LLM：v1.0 的 Add 和 Search 都不调用。代码里虽然有可选的查询改写开关（`llm_model`），但部署中关闭。以后的版本如果启用，只用 `gpt-4o-mini`，并另行披露。
- 重排模型：不使用。

## 运行

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
cd service
export DASHSCOPE_API_KEY=...   # text-embedding-v4
export AML_TOKEN=...           # memory-system key that the platform sends
AML_DB=./data/mem.sqlite AML_CACHE=./cache/emb.sqlite uvicorn server:app --host 127.0.0.1 --port 8000
```

`scripts/smoke_local.sh` 会在本机启动服务，依次检查写入、重复写入、检索、跨用户隔离和鉴权。

## 部署

- **systemd 服务**：`deploy/starwingmem.service`。服务只监听 127.0.0.1:8000，必须单进程运行，因为记忆保存在进程内存里。
- **HTTPS 前置**：`deploy/Caddyfile`。用 Caddy 申请 Let's Encrypt 的 IP 证书，只把 `/add`、`/search`、`/health` 转发给服务，其他路径一律返回 404。客户端访问 IP 地址时不发送 SNI，所以要设置 `default_sni`。
- **费用保护**：环境变量 `AML_EMB_BUDGET_YUAN`（默认 65 元）是向量调用的费用上限，花费累计到上限后，新的向量调用会报错。

## 数据处理

评测数据只用于完成本期评测，不用于训练，评测结束后 30 天内删除。

## 版本记录

| 版本 | commit / tag | 改动 |
|---|---|---|
| v1.0 | `8584168` | 首个版本 |
| v1.0.1 | tag `v1.0.1` | 修复：建议、偏好类问题的摘要条目与只有一条消息的会话窗口 id 相同，一次 Search 返回了重复 id，平台按接口契约拒收（文本赛道 Smoke 中 scriptmem 1/18 题）。现在摘要条目用 `digest:` 前缀，并保证同一次返回中 id 唯一。检索和打包逻辑不变。回归测试：`tests/test_search_ids.py`。 |
| v1.0.2 | tag `v1.0.2` | 按赛道分接口。平台不允许，未绑定、未评测，见下方说明。 |
| v1.0.3 | tag `v1.0.3` | 多模态用户返回 ContentPart[]；Search 诊断日志。见下方说明。 |

## 版本记录补充

**v1.0.2（未绑定）**：曾尝试让三个赛道各用一组接口（`/mm/*`、`/coding/*`）。平台不允许同一个 Key 的不同版本使用不同接口（HTTP 409 "endpoint is immutable for this leaderboard key"），所以这个版本没有绑定，也没有参加评测。

**v1.0.3**：
- 三个赛道共用 `/add`、`/search`。平台发来的 id 都是匿名哈希，所以按内容形态区分：某个用户的 Add 里只要出现过 content 数组（多模态赛道的图文混排），这个用户的 Search 结果就按接口规范返回有序的 ContentPart[]；其他用户仍返回字符串。v1.0.1 在多模态赛道也返回字符串，那次多模态 Smoke 13 道题全部 0 分。
- 检索和打包逻辑与 v1.0.1 完全相同：离线测试中，文本用户 40 次 Search 的返回逐字节一致。
- Search 诊断日志：只记录问题（截断到 500 字）、返回条数、字符数和耗时，写在服务器本地。日志属于评测数据，和其他评测数据一样在本期结束后 30 天内删除。
- 测试：`tests/test_multimodal_parts.py`。

## 许可证

MIT，见 `LICENSE`。
