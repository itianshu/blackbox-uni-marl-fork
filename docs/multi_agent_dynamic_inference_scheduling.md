# RFC：多策略动态推理调度（separate-async，按异构场景设计、机制兼容同构）

Status: Final design
Authors: Uni-Agent contributors
Last updated: 2026-09-08

## 概述（Summary）

本 RFC 面向多智能体黑盒训练场景：每个 agent 使用不同的 policy，**模型架构不同**（如一个 2B、一个 30B），**replica 结构（TP/卡数/节点布局）亦可不同**（如 agentA 一个 replica 是单节点 8 卡、agentB 一个 replica 是双节点16卡）；训练与推理**异步**、部署形式为 **separate**（训练与推理在各自独立的 GPU 池上并发），与 verl V1 的 `separate_async` 一致。**问题**：一条多 agent 轨迹（如 A->B->A->C）须整条生成完才算一次采样进入 TransferQueue 供训练消费；各 agent 推理负载不均时有两个恶果--(1) **训推吞吐失配、一侧空等**：完整轨迹的产出被最慢环节门控，推理侧吞吐下降而训练侧消费速率不变（典型：训练侧把 TQ 消费空后空等）；(2) **生成轨迹陈旧度过高**：瓶颈 agent 队列里积压的基本是上几个 step 的轨迹请求，被处理时已横跨多次权重更新（旧版本生成），且一条轨迹前后多个 turn 命中不同版本、轨迹内版本差距过大，off-policy 加剧、训练不稳定。

![alt text](image-1.png)

![alt text](image-2.png)

![alt text](image-3.png)

**本 RFC 的机制按 POLICY 异构场景设计（实际业务场景通常异构，如一个 2B、一个 30B），并天然兼容同构场景**：初始化 standalone 推理实例时，在可借资源上预创建 donor 架构的 guest replica；未借用时 guest 不加载真实权重并常驻 vLLM level-2 sleep。假定 home 为借出方、donor 为借入方，借用时先撤掉 home 的推理与权重同步路由，将在途请求迁移到 home 剩余 replica，随后 sleep home、wake guest、定向同步 donor 最新权重并把 guest 纳入 donor LB。调度器在完整 metrics 样本到达时计算目标分配，只要存在可借单元且方案通过防抖，便立即执行相对当前分配的借还差量；仍在目标中的借用保持原地续借，只在 donor 更新权重后刷新 guest。

**最终方案定案**：

- **调度信号**：以 vLLM `/metrics` 的 **KV cache 利用率**作为借卡、普通归还和安全早归还的统一主信号；LB per-server in-flight 仅作为迁移成本排序等辅助信息；
- **调度时机**：poll 线程持续刷新负载信号；完整 metrics 样本确认需要调度且存在可借单元时，普通借入立即串行执行；普通归还要求 donor 连续 10 次低 KV 且缩容后预测 KV 仍安全，可只归还部分单元；成功借出后进入 10 秒全局借卡冷却，冷却期只禁止新增借卡，不阻塞归还；step boundary 只负责权重更新后的 guest 续借刷新和失败恢复；
- **借用量**：采用护栏约束下的离散容量分配：一次借卡事件只允许一个 `home → donor` 方向，可沿该方向借多个 N:M 原子单元；预测利用率只用于安排候选顺序，不作为借卡准入或停止条件；持续选择直到该方向候选耗尽或触发借出方护栏；
- **多入边/多出边**：借还映射是一般有向图；同一 home 可为多个 donor 预创建 guest slot，同一 donor 也可随多轮调度累计接收多个 home 的单元，但单次事件只新增一个方向。一个存量分配可同时包含 `A→B` 与 `A→C`，但二者必须使用不重叠的 home replica；
- **防抖**：进入/退出阈值滞回之外，借卡目标需要连续确认；一次只新增一个方向；成功借出后进入 10 秒全局冷却；物理切换后等待稳定采样窗口；普通归还要求连续 10 次低 KV、缩容预测护栏和最短持有期；同等目标优先保留当前单元；
- **原地续借**：目标中不变的 guest 保持路由和唤醒状态，仅在 donor `update_weights` 后定向刷新权重。

## 动机（Motivation）

**核心问题一：局部负载不均 -> 训推吞吐失配 -> TQ 水位下降 -> 长期干涸。** separate-async 下训练与推理并发：一条轨迹（如 A->B->A->C）**整条生成完才算一次采样**，finish 后进 TransferQueue（TQ）供训练侧消费；训练侧攒够 mini-batch 触发一轮训练，A/B/C 共享同一 global step。若某 agent（如 C）推理资源不足，完整轨迹的生成卡在 C 环节--A/B 的调用各自完成了，轨迹却出不来--推理侧吞吐随之下降；训练侧消费速率不变，长期下去 TQ 被抽干、训练侧空等（反之若推理快于训练，则推理侧积压空等）。

**核心问题二：生成轨迹陈旧度过高 -> off-policy 加剧、训练不稳定。** C 过载时，其队列里排的基本是**上几个 step 产生的轨迹请求**：这些轨迹的 C 环节从产出到被处理横跨了 C 的多次 `on_step_end` 权重更新，是用已过时的策略版本生成的，轨迹陈旧度（实际推理版本 vs 当前训练版本）随排队深度持续走高；且一条轨迹（如 A->C->A->C）的前后多个 turn 分别命中不同权重版本，**同一轨迹内部的版本差距也随之更大**--训练侧重要性采样比率恶化、梯度方差增大，训练趋于不稳定。

**借出的形态：相对低载而非绝对空闲。** separate-async 持续生成下**不存在完全空闲**--多批 rollout 生命周期重叠，A 的队列里始终有其他轨迹/其他 step 的请求，不会排空。富余是**相对**的：A 的请求到达率低于其服务能力。separate 部署下训练在另一组卡上并发，低载推理卡无法用于训练；唯一出路是**把 A 的低载借出单元借给 C**（单元内的小量在途请求迁移到 `A` 剩余 replica 续推，不丢弃），增加 C 的吞吐。C 侧积压深（队列深、剩余工作量大），足以摊销 swap 与迁移成本。

这比"笼统的负载不均"更具体，且**无法靠静态配吞吐消除**：长尾是结构性的（agent 角色决定 A 短 C 长），A 的负载必然相对偏低；静态消除需预知长尾比例并把更多卡给 C，但长尾比例常动态/不定，静态分配抓不住，只有运行时动态借用才行。

架构不同时（A=2B 低载、C=30B 过载），2B 引擎装不下 30B 权重，必须在该卡组上运行 30B 引擎--通过预创建的 30B 结构借调副本（sleep）唤醒实现，并靠 sleep 释放显存让引擎分时占用同一组物理卡。若 A 的单个 replica 小于 30B 借调副本所需卡数，则聚合多个 A replica 凑卡；反之一个大 home replica 可拆出多个小借调副本。

目标：**在 separate-async 持续生成期间，跨 policy 借用低载卡组（异构或同构同一机制）--sleep home 引擎、wake donor replica、定向同步一次 donor 权重、纳入 donor 路由；借调副本不在任何 checkpoint manager、不被任何 policy 持续同步/冲刷，在归还 home 的时候再更新 home 的权重。**

---

## 推理生命周期与触发条件（Inference Lifecycle & Triggers）

推理请求生命周期（separate-async，dispatch 链：MAS runner -> gateway -> `PolicyRoutingLLMClient` -> 各 policy `LLMServerClient` -> 各 policy `GlobalRequestLoadBalancer`（LB）-> `vLLMHttpServer`）：

1. **添加新推理任务（new task）**：MAS runner 推进到下一步（下一 turn / 下一 agent / 下一段生成），对 policy `X` 发 `generate(policy_name=X)`。流向：gateway -> `PolicyRoutingLLMClient.generate` -> `X` 的 `LLMServerClient.generate(request_id, ...)` -> `X` 的 LB。
2. **replica 被调用到（invoked）**：`X` 的 LB `acquire_server(request_id)` 按 least-loaded + sticky 选一个 replica，调其 `vLLMHttpServer.generate`。
3. **推理完成（complete）**：`vLLMHttpServer.generate` 返回（vLLM 生成完请求 token：EOS 或 max_tokens）。流向：结果 -> `LLMServerClient` -> LB `release_server`（in-flight **−1**）-> 结果回 MAS runner -> MAS runner 据此发下一个 `generate`（回到步骤 1）。
4. **abort/重试**：若请求被 abort（权重同步或借还期间），`FullyAsyncLLMServerClient` 透明重试（重新 `acquire_server` 再 `generate`），agent 侧无感。

---

### 调度信号与观测量：vLLM metrics + LB 在途

**信号采集（零 verl 改动）**。每个 standalone replica 的 `vLLMHttpServer` 自带 Prometheus `/metrics` 端点。调度器以 `metrics_scrape_interval_s` 周期并发抓取各 replica，避免单个不可达服务把一次采样串行拖长。

- **完整 vLLM metrics 快照**：保留 `/metrics` 中全部数值型 `vllm:*` series 及 labels，未知的新版本指标也不会丢弃。观测汇总包含 running/waiting/swapped 请求数、prompt/generation token counter 与窗口吞吐、请求成功/失败和抢占计数、prefix-cache 命中率，以及 queue time、TTFT、TPOT/ITL、端到端、prefill、decode、inference 等 histogram 的 mean/p90。
- **KV cache 利用率（主判定信号）**：从完整快照中选择 gauge。**metric 名随 vLLM 版本有更名**（如 `gpu_cache_usage_perc`、`kv_cache_usage_perc`、`kv_cache_usage_ratio`），配置 `kv_metric_names` 按序尝试，取第一个命中的系列。
- **LB 在途（辅助，仍采集）**：`get_status()` per-server in-flight / `get_total_inflight()`，用于借出单元排序（per-replica 在途 = abort 迁移的回放成本）和观测，不再单独触发归还。借出时请求会迁移到剩余 home replica，in-flight 必然瞬时升高，因此不能把它作为归还主判据。

**policy 级聚合**：waiting/running 等 gauge 同时保留即时值和窗口 p90；counter 保留累计值并基于窗口内同一 replica 计算每秒速率；histogram 汇总 mean/p90。KV 每次成功抓取取 per-replica 最大值，再在约 20 次抓取跨度的滚动窗口上取 p90，最后按 `ema_alpha` 做 EMA。EMA、借还确认和 `rebalance_settle_polls` 只随完整的新 `/metrics` 抓取推进，不会被更高频的 LB poll 重复推进。缺失或非法值会跳过，不会当作 0。

### 调度时机：完整 metrics 样本驱动即时调度

poll 线程持续采样并更新各 policy 的 EMA，同时检查 home 是否需要安全早归还。一次抓取中所有 policy 都获得合法 KV 值后，直接在 poll 线程计算目标分配；若有可借单元且方案通过防抖，立即执行借还，无需等待下一个 step end：

- **调度判入**：`kv_util ≥ kv_enter` 的最高 policy 驱动防抖 FSM；连续 `bottleneck_confirm_polls` 次完整 metrics 判定后开启一次调度 episode。
- **多 donor 候选**：episode 开启后，所有 `kv_util ≥ kv_enter` 且存在入边的 policy 都可作为本轮 donor；最高 policy 仍作为兼容指标 `bottleneck` 暴露。
- **解除**：驱动 episode 的 policy 降到 `kv_exit` 或以下时退出瓶颈态，但不会直接清空其全部借用；普通归还还须通过连续低 KV 和缩容后预测 KV 护栏。
- **借卡目标确认**：离散容量分配产生的目标须连续 `rebalance_confirm_polls` 次完整 metrics 判定一致，才执行新增借卡；普通归还已有独立的 10 次低 KV 确认，不再额外等待这层确认。
- **稳定窗口**：物理借还后等待 `rebalance_settle_polls` 次 poll；窗口内保持当前分配，避免 guest 空 KV、请求回放和 EMA 延迟形成瞬态反馈。
- **借卡冷却**：任一借卡事件成功后，`borrow_cooldown_s` 秒内不再新增任何 `home → donor` 借用；普通归还与安全早归还不受影响。冷却结束后，新的完整 metrics 样本才可能确认并触发下一次借卡。
- **早归还（实时，负载驱动）**：见下节，时机不固定。

普通借还和安全早归还都由 poll 线程经互斥 gate 在 trainer 进程执行，并与 `update_weights` 串行。采样发生在 gate 外；如果采样期间刚好跨过 step boundary，该次旧采样会被丢弃，下一次 poll 重新判断。

**step 边界顺序**：恢复未完成的归还 → 各 policy `on_step_end` 常规同步 → 对仍在借用的 guest 定向同步 donor 最新权重。step boundary 不再发起普通借卡；续借单元不切 LB、不 sleep/wake。

### 借用量：护栏约束下的离散容量分配

**借多少**由 `quantity.py` 计算。算法不假定只能借一个 replica，而是以配置声明的 N:M 借出单元为不可拆分的原子单位。定量层可以产生完整候选，调度器只放行排序最靠前的一个 `home → donor` 方向；该方向可以包含多个原子单元，其他方向须等待至少 10 秒后的下一轮借卡。调度器负责"何时和哪个方向"，定量策略负责该方向"多少"。

计算步骤：

1. 由观测时的服务容量还原各 policy 的固定需求：`demand[p] = ema_util[p] × current_serving_cards[p]`。这样即使上一个 step 仍有借用，需求估计也不会把借来的容量误当作新增请求。
2. 枚举完整目标分配时以各 policy 的初始卡数为基线：`capacity[p] = initial_cards[p]`；当前仍在使用的借用单元也属于候选。
3. 枚举所有指向当前热 donor（`kv_util ≥ kv_enter`）的未使用、拓扑合法且未被失败封锁的 N:M 单元，模拟卡数沿对应边从 home 转移到 donor，并计算 `predicted_util[p] = demand[p] / capacity_after[p]`。
4. 将相关 policy 的预测利用率从高到低排列成向量，按字典序为候选排序：优先处理预测峰值更低的方案；同分时优先保留当前单元，再选择在途请求更少的单元。该排序只决定先借哪个单元，不决定是否允许借卡。
5. 选中后更新预测容量并继续计算，直到候选耗尽，或剩余候选都触发借出方安全护栏。

每个候选还必须同时满足：借出方保留至少 1 个完整 home replica；借出方预测利用率不超过 `kv_post_lend_max`；N:M 卡数闭合；pair 未处于失败封锁期；同一 home replica 不得被两个计划重复使用。若旧借用归还失败，本轮不执行任何新借用，避免按错误容量继续调度。

算法不再检查“是否改善整体负载”，也没有收益死区；借卡目标仍须连续确认，最终只执行目标与当前分配的差量。已有借用不会因退出瓶颈而整批清空，单方向限制只约束本轮新增单元。

### 普通归还与安全早归还（KV 驱动）

**普通归还**监测借入方 donor。仅当 donor 的 EMA KV 连续 `return_confirm_polls` 次完整抓取不高于 `kv_exit`，才开始尝试缩容。调度器以 `demand = ema_kv × current_serving_cards` 固定当前需求，将候选借用单元按 guest 卡数从小到大逐个模拟撤除；仅当 `demand / capacity_after <= kv_post_lend_max` 时才归还该单元，并基于缩容后的容量继续尝试下一个。因此一次可以归还零个、一个或多个单元，不要求归还该 donor 的全部借入 replica。完成一批归还后，剩余借用重新累计 10 次低 KV。

**安全早归还**监测借出方 home。借用期间 poll 线程使用与借卡相同的窗口 p90 + EMA KV 信号；当 home 的 EMA KV 连续 `early_return_confirm_polls` 次完整抓取达到 `kv_enter` 时，立即归还一个借出单元。不完整或非法 KV 不计数；LB in-flight 只作为辅助观测。home 真正回 LB 前会同步当前权重；若同一单元立即转借给另一 donor，则 home 保持 sleep，不做无用的 home 权重同步。安全早归还不受普通归还的缩容预测、稳定窗口或最短持有期限制。

---

## 背景与术语（Background & Glossary）

基于 verl V1 `PPOTrainerSeparateAsync`：

- **Policy**：一个 RL 策略，有自己的 actor（FSDP，在 `global_pool` 训练卡）、自己的 `LLMServerManager` + `GlobalRequestLoadBalancer`（LB）+ `CheckpointEngineManager`。
- **两个 checkpoint manager**：`checkpoint_manager`（hybrid 共置，强制 naive，`trainer_base.py:357`）；`standalone_checkpoint_manager`（standalone replica，**非 naive**，断言要求 nccl/nixl/mooncake，`trainer_separate_async.py:59-61`）。**跨池同步走后者，本特性的 add/remove 都针对它。**
- **Standalone replica**：独立推理卡池（`rollout_pool_{rank}`）上的 vLLM 引擎。`replica.workers` = `CheckpointEngineWorker` actor（NCCL/NIXL 接收端），**不是** FusedWorker。-- 这使得跨池 W1 推送对任意 standalone replica（含借调副本）都直接可用。
- **借调副本（英文/代码：guest replica）**：预创建在**借出单元**（一个 home replica 的卡组，或多个 home replica 聚合的卡集合）上的 standalone replica。取名"借调"取其本义--编制（卡）在 home、临时为 donor 工作、用完归还：**结构与 donor 自有 replica 完全一致**（同架构、同 TP/卡数/节点布局，不按 home 卡组适配），常驻 level-2 sleep（权重落 CPU）。一个借出单元上可放置整数个借调副本（按总卡数折算）。它就是一个标准 standalone replica（同样有 `CheckpointEngineWorker` 接收端），只是初始沉睡、与 home 引擎共占同一批物理卡。
- **Home / Donor / 借出单元（borrow unit）**：home 为被借出卡的所属 policy；donor 为借用卡的过载 policy；借出单元为一次借用所涉及的 1..N 个 home replica，其卡集合须恰能切分成 M 个 donor 结构借调副本（N、M 与分组由映射表显式声明，如 2×8 卡 home -> 1×16 卡借调副本、1×16 卡 home -> 2×8 卡借调副本）。
- **持续生成，无窗口**：`ReplayBufferAsync` 轮询连续到达的轨迹；训练与生成并发。`update_weights` 在 `on_step_end` **离散触发**（abort->跨池推->resume）；被 abort 请求由 `FullyAsyncLLMServerClient` 透明重试。
- **LB 与 manager 是两件事**：加进 `standalone_checkpoint_manager.replicas`（同步）≠ 加进 `GlobalRequestLoadBalancer`（路由），须分别调（参照 `trainer_separate_async.py:194-203`）。

## 假设与硬约束（Assumptions and Hard Constraints）

- **模型架构可异**（2B/30B 等）；机制按异构设计，同构 policy 对同样适用（同一流程，无分叉）。
- **replica 结构可不一致，但折算须闭合**：各 policy 的 replica 可有不同 TP/卡数/节点布局（如 A 单节点 8 卡/replica、B 双节点 16 卡/replica）。借调副本结构一律跟随 donor 自身 replica 结构。每个借还方向声明借出单元：N 个 home replica 的卡集合须恰能按借调副本拓扑切分出 M 个 donor 结构借调副本（N=2,M=1 即"借出 2 个 home replica、放 1 个借调副本"；N=1,M=2 即"借出 1 个、拆放 2 个"）。折算与分组由映射表显式声明、初始化校验。这是借用的前提。
- **separate-async 部署**：训练在 `global_pool`，推理 standalone replica 在各自 `rollout_pool`，分卡池并发。权重同步天然跨池、非 naive。
- **`update_weights` 在 `on_step_end` 离散触发**，遍历 `standalone_checkpoint_manager.replicas` 跨池推送。借调副本**从不在任何 scm**（任何 policy 的 `on_step_end` 都不碰它），home 引擎借用期间须已移出 home scm，否则被 home 同步冲刷。
- **借用可跨 global step 原地续借**：目标不变时 guest 保持在 donor LB，边界上在 donor 常规 `update_weights` 完成后做一次定向权重刷新。普通归还同时受 `return_confirm_polls`、缩容后预测 KV 和 `min_lend_polls` 约束；安全早归还不受此限制。
- **借还图支持多入边与多出边**：每条 `(home, donor)` 边有独立的预创建 guest 集合。不同边可以经多轮调度累计激活不重叠的 home replica 单元，但每次借卡事件只新增一个方向；同一 home replica 在任意时刻只能服务一条出边。
- **`add_replicas`/`remove_replicas` 无锁 list 修改**（上游流程不调用）。metrics 驱动借还、续借权重刷新和安全早归还均由 gate 与 `update_weights` 串行。
- **异构需 2 处 verl 行为补丁，并由调度侧提供独立权重管线**：见后文

---

## 设计总览（Design Overview）

### 机制：sleep/wake 借调副本 + 跨 manager add/remove

"借 home `A` 的卡组给 donor `B`"= 在一个借出单元（1..N 个 `A` replica 聚合的卡集合）上激活整数个 `B` 结构的 standalone replica（预创建的借调副本，sleep）、纳入 `B` 的路由，并在开始服务前对每个借调副本同步一次 `B` 权重：

1. 借用：撤路由 + **单元内在途请求迁移到 `A` 剩余 replica 续推**（不丢弃）-> sleep 借出单元内**全部** home 引擎（权重落 CPU）-> wake 全部借调副本 -> 定向推送 `B` 当前权重 -> 借调副本进 `B` 的 LB。借调副本**不**加进 `B.scm`。
2. 借用期间：home 不在 `A.scm`，guest 不在 `B.scm`。每个边界在 `B` 常规更新完成后，对续借 guest 定向推送一次 `B` 最新权重，再继续接流量。
3. 归还：全部 guest 出 `B` LB -> guest `sleep()`。若卡回到 home 服务，则 home wake -> 定向同步 home 当前权重 -> 成组回 `A.scm` 与 `A` LB；若同一单元马上改借给另一个 donor，则保持 home sleep/脱离 scm/LB，直接 wake 新 guest 并推送新 donor 权重，省去无用的 home 权重同步和 wake/sleep。

**借调副本是无权重外壳**：预创建不加载权重、归还不回写权重，也不加入 donor 的持久 SCM。首次借入前同步 donor 权重，原地续借时在每个边界定向刷新。

### 借用 / 归还序列

记 `A.scm`/`B.scm` 为 `standalone_checkpoint_manager`，`lb_A`/`lb_B` 为 LB。普通分配变更由完整 metrics 样本即时驱动，未变化的借用原地续借；所有物理切换和权重刷新均经 gate 与 `update_weights` 串行。

设借出单元为 home `A`（2B，单节点 8 卡/replica）的 N 个 replica（记 `A_replica_1..N`），其卡集合上预创建有 donor `B`（30B，双节点 16 卡/replica）结构借调副本 `B_replica_1..M`（按折算，A 借出 2 个 replica 时，聚合 1 个 16 卡 replicaB 借调副本；反向拆分借出 1 个 16 卡可以拆出 2 个 replicaA 借调副本；等卡数互换时比例为 1）。

**借 `A` 的借出单元给 `B`**：
1. `lb_A.remove_servers([r.server_id for r in unit])` -- 停止路由新 `A` 请求到单元内 home 引擎；**必须先于 abort**，使被中断请求的重试落到 `A` 的剩余 replica。
2. **在途任务迁移（而非 abort 丢弃）**：对单元内各 `r.abort_all_requests(reset_prefix_cache=True)`；被中断请求由 `FullyAsyncLLMServerClient` 透明重试（检测 `stop_reason=="aborted"` 后以 `prompt_ids + 已生成 token` 重发），重新 `acquire_server` 时 sticky 指向已移除 server 自动失效、按 least-loaded 落到 `A` **剩余 replica** 续推（部分回放，不必从头生成）。**前提**：`A` 的 LB 保留 ≥1 个 replica 承接（安全不变量）；排空超时（`borrow_drain_timeout_s`）兜底。
3. `A.scm.remove_replicas(unit)` -- `A` 同步跳过单元内全部 home 同步权重。
4. 单元内 home 引擎逐个 `sleep()` -- level-2，`A` 权重落 CPU，腾显存。
5. 全部借调副本 `wake_up()` -- 借调副本此前无真实权重（预创建 dummy），唤醒后是空壳。
6. **定向推 `B` 当前权重到全部借调副本**：临时 `CheckpointEngineManager(非naive, actor_wg=B.actor_rollout_wg, replicas=[借来的replica]).update_weights()`。首次借入执行一次，后续续借在每个边界刷新；同一 donor 的续借单元合并成一次 manager 调用。
7. （可选）`lb_B.clear_sticky_cache()`；`lb_B.add_servers({G.server_id: G.server_handle for G in unit_guests})` -- 借调副本开始服务 `B`。

**归还（metrics 驱动的正常路径）**：
1. `lb_B.remove_servers([G.server_id for G in unit_guests])` + 各 `G.abort_all_requests(reset_prefix_cache=True)`（被中断请求由透明重试落到 `B` 剩余 replica 续推）。
2. 各借调副本 `G.sleep()` -- **不回写任何权重**；借调副本回到无权重态。
3. 单元内 home 引擎逐个 `wake_up()` -- home 权重（sleep 前版本）载回 GPU。
4. 对 home 定向推送 `A` 当前权重，随后 `A.scm.add_replicas(unit)`，最后回 `A` LB。本步常规 `update_weights` 随后再次覆盖它；提前定向推送保证 home 在重新接流量前已经是当前版本。

**安全早归还（home KV 持续饱和）**：使用同一归还序列；home 在回 `A.scm` 与 `A` LB 前同样先定向同步当前权重。若该单元在同一次分配变更中直接转借，则跳过 home 唤醒、同步和重新挂载。

**执行失败处理**：
- 借用中途失败：逆序回滚到借用前状态（guest 回 sleep、home wake、回 scm、回 LB）；回滚本身也失败则封锁该 pair 并把整体调度退化为 static。
- 归还中途失败：归还序列记录断点（return_stage），下一次完整 metrics 决策或 step boundary 从断点续传，该 lend 保持 active；部分归还中的单元禁用直接转借优化。

### 借调副本的归属与同步

借调副本归其借出单元的 home policy 所有，**从不注册进任何 policy 的 `standalone_checkpoint_manager`**。未借用时借调副本沉睡、不在任何 LB、不在任何 scm、以 dummy 权重占位。首次借用和每次跨 step 续借都从 donor actor 定向同步当前权重；归还时只 sleep、不回写权重。

### 调度器与执行边界

- **`MultiPolicyInferenceScheduler`**（trainer 进程内的普通类，非 Ray actor）：poll 时更新 EMA、稳定窗口和早归还状态；完整 metrics 样本到达后计算并确认目标分配，输出归还、续借和新增借用三类差量。
- **执行边界**：归还、续借权重刷新、借入和 `update_weights` 都在 trainer 进程由 gate 串行。`standalone_checkpoint_manager` 持不可序列化的 `actor_wg`，不能整体传给调度器 actor。

### Trainer 接线（Trainer Wiring）

- `_init_policy_runtimes()` 之前：展开借还图并统计每个 home 的出边数，向该 policy 的 rollout 注入 `dynamic_inference_max_colocate_count = 1 + out_degree(home)`；运行时补丁据此创建含多个 fractional worker slot 的 standalone PG。
- `_init_policy_runtimes()` 之后（真实 replica 创建及首次权重加载完成之后）：校验折算约束及 PG slot 数（每个借出单元对每个借用方向：N×home 卡数 == M×donor 卡数可整除组合、分组与节点布局可放置、借调副本结构=donor 自有 replica 结构）；按 config 构建**借还映射**；为每条出边的每个借出单元在其卡集合上**预创建全部借调副本**--按单元执行"home 引擎 sleep -> 借调副本 init（dummy，含显存 profiling/建图）-> 借调副本 sleep（level-2）-> home 引擎 wake"的过场；构建调度器与 poll 线程。
- poll 循环：每次完整 metrics 抓取后判断目标分配，满足条件时立即执行删减归还和增量借入；主循环的 step boundary 仅执行“未完成归还恢复 -> `on_step_end`（`update_weights`）-> 存量 guest 定向同步”。
- 各 policy 的 `standalone_checkpoint_manager.update_weights` 只同步当时未借出的 home replica；续借 guest 随后由临时 manager 定向同步 donor 最新权重。

### 安全不变量（Safety Invariants）

- **唯一归属**：每个借调副本**从不在任何 scm**；home 引擎要么在 `A.scm`（未借用）、要么移出（借用中，以借出单元为单位成组移出/移回）。任一 `on_step_end` 时刻每个 replica 至多在一个 scm、至多在一个 LB，杜绝双重归属与同步冲突。home 引擎的 remove（借）/add（还）按单元配对、相对 `update_weights` 串行。
- **串行**：add/remove、sleep/wake 与 `update_weights` 串行（trainer 主线程安全点）。
- **一卡一唤醒**：同一物理卡上任意时刻至多一个唤醒引擎（home 引擎或某个借调副本）；聚合/拆分单元靠"单元整体 sleep 全部 home 引擎、整体 wake 全部借调副本"配对保证。home 引擎 wake 的前提是其单元上全部借调副本已归还 sleep。
- **跨出边互斥**：多条出边的 sleeping guest 可以共享同一 home PG，但若两个借出单元包含同一个 home replica，guest registry 拒绝同时激活；调度器的全局 `used` 集合同样在计划阶段排除重叠单元。
- **home 保留承接能力**：任一 policy 被借出后，其 LB 须保留 ≥1 个 replica（承接借出单元在途请求的迁移，及 home 下一 turn 新请求的路由；不可全借）。
- **先撤路由再换权**：`remove_servers` **先行**（使被中断请求的重试落到剩余 replica）+ 在途迁移/`abort(reset_prefix_cache=True)` + 排空兜底，之后才 add/remove、sleep/wake、推权重；换完且定向推送完成后再 `add_servers`。
- **权重与所属一致**：guest 首次进 donor LB 前持 donor 权重，续借时在每次 donor 更新后定向刷新；home 只有真正回 home LB 时才同步当前权重，直接换借时保持 sleep，不执行无效 home 同步；如果新 guest 激活失败，回滚路径先同步 home 再恢复其 scm/LB。
- 折算约束（卡数闭合、分组可放置、借调副本结构=donor 结构）在初始化校验，违反则关闭调度。
- **借出方护栏**：借出方 LB 保留 ≥1 replica，且借后预测利用率不得超过 `kv_post_lend_max`；默认阈值满足 `kv_exit < kv_post_lend_max < kv_enter`，在允许借出与安全早归还之间保留滞回余量。

## 新增组件（New Components）

- `uni_agent/trainer/dynamic_inference/types.py`：配置/记录/校验（借调句柄、借出单元/折算映射、KV 阈值）。
- `signals.py`：LB 在途轮询 + 完整 vLLM `/metrics` 并发抓取、归一化与滚动窗口统计。
- `scheduler.py`：`MultiPolicyInferenceScheduler`（KV 瓶颈判定、防抖和分配差量，只决策不执行）。
- `quantity.py`：护栏约束下的离散容量分配，一次可选择多个 N:M 原子单元。
- `executor.py`：借/还原语（add/remove + 定向 W1 推送 + LB 切换 + 回滚/断点续传），在 trainer 进程执行。
- `guest_engine.py`：借调副本的预创建（`load_format="dummy"`）、按折算的 PG 放置（home 卡子集拆分 / 跨 home PG 聚合）、sleep/wake 生命周期。
- `controller.py`：trainer 接线（setup/边界/poll 线程/早归还互斥/关闭清理）。
- `patch.py`：verl 运行时补丁（standalone sleep/wake、guest 外部池），零源码侵入。
- 对 `multi_agents_ppo_trainer.py` 的改动（含放开 sync-only、接入 separate-async、插借还安全点）。
- config 块 `dynamic_inference_scheduling` + 2B/30B 异构示例。

## 配置（Configuration）

```yaml
dynamic_inference_scheduling:
  enable: true
  mode: resource                  # static（关闭借用，仅归还）| resource（完整 metrics 驱动调度）
  # --- 信号（/metrics 抓取）---
  resource_usage:
    kv_enter: 0.85                # 判入瓶颈的 KV 利用率阈值
    kv_exit: 0.6                  # 退出瓶颈态的阈值（非对称滞回）
    kv_post_lend_max: 0.7         # 须满足 kv_exit < 此值 < kv_enter
    kv_metric_names:              # /metrics 中 KV gauge 候选名，按序取首个命中
      - kv_cache_usage_perc
      - gpu_cache_usage_perc
      - kv_cache_usage_ratio
    ema_alpha: 0.3                # 信号 EMA 平滑
  metrics_scrape_interval_s: 1.0  # /metrics 抓取周期
  # --- 瓶颈确认 ---
  bottleneck_confirm_polls: 10    # 连续 N 次完整 metrics 判同一 policy 过载才借
  rebalance_confirm_polls: 2      # 同一目标分配连续 N 次完整 metrics 后立即执行
  rebalance_settle_polls: 3       # 物理借还后等待 N 个 poll 再重新决策
  min_lend_polls: 2               # 普通路径至少经过 N 次完整 metrics；安全早归还绕过
  borrow_cooldown_s: 10.0         # 成功借出后 N 秒内禁止新增借卡；不阻塞归还
  return_confirm_polls: 10        # donor EMA KV <= kv_exit 连续 N 次后尝试部分归还
  # --- 早归还（与借卡统一使用 KV 主信号）---
  early_return_confirm_polls: 10  # home EMA KV >= kv_enter 连续 N 次后归还一个单元
  # --- 轮询 / 执行 ---
  poll_interval_s: 0.2            # LB 辅助指标轮询周期；不用于决定归还
  borrow_drain_timeout_s: 10.0
  sleep_patch_mode: patched       # patched | collective_rpc（DP=1 兜底）
  # --- 借还映射 ---
  borrowing:
    # 空列表会对任意数量的 policy 展开为 1:1 全有向图；仅适合同构拓扑。
    # 显式列表可表达一般有向图，同一 policy 可有多条入边和多条出边。
    # 每个方向声明借出单元（N 个 home replica）与借调副本折算（M 个 donor 结构借调副本）
    # 须满足 N × home卡数 == M × donor卡数；初始化时按物理节点优先组合 home replica，
    # 并校验组合后每个 guest 的节点内卡布局（借调副本结构=donor 自有 replica 结构）。
    pairs:
      - home: policyA                 # 借出侧（home）
        donor: policyB                # 接收侧（donor，借调副本结构=其自有 replica 结构）
        home_replicas_per_unit: 2     # 每个借出单元聚合 2 个 home replica（如 2×8 卡 = 16 卡）
        guest_replicas_per_unit: 1    # 放置 1 个 donor 结构借调副本（16 卡，聚合借出）
      - home: policyB
        donor: policyA
        home_replicas_per_unit: 1     # 1 个 16 卡 home replica
        guest_replicas_per_unit: 2    # 拆成 2 个 8 卡借调副本（拆分借出）
      - home: policyA                 # policyA 的第二条出边
        donor: policyC
        home_replicas_per_unit: 1
        guest_replicas_per_unit: 1
    guest_replica_rank_offset: 10000  # guest vLLM actor rank 起始偏移，保证名称唯一
  # separate-async 自身要求 checkpoint_engine.backend 为 nixl/nccl/mooncake（非 naive）
```

---

## 风险与待解问题（Risks & Open Issues）

### B5. 一卡多引擎需要 2 处 verl 行为补丁和独立权重管线

异构要求借出单元的卡上同时存在 home 引擎与借调副本引擎（借调副本数按折算可为多个）、任意时刻每张卡只一个唤醒。核实 vendored verl（vLLM 本身未装，部分行为须对着实际 vLLM 再确认）后，**当前 verl 无法开箱支持**：前两项由运行时补丁 `patch.py` 落地，verl 子模块保持 pristine；第三项由动态调度代码实现，不属于 verl 补丁。

1. **standalone sleep/wake 当前是 no-op**（`vllm_async_server.py` STANDALONE 分支只 `logger.info("skip sleep in standalone mode")`，只有 HYBRID 才调 `engine.sleep(level=2)`/`wake_up`）。引擎本身 sleep-capable（`enable_sleep_mode=True` 无条件传入，含 standalone），只是 verl 不调。**改动**：让 STANDALONE 分支也调 `engine.sleep(level=2)`/`wake_up(...)`。**未决**：`engine.sleep(level=2)` 是否真正把显存降到近零（权重+KV 落 CPU，仅留 CUDA context/固定 buffer）--这是整个机制最关键的不确定项。缓解：初始化过场中做 **sleep 探针**（真实 level-2 sleep 需秒级；未打上补丁的 no-op 毫秒级返回），setup 时 fail-fast。

2. **借调副本 PG 放不进"借出单元的卡"**（每 replica 独占一个 PG `rollout_pool_{rank}`；另建 guest PG 无法调度到已被 home PG 保留的卡）。**改动**：借调副本不新建 PG，改为对 home 既有 PG 做 **merge/split 资源池视图**。home PG 的 `max_colocate_count` 在 replica 初始化前自动计算为 `1 + out_degree(home)`；bundle 为 `{CPU: slots, GPU: 1}`，home 与每条出边的 guest worker 各预留 `1/slots` GPU。该内部值由借还图推导，不要求用户手工配置。多个 sleeping guest 只占 Ray slot 与残留 CPU 内存；运行时通过跨出边互斥保证同一卡仍只有一个唤醒引擎。

3. **借调副本须独立 donor 架构权重管线**：用 donor 架构 `actor_wg` + 借调副本的独立临时 `CheckpointEngineManager` 做定向推送，不能复用 home 的 `standalone_checkpoint_manager`（借调副本也不进 donor scm，不做持续同步）。

### B7. 借调副本常驻 CPU 内存与预创建成本

sleep(level=2) 把 GPU 显存卸载，但模型缓冲区（如 RoPE 缩放张量）会保留在 CPU，wake up 从 CPU 载回。在本场景中这部分 CPU 上的缓存权重可能是完全没用的（借调副本借用时反正会定向推送覆盖）。

多出边会线性增加每个 home 卡上的 sleeping guest worker 数；空 `pairs` 生成全有向图时，全局 guest 数按 `O(policy_count² × home_units)` 增长。policy 较多或模型较大时应显式声明实际需要的边，控制 Ray actor、CPU 内存和启动预创建时间。

### B8. replica 的映射关系，尤其是异构 replica

### B9. 每次 `update_weights` 的 PG 成本

非 naive `update_weights` 每次新建 NCCL/NIXL PG 并 `finalize` 销毁（无泄漏，已核实），秒级。首次借入按借出单元推送；续借则把同一 donor 的全部 guest 合并到一次临时 manager 调用中。续借避免了 abort、sleep/wake 和 LB 切换，但仍有权重同步成本。

### B11. 借入 replica 的流量获取依赖后续到达；存量请求无法原地迁移（abort+重试是唯一原语）

`add_servers` 把新 replica 在途初始化为 0；新请求经 least-loaded 取全局最小--guest 加入后**新请求天然优先落到它**。反之，已派发的请求在原 replica 上运行到完成，guest 帮不到它们。

- **成立前提**：借入后仍有新请求到达。稳态门控下 donor 的到达率 λ = 其旧容量（门控的定义本身），借用后系统级排空率增量 ≈ μ_guest。
- **存量重分配**：当前不做；新到达请求由 LB 自然分流到 guest，已经排队的请求不主动迁移。

### B12. KV cache 利用率作为单一调度信号的局限

- KV 利用率是**显存维度**信号：KV 池大、序列短的 compute-bound 负载可能漏判；长上下文但无排队的负载也可能误判为瓶颈。当前实现已经采集并汇总 vLLM waiting/running、排队延迟和吞吐等指标，但尚未将它们并入调度判定；`kv_enter`、`kv_exit` 仍须按实际业务标定，并结合新增观测量评估误判率。
- 借出瞬间 guest 的 KV 池为空、home 的 KV 池在 sleep 前打满，首个观测窗口存在偏差；调度器在 `rebalance_settle_polls` 内保持当前分配，等待新拓扑下的信号稳定。

### B13. 动态借用的执行成本与窗口

- **借用执行是秒级阻塞**（abort 迁移 + sleep/wake + 定向推送建 PG，见 B9）；增量执行、10 次瓶颈确认和目标确认限制短时间内的无效换手。
- **在途迁移的瞬时压力**：借用时刻 home 剩余 replica 须立即承接被迁移请求，因此保留 `kv_post_lend_max` 护栏；安全早归还统一使用 KV，避免把迁移必然造成的 in-flight 上升误判为持续过载。
- **与 step 边界的互斥**：普通分配变更和安全早归还都由 poll 线程触发，并经 gate 与 step boundary 串行；`update_weights` 期间不执行 sleep/wake。
