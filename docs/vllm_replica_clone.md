# vLLM replica 权重克隆

借还初始化从同 policy 的已发布 vLLM 复制权重，不调用训练 actor。跨 step 的 guest
保留在目标 policy 的 checkpoint manager 中，每个 step 接收一次常规 actor 更新。

## 当前实现

- `replica_clone.py`：具名 worker extension RPC（继承 verl 原扩展，不传输函数对象），逐 rank 匹配 fingerprint/manifest。
- 独立 NCCL 双端 communicator，GPU 分块传输（最多 64 MiB 临时 buffer）。同节点和
  跨节点使用相同协议；传输不经过 driver，CPU 仅处理用于验证的哈希副本。
- source 和 destination 暂停生成并清空请求缓存，成功后恢复；其余 replicas 可继续推理。
- 每个 rank 对源和目标实际存储计算完整 SHA256；所有 rank 校验通过才接入 LB。
- communicator 按有向 replica 对缓存，每 worker 上限 16 个，随 worker 退出回收。
- 原地 copy 保持参数存储地址；不把已加工的权重再次送入 HF load_weights。
- checkpoint manager 每次更新固定 recipients tuple，并校验成员没有中途变化。
- 全部接收端成功才推进发布版本；克隆要求源版本等于 manager 已发布版本。
- abort 返回字典中的 error（含 server_results）会抛异常，不视为成功。

## 并发与删除的旧路径

删除 actor phase 的 boundary 持锁以及 `_directed_push` 临时 actor checkpoint manager。
训练前反向/optimizer 不获取拓扑锁。保留全局 gate，串行化完整借还事务和 step-end
发布/保存；这是第一版的保守粒度，尚未实现 per-policy 并行拓扑事务。
指标采样不再因 snapshot_generation 读取而阻塞，发布期间采样可以进行，过期决策丢弃。

任何失败均停止事务并使 executor/gate 失效，不自动回滚、不静默回退 actor 同步。
超时不是远端取消，失败后需要结束该实验并释放 worker，不能在不确定显存状态下
继续 wake 另一个 engine。没有实现跨进程崩溃恢复和 CAS 重放协议。

## 支持边界

第一版面向同构、未量化、无 LoRA/MTP/EP、rollout DP=1 的模型。通用枚举涵盖
parameters 和 registered buffers（含非持久 buffer）；已知 Attention.kv_cache 是请求
状态，不复制。vLLM Attention 的 q_range/k_range/v_range 标量虽未注册，也纳入
传输与哈希；发现其它未注册 tensor 属性或非连续布局会拒绝。
量化、FP8 KV、LoRA、MTP、EP 明确拒绝；需添加并验证专用运行时状态适配。
PP 按 world rank 对应并检查逐 rank manifest，但未经 PP GPU 实验不得称为已支持。
完整 hash 会引入 D2H 和 CPU 开销；当前优先验证正确性，尚未优化成可选抽样验证。

## 验证入口

```bash
bash examples/multi_agent_blackbox/scripts/run_vllm_clone_smoke.sh
```

基于原 transaction smoke 的两个训练 step，先借一次，在第一个 actor update
调用期间归还再借一次，然后跨 step 常规更新，第二个 step 前归还。断言
拓扑事务与 actor update 调用时间重叠、所有 clone hash 一致、guest 参与且仅参与
所属 policy 的同步、各 policy 每个 step 只有一次常规发布。

此次修改前的 2026-09-22 transaction smoke 报告属于旧 actor 定向同步路径，不能
作为本实现的验证结果。新结果另写报告。

冒烟配置：三个 policy 各 4 张训练卡、4 张推理卡，每个 policy 两个 TP2 home
replica，总计 24 张 GPU，仅预创建被测单元的一个 guest engine。模型为 Qwen2.5-0.5B-Instruct。

启动失败与运行期失败分开处理：guest 初始化的端口冲突，以及 vLLM 隐藏子进程根因后的
`Engine core initialization failed`，按配置次数有限重建。先 kill 全部失败 actor，并等待
其返回 ActorDiedError；清理不完整或等待超时不重建。初始化成功后的 sleep 错误、
权重克隆错误和借还中途异常不走这条重试路径。退出清理异常会单独记录，保留最初实验异常。

vLLM 服务默认从本机 24000–31999 范围租用独立 32 端口块，通过 flock 保留到
服务进程退出，并把 `VLLM_PORT` 传给引擎子进程。该默认值避开常见 Linux 临时
端口范围；其它环境应按网络策略设置 `UNI_AGENT_VLLM_PORT_RANGE=start:stop`、
`UNI_AGENT_VLLM_PORT_BLOCK_SIZE` 和 `UNI_AGENT_VLLM_PORT_LOCK_DIR`。显式
`VLLM_PORT` 会保留。锁只协调使用本协议的服务，外部进程竞争仍由启动失败处理兜底。
