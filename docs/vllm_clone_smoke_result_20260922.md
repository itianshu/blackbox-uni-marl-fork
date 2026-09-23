# vLLM 权重克隆与借还冒烟结果（2026-09-22）

完整 GPU 冒烟通过：2 个训练 step、2 次借卡、2 次还卡、4 次 vLLM→vLLM 克隆，进程退出码 0。结束后 Ray 可用 GPU 恢复为 32。没有重启 Ray 集群，没有修改环境中安装的 verl/vLLM。

原始运行日志包含集群路径和节点信息，保留在验证环境中，不纳入仓库。本文只记录可复现配置、结构化结论和已知限制；复现入口见下文。

## 本次修复

1. vLLM 0.19 默认拒绝 RPC 中的函数对象。改为具名 worker RPC，扩展继承 verl 的原始 worker extension，保留 checkpoint/sleep 支持，不开启不安全序列化。
2. Attention 的 q_range/k_range/v_range 是未注册的 FP32 标量。它们现在参与传输和完整哈希校验；KV cache 不复制，其它未知张量仍拒绝。
3. 冒烟的生成探针改用 verl 接口的 `logprobs=True`。原生 vLLM 的 `logprobs=0` 在 verl 接口会被转成关闭。
4. 实际重跑发现 TCPStore 端口 38681、58109 冲突，分别影响 guest 和普通引擎。服务默认租用 24000–31999 范围内的独立 32 端口块，以 flock 协调，VLLM_PORT 传递给子进程，避开本集群系统临时端口范围。已在三个远端节点检查到互不重叠的有效租约。
5. guest 尚未初始化成功时，对端口冲突和隐藏根因的 vLLM 启动失败做有限重建；检查 kill 数量并等待 ActorDiedError，清理不完整或超时不重建。初始化后的 sleep、借还、克隆异常不走此重试。
6. 保留原始实验异常，退出清理的次生异常单独记录，不再覆盖根因。

此前实现的 vLLM 克隆、删除 actor 阶段拓扑锁、保留借还事务与 step end 发布互斥，均在本轮实际运行中使用。没有恢复临时 actor checkpoint manager 的旧同步路径。

## 实验配置

- zzh_env Python，环境中安装的 verl；vLLM 0.19。
- Qwen2.5-0.5B-Instruct，未量化 BF16，TP=2、PP=1。
- 三个 policy 各 4 张训练 GPU、4 张推理 GPU，各两个 home replica；共 24 张 GPU。被测 guest 复用 policy2 的一个 TP2 单元，只预创建这一个 guest。
- 2 step，batch=8，rollout n=2；随机路由 harness 为 3–5 轮，输出预算 512–1024 token，max_model_len=6144。
- drain 超时 180 秒，权重发布/克隆超时 600 秒，clone 临时 buffer 上限 64 MiB/rank。
- 自动负载决策在冒烟中停止，强制执行事务序列。所有引擎、训练、LB 和 checkpoint 操作均为真实操作。

复现入口：`bash examples/multi_agent_blackbox/scripts/run_vllm_clone_smoke.sh`。原外层脚本的默认资源横幅可能仍显示 4/12/4；本脚本覆盖后的实际推理资源为 4/4/4。

## 借还和权重一致性

| 操作 | 时机 | 克隆路径 | 克隆耗时 | 完整事务耗时 |
| --- | --- | --- | ---: | ---: |
| 第一次借出 | step1 前 | policy1 home → policy1 guest，同节点 | 3.430 秒 | 10.739 秒 |
| 第一次归还 | step1 actor update 调用期间 | policy2 保留 replica → policy2 home，跨节点 | 2.045 秒 | 4.767 秒 |
| 第二次借出 | 与上述归还同一并发测试区间 | policy1 home → policy1 guest，同节点 | 1.695 秒 | 5.028 秒 |
| 第二次归还 | step2 前 | policy2 保留 replica → policy2 home，跨节点 | 1.676 秒 | 4.749 秒 |

借出方向均为 policy2 提供 GPU 给 policy1。归还克隆跨两个物理节点，覆盖跨节点传输。

每次实际传输 996,543,296 字节，两个 TP rank 各 498,271,648 字节。4 次源/目标完整 SHA256 均匹配；每次生成的 8 个 token 相同，最大 log probability 差值均为 0。

克隆耗时包含暂停、检查、传输、完整哈希和恢复，不是纯网络耗时，也不是单独精确测量的 source 暂停时长。source 仍在 LB 中，暂停会影响被分配到它的请求；本轮记录的已消费训练 batch 的 aborted_ratio 均为 0，不能据此推断高负载下一定没有中断。

## 并发及 step end

actor update 调用与归还/再次借出区间重叠 **3.955 秒**。actor 调用约 3.955 秒，拓扑事务区间约 11.069 秒；actor 调用结束后，边界等待拓扑事务完成再发布参数。这是调用区间验证，不是 GPU kernel profiler 证明。

| 边界 | policy1 接收 replica 数 | policy2 | policy3 | 每 policy 发布次数 |
| --- | ---: | ---: | ---: | ---: |
| step1：仍借用 | 3（含 guest） | 1 | 2 | 1 |
| step2：已归还 | 2 | 2 | 2 | 1 |

包括初始化在内，共 9 条成功权重发布记录。step1 三组发布耗时为 26.838 / 9.518 / 15.892 秒；step2 为约 11.178 秒。无借还失败、同步超时或未完成事务穿越边界。

日志 `timing_s/step` 为 step1 91.764 秒、step2 16.896 秒；这是训练函数计时，不包含全部初始化，也不包含外层探针在 step 前执行的所有借还操作，不宜作为性能对比。

## 验证范围和残留项

- 相关回归测试：121 项通过；guest 生命周期测试 26 项通过；端口分配与补丁测试 21 项通过（测试组之间有重叠，不直接相加）。真实 Ray actor 销毁确认探针通过。
- 另外完成一个 eager 模式的两个 TP2 引擎短验证，哈希及输出一致，退出码 0；完整冒烟使用 CUDA graph。
- step1 三个 policy 的梯度为 0，step2 梯度范数分别为 0.847 / 0.459 / 1.863。四次克隆发生在 step2 更新前，所以克隆哈希相同符合本次数据表现。**尚未验证 step2 非零更新后再发起一次克隆**，不能把版本号前进等同于已验证任意权重变化下的克隆。
- 本轮验证的是事务、通信、版本路由与边界发布机制，不证明 KV 调度阈值合理或负载均衡带来性能收益，也不是量化、PP、多机单个 replica 的支持认证。
- 正常退出清理仍出现第三方 resource_tracker 共享内存 KeyError 与 Gateway lifespan GeneratorExit。最终退出码为 0，32 张 GPU 全部释放；这些退出告警尚未彻底消除。
- 端口锁只协调采用同协议的服务，不能阻止外部程序主动绑定同一端口。显式 `VLLM_PORT` 会保留；其它环境可用 `UNI_AGENT_VLLM_PORT_RANGE`、`UNI_AGENT_VLLM_PORT_BLOCK_SIZE` 和 `UNI_AGENT_VLLM_PORT_LOCK_DIR` 调整租约范围。

失败尝试的日志均保留，依次位于 vllm_clone_verified（RPC）、vllm_clone_rpc_fixed（范围常量）、vllm_clone_ranges_fixed（探针 logprobs 参数）、vllm_clone_final 和 vllm_clone_retry_verified（端口冲突）。这些尝试不计入通过结果。
