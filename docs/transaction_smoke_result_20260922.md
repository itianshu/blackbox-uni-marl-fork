# 借还事务与单次同步冒烟结果（2026-09-22）

结果：通过。真实 GPU 实验完成 2 个训练 step、1 次借入、1 次归还，
脚本退出码为 0。退出后 Ray 可用 GPU 为 32/32。

## 配置与覆盖范围

- 解释器：`/mnt/bn/chenghao1026/resouces/libs/zzh_env/bin/python3`。
- verl：该环境中的 `/mnt/bn/chenghao1026/zzh/zzh_0813/verl`；未修改其源码。
- 使用现有 Ray 集群，未重启集群。
- 三个 policy 均使用 Qwen2.5-0.5B-Instruct；训练 GPU 为 4/4/4，
  standalone 推理 GPU 为 4/12/4，TP2 replicas 为 2/6/2。
- batch=8，rollout n=2，prompt 上限 4096，response 上限 1024，
  model length 上限 6144；随机路由 3–5 轮，输出预算 512–1024。
- abort/sleep/wake 超时为 60 秒；完整权重同步超时为 300 秒。
- 本次只预创建 policy2 → policy1 的 guest 方向；正式实验的互借图不变。
- 初始化后停止自动调度，固定执行借还，以保证测试覆盖；此结果不用于
  评价 KV 阈值、自动调度触发概率或负载均衡收益。

## 实测结果

| 阶段 | policy1 同步 replicas | policy2 同步 replicas | policy3 同步 replicas | 每个 policy 常规同步次数 |
|---|---:|---:|---:|---:|
| Step 1：policy2 借出一个 TP2 unit 给 policy1 | 3 | 5 | 2 | 1 |
| Step 2：归还后 | 2 | 6 | 2 | 1 |

第一个边界断言 guest 包含在 policy1 的常规同步中；第二个边界断言它已退出
policy1 的同步列表。两个边界都校验 checkpoint manager 与 active_lends 的
成员归属一致，且每个 policy 的常规同步只调用一次。

- 借入执行耗时：22.096 秒，其中 guest 定向权重装载为 19.477 秒。
- 归还执行耗时：14.996 秒，其中 home 定向权重恢复为 12.755 秒。
- 日志定义的持有时间（借入提交至归还提交）：88.913 秒。
- Step 1 / Step 2 的 `timing_s/step`：74.350 / 31.808 秒。
  测试入口在这段计时之前执行借还，所以这些时间不包含上述借还耗时，
  也不包含初始化，不应当用于性能比较。
- `CHECKPOINT_SYNC_FAILED`：0。

| Policy | Step 1 常规同步耗时（秒） | Step 2 常规同步耗时（秒） |
|---|---:|---:|
| policy1 | 10.301 | 9.026 |
| policy2 | 19.469 | 16.468 |
| policy3 | 12.252 | 12.123 |

“单次同步”指 step end 的常规同步。新借入时的模型装载、即时归还时的 home
权重恢复仍分别需要一次定向传输；已持有 guest 不再另走 renew 传输。

## 错误与并发验证

相关测试共 210 项通过：205 项轻量测试，以及 zzh_env 中 5 项 trainer 接线测试。
覆盖借还顺序、失败阶段保留、abort/sleep/wake/transfer 故障注入、失败后禁止
进入后续阶段、边界等待正在执行的事务、同步成员变更与去重、真实生效的传输
等待超时，以及冒烟入口对重复同步和 guest 漏同步的拒绝。

失败处理采用停止实验的方式：超时不等于远端操作已取消，因此不进行不确定的
自动回滚或重试，也不唤醒另一套 engine。需要重启实验；这不是崩溃恢复协议。
并发控制和错误语义见 [实现说明](dynamic_inference_transactions.md)。

## 复现与证据

```bash
bash examples/multi_agent_blackbox/scripts/run_transaction_smoke.sh
```

入口仍调用原来的 `run_e2e_borrow_verify.sh`，仅显式选择测试专用 patch。

- [完整运行日志](../examples/multi_agent_blackbox/logs/transaction_smoke_verified/train.log)
- [借还事务 JSONL](../examples/multi_agent_blackbox/logs/transaction_smoke_verified/checkpoints/dynamic_inference.jsonl)
- [结构化验收结果](../examples/multi_agent_blackbox/logs/transaction_smoke_verified/verification.json)
- [确定性测试入口](../examples/multi_agent_blackbox/transaction_smoke_patch.py)

本次验证了一个方向、一个 TP2 unit、两个 step 的真实执行链路；没有据此宣称
所有方向、长时间压力或外部 NCCL 故障均已被覆盖。
