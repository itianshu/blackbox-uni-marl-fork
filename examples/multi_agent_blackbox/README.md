# Multi-Agent Blackbox Training Example

This example shows the minimal Uni-Agent wiring for multi-agent blackbox RL
training.

- `framework.py` provides the example-specific `RemoteMultiAgentFramework`.
- `remote_runner.py` executes each MAS rollout runner as an independent Ray
  remote task.
- `multi_agent_runner.py` is the in-process Python MAS entry point.
- `scripts/three_agent_external_mas.py` is the command-driven standalone MAS
  entry point used by the external recipe.
- `config/mas_config.yaml` defines abstract MAS roles such as `agent_1`,
  `agent_2`, and `agent_3`.
- `config/multi_agent_blackbox.yaml` maps roles to trainable policies. Each
  policy root-composes the public `verl.trainer.config/ppo_trainer.yaml` base
  and then applies its `policies.<policy_name>.ppo_trainer_overrides`.

The runner uses one rollout-level Gateway URL. Each OpenAI-compatible
`/chat/completions` request sets `model` to the MAS role name, and the Gateway
routes that role to the policy configured in `role_policy_mapping`.

## Command-Driven External MAS

`config/multi_agent_blackbox_external.yaml` is an opt-in recipe for an MAS
that is launched as a command instead of imported as a Python callable. Each
rollout remains one Ray remote task. On the node selected by Ray, that task:

1. copies the configured MAS YAML template;
2. injects the rollout-scoped Gateway URL into the global and per-agent LLM
   settings;
3. sets each agent model to its role name;
4. starts the configured MAS child process;
5. waits for its result and removes the temporary config.

The bundled recipe is directly runnable and starts the migrated three-agent
example:

```yaml
command:
  argv:
    - python
    - -m
    - examples.multi_agent_blackbox.scripts.three_agent_external_mas
    - --config
    - "{config_path}"
    - --prompt
    - "{prompt}"
```

The same interface can launch any external MAS by replacing the module with a
user-provided command. The command receives a rollout-local YAML file and the
current task as separate arguments.

`argv` is preferred to `shell: true`: placeholders containing spaces or shell
characters remain individual arguments. The external MAS must print a JSON
object on its last stdout line when `result.mode: stdout_json`; the object can
contain `final_result` and `reward_info`.

All agents normally receive the same dynamic `rollout.base_url`. They select
different trainable policies by sending their role as the request model:

```yaml
llm:
  base_url: <injected rollout Gateway URL>
agents:
  planner:
    model: planner
  reviewer:
    model: reviewer
```

Do not expose the underlying vLLM server addresses to the MAS. The Gateway
maps `model=<role>` through `role_policy_mapping` and records the corresponding
trajectory. Request temperature is intentionally ignored by the Gateway so
generation uses each policy's rollout temperature; `top_p`, `top_k`, and
`max_tokens` may still be supplied by the MAS.

The external recipe requests one Ray CPU and uses `SPREAD`, so concurrent MAS
tasks tend to distribute across worker nodes. To exclude the head node or cap
per-node concurrency, advertise a custom Ray resource such as
`mas_runner_slot` only on eligible workers and configure:

```yaml
execution:
  ray:
    resources:
      mas_runner_slot: 1
```

The bundled three-agent MAS preserves the callable example's sequential
topology (`agent_1 -> agent_2 -> agent_3`), per-agent prompts and token limits;
each later agent receives the outputs produced by earlier agents. The script
prints one JSON object containing `final_result`, `agent_outputs`, and
`reward_info`.

The first implementation supports `execution.backend: local_process`.
`sandbox` is a reserved backend boundary for Docker or OpenYuanRong execution;
selecting it currently raises `NotImplementedError` rather than silently
running in the wrong environment.

The example uses verl's standard `reward.custom_reward_function` path. The
runner returns rollout-level `final_result` and `agent_outputs`; before
RewardLoopWorker scoring, `MultiAgentFramework` injects them into `extra_info`.
Production runs should replace `examples.multi_agent_blackbox.reward` with a
task evaluator, rule checker, or judge model that scores the final MAS result.

## Remote MAS Rollout Execution

The example config selects
`examples.multi_agent_blackbox.framework.RemoteMultiAgentFramework`. For every
prompt sample and rollout index, the parent framework first creates the Gateway
rollout and then submits one `remote_multi_agent_run` Ray task. The task receives
only serializable prompt, rollout-handle, role mapping, and runner arguments;
the live Gateway runtime and TransferQueue remain owned by the parent framework.
The remote `session_runtime` argument is only a compatibility stub for
capturing `complete_session`/`complete_multi_agent_rollout` reward metadata. A
runner must use `rollout.base_url` for model requests and must not depend on
live create, wait, finalize, or abort methods from the parent Gateway runtime.

After the Ray task returns, the parent completes and finalizes the Gateway
rollout, annotates its trajectories, and writes them to TransferQueue. A task
failure or cancellation causes the parent to abort the Gateway rollout. During
a successful trainer cleanup, outstanding Ray tasks reach a terminal state
before Gateway actors and policy resources are released; a cleanup timeout
preserves those resources and surfaces an error to the training entrypoint.
The Ray task reserves zero CPUs
by default, matching the SWE blackbox example; `max_concurrent_rollouts`
remains the limit on simultaneously submitted MAS rollouts.

## Ray Placement-Group Name Collision (verl is not modified)

verl hard-codes every policy's Ray placement-group name prefix to
`global_pool`. Ray 2.55+ enforces unique placement-group names, so initializing
the second policy trainer fails with
`name 'global_poolverl_group_...' already exists`.

This example ships a runtime patch (`verl_patch.py`) that makes each policy's
placement-group name prefix unique and labels it with the policy's
`policy_name` (e.g. `policy_1`), producing names like
`policy_1_<random>verl_group_8_8:0`. Reward-worker and vLLM-server actor names
use the same policy label. The patch is applied automatically before the
policy trainers are constructed, driven by
`config.example_patch_fqn`:

```yaml
example_patch_fqn: examples.multi_agent_blackbox.verl_patch
```

The patch is split across the processes that create and look up Ray-global
objects:

- TaskRunner `apply_patch()` installs the placement-group, reward-actor, and
  vLLM creator-side naming patches.
- Ray worker `apply_worker_patch()` installs only the `ServerAdapter`
  lookup-side naming patch before actor or task code runs.

Policy-first names keep all resources for one policy adjacent in Ray tooling,
for example `policy_1_reward_loop_worker_0` and
`policy_1_vllm_server_0_0`.

The worker hook is registered by the example config:

```yaml
ray_kwargs:
  ray_init:
    runtime_env:
      worker_process_setup_hook: examples.multi_agent_blackbox.verl_patch.apply_worker_patch
```

Ray runs the lightweight hook once in every Python worker process. Rollout
workers use the patched `ServerAdapter`; reward and Gateway workers only pay
the one-time import cost. Every Ray node must be able to import this repository
through a shared checkout, editable installation, or equivalent `PYTHONPATH`.

The Gateway actor pool remains fixed at `gateway_count` (currently 8 in the
example config) and is created once before training. Sessions are dynamic, but
training does not dynamically add Gateway actors. Changing the hook requires a
fresh training Driver so the next Ray job receives the new runtime environment;
the existing Ray cluster does not need to restart.

No verl file is modified by this runtime path. The file-level patch under
`examples/multi_agent_blackbox/patches/` is an archival alternative and is not
used by the launch scripts.

## Dynamic Inference Scheduling (cross-policy replica borrowing)

`dynamic_inference_scheduling`（顶层配置块，默认 `enable: false`，零行为变化）
implements the RFC for borrowing GPU replicas across policies at step
boundaries: an underloaded policy (home) sleeps one topology-valid N:M
borrow units (vLLM level-2), and pre-created replicas with the bottleneck
policy's (donor's) architecture wake on the same cards, receive a directed
weight push, and join the donor's load balancer. Every decision adds at most
one atomic unit; if no lender passes the home-capacity and post-lend KV guards,
the scheduler does not borrow. Decision and execution logic lives in
`uni_agent/trainer/dynamic_inference/`; verl itself stays pristine — the patches (STANDALONE
engine sleep/wake, guest external resource pool) chain on top of
`verl_patch.py` at runtime.

### 前置条件（不满足会在启动时立即报错）

1. `trainer.v1.trainer_mode: separate_async`（借用对象是 standalone rollout
   replica，sync 模式无此结构）；
2. 每个 home policy 至少有 `home_replicas_per_unit + 1` 个 standalone
   replica；每次最多新增一个完整单元，并始终保留 ≥1 个继续服务；
3. 支持异构 replica 的 N:M 折算，但必须满足
   `N × home_cards_per_replica = M × donor_cards_per_replica`，且两侧每节点
   GPU 数一致。例如 `2×(单节点8卡) → 1×(双节点16卡)`，反向配置为
   `1×(双节点16卡) → 2×(单节点8卡)`；
4. 每个 policy 的 rollout 配置满足：`name: vllm`、
   `enable_sleep_mode: true`、`free_cache_engine: true`、
   `checkpoint_engine.backend: nccl`（示例 yaml 已满足）；
5. `borrowing.pairs: []` 会对任意数量的 policy 自动生成 1:1 全有向图，
   适合同构 topology；异构 policy 必须显式声明合法的 N:M 边。借还图支持
   多入边和多出边，home placement group 的 worker slot 数会自动设置为
   `1 + out_degree(home)`，无需手工配置。

异构多向借用需要显式写出每个方向，例如 `policy_8gpu` 可同时具有两条出边：

```yaml
borrowing:
  pairs:
    - {home: policy_8gpu, donor: policy_16gpu,
       home_replicas_per_unit: 2, guest_replicas_per_unit: 1}
    - {home: policy_16gpu, donor: policy_8gpu,
       home_replicas_per_unit: 1, guest_replicas_per_unit: 2}
    - {home: policy_8gpu, donor: policy_c,
       home_replicas_per_unit: 1, guest_replicas_per_unit: 1}
```

每条出边会在相应 home 卡上预创建一套 sleeping guest。不同 home replica
单元可同时沿不同边借出；共享同一 home replica 的两套 guest 不能同时唤醒。

### 启用步骤

1. 把顶层 `dynamic_inference_scheduling.enable` 改为 `true`，按需调整
   阈值（各字段含义见 yaml 注释）；
2. 把 worker hook 切换到调度补丁（内部链式调用原 example hook）：

   ```yaml
   ray_kwargs:
     ray_init:
       runtime_env:
         worker_process_setup_hook: uni_agent.trainer.dynamic_inference.patch.apply_worker_patch
   ```

3. 重启训练 Driver（Ray worker hook 只在新 Ray job 生效）。

### 额外资源预算

- **host RAM**：每个沉睡 guest replica 持有完整的 vLLM worker actor
  外壳与 CPU 侧权重缓冲；一个借出单元的额外占用按
  `guest_replicas_per_unit` 计算。所有 guest 都会在训练开始前预创建。
- **启动时间**：预创建过场（home sleep → guest init → guest sleep →
  home wake）每单元需数秒至数十秒；precreate 在训练开始前一次性完成，
  且内置探针——若 worker 侧 sleep 补丁未生效（sleep 瞬间返回），启动
  立即失败并提示切换 `sleep_patch_mode: collective_rpc` 降级（仅 DP=1）。

### 运行时安全与降级

- 借还与 `update_weights` 严格串行（边界顺序：删减归还 → 各 policy
  `update_weights` → 续借 guest 定向同步 → 增量借入）；
- 借用失败自动回滚并屏蔽该 pair 一个 swap 窗口；重复失败触发熔断，
  调度器降级为 static（只归还不再借用）；
- 瓶颈/目标连续确认、稳定观察窗口、最短持有期和全局冷却共同限制换手；
- 训练 cleanup 时先归还全部活跃借出、kill 全部 guest actor，再走
  per-policy 清理。
- 指标以 `dynamic_inference/*` 前缀合并进 trainer metrics
  （swap_episodes / renewals / returns / bottleneck / kv_util / disabled 等）。

### 三 policy 长时间借还卡实验

```bash
bash examples/multi_agent_blackbox/scripts/run_e2e_borrow_verify.sh
```

该脚本默认使用 `zzh_env/bin/python3` 及其已安装的 verl，并清理系统
`PYTHONPATH` 对 verl 的覆盖。Ray 地址从
`/tmp/ray_zzh_255/ray_current_cluster` 读取，也可通过 `RAY_ADDRESS` 指定。
需要启动集群时使用 `scripts/start_ray_cluster.sh`；本实验需要合计 32 张 GPU。

| Agent | Policy | 训练 GPU | 独立推理 GPU | TP | 独立推理 replica |
| --- | --- | ---: | ---: | ---: | ---: |
| agent_1 | policy_1 | 4 | 4 | 2 | 2 |
| agent_2 | policy_2 | 4 | 12 | 2 | 6 |
| agent_3 | policy_3 | 4 | 4 | 2 | 2 |

默认运行 1000 step，每批 64 个 prompt、每个 prompt 4 个 rollout，最多同时运行
128 个 MAS 请求，使借来的 replica 有后续请求可以接收。agent_1/3 输出 4096 token，
agent_2 输出 512 token；均设置 `ignore_eos: true`，持续生成到 token 上限。
不设置 `min_tokens`，避免异步中断续写时剩余 token 数小于原始下限。三个 agent 顺序执行，使 KV 压力在 policy 之间转移；
借用图包含三个 policy 间的全部六个方向，每个事件最多新增 1 个 TP2 replica
（2 GPU），但不再设置 `max_units`，因此后续确认事件可以继续借用。每个 lender
仍至少保留一个 home replica，并受借出后 KV 上限保护。
实测 5% enter 无法触发借用，校准后 KV 阈值为 enter=1%、exit=0.3%、post_lend_max=0.7%，
EMA=0.3；LB 与 vLLM metrics 使用同一个 0.5 秒采样。本轮设置
`queue_signal_enabled=false`，排队数量和时间只记录、不参与借还决策；
借卡确认 10 次、目标确认 5 次、切换后等待 10 次、最短保留 60 次决策，
冷却 30 秒、归还确认 20 次、提前收回确认 10 次。轮数不是精确墙钟时间，
安全提前收回不受最短保留限制。关闭周期验证，避免打断长跑。
这些是低 KV 工作负载的实验参数，需根据真实吞吐、排队和切换成本验证收益。

`${trainer.default_local_dir}/dynamic_inference.jsonl` 每 5 秒记录一次负载，并在
决策、借还开始/完成、提前收回、续借时记录事件。包含 lend_id、方向、持有时长、
切换耗时、各 policy 的调度 KV、每 replica 等待数、排队时长、逐 replica
KV/请求数/计数器、吞吐及延迟摘要。
`sample_age_s` 标明测量距事件多久；切换完成时旧采样不应解释为切换后的收益。
KV 调度信号在每个采样点先取 policy 内所有可抓取 replica 的平均值，再取时间窗
P90 并做 EMA。排队耗时使用窗口内 Prometheus 直方图增量的 P90，同时用持续深排队
时长覆盖“请求尚未出队、直方图还未产生新观测”的情况。
需要物理 GPU 观测时，使用 `scripts/monitor_borrow_gpus.py --address <ray-address>
--launcher-pid <pid> --output <gpu.jsonl>`，每 15 秒采集各节点 GPU 利用率、显存和进程。
可设置 `TOTAL_TRAINING_STEPS` 延长或缩短实验；生成长度在
`config/mas_config_borrow_verify.yaml` 中配置。脚本末尾参数会透传给 Hydra。

对照实验可运行（同一初始模型，dynamic/static 各 20 step）：

```bash
RAY_ADDRESS=ray-gcs-host:port RAY_DASHBOARD_ADDRESS=http://dashboard-host:port \
PYTHONPATH="$PWD" /mnt/bn/chenghao1026/resouces/libs/zzh_env/bin/python3 \
  examples/multi_agent_blackbox/scripts/run_borrow_comparison.py \
  --steps 20 --output-dir /absolute/path/to/comparison
```

脚本自动启动 `observe_comparison_steps.py`，记录每个 step 日志到达时间以对齐监控窗口，
两组完成后自动调用分析脚本生成 Markdown、JSON、CSV 和图表。
也可执行 `analyze_borrow_comparison.py /absolute/path/to/comparison` 重新生成报告。
脚本默认生成独立的 1024 行重复 mock 数据，为 64-prompt batch 提供更多独立行索引。
seed 固定为 42；前 2 step 单列为预热。两组都被动采集 GPU 与 vLLM 指标。
mock reward 为答案子串命中，不是独立测试集的严格正确率。
训练进程组使用环境变量 `UNI_AGENT_MASTER_PORT_RANGE=21000:22000` 的互不重叠端口段，
guest rendezvous 仍使用 20010–20521。该设置仅对本实验启用，不修改已安装的 verl。

## Verification

Check the example configuration, launchers, and replay-buffer contract before
allocating policy GPUs:

```bash
pytest -q tests/test_multi_agent_blackbox_example.py \
  tests/uni_agent/trainer/test_multi_agent_replay_buffer_contract.py
```

For a cluster validation, launch either bundled recipe with deliberately small
models and data. A successful run must not contain a placement-group collision,
reward-actor collision, or `Failed to look up actor` error.

## Policy Resource Isolation

`role_policy_mapping` maps MAS roles to trainable policy names. Multiple roles
can map to the same policy, and only the unique policy names instantiate v1
PPOTrainer runtimes.

The root `ppo_trainer_config_source` and `ppo_trainer_config_name` select the
single verl PPO base shared by every policy. Each `policies.<policy_name>` block
provides `ppo_trainer_overrides`; policy entries do not select a different
Hydra source or PPO trainer base. The multi-agent trainer composes the
root-selected base, applies each policy's overrides, and passes the resolved
config to that policy's v1 PPOTrainer. Put policy-specific model paths, optional Prometheus served
model names, Ray resource pools, GPU counts, tensor parallel sizes, rollout
memory settings, and checkpoint directories in the overrides.

## Launch

The command-driven external MAS example has a dedicated launcher:

```bash
bash examples/multi_agent_blackbox/scripts/run_external_mas_train.sh
```

It selects `multi_agent_blackbox_external`, which starts
`three_agent_external_mas.py` inside one Ray task per rollout. All settings can
be overridden via same-name environment variables; defaults are defined inside
the script.

The original in-process callable example can still use its existing launcher:

```bash
bash examples/multi_agent_blackbox/scripts/run_e2e_train.sh
```

Both launchers drive:

```bash
python -m uni_agent.trainer.main_multi_agents_ppo
```

Set at least these environment variables for a real run (the example defaults
point at the bundled mock models and mock data):

```bash
export POLICY_1_MODEL_PATH=/path/to/policy_1_model
export POLICY_2_MODEL_PATH=/path/to/policy_2_model
export TRAIN_DATA=/path/to/train.parquet
export VAL_DATA=/path/to/val.parquet
```

Optional per-policy resources (defaults: 1 node, 2 GPUs/node, TP=2):

```bash
export POLICY_1_N_GPUS_PER_NODE=2
export POLICY_2_N_GPUS_PER_NODE=4
export POLICY_1_ROLLOUT_N_GPUS_PER_NODE=2
export POLICY_2_ROLLOUT_N_GPUS_PER_NODE=4
export POLICY_1_TENSOR_PARALLEL_SIZE=2
export POLICY_2_TENSOR_PARALLEL_SIZE=4
```

For external command-driven training, use `run_external_mas_train.sh` as the
recommended starting point. For the original in-process callable runner, use
`run_e2e_train.sh`. Adapt the per-policy PPO config blocks and the MAS command
configuration for your production system before launching a real training run.

## Config Field Modification Guide

`config/multi_agent_blackbox.yaml` 里的字段分三类，改之前先分清所有权：

### 1. 外层共享字段（改顶层）

这些字段只在 outer trainer 配置中定义，由 `MultiAgentsPPOTrainer` 投影到每个
policy 的 v1 runtime。`algorithm.*`、`reward.*` 和 `trainer.v1` 由
`_compose_policy_ppo_config()` 统一合并；其余共享字段仍按 YAML interpolation
处理。改顶层即可；不需要在 per-policy 配置中重复声明。

| 想改什么 | 改哪里 |
|---|---|
| 训练总步数 | `trainer.total_training_steps` |
| 训练模式 sync/separate_async | `trainer.v1.trainer_mode` |
| 每步同步频率 | `trainer.v1.separate_async.parameter_sync_step` |
| 每步 batch 大小 | `data.train_batch_size` |
| 序列长度上限 | `data.max_prompt_length` / `data.max_response_length` |
| GRPO 算法配置 | `algorithm.*` |
| reward 函数 / worker 数 | `reward.*` |
| rollout 失败处理 | `sampler.*` |
| 每问题 rollout 数 | `actor_rollout_ref.rollout.n` |

注意：`algorithm.*` 尤其不能 per-policy 分叉——合并 advantage 计算只读
`policy_1` 的配置，两个 policy 改成不同的值会静默不一致。reward 同理（当前
框架按共享 reward 设计）。

### 2. per-policy 独立字段（未标注 = 可自由调整）

`policies.*.ppo_trainer_overrides` 里未标注"外层共享"的字段都属于 per-policy
独立（每个 policy 可以不同）：

- `actor_rollout_ref.model.path`（每个 policy 可用不同模型）
- `actor_rollout_ref.rollout.temperature`（per-policy 采样温度；训练侧重算
  log-prob 用同一个值，需保持一致语义）
- `actor_rollout_ref.rollout.tensor_model_parallel_size`
- `actor_rollout_ref.rollout.nnodes` / `n_gpus_per_node`（standalone rollout 资源）
- `actor_rollout_ref.rollout.multi_turn.format`（per-policy 对话格式/工具解析）
- `rollout.prompt_length` / `response_length` / `max_model_len` /
  `gpu_memory_utilization`
- `actor` 的 `lr` / `ppo_mini_batch_size` / `ppo_max_token_len_per_gpu` /
  `fsdp_config.fsdp_size` / `optim.*`
- `trainer.nnodes` / `n_gpus_per_node` / `default_local_dir`
- `checkpoint_engine.engine_kwargs.nccl.group_name`（每 policy 必须唯一）

### 3. 必须由外部提供的字段（`???`）

- `data.train_files` / `data.val_files`
- 每个 policy 的 `model.path`

通过 hydra override 提供，例如：

```bash
policies.policy_1.ppo_trainer_overrides.actor_rollout_ref.model.path=/path/to/model1 \
policies.policy_2.ppo_trainer_overrides.actor_rollout_ref.model.path=/path/to/model2 \
data.train_files=/path/to/train.parquet \
data.val_files=/path/to/val.parquet
```

保持 `???`：强制显式提供，避免环境变量缺失时静默回退到 verl 默认模型。

### 一致性约束（改动时注意）

- `data.train_batch_size == parameter_sync_step(1) * actor.ppo_mini_batch_size`
  （separate_async 断言）；
- `actor.fsdp_config.fsdp_size` 须整除训练总卡数
  （`trainer.nnodes × trainer.n_gpus_per_node`；默认取全部卡数）；
- `rollout.tensor_model_parallel_size` 须与 rollout 卡数匹配；
- `trainer.v1.trainer_mode` 与 `parameter_sync_step` 以 outer 配置为准，旧的
  per-policy 值会被覆盖。

完整字段注释见 `config/multi_agent_blackbox.yaml` 的 `policies:` 段。

### 高 KV 负载动态调度对照

`run_e2e_borrow_verify.sh` 的 `BORROW_LOAD_PROFILE=high_kv` 选择
`multi_agent_blackbox_borrow_high_kv.yaml` 和 `mas_config_borrow_high_kv.yaml`。
沿用 zzh_env 中安装的 verl、Qwen2.5-0.5B-Instruct；训练卡为 4/4/4，
独立推理卡为 4/12/4，TP2、replica 数为 2/6/2。

| 配置 | 高负载值 |
|---|---|
| 每步 prompt × rollout | 256 × 4 |
| MAS 并发上限 | 1024，外部 Ray 任务 SPREAD |
| agent_1/2/3 输出上限 | 10240 / 512 / 4096，ignore_eos=true |
| prompt / 总上下文上限 | 12288 / 24576 |
| 每 replica max_num_seqs | 256 / 256 / 160 |
| max_num_batched_tokens | 4096 |
| gpu_memory_utilization | 0.5 |
| checkpoint 同步分桶 | 128 MiB |
| KV 进入 / 退出 / 借出后预测上限 | 0.55 / 0.30 / 0.40 |
| 调度信号 | 仅 KV；排队指标只观测 |
| 借用拓扑 | 1/2/3 全方向互借，无 `max_units` |

agent_3 输入包含前两个 agent 的回答；`max_tokens` 仅限制其自己的输出，
因此输入预算需另行检查。高负载下 vLLM 临时张量和跨进程权重同步缓冲也会
占用显存；`gpu_memory_utilization` 不是整个进程的运行时显存硬上限。
较低的预算和较小同步分桶用于预留这部分空间。

从仓库根目录运行匹配的各 20 step 对照：

```bash
RAY_ADDRESS=ray-gcs-host:port RAY_DASHBOARD_ADDRESS=http://dashboard-host:port \
/mnt/bn/chenghao1026/resouces/libs/zzh_env/bin/python3 \
  examples/multi_agent_blackbox/scripts/run_borrow_comparison.py \
  --profile high_kv --steps 20 --modes static dynamic \
  --output-dir examples/multi_agent_blackbox/logs/my_high_kv_comparison
```

两组顺序运行，均调用 `run_e2e_borrow_verify.sh`。使用不同输出目录保留校准
和正式运行；负载或显存配置改变后，应重新开始匹配的对照。Ray 集群需使用
本目录脚本 `scripts/start_ray_cluster.sh` 启动。

每组保留训练日志、step 指标、GPU 及 vLLM 采样；GPU/serving 监控另写本机
`/tmp` 备份，结束时恢复到实验目录，以抵御共享文件系统的短暂写入失败。
两组成功完成后生成 `comparison_report.md`、JSON/CSV 数据和对比图。
KV 报告区分整个统计窗口、忙碌 replica 和 KV≥50% 的有效采样占比；
reward 是重复 mock 题上的答案子串命中率，不能作为独立测试集精度。

### 随机路由业务负载 harness

`random_routing_harness.py` 用于替代固定 `agent_1 → agent_2 → agent_3`
串行桩。请求从 `agent_1` 入口进入，入口为每个处理轮次独立随机选择
`agent_1/2/3`；总轮数均匀采样 1–10。每轮只传递原始任务和上一轮输出，
因此多轮交互不会无限累积上下文。seed、样本编号和任务文本共同决定路由与
token budget，使 dynamic/static 两组能够复现相同的工作负载。

默认每轮从 4096–8192 token 中按 512 对齐随机选择 `max_tokens`，并将
`min_tokens` 设为预算的 75%。没有设置 `ignore_eos`，达到随机下限后模型可按
自然 EOS 结束。三个 agent 的路由权重相同；在 2/6/2 replicas 配置下，
policy_1/3 的单 replica 请求压力预期约为 policy_2 的三倍。

运行使用该 harness 的匹配对照：

```bash
RAY_ADDRESS=ray-gcs-host:port RAY_DASHBOARD_ADDRESS=http://dashboard-host:port \
/mnt/bn/chenghao1026/resouces/libs/zzh_env/bin/python3 \
  examples/multi_agent_blackbox/scripts/run_borrow_comparison.py \
  --profile random_routing --steps 10 --modes dynamic static \
  --output-dir examples/multi_agent_blackbox/logs/my_random_routing_comparison
```

可在 `config/mas_config_random_routing.yaml` 中调整轮数、路由权重、token 范围、
对齐步长和最低生成比例。训练入口使用
`config/multi_agent_blackbox_borrow_random_routing.yaml`，其余 GPU、TP、replica、
KV 阈值和高 KV 对照保持一致。
