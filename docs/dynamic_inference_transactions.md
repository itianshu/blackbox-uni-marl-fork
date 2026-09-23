# Dynamic inference ownership transactions

A successful borrow removes the sleeping home replica from the home checkpoint
manager and adds its awake guest to the donor checkpoint manager. Both managers
are persistent; step end invokes each policy hook once and has no guest renewal
transfer. Initial borrow and immediate return clone the published weights from
a same-policy active vLLM replica before the receiving engine accepts requests.
See [vLLM clone details](vllm_replica_clone.md) for implementation and limits.

All topology operations run under BoundaryGate, also held by checkpoint saving
and step-end synchronization. Actor computation does not hold this gate. The lock covers execution, not
just decisions. A boundary waits for a currently running operation to return.
Before policy hooks, membership is checked against active_lends: no duplicates,
no cross-policy ownership, no incomplete return and no failed transaction.

Errors propagate as exceptions. A failed executor retains the original error and
its transaction state; a failed gate rejects all later boundary/poll work.
An actor update already running may finish, but cannot publish weights afterward.
The poll thread stops after a fatal gate error. A local timeout does not prove a
Ray method was canceled. Consequently there is deliberately NO automatic retry,
rollback, alternate-engine wake or best-effort continuation after such an error.
Shutdown kills guest actors and tears down job resources; restart is required.
This is a fail-stop transaction protocol, not crash-persistent recovery.

Abort/sleep/wake timeout defaults to 60 seconds (formerly 10). Checkpoint sync
uses weight_sync_timeout_s (default 300 seconds) across abort, KV release,
prepare/init, transfer, finalize and resume. Ray waits in prepare/init have the
same remaining deadline, and transfer/finalize use awaitable ObjectRefs. On
failure, no finalize/resume is attempted against potentially running kernels.

The installed zzh_env verl is used unchanged. The repository runtime patch
creates a fresh shared NCCL group for each synchronization; rebuild_group=true
must remain enabled on sender and receiver engines. Completion/failure logging
includes step, target addresses, phase and elapsed time. A timeout causes fast
failure, not guaranteed cancellation of distributed GPU work.

Validation must include ownership round trips, injected abort/sleep/wake and
transfer failures, no hooks after a return failure, waiting for an active
operation at a boundary, and an actually bounded transfer timeout. GPU smoke
must additionally confirm changing receiver sets and guest inclusion in regular
policy updates. Successful unit tests alone do not establish NCCL liveness.

The opt-in `examples/multi_agent_blackbox/scripts/run_transaction_smoke.sh`
uses an example-only patch to stop automatic scheduling after initialization.
Before training step 1 it borrows one TP2 unit from policy_2 for policy_1;
before step 2 it returns that unit. Every boundary asserts one call to each
persistent policy checkpoint manager, guest inclusion during the first update,
and guest exclusion after return. `TRANSACTION_SMOKE passed` is printed only
after both real training steps succeed. This test validates execution and
membership, not the effectiveness of KV-based scheduling thresholds.
The smoke script keeps the 4/4/4 training and 4/12/4 inference GPU layout,
but pre-creates only the tested policy_2-to-policy_1 guest direction to reduce
startup time. Ordinary experiment configs retain their full borrowing graph.
