"""Tests for the physical borrow/return sequences.

All verl/Ray touchpoints are fakes with a shared operation log, so the tests
assert the exact choreography (ordering invariants I1-I6) without a cluster.
"""

import sys
import types

import pytest

from uni_agent.trainer.dynamic_inference.executor import BoundaryExecutor
from uni_agent.trainer.dynamic_inference.scheduler import MultiPolicyInferenceScheduler
from uni_agent.trainer.dynamic_inference.signals import SignalStore
from uni_agent.trainer.dynamic_inference.types import (
    BorrowPlan,
    GuestUnit,
    PolicyInferenceHandles,
)


# --------------------------------------------------------------------- fakes
class Ref:
    """Stand-in for a Ray ObjectRef."""

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class RemoteMethod:
    def __init__(self, log, name, fn):
        self._log = log
        self._name = name
        self._fn = fn

    def remote(self, *args, **kwargs):
        result = self._fn(*args, **kwargs)
        self._log.append((self._name, result))
        return Ref(result)


class FakeLB:
    def __init__(self, log, tag):
        self._log = log
        self._tag = tag
        self.remove_servers = RemoteMethod(log, f"lb:{tag}:remove", lambda addrs: list(addrs))
        self.add_servers = RemoteMethod(log, f"lb:{tag}:add", lambda servers: dict(servers))


class FakeReplica:
    def __init__(self, log, name, rank, fail=None):
        self._log = log
        self.name = name
        self.replica_rank = rank
        self.server_address = f"{name}:{rank}"
        self.server_handle = f"h:{name}:{rank}"
        self.world_size = 8
        self.nnodes = 1
        self._fail = fail or {}

    async def abort_all_requests(self):
        if "abort_all_requests" in self._fail:
            raise self._fail["abort_all_requests"]
        self._log.append((f"abort:{self.server_address}",))

    async def sleep(self):
        if "sleep" in self._fail:
            raise self._fail["sleep"]
        self._log.append((f"sleep:{self.server_address}",))

    async def wake_up(self):
        if "wake_up" in self._fail:
            raise self._fail["wake_up"]
        self._log.append((f"wake:{self.server_address}",))


class FakeSCM:
    def __init__(self, log, tag, replicas):
        self._log = log
        self._tag = tag
        self.replicas = list(replicas)
        self.config = f"scm-config:{tag}"

    def remove_replicas(self, replicas):
        self._log.append((f"scm:{self._tag}:remove", tuple(r.server_address for r in replicas)))
        removed = set(id(r) for r in replicas)
        self.replicas = [r for r in self.replicas if id(r) not in removed]

    def add_replicas(self, replicas):
        self._log.append((f"scm:{self._tag}:add", tuple(r.server_address for r in replicas)))
        self.replicas.extend(replicas)


class FakeGuestEngine:
    def __init__(self, log, guest_map):
        self._log = log
        self._guest_map = guest_map

    async def guests_for(self, home, donor, home_replicas):
        self._log.append(("guests_for", home, donor,
                          tuple(r.server_address for r in home_replicas)))
        return self._guest_map[(home, donor)]

    def release(self, home, donor, home_replicas):
        self._log.append(("guest_release", home, donor))


class FakePushManager:
    """Injected as verl.checkpoint_engine.base.CheckpointEngineManager."""

    instances = []
    fail_next = False
    log = None  # set by the test to the shared op log for ordering asserts

    def __init__(self, *, config, actor_wg, replicas):
        self.config = config
        self.actor_wg = actor_wg
        self.replicas = list(replicas)
        self.update_calls = []
        FakePushManager.instances.append(self)

    async def update_weights(self, global_steps=None):
        self.update_calls.append(global_steps)
        if FakePushManager.log is not None:
            FakePushManager.log.append(("push", self.actor_wg, global_steps))
        if FakePushManager.fail_next:
            FakePushManager.fail_next = False
            raise RuntimeError("push failed")


@pytest.fixture()
def push_stub(monkeypatch):
    module = types.ModuleType("verl.checkpoint_engine.base")
    module.CheckpointEngineManager = FakePushManager
    FakePushManager.instances = []
    FakePushManager.fail_next = False
    FakePushManager.log = None
    monkeypatch.setitem(sys.modules, "verl.checkpoint_engine.base", module)
    return FakePushManager


class Registry:
    """Minimal guest-unit registry for the scheduler hooks."""

    def __init__(self):
        self.units = {}

    def unit_for(self, home, donor, replica):
        key = (home, donor, id(replica))
        if key not in self.units:
            self.units[key] = GuestUnit(home_policy=home, donor=donor, home_replicas=[replica])
        return self.units[key]


class Env:
    """One borrowable world: two policies, fakes wired to a shared op log."""

    def __init__(self):
        self.log = []
        self.handles = {}
        for name in ("a", "b"):
            replicas = [FakeReplica(self.log, name, i) for i in range(2)]
            self.handles[name] = PolicyInferenceHandles(
                actor_rollout_wg=f"actor_wg:{name}",
                standalone_checkpoint_manager=FakeSCM(self.log, name, replicas),
                lb_handle=FakeLB(self.log, name),
                rollout_config=None,
                model_config=None,
                replicas=replicas,
                cards_per_replica=8,
                nnodes=1,
            )
        guests = {
            ("b", "a"): [FakeReplica(self.log, "guest-ba", 0)],
            ("a", "b"): [FakeReplica(self.log, "guest-ab", 0)],
        }
        self.guest_engine = FakeGuestEngine(self.log, guests)
        from uni_agent.trainer.dynamic_inference.types import parse_scheduling_config

        self.config = parse_scheduling_config({"enable": True, "mode": "resource"})
        self.scheduler = MultiPolicyInferenceScheduler(
            config=self.config, handles=self.handles,
            signals=SignalStore(["a", "b"], max_num_seqs={"a": 32, "b": 32}),
            guest_units=Registry(),
        )
        self.executor = BoundaryExecutor(
            config=self.config, handles=self.handles,
            guest_engine=self.guest_engine, scheduler=self.scheduler,
        )

    def ops(self):
        return [entry[0] for entry in self.log]


# ---------------------------------------------------------------------- tests
class TestBorrow:
    def test_borrow_full_sequence(self, push_stub):
        env = Env()
        push_stub.log = env.log
        home = env.handles["b"].replicas[0]
        plan = BorrowPlan(home_policy="b", donor="a", home_replicas=[home])
        lend = env.executor.borrow(plan, step=7)

        assert lend is not None
        # exact choreography: I6 (LB removal first), I3 (home asleep before
        # guest wakes), I5 (push before the donor LB add)
        assert env.log == [
            ("lb:b:remove", ["b:0"]),
            ("abort:b:0",),
            ("scm:b:remove", ("b:0",)),
            ("sleep:b:0",),
            ("guests_for", "b", "a", ("b:0",)),
            ("wake:guest-ba:0",),
            ("push", "actor_wg:a", 7),
            ("lb:a:add", {"guest-ba:0": "h:guest-ba:0"}),
        ]

        # scheduler bookkeeping
        assert env.scheduler.active_lends == [lend]
        # home scm no longer holds the replica; donor scm never gained it (I1)
        assert home not in env.handles["b"].standalone_checkpoint_manager.replicas
        assert all(r.server_address != "guest-ba:0"
                   for p in env.handles.values()
                   for r in p.standalone_checkpoint_manager.replicas)

    def test_push_failure_rolls_back(self, push_stub):
        env = Env()
        home = env.handles["b"].replicas[0]
        plan = BorrowPlan(home_policy="b", donor="a", home_replicas=[home])
        push_stub.fail_next = True

        lend = env.executor.borrow(plan, step=7)
        assert lend is None
        assert env.scheduler.active_lends == []
        ops = env.ops()
        # rollback: guest sleep -> guest release -> home wake -> scm add -> lb add
        assert ops.index("sleep:guest-ba:0") < ops.index("wake:b:0") \
            < ops.index("scm:b:add") < ops.index("lb:b:add")
        # home fully restored
        assert home in env.handles["b"].standalone_checkpoint_manager.replicas
        # pair blocked for a long window
        assert env.scheduler.pair_block_until[("b", "a")] > 7

    def test_partial_home_sleep_failure_wakes_entire_unit(self, push_stub):
        env = Env()
        homes = env.handles["b"].replicas
        homes[1]._fail["sleep"] = RuntimeError("sleep failed")
        plan = BorrowPlan(home_policy="b", donor="a", home_replicas=homes)

        assert env.executor.borrow(plan, step=7) is None
        ops = env.ops()
        assert "wake:b:0" in ops
        assert "wake:b:1" in ops
        assert all(home in env.handles["b"].standalone_checkpoint_manager.replicas
                   for home in homes)

    def test_guest_wake_failure_releases_reserved_unit(self, push_stub):
        env = Env()
        guest = env.guest_engine._guest_map[("b", "a")][0]
        guest._fail["wake_up"] = RuntimeError("wake failed")
        home = env.handles["b"].replicas[0]
        plan = BorrowPlan(home_policy="b", donor="a", home_replicas=[home])

        assert env.executor.borrow(plan, step=7) is None
        ops = env.ops()
        assert "sleep:guest-ba:0" in ops
        assert "guest_release" in ops
        assert "wake:b:0" in ops
        assert home in env.handles["b"].standalone_checkpoint_manager.replicas

    def test_repeated_failures_disable_scheduling(self, push_stub):
        env = Env()
        plan = BorrowPlan(home_policy="b", donor="a",
                          home_replicas=[env.handles["b"].replicas[0]])
        for _ in range(2):
            push_stub.fail_next = True
            env.executor.borrow(plan, step=7)
        assert env.scheduler.disabled is True


class TestReturn:
    def _lend(self, env):
        home = env.handles["b"].replicas[0]
        guest = env.guest_engine._guest_map[("b", "a")][0]
        from uni_agent.trainer.dynamic_inference.types import LendRecord

        lend = LendRecord(lend_id=1, home_policy="b", donor="a",
                          home_replicas=[home], guests=[guest], since_step=7)
        env.scheduler.note_borrow_started(lend)
        return lend

    def test_return_full_sequence(self, push_stub):
        env = Env()
        lend = self._lend(env)
        assert env.executor.return_(lend, step=8) is True

        ops = env.ops()
        # I6: donor LB removal first; guests asleep before home wakes (I3).
        # A directed home push occurs before SCM/LB re-attachment; the fake
        # push logger is disabled here, so only the surrounding operations show.
        assert ops == [
            "lb:a:remove",
            "abort:guest-ba:0",
            "sleep:guest-ba:0",
            "wake:b:0",
            "scm:b:add",
            "lb:b:add",
            "guest_release",
        ]
        assert len(push_stub.instances) == 1
        assert push_stub.instances[0].actor_wg == "actor_wg:b"
        assert env.scheduler.active_lends == []
        assert lend.home_replicas[0] in env.handles["b"].standalone_checkpoint_manager.replicas

    def test_early_return_pushes_home_weights(self, push_stub):
        env = Env()
        push_stub.log = env.log
        lend = self._lend(env)
        assert env.executor.return_(lend, step=8, early=True) is True
        # exactly one push, from the home policy's actor to the home replicas,
        # after the home wake and before the home scm/LB re-attach
        ops = env.ops()
        assert ops == [
            "lb:a:remove", "abort:guest-ba:0", "sleep:guest-ba:0", "wake:b:0",
            "push", "scm:b:add", "lb:b:add", "guest_release",
        ]
        push = push_stub.instances[0]
        assert push.actor_wg == "actor_wg:b"
        assert [r.server_address for r in push.replicas] == ["b:0"]
        assert push.update_calls == [8]

    def test_direct_handoff_skips_home_wake_and_weight_push(self, push_stub):
        env = Env()
        push_stub.log = env.log
        lend = self._lend(env)
        home = lend.home_replicas[0]
        env.handles["b"].standalone_checkpoint_manager.remove_replicas([home])
        env.log.clear()

        assert env.executor.return_(
            lend, step=8, reactivate_home=False,
        ) is True
        replacement = env.executor.borrow(
            BorrowPlan(home_policy="b", donor="a", home_replicas=[home]),
            step=8,
            home_already_sleeping=True,
        )

        assert replacement is not None
        assert env.ops() == [
            "lb:a:remove", "abort:guest-ba:0", "sleep:guest-ba:0",
            "guest_release", "guests_for", "wake:guest-ba:0", "push", "lb:a:add",
        ]
        assert "wake:b:0" not in env.ops()
        assert all(instance.actor_wg != "actor_wg:b" for instance in push_stub.instances)

    def test_failed_direct_handoff_refreshes_home_during_rollback(self, push_stub):
        env = Env()
        push_stub.log = env.log
        lend = self._lend(env)
        home = lend.home_replicas[0]
        env.handles["b"].standalone_checkpoint_manager.remove_replicas([home])
        assert env.executor.return_(lend, step=8, reactivate_home=False) is True
        env.log.clear()
        push_stub.fail_next = True

        replacement = env.executor.borrow(
            BorrowPlan(home_policy="b", donor="a", home_replicas=[home]),
            step=8,
            home_already_sleeping=True,
        )

        assert replacement is None
        ops = env.ops()
        push_indices = [index for index, op in enumerate(ops) if op == "push"]
        assert len(push_indices) == 2
        assert ops.index("wake:b:0") < push_indices[-1] \
            < ops.index("scm:b:add") < ops.index("lb:b:add")
        home_pushes = [
            instance for instance in push_stub.instances
            if instance.actor_wg == "actor_wg:b"
        ]
        assert len(home_pushes) == 1
        assert home in env.handles["b"].standalone_checkpoint_manager.replicas

    def test_renew_only_pushes_latest_donor_weights(self, push_stub):
        env = Env()
        push_stub.log = env.log
        lend = self._lend(env)
        env.log.clear()

        assert env.executor.renew(lend, step=8) is True
        assert env.log == [("push", "actor_wg:a", 8)]
        assert env.scheduler.active_lends == [lend]
        assert env.scheduler.metrics_snapshot()["renewals"] == 1

    def test_renewal_batches_all_units_for_the_same_donor(self, push_stub):
        env = Env()
        from uni_agent.trainer.dynamic_inference.types import LendRecord

        first = self._lend(env)
        second = LendRecord(
            lend_id=2, home_policy="b", donor="a",
            home_replicas=[env.handles["b"].replicas[1]],
            guests=[FakeReplica(env.log, "guest-ba", 1)], since_step=7,
        )
        env.scheduler.note_borrow_started(second)

        assert env.executor.renew_many([first, second], step=8) is True
        assert len(push_stub.instances) == 1
        assert [r.server_address for r in push_stub.instances[0].replicas] == [
            "guest-ba:0", "guest-ba:1",
        ]
        assert env.scheduler.metrics_snapshot()["renewals"] == 2

    def test_failed_renewal_keeps_lend_active_for_safe_return(self, push_stub):
        env = Env()
        lend = self._lend(env)
        push_stub.fail_next = True

        assert env.executor.renew(lend, step=8) is False
        assert env.scheduler.active_lends == [lend]
        assert env.scheduler.pair_block_until[("b", "a")] > 8

    def test_one_failed_batch_counts_once_for_multiple_lends_on_same_pair(self, push_stub):
        env = Env()
        first = self._lend(env)
        from uni_agent.trainer.dynamic_inference.types import LendRecord

        second = LendRecord(
            lend_id=2, home_policy="b", donor="a",
            home_replicas=[env.handles["b"].replicas[1]],
            guests=[FakeReplica(env.log, "guest-ba", 1)], since_step=7,
        )
        env.scheduler.note_borrow_started(second)
        push_stub.fail_next = True

        assert env.executor.renew_many([first, second], step=8) is False
        assert env.executor._pair_failures[("b", "a")] == 1
        assert env.scheduler.disabled is False

    def test_successful_return_clears_previous_pair_failure(self, push_stub):
        env = Env()
        lend = self._lend(env)
        env.executor._pair_failures[("b", "a")] = 1

        assert env.executor.return_(lend, step=8) is True
        assert ("b", "a") not in env.executor._pair_failures

    def test_lb_add_failure_retries_without_duplicate_scm_entry(self, push_stub):
        env = Env()
        lend = self._lend(env)
        calls = 0

        class FlakyAdd:
            def remote(self, servers):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("LB add failed")
                return Ref(dict(servers))

        env.handles["b"].lb_handle.add_servers = FlakyAdd()
        assert env.executor.return_(lend, step=8) is False
        assert lend.return_stage == 6
        scm = env.handles["b"].standalone_checkpoint_manager
        assert sum(r is lend.home_replicas[0] for r in scm.replicas) == 2

        assert env.executor.return_(lend, step=9) is True
        assert sum(r is lend.home_replicas[0] for r in scm.replicas) == 2

    def test_partial_failure_resumes_from_recorded_stage(self, push_stub):
        env = Env()
        lend = self._lend(env)
        lend.guests[0]._fail["sleep"] = RuntimeError("guest sleep hung")

        assert env.executor.return_(lend, step=8) is False
        assert lend.return_stage == 2  # LB removed + aborted, sleep pending
        assert env.scheduler.active_lends == [lend]  # still active

        # the guest recovers; the retry must NOT redo stages 1-2
        del lend.guests[0]._fail["sleep"]
        env.log.clear()
        assert env.executor.return_(lend, step=9) is True
        assert env.ops() == [
            "sleep:guest-ba:0",
            "wake:b:0",
            "scm:b:add",
            "lb:b:add",
            "guest_release",
        ]
