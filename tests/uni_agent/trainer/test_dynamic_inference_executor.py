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
        self._dynamic_published_version = (0, 1)
        for replica in self.replicas:
            replica._dynamic_weight_version = (0, 1)

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


@pytest.fixture()
def clone_stub(monkeypatch):
    from uni_agent.trainer.dynamic_inference import replica_clone
    state = types.SimpleNamespace(instances=[], fail_next=False, log=None)
    async def clone(source, target, **kwargs):
        state.instances.append((source, target, kwargs))
        if state.log is not None:
            state.log.append(("clone", source.server_address, target.server_address))
        if state.fail_next:
            state.fail_next = False
            raise RuntimeError("clone failed")
        target._dynamic_weight_version = kwargs['version']
        return {'source': source.server_address, 'target': target.server_address}
    monkeypatch.setattr(replica_clone, 'clone_replica', clone)
    return state


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

        self.config = parse_scheduling_config({"enable": True})
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



# Transaction contract: membership is committed once, and failures are fatal.
def borrow(env):
    return env.executor.borrow(BorrowPlan(home_policy="b", donor="a",
        home_replicas=[env.handles["b"].replicas[0]]), step=7)


def test_membership_roundtrip(clone_stub):
    env = Env()
    lend = borrow(env)
    assert lend.guests[0] in env.handles["a"].standalone_checkpoint_manager.replicas
    assert lend.home_replicas[0] not in env.handles["b"].standalone_checkpoint_manager.replicas
    env.executor.validate_membership()
    env.executor.return_(lend, 8)
    assert lend.guests[0] not in env.handles["a"].standalone_checkpoint_manager.replicas
    assert lend.home_replicas[0] in env.handles["b"].standalone_checkpoint_manager.replicas
    assert not env.scheduler.active_lends
    env.executor.validate_membership()
    assert len(clone_stub.instances) == 2  # initial guest + home refresh only


@pytest.mark.parametrize("operation", ["abort_all_requests", "sleep", "wake_up"])
def test_borrow_failure_stops_transaction(clone_stub, operation):
    env = Env()
    target = (env.guest_engine._guest_map[("b", "a")][0] if operation == "wake_up"
              else env.handles["b"].replicas[0])
    target._fail[operation] = TimeoutError("injected")
    with pytest.raises(RuntimeError, match="TimeoutError"):
        borrow(env)
    before = list(env.log)
    with pytest.raises(RuntimeError, match="restart required"):
        env.executor.validate_membership()
    with pytest.raises(RuntimeError):
        borrow(env)
    assert env.log == before
    assert not env.scheduler.active_lends


@pytest.mark.parametrize("operation,stage", [("abort_all_requests", 1), ("sleep", 2), ("wake_up", 3)])
def test_return_failure_never_advances_or_retries_unknown_work(clone_stub, operation, stage):
    env = Env()
    lend = borrow(env)
    target = lend.home_replicas[0] if operation == "wake_up" else lend.guests[0]
    target._fail[operation] = TimeoutError("injected")
    with pytest.raises(RuntimeError, match="TimeoutError"):
        env.executor.return_(lend, 8)
    assert lend.return_stage == stage
    assert lend in env.scheduler.active_lends
    before = list(env.log)
    with pytest.raises(RuntimeError, match="restart required"):
        env.executor.return_(lend, 9)
    assert env.log == before


def test_sync_failure_does_not_wake_home_for_rollback(clone_stub):
    env = Env()
    clone_stub.fail_next = True
    with pytest.raises(RuntimeError, match="clone failed"):
        borrow(env)
    assert "wake:b:0" not in env.ops()
    assert env.executor.failure is not None


def test_duplicate_membership_rejected(clone_stub):
    env = Env()
    lend = borrow(env)
    env.handles["a"].standalone_checkpoint_manager.replicas.extend(lend.guests)
    with pytest.raises(RuntimeError, match="Duplicate"):
        env.executor.validate_membership()


def test_add_is_idempotent(clone_stub):
    env = Env()
    manager = env.handles["a"].standalone_checkpoint_manager
    env.executor._add_replicas(manager, list(manager.replicas))
    assert len(manager.replicas) == 2


def test_borrow_return_order_keeps_home_and_guest_exclusive(clone_stub):
    env = Env()
    clone_stub.log = env.log
    lend = borrow(env)
    ops = env.ops()
    assert ops.index('sleep:b:0') < ops.index('wake:guest-ba:0')
    assert ops.index('scm:a:add') < ops.index('lb:a:add')
    env.log.clear()
    env.executor.return_(lend, 8)
    ops = env.ops()
    assert ops.index('lb:a:remove') < ops.index('abort:guest-ba:0')
    assert ops.index('scm:a:remove') < ops.index('sleep:guest-ba:0')
    assert ops.index('sleep:guest-ba:0') < ops.index('wake:b:0')
    assert ops.index('scm:b:add') < ops.index('lb:b:add')


def test_boundary_uses_donor_manager_once_with_guest(clone_stub):
    from concurrent.futures import ThreadPoolExecutor
    from uni_agent.trainer.dynamic_inference.controller import DynamicInferenceController, BoundaryGate
    env = Env()
    lend = borrow(env)
    syncs = []
    trainer = types.SimpleNamespace(global_steps=8, timing_raw={}, policy_trainers={})
    for policy, handle in env.handles.items():
        trainer.policy_trainers[policy] = types.SimpleNamespace(on_step_end=
            lambda policy=policy, handle=handle: syncs.append((policy,
                tuple(handle.standalone_checkpoint_manager.replicas))))
    controller = DynamicInferenceController(env.config, trainer)
    controller.executor = env.executor
    controller.scheduler = env.scheduler
    controller.gate = BoundaryGate()
    with ThreadPoolExecutor(max_workers=2) as pool:
        trainer._policy_pool = pool
        controller.run_boundary(trainer)
    assert len(syncs) == 2
    assert lend.guests[0] in dict(syncs)['a']
    assert lend.home_replicas[0] not in dict(syncs)['b']
    assert len(clone_stub.instances) == 1  # no additional guest-only transfer


def test_last_seed_cannot_be_lent(clone_stub):
    env = Env()
    unit = list(env.handles['b'].replicas)
    with pytest.raises(RuntimeError, match='last published seed'):
        env.executor.borrow(BorrowPlan('b', 'a', unit), 7)
    assert not env.log


def test_stale_source_is_rejected(clone_stub):
    env = Env()
    for r in env.handles['a'].replicas:
        r._dynamic_weight_version = (0, 0)
    with pytest.raises(RuntimeError, match='No verified published'):
        borrow(env)
    assert not clone_stub.instances
    assert 'lb:a:add' not in env.ops()


def test_incoming_guest_does_not_replace_home_seed(clone_stub):
    env = Env()
    env.handles['b'].standalone_checkpoint_manager.replicas.append(
        FakeReplica(env.log, 'incoming', 9))
    with pytest.raises(RuntimeError, match='last published seed'):
        env.executor.borrow(BorrowPlan('b', 'a', list(env.handles['b'].replicas)), 7)
    assert not env.log
