"""Tests for guest-replica pre-creation and lifecycle."""

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from uni_agent.trainer.dynamic_inference import guest_engine as ge
from uni_agent.trainer.dynamic_inference.guest_engine import GuestEngineManager
from uni_agent.trainer.dynamic_inference.types import (
    BorrowPairSpec,
    PolicyInferenceHandles,
    parse_scheduling_config,
)


# --------------------------------------------------------------------- fakes
class FakeHomeReplica:
    def __init__(self, log, policy, rank, *, world_size=8, node_id=None):
        self._log = log
        self.policy = policy
        self.replica_rank = rank
        self.world_size = world_size
        self.nnodes = 1
        self.resource_pool = f"pool:{policy}:{rank}"
        if node_id is not None:
            self.node_ids = [node_id] * world_size

    async def sleep(self):
        self._log.append((f"home_sleep:{self.policy}:{self.replica_rank}",))

    async def wake_up(self):
        self._log.append((f"home_wake:{self.policy}:{self.replica_rank}",))


class FakeGuestReplica:
    """Constructed by the stubbed get_rollout_replica_class."""

    created = []

    def __init__(self, *, replica_rank, config, model_config, gpus_per_node, name_suffix=""):
        self.kwargs = dict(replica_rank=replica_rank, config=config,
                           model_config=model_config, gpus_per_node=gpus_per_node,
                           name_suffix=name_suffix)
        self.replica_rank = replica_rank
        self.config = config
        self.gpus_per_node = gpus_per_node
        self.name_suffix = name_suffix
        self.workers = []
        self.servers = []
        FakeGuestReplica.created.append(self)

    async def init_standalone(self):
        self._log.append((f"guest_init:{self.replica_rank}",))

    async def sleep(self):
        self._log.append((f"guest_sleep:{self.replica_rank}",))

    async def wake_up(self):
        self._log.append((f"guest_wake:{self.replica_rank}",))


class Env:
    def __init__(self, monkeypatch, *, policies=("a", "b")):
        self.log = []
        FakeGuestReplica.created = []
        # bind the shared log onto the guest class (set in __init__ path)
        FakeGuestReplica._log = self.log

        replica_mod = types.ModuleType("verl.workers.rollout.replica")
        replica_mod.get_rollout_replica_class = lambda name: FakeGuestReplica
        monkeypatch.setitem(sys.modules, "verl.workers.rollout.replica", replica_mod)

        self.handles = {}
        for name in policies:
            replicas = [FakeHomeReplica(self.log, name, i) for i in range(2)]
            rollout = SimpleNamespace(
                name="vllm", load_format="safetensors", n_gpus_per_node=8,
                max_num_seqs=32,
            )
            self.handles[name] = PolicyInferenceHandles(
                actor_rollout_wg=None,
                standalone_checkpoint_manager=None,
                lb_handle=None, rollout_config=rollout, model_config=f"model:{name}",
                replicas=replicas, cards_per_replica=8, nnodes=1, max_num_seqs=32,
            )

        config = parse_scheduling_config({
            "enable": True, "mode": "resource",
        })
        self.config = config
        pairs = [BorrowPairSpec(home="a", donor="b"), BorrowPairSpec(home="b", donor="a")]
        self.manager = GuestEngineManager(
            config=config, handles=self.handles, pairs=pairs,
        )
        # real level-2 sleeps take seconds; the fakes return instantly, so the
        # probe threshold must be zeroed for the happy paths
        monkeypatch.setattr(ge, "_SLEEP_PROBE_MIN_S", 0.0)

    def ops(self):
        return [entry[0] for entry in self.log]


@pytest.fixture()
def env(monkeypatch):
    return Env(monkeypatch)


class TestPrecreate:
    def test_transition_order_and_unit_registry(self, env):
        count = asyncio.run(env.manager.precreate_all())
        # 2 pairs x 2 home replicas = 4 guests
        assert count == 4
        assert len(env.manager.units) == 4
        # per pair the four-step transition: home sleep -> guest init ->
        # guest sleep -> home wake
        ops = env.ops()
        assert ops[:4] == [
            "home_sleep:a:0", "guest_init:10000", "guest_sleep:10000", "home_wake:a:0",
        ]
        assert ops[4:8] == [
            "home_sleep:a:1", "guest_init:10001", "guest_sleep:10001", "home_wake:a:1",
        ]

    def test_guest_config_isolated_from_donor(self, env):
        asyncio.run(env.manager.precreate_all())
        guests = FakeGuestReplica.created
        # ranks are offset and unique
        assert [g.replica_rank for g in guests] == [10000, 10001, 10002, 10003]
        # donor rollout config was deep-copied and forced to dummy weights
        donor_rollout = env.handles["b"].rollout_config
        for g in guests[:2]:  # guests for pair a->b use b's architecture
            assert g.kwargs["model_config"] == "model:b"
            assert g.config is not donor_rollout
            assert g.config.load_format == "dummy"
            assert donor_rollout.load_format == "safetensors"  # donor untouched
            assert "guest_a_" in g.name_suffix

    def test_guest_bound_to_home_pool(self, env):
        asyncio.run(env.manager.precreate_all())
        for unit in env.manager.units:
            assert unit.guests[0]._guest_external_pool == unit.home_replicas[0].resource_pool
            assert unit.in_use is False

    def test_sleep_probe_fails_fast_when_patch_missing(self, monkeypatch):
        env = Env(monkeypatch)
        # restore the real threshold: the fake sleep returns in ~0ms, which is
        # exactly the unpatched-no-op signature the probe must catch
        monkeypatch.setattr(ge, "_SLEEP_PROBE_MIN_S", 0.5)
        with pytest.raises(RuntimeError, match="sleep\\(\\) returned in"):
            asyncio.run(env.manager.precreate_all())
        assert "home_wake:a:0" in env.ops()

    def test_init_failure_restores_home(self, env, monkeypatch):
        async def fail_init(self):
            self._log.append((f"guest_init:{self.replica_rank}",))
            raise RuntimeError("init failed")

        monkeypatch.setattr(FakeGuestReplica, "init_standalone", fail_init)
        with pytest.raises(RuntimeError, match="init failed"):
            asyncio.run(env.manager.precreate_all())
        assert env.ops()[:3] == [
            "home_sleep:a:0", "guest_init:10000", "home_wake:a:0",
        ]

    def test_heterogeneous_aggregate_two_8_card_homes_into_one_16_card_guest(self, monkeypatch):
        env = Env(monkeypatch)
        env.handles["a"].replicas.append(FakeHomeReplica(env.log, "a", 2))
        env.handles["b"].cards_per_replica = 16
        env.handles["b"].nnodes = 2

        base_mod = types.ModuleType("verl.single_controller.ray.base")
        base_mod.merge_resource_pool = lambda left, right: ("merged", left, right)
        monkeypatch.setitem(sys.modules, "verl.single_controller.ray.base", base_mod)

        pair = BorrowPairSpec(
            home="a", donor="b", home_replicas_per_unit=2,
            guest_replicas_per_unit=1,
        )
        manager = GuestEngineManager(env.config, env.handles, [pair])
        count = asyncio.run(manager.precreate_all())

        assert count == 1
        unit = manager.units[0]
        assert [r.replica_rank for r in unit.home_replicas] == [0, 1]
        assert len(unit.guests) == 1
        assert unit.guests[0]._guest_external_pool == (
            "merged", "pool:a:0", "pool:a:1")

    def test_heterogeneous_fold_prefers_same_node_over_replica_order(self, monkeypatch):
        env = Env(monkeypatch)
        env.handles["a"].replicas = [
            FakeHomeReplica(env.log, "a", 0, world_size=4, node_id="node-a"),
            FakeHomeReplica(env.log, "a", 1, world_size=4, node_id="node-b"),
            FakeHomeReplica(env.log, "a", 2, world_size=4, node_id="node-a"),
            FakeHomeReplica(env.log, "a", 3, world_size=4, node_id="node-b"),
        ]
        env.handles["a"].cards_per_replica = 4

        base_mod = types.ModuleType("verl.single_controller.ray.base")
        base_mod.merge_resource_pool = lambda left, right: ("merged", left, right)
        monkeypatch.setitem(sys.modules, "verl.single_controller.ray.base", base_mod)

        pair = BorrowPairSpec(
            home="a", donor="b", home_replicas_per_unit=2,
            guest_replicas_per_unit=1,
        )
        manager = GuestEngineManager(env.config, env.handles, [pair])
        assert asyncio.run(manager.precreate_all()) == 2
        assert [
            [replica.replica_rank for replica in unit.home_replicas]
            for unit in manager.units
        ] == [[0, 2], [1, 3]]

    def test_heterogeneous_fold_rejects_cross_node_only_groups(self, monkeypatch):
        env = Env(monkeypatch)
        env.handles["a"].replicas = [
            FakeHomeReplica(env.log, "a", rank, world_size=4, node_id=f"node-{rank}")
            for rank in range(3)
        ]
        env.handles["a"].cards_per_replica = 4
        pair = BorrowPairSpec(
            home="a", donor="b", home_replicas_per_unit=2,
            guest_replicas_per_unit=1,
        )
        manager = GuestEngineManager(env.config, env.handles, [pair])
        with pytest.raises(ValueError, match="no node-local group"):
            asyncio.run(manager.precreate_all())

    def test_heterogeneous_split_one_16_card_home_into_two_8_card_guests(self, monkeypatch):
        env = Env(monkeypatch)
        env.handles["a"].cards_per_replica = 16
        env.handles["a"].nnodes = 2

        base_mod = types.ModuleType("verl.single_controller.ray.base")
        base_mod.split_resource_pool = lambda pool, sizes: [
            ("split", pool, idx, size) for idx, size in enumerate(sizes)
        ]
        monkeypatch.setitem(sys.modules, "verl.single_controller.ray.base", base_mod)

        pair = BorrowPairSpec(
            home="a", donor="b", home_replicas_per_unit=1,
            guest_replicas_per_unit=2,
        )
        manager = GuestEngineManager(env.config, env.handles, [pair])
        count = asyncio.run(manager.precreate_all())

        assert count == 2
        assert all(len(unit.guests) == 2 for unit in manager.units)
        assert [g._guest_external_pool for g in manager.units[0].guests] == [
            ("split", "pool:a:0", 0, 8),
            ("split", "pool:a:0", 1, 8),
        ]


class TestGuestsFor:
    def test_precreated_units_marked_in_use(self, env):
        asyncio.run(env.manager.precreate_all())
        home = env.handles["a"].replicas[0]
        guests = asyncio.run(env.manager.guests_for("a", "b", [home]))
        assert len(guests) == 1
        unit = env.manager.unit_for("a", "b", [home])
        assert unit.in_use is True
        # re-entrancy guard: the same unit cannot serve two lends
        with pytest.raises(RuntimeError, match="already lent out"):
            asyncio.run(env.manager.guests_for("a", "b", [home]))
        env.manager.release("a", "b", [home])
        assert unit.in_use is False

    def test_missing_precreated_unit_raises(self, env):
        home = env.handles["a"].replicas[0]
        with pytest.raises(RuntimeError, match="precreate_all"):
            asyncio.run(env.manager.guests_for("a", "b", [home]))

    def test_different_donor_slots_cannot_wake_on_same_home_replica(self, monkeypatch):
        env = Env(monkeypatch, policies=("a", "b", "c"))
        manager = GuestEngineManager(env.config, env.handles, [
            BorrowPairSpec(home="a", donor="b"),
            BorrowPairSpec(home="a", donor="c"),
        ])
        asyncio.run(manager.precreate_all())
        home = env.handles["a"].replicas[0]

        asyncio.run(manager.guests_for("a", "b", [home]))
        with pytest.raises(RuntimeError, match="cannot also activate a->c"):
            asyncio.run(manager.guests_for("a", "c", [home]))

        manager.release("a", "b", [home])
        assert len(asyncio.run(manager.guests_for("a", "c", [home]))) == 1

class TestKillAll:
    def test_kills_every_guest_actor(self, env, monkeypatch):
        asyncio.run(env.manager.precreate_all())
        killed = []
        ray_stub = types.SimpleNamespace(kill=lambda actor, no_restart=False: killed.append(actor))
        monkeypatch.setitem(sys.modules, "ray", ray_stub)
        guest = FakeGuestReplica.created[0]
        guest.workers = ["w1", "w2"]
        guest.servers = ["s1"]

        env.manager.kill_all()
        assert killed == ["s1", "w1", "w2"]
        assert env.manager.units == []
        # unit_for no longer resolves after the clear
        assert env.manager.unit_for("a", "b", [env.handles["a"].replicas[0]]) is None

    def test_kill_failure_is_isolated(self, env, monkeypatch):
        asyncio.run(env.manager.precreate_all())

        def bad_kill(actor, no_restart=False):
            raise RuntimeError("already dead")

        monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(kill=bad_kill))
        env.manager.kill_all()  # must not raise
        assert env.manager.units == []
