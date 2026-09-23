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
        self.node_ids = [node_id or f"node-{policy}-{rank}"] * world_size

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
            "enable": True,
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
    def test_disjoint_home_replicas_initialize_in_parallel_and_shared_ones_serialize(
        self, monkeypatch,
    ):
        env = Env(monkeypatch, policies=("a", "b", "c"))
        env.manager.pairs = [
            BorrowPairSpec(home="a", donor="b"),
            BorrowPairSpec(home="a", donor="c"),
        ]
        active_by_home = {0: 0, 1: 0}
        max_active_by_home = {0: 0, 1: 0}
        active_total = 0
        max_active_total = 0
        first_wave_started = set()
        first_wave_ready = asyncio.Event()

        async def tracked_init(guest):
            nonlocal active_total, max_active_total
            home_rank = int(guest._guest_external_pool.rsplit(":", 1)[1])
            active_by_home[home_rank] += 1
            active_total += 1
            max_active_by_home[home_rank] = max(
                max_active_by_home[home_rank], active_by_home[home_rank],
            )
            max_active_total = max(max_active_total, active_total)
            if guest.replica_rank in {10000, 10001}:
                first_wave_started.add(home_rank)
                if len(first_wave_started) == 2:
                    first_wave_ready.set()
            try:
                await asyncio.wait_for(first_wave_ready.wait(), timeout=1)
                guest._log.append((f"guest_init:{guest.replica_rank}",))
            finally:
                active_by_home[home_rank] -= 1
                active_total -= 1

        monkeypatch.setattr(FakeGuestReplica, "init_standalone", tracked_init)

        assert asyncio.run(env.manager.precreate_all()) == 4
        assert max_active_by_home == {0: 1, 1: 1}
        assert max_active_total == 2
        ops = env.ops()
        assert ops.index("guest_sleep:10000") < ops.index("guest_init:10002")
        assert ops.index("guest_sleep:10001") < ops.index("guest_init:10003")
        for rank in (0, 1):
            assert ops.count(f"home_sleep:a:{rank}") == 1
            assert ops.count(f"home_wake:a:{rank}") == 1

    def test_transition_order_and_unit_registry(self, env):
        count = asyncio.run(env.manager.precreate_all())
        # 2 pairs x 2 home replicas = 4 guests
        assert count == 4
        assert len(env.manager.units) == 4
        ops = env.ops()
        assert set(ops[:4]) == {
            "home_sleep:a:0", "home_sleep:a:1",
            "home_sleep:b:0", "home_sleep:b:1",
        }
        for policy, replica_rank, guest_rank in [
            ("a", 0, 10000), ("a", 1, 10001),
            ("b", 0, 10002), ("b", 1, 10003),
        ]:
            assert ops.index(f"home_sleep:{policy}:{replica_rank}") < ops.index(
                f"guest_init:{guest_rank}"
            ) < ops.index(f"guest_sleep:{guest_rank}") < ops.index(
                f"home_wake:{policy}:{replica_rank}"
            )

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

    def test_guest_checkpoint_group_is_unique_from_donor(self, env):
        checkpoint = SimpleNamespace(
            engine_kwargs=SimpleNamespace(
                nccl=SimpleNamespace(group_name="checkpoint_donor"),
            ),
        )
        env.handles["b"].rollout_config.checkpoint_engine = checkpoint

        asyncio.run(env.manager.precreate_all())
        guests = FakeGuestReplica.created[:2]
        assert [g.config.checkpoint_engine.engine_kwargs.nccl.group_name for g in guests] == [
            "dynamic_guest_a_b_10000", "dynamic_guest_a_b_10001",
        ]
        assert env.handles["b"].rollout_config.checkpoint_engine.engine_kwargs.nccl.group_name == (
            "checkpoint_donor"
        )

    def test_guest_bound_to_home_pool(self, env):
        asyncio.run(env.manager.precreate_all())
        for unit in env.manager.units:
            assert unit.guests[0]._guest_external_pool == unit.home_replicas[0].resource_pool
            assert unit.in_use is False

    def test_reserved_master_port_pool_is_split_per_guest(self, monkeypatch):
        env = Env(monkeypatch)
        env.config.borrowing.guest_master_port_range = [32000, 32016]
        env.config.borrowing.guest_master_port_stride = 4
        manager = GuestEngineManager(env.config, env.handles, env.manager.pairs)
        # The two policies share the same pair of physical nodes.  Slots must
        # be isolated across policies on one node and reusable across nodes.
        monkeypatch.setattr(
            ge,
            "_resource_pool_master_node_id",
            lambda pool: f"node-{pool.rsplit(':', 1)[1]}",
        )

        assert asyncio.run(manager.precreate_all()) == 4
        assert [guest._guest_master_port_range for guest in FakeGuestReplica.created] == [
            [32000, 32004],
            [32000, 32004],
            [32004, 32008],
            [32004, 32008],
        ]
        assert [guest._guest_master_node_id for guest in FakeGuestReplica.created] == [
            "node-0", "node-1", "node-0", "node-1",
        ]

    def test_reserved_master_port_pool_exhaustion_is_explicit(self, monkeypatch):
        env = Env(monkeypatch)
        env.config.borrowing.guest_master_port_range = [32000, 32004]
        manager = GuestEngineManager(env.config, env.handles, env.manager.pairs)
        monkeypatch.setattr(
            ge, "_resource_pool_master_node_id", lambda pool: "same-node",
        )

        with pytest.raises(
            RuntimeError,
            match="guest master-port pool exhausted on node same-node",
        ):
            asyncio.run(manager.precreate_all())

    def test_master_node_resolution_matches_worker_group_pool_selection(self, monkeypatch):
        class FakePlacementGroup:
            def __init__(self, pg_id):
                self.id = pg_id

        class FakePool:
            def __init__(self, pgs):
                self.pgs = pgs

            def get_placement_groups(self, **kwargs):
                assert kwargs == {"strategy": "PACK", "device_name": "cuda"}
                return self.pgs

        class FakeSubPool(FakePool):
            store = [4, 4]
            start_bundle_index = 5

        pg_0 = FakePlacementGroup("pg-0")
        pg_1 = FakePlacementGroup("pg-1")
        tables = {
            "pg-0": {"bundles_to_node_id": {0: "node-0", 1: "node-0"}},
            "pg-1": {"bundles_to_node_id": {0: "node-1", 1: "node-1"}},
        }
        ray_stub = SimpleNamespace(
            _private=SimpleNamespace(
                state=SimpleNamespace(
                    state=SimpleNamespace(
                        placement_group_table=lambda pg_id: tables[pg_id],
                    ),
                ),
            ),
        )
        ray_base_stub = types.ModuleType("verl.single_controller.ray.base")
        ray_base_stub.SubRayResourcePool = FakeSubPool
        ray_base_stub.sort_placement_group_by_node_ip = lambda pgs: list(reversed(pgs))
        device_stub = types.ModuleType("verl.utils.device")
        device_stub.get_device_name = lambda: "cuda"
        monkeypatch.setitem(sys.modules, "ray", ray_stub)
        monkeypatch.setitem(sys.modules, "verl.single_controller.ray.base", ray_base_stub)
        monkeypatch.setitem(sys.modules, "verl.utils.device", device_stub)

        # A normal pool uses the first PG after IP sorting (pg-1 here).
        assert ge._resource_pool_master_node_id(FakePool([pg_0, pg_1])) == "node-1"
        # A sub-pool mirrors divmod(start_bundle_index, local_world_size):
        # divmod(5, 4) -> placement group 1, bundle 1.
        assert ge._resource_pool_master_node_id(FakeSubPool([pg_0, pg_1])) == "node-1"

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
        assert env.manager.units == []
        for policy in ("a", "b"):
            for rank in (0, 1):
                assert f"home_wake:{policy}:{rank}" in env.ops()

    @pytest.mark.parametrize("message", [
        "DistNetworkError: server socket has failed to listen; EADDRINUSE: address already in use",
        "Engine core initialization failed. See root cause above. Failed core proc(s): {}",
    ])
    def test_port_collision_rebuilds_guest_and_retries(self, env, monkeypatch, message):
        failed_guest = None
        original_init = FakeGuestReplica.init_standalone

        async def collide_once(guest):
            nonlocal failed_guest
            if failed_guest is None:
                failed_guest = guest
                guest.workers = ["partial-worker"]
                guest.servers = ["partial-server"]
                raise RuntimeError(message)
            await original_init(guest)

        killed = []
        monkeypatch.setattr(FakeGuestReplica, "init_standalone", collide_once)
        monkeypatch.setattr(
            env.manager,
            "_kill_guest_actors",
            lambda guests: killed.extend(guests) or 2,
        )
        async def stopped(guest):
            assert guest in killed
        monkeypatch.setattr(env.manager, "_wait_guest_actors_stopped", stopped)
        env.config.borrowing.guest_init_retry_backoff_s = 0
        env.config.borrowing.guest_init_retry_jitter_s = 0

        assert asyncio.run(env.manager.precreate_all()) == 4
        assert failed_guest in killed
        assert failed_guest not in [
            guest for unit in env.manager.units for guest in unit.guests
        ]
        assert len(FakeGuestReplica.created) == 5
        assert any(op.startswith("guest_init:10004") for op in env.ops())

    def test_non_port_init_failure_is_not_retried(self, env, monkeypatch):
        calls = 0

        async def fail_init(guest):
            nonlocal calls
            calls += 1
            raise RuntimeError("model load failed")

        monkeypatch.setattr(FakeGuestReplica, "init_standalone", fail_init)
        env.config.borrowing.guest_init_retry_backoff_s = 0
        env.config.borrowing.guest_init_retry_jitter_s = 0

        with pytest.raises(RuntimeError, match="model load failed"):
            asyncio.run(env.manager.precreate_all())
        # All units start concurrently, but none gets a second attempt.
        assert calls == 4

    def test_wake_failure_is_reported_after_all_homes_attempt_restore(
        self, env, monkeypatch,
    ):
        async def fail_one_wake(replica):
            replica._log.append((f"home_wake:{replica.policy}:{replica.replica_rank}",))
            if replica.policy == "a" and replica.replica_rank == 0:
                raise RuntimeError("wake failed")

        monkeypatch.setattr(FakeHomeReplica, "wake_up", fail_one_wake)

        with pytest.raises(RuntimeError, match="failed to restore homes") as exc_info:
            asyncio.run(env.manager.precreate_all())

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "wake failed"
        assert env.manager.units == []
        for policy in ("a", "b"):
            for rank in (0, 1):
                assert f"home_wake:{policy}:{rank}" in env.ops()

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

    def test_missing_home_layout_is_rejected(self, monkeypatch):
        env = Env(monkeypatch)
        for replica in env.handles["a"].replicas:
            del replica.node_ids
        manager = GuestEngineManager(
            env.config,
            env.handles,
            [BorrowPairSpec(home="a", donor="b")],
        )

        with pytest.raises(RuntimeError, match="cannot determine the physical node"):
            asyncio.run(manager.precreate_all())

    def test_heterogeneous_split_one_16_card_home_into_two_8_card_guests(self, monkeypatch):
        env = Env(monkeypatch)
        env.handles["a"].cards_per_replica = 16
        env.handles["a"].nnodes = 2
        for replica in env.handles["a"].replicas:
            replica.node_ids = (
                [f"node-{replica.replica_rank}-0"] * 8
                + [f"node-{replica.replica_rank}-1"] * 8
            )

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


def test_retry_cleanup_waits_for_actor_death(monkeypatch):
    class ActorDiedError(Exception):
        pass
    calls = []
    async def ready():
        calls.append('ready')
        if len(calls) > 1:
            raise ActorDiedError()
    monkeypatch.setitem(sys.modules, 'ray', SimpleNamespace(
        exceptions=SimpleNamespace(ActorDiedError=ActorDiedError)))
    guest = SimpleNamespace(servers=[SimpleNamespace(__ray_ready__=SimpleNamespace(remote=ready))], workers=[])
    manager = object.__new__(GuestEngineManager)
    asyncio.run(manager._wait_guest_actors_stopped(guest, timeout=1))
    assert calls == ['ready', 'ready']


def test_retry_cleanup_timeout_is_not_success(monkeypatch):
    class ActorDiedError(Exception):
        pass
    async def ready():
        await asyncio.Event().wait()
    monkeypatch.setitem(sys.modules, 'ray', SimpleNamespace(
        exceptions=SimpleNamespace(ActorDiedError=ActorDiedError)))
    guest = SimpleNamespace(servers=[SimpleNamespace(__ray_ready__=SimpleNamespace(remote=ready))], workers=[])
    manager = object.__new__(GuestEngineManager)
    with pytest.raises(TimeoutError):
        asyncio.run(manager._wait_guest_actors_stopped(guest, timeout=0.01))
