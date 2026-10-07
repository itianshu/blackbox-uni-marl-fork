"""Tests for the decision layer (scheduler) and the SignalStore.

The scheduler is exercised against the real ``SignalStore`` fed synthetic
``PolicySample`` windows. KV utilisations are injected either through samples
(end-to-end through ``poll``'s EMA) or directly into ``ema_kv`` (bypassing
EMA warmup) depending on what the test measures.
"""

import time

import pytest

from uni_agent.trainer.dynamic_inference import scheduler as scheduler_module
from uni_agent.trainer.dynamic_inference.scheduler import MultiPolicyInferenceScheduler
from uni_agent.trainer.dynamic_inference.signals import PolicySample, SignalStore
from uni_agent.trainer.dynamic_inference.types import (
    BorrowPlan,
    BorrowPairSpec,
    GuestUnit,
    LendRecord,
    PolicyInferenceHandles,
    SchedulingConfig,
    StepStats,
)


# --------------------------------------------------------------------- fakes
class FakeReplica:
    def __init__(self, policy, rank, addr=None):
        self.policy = policy
        self.replica_rank = rank
        self.server_address = addr or f"{policy}-rep{rank}"


class FakeGuestUnits:
    """Registry fake: one persistent GuestUnit per (home, donor, replica)."""

    def __init__(self):
        self.units = {}

    def unit_for(self, home, donor, replica):
        key = (home, donor, id(replica))
        if key not in self.units:
            self.units[key] = GuestUnit(home_policy=home, donor=donor, home_replicas=[replica])
        return self.units[key]


class FakeGroupedGuestUnits(FakeGuestUnits):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def candidate_units(self, home, donor):
        return self.groups.get((home, donor), [])

    def unit_for(self, home, donor, replica):
        replicas = replica if isinstance(replica, list) else [replica]
        key = (home, donor, tuple(sorted(id(r) for r in replicas)))
        if key not in self.units:
            self.units[key] = GuestUnit(
                home_policy=home, donor=donor, home_replicas=list(replicas))
        return self.units[key]


def _handle(policy, n_replicas=2, cards=8, max_num_seqs=32):
    return PolicyInferenceHandles(
        actor_rollout_wg=None,
        standalone_checkpoint_manager=None,
        lb_handle=None, rollout_config=None, model_config=None,
        replicas=[FakeReplica(policy, i) for i in range(n_replicas)],
        cards_per_replica=cards, nnodes=1,
        max_num_seqs=max_num_seqs,
    )


def _config(**overrides):
    """parse via the real parser with the pairs defaulting to a<->b."""
    from uni_agent.trainer.dynamic_inference.types import parse_scheduling_config

    # Most scheduler unit tests use a short confirmation loop; production
    # defaults are covered independently by test_dynamic_inference_types.py.
    data = {
        "enable": True,
        "rebalance_confirm_polls": 2,
        "borrow_cooldown_s": 0,
        "return_confirm_polls": 1,
    }
    data.update(overrides)
    return parse_scheduling_config(data)


def _scheduler(policies=("a", "b"), *, signals=None, units=None, **overrides):
    handles = {p: _handle(p) for p in policies}
    signals = signals or SignalStore(list(policies), max_num_seqs={p: 32 for p in policies})
    units = units or FakeGuestUnits()
    sched = MultiPolicyInferenceScheduler(
        config=_config(**overrides),
        handles=handles,
        signals=signals,
        guest_units=units,
    )
    return sched


def _feed(store: SignalStore, policy, per_replica, kv=None, n=5, total=None):
    """Feed a stationary window of samples for one policy.

    ``per_replica``: server -> in-flight; ``kv``: server -> KV gauge.
    """
    for _ in range(n):
        store.record(policy, PolicySample(
            t=time.monotonic(),
            inflight_per_replica=dict(per_replica),
            total_inflight=total if total is not None else sum(per_replica.values()),
            kv_cache_usage=dict(kv) if kv else {},
        ))


def _stats(step):
    return StepStats(step=step)


def _saturate_kv(sched, n=30):
    """Poll enough times for the KV EMA to reach its stationary value."""
    for _ in range(n):
        sched.poll()


def _decides(sched, steps, **stats_kwargs):
    """poll + decide for each step; returns the last plan."""
    plan = None
    for step in steps:
        sched.poll()
        plan = sched.decide(_stats(step=step, **stats_kwargs))
    return plan


def _prepared_scheduler(*, kv_a, kv_b, b_replicas=2, a_replicas=2):
    """Build a saturated two-policy scheduler for borrow-planning tests."""
    sched = _scheduler()
    sched.handles["b"].replicas = [FakeReplica("b", i) for i in range(b_replicas)]
    sched.handles["a"].replicas = [FakeReplica("a", i) for i in range(a_replicas)]
    _feed(sched.signals, "a", {"a-rep0": 4}, kv={"a-rep0": kv_a})
    addrs = {f"b-rep{i}": 4 for i in range(b_replicas)}
    _feed(sched.signals, "b", addrs, kv={addr: kv_b for addr in addrs})
    _saturate_kv(sched)
    return sched


def test_one_quantity_plan_selects_only_the_requested_borrowers_best_unit():
    sched = _scheduler(policies=('a', 'b', 'c'), borrowing={'pairs': [
        {'home': 'b', 'donor': 'a', 'max_units': 1},
        {'home': 'b', 'donor': 'c', 'max_units': 1},
    ]})
    sched.handles['b'].replicas = [FakeReplica('b', i) for i in range(4)]
    sched.ema_kv.update(a=.9, b=.01, c=.95)
    plan = sched.quantity.select_borrow(sched, 'c', step=1)
    assert plan.donor == 'c'
    assert len(plan.home_replicas) == 1


def test_diluted_incumbent_does_not_block_another_borrower_or_move_physical_units():
    sched = _scheduler(policies=('a', 'b', 'c'), borrowing={'pairs': [
        {'home': 'b', 'donor': 'a', 'max_units': 1},
        {'home': 'b', 'donor': 'c', 'max_units': 1},
    ]}, resource_usage={'kv_enter': .01, 'kv_exit': .003, 'kv_post_lend_max': .007})
    sched.handles['b'].replicas = [FakeReplica('b', i) for i in range(4)]
    incumbent_replica = sched.handles['b'].replicas[-1]
    lend = LendRecord(lend_id=1, home_policy='b', donor='a',
        home_replicas=[incumbent_replica], guests=[FakeReplica('a', 10000)], since_step=1)
    sched.note_borrow_started(lend)
    sched.ema_kv.update(a=.007, b=.002, c=.03)
    sched._settle_polls_remaining = 0
    plan = None
    for step in range(sched.config.rebalance_confirm_polls):
        plan = sched.decide(_stats(step))
    assert plan.returns == []
    assert [borrow.donor for borrow in plan.borrows] == ['c']
    assert incumbent_replica not in plan.borrows[0].home_replicas
    assert sched.active_lends == [lend]


# ------------------------------------------------------------ SignalStore
class TestSignalStore:
    def test_kv_util_uses_policy_replica_mean(self):
        store = SignalStore(["a"], max_num_seqs={"a": 32})
        _feed(store, "a", {"s0": 4, "s1": 4}, kv={"s0": 0.4, "s1": 0.9})
        assert store.kv_util("a") == pytest.approx(0.65)

    def test_kv_util_p90_window(self):
        store = SignalStore(["a"], max_num_seqs={"a": 32})
        _feed(store, "a", {"s0": 4}, kv={"s0": 0.2}, n=1)
        _feed(store, "a", {"s0": 4}, kv={"s0": 0.8}, n=10)
        kv = store.kv_util("a")
        assert kv >= 0.8  # p90 dominated by the stationary value
        assert kv <= 0.8

    def test_kv_util_without_samples(self):
        store = SignalStore(["a"], max_num_seqs={"a": 32})
        assert store.kv_util("a") == 0.0

    def test_stat_modes(self):
        store = SignalStore(["a"], max_num_seqs={"a": 32})
        for total in (1, 2, 3, 4, 100):
            store.record("a", PolicySample(t=time.monotonic(),
                                           inflight_per_replica={"s0": total}, total_inflight=total))
        assert store.stat_inflight("a", "max") == 100
        assert store.stat_inflight("a", "instant") == 100
        p90 = store.stat_inflight("a", "p90")
        assert p90 >= 3  # robust to the exact quantile convention

    def test_window_eviction(self):
        store = SignalStore(["a"], window_s=0.0, max_num_seqs={"a": 32})
        store.record("a", PolicySample(t=1.0, inflight_per_replica={"s0": 5}, total_inflight=5))
        store.record("a", PolicySample(t=2.0, inflight_per_replica={"s0": 7}, total_inflight=7))
        # window_s=0 evicts everything older than the newest sample
        assert store.stat_inflight("a", "instant") == 7


# ------------------------------------------------------------- borrow confirmation
class TestBorrowConfirmation:
    def test_only_one_confirmation_stage(self):
        sched = _prepared_scheduler(kv_a=.95, kv_b=.1)
        assert sched.decide(_stats(10)).borrows == []
        assert sched._pending_borrow_pair == ("b", "a")
        assert sched._pending_borrow_polls == 1
        assert len(sched.decide(_stats(11)).borrows) == 1
        assert not hasattr(sched, "bottleneck")
        assert not hasattr(sched, "confirm")

    def test_no_safe_plan_clears_confirmation(self):
        sched = _prepared_scheduler(kv_a=.95, kv_b=.1)
        sched.decide(_stats(10))
        sched.pair_block_until[("b", "a")] = 20
        assert sched.decide(_stats(11)).borrows == []
        assert sched._pending_borrow_pair is None
        assert sched._pending_borrow_polls == 0
        assert sched.decide(_stats(20)).borrows == []
        assert len(sched.decide(_stats(21)).borrows) == 1

    def test_below_enter_clears_confirmation(self):
        sched = _prepared_scheduler(kv_a=.95, kv_b=.1)
        sched.decide(_stats(10))
        sched.ema_kv["a"] = .7
        assert sched.decide(_stats(11)).borrows == []
        assert sched._pending_borrow_pair is None
        assert sched._pending_borrow_polls == 0

    def test_highest_load_is_candidate_but_not_necessarily_borrowable(self):
        sched = _scheduler()
        sched.ema_kv.update(a=.9, b=.97)
        assert sched._pick_candidate() == "b"
        for step in range(3):
            assert sched.decide(_stats(step)).borrows == []
        assert sched._pending_borrow_pair is None

    @pytest.mark.parametrize("switch_borrower", [False, True])
    def test_pair_change_restarts_confirmation(self, switch_borrower):
        sched = _scheduler(policies=("a", "b", "c"))
        sched.ema_kv.update(a=.95, b=.1, c=.2)
        assert sched.decide(_stats(10)).borrows == []
        assert sched._pending_borrow_pair == ("b", "a")
        if switch_borrower:
            sched.ema_kv.update(a=.1, b=.2, c=.95)
            expected = ("a", "c")
        else:
            sched.ema_kv.update(b=.2, c=.1)
            expected = ("c", "a")
        assert sched.decide(_stats(11)).borrows == []
        assert sched._pending_borrow_pair == expected
        assert sched._pending_borrow_polls == 1
        plan = sched.decide(_stats(12))
        assert [(p.home_policy, p.donor) for p in plan.borrows] == [expected]

    def test_long_deep_queue_can_trigger_when_kv_is_low(self):
        sched = _scheduler()
        sched.ema_kv.update({"a": 0.2, "b": 0.1})
        sched.waiting_per_replica["a"] = sched.ru.waiting_enter_per_replica + 1
        sched.queue_time_s["a"] = sched.ru.queue_time_enter_s + 1

        sched.decide(_stats(step=10))
        sched.decide(_stats(step=11))

        assert sched._pick_candidate() == "a"

    def test_queue_trigger_requires_both_depth_and_duration(self):
        sched = _scheduler()
        sched.ema_kv.update({"a": 0.2, "b": 0.1})
        sched.waiting_per_replica["a"] = sched.ru.waiting_enter_per_replica + 1
        sched.queue_time_s["a"] = sched.ru.queue_time_enter_s - 0.1
        assert sched._pick_candidate() is None

        sched.waiting_per_replica["a"] = sched.ru.waiting_enter_per_replica - 0.1
        sched.queue_time_s["a"] = sched.ru.queue_time_enter_s + 1
        assert sched._pick_candidate() is None

    def test_queue_signal_can_be_disabled_for_kv_only_experiments(self):
        sched = _scheduler(resource_usage={
            "queue_signal_enabled": False,
            "kv_enter": 0.85,
            "kv_exit": 0.6,
            "kv_post_lend_max": 0.7,
        })
        sched.ema_kv.update({"a": 0.2, "b": 0.1})
        sched.waiting_per_replica["a"] = 1000
        sched.queue_time_s["a"] = 1000

        assert sched._pick_candidate() is None
        assert sched._is_relieved("a") is True

    def test_composite_exit_requires_kv_and_queue_to_recover(self):
        sched = _scheduler()
        sched.ema_kv["a"] = sched.ru.kv_exit
        sched.waiting_per_replica["a"] = sched.ru.waiting_exit_per_replica + 1
        sched.queue_time_s["a"] = sched.ru.queue_time_exit_s + 1
        sched.decide(_stats(step=10))
        assert not sched._is_relieved("a")

        sched.waiting_per_replica["a"] = 0
        sched.queue_time_s["a"] = 0
        sched.decide(_stats(step=11))
        assert sched._is_relieved("a")


# ------------------------------------------------------------- borrow planning
class TestBorrowPlanning:
    """The default strategy ranks and allocates safe atomic borrow units."""

    def test_borrows_one_unit_when_that_is_all_the_lender_can_spare(self):
        sched = _prepared_scheduler(kv_a=0.95, kv_b=0.1)
        plan = _decides(sched, [10, 11, 12])
        assert sched._pick_candidate() == "a"
        assert len(plan.borrows) == 1
        assert plan.borrows[0].home_policy == "b"
        assert plan.borrows[0].donor == "a"
        assert len(plan.borrows[0].home_replicas) == 1

    def test_one_event_selects_only_the_best_lender(self):
        # c is the colder lender and is selected for this event; b can only be
        # considered by a later event after the global borrow cooldown.
        config = _config(
            borrowing={"pairs": [{"home": "b", "donor": "a"},
                                 {"home": "c", "donor": "a"}]},
        )
        handles = {p: _handle(p) for p in ("a", "b", "c")}
        store = SignalStore(["a", "b", "c"], max_num_seqs={p: 32 for p in "abc"})
        sched = MultiPolicyInferenceScheduler(config, handles, store, FakeGuestUnits())
        _feed(store, "a", {"a-rep0": 4}, kv={"a-rep0": 0.95})
        _feed(store, "b", {"b-rep0": 4}, kv={"b-rep0": 0.3})
        _feed(store, "c", {"c-rep0": 4}, kv={"c-rep0": 0.05})
        _saturate_kv(sched)
        plan = _decides(sched, [10, 11, 12])
        assert len(plan.borrows) == 1
        assert plan.borrows[0].home_policy == "c"
        assert all(borrow.donor == "a" for borrow in plan.borrows)

    def test_one_event_selects_only_one_donor_for_a_shared_home(self):
        config = _config(borrowing={"pairs": [
            {"home": "a", "donor": "b"},
            {"home": "a", "donor": "c"},
        ]})
        handles = {
            "a": _handle("a", n_replicas=5),
            "b": _handle("b", n_replicas=2),
            "c": _handle("c", n_replicas=2),
        }
        store = SignalStore(["a", "b", "c"], max_num_seqs={p: 32 for p in "abc"})
        sched = MultiPolicyInferenceScheduler(config, handles, store, FakeGuestUnits())
        _feed(store, "a", {f"a-rep{i}": 1 for i in range(5)},
              kv={f"a-rep{i}": 0.05 for i in range(5)})
        _feed(store, "b", {f"b-rep{i}": 4 for i in range(2)},
              kv={f"b-rep{i}": 0.95 for i in range(2)})
        _feed(store, "c", {f"c-rep{i}": 4 for i in range(2)},
              kv={f"c-rep{i}": 0.90 for i in range(2)})
        _saturate_kv(sched)

        plan = _decides(sched, [10, 11, 12])

        assert {borrow.donor for borrow in plan.borrows} == {"b"}
        replica_ids = [
            id(replica)
            for borrow in plan.borrows
            for replica in borrow.home_replicas
        ]
        assert len(replica_ids) == len(set(replica_ids))

    def test_no_borrow_when_no_spare_unit(self):
        # lender's single spare replica must stay home (keep >= 1)
        sched = _prepared_scheduler(kv_a=0.95, kv_b=0.1, b_replicas=1)
        plan = _decides(sched, [10, 11, 12])
        assert sched._pick_candidate() == "a"
        assert plan.borrows == []

    def test_failed_pair_is_blocked(self):
        sched = _prepared_scheduler(kv_a=0.95, kv_b=0.1)
        sched.pair_block_until[("b", "a")] = 99
        plan = _decides(sched, [10, 11, 12])
        assert sched._pick_candidate() == "a"
        assert plan.borrows == []
        # after the block expires the pair is eligible again
        sched.pair_block_until[("b", "a")] = 10
        plan = _decides(sched, [11, 12])
        assert len(plan.borrows) == 1


class TestSingleUnitPlanning:
    """Detailed tests for single-unit selection and lender safety guards."""

    def test_only_one_safe_unit_is_selected_per_event(self):
        sched = _prepared_scheduler(kv_a=0.95, kv_b=0.1, b_replicas=5)
        plan = _decides(sched, [10, 11, 12])
        assert len(plan.borrows) == 1
        ids = [id(r) for bp in plan.borrows for r in bp.home_replicas]
        assert len(set(ids)) == 1

        # A small overloaded policy can still receive one safe unit.
        small_gain = _prepared_scheduler(
            kv_a=0.86, kv_b=0.1, a_replicas=100, b_replicas=2)
        small_gain_plan = _decides(small_gain, [10, 11, 12])
        assert len(small_gain_plan.borrows) == 1

    def test_lender_post_lend_guard(self):
        # A donor with a large card pool and a lender whose
        # predicted post-lend utilisation exceeds kv_post_lend_max: the guard
        # blocks the unit regardless of candidate ranking.
        # donor a: 10 reps x 8 cards, kv 1.0 -> donor_after = 80/88 = .91
        # lender b: 2 reps x 8 cards, kv .48 -> lender_after = .48*16/8 = .96 > .9
        sched = _prepared_scheduler(
            kv_a=1.0, kv_b=0.48, b_replicas=2, a_replicas=10)
        plan = _decides(sched, [10, 11])
        assert sched._pick_candidate() == "a"
        assert plan.borrows == []
        # a cooler lender (.35 -> lender_after .7) is allowed
        sched2 = _prepared_scheduler(
            kv_a=1.0, kv_b=0.35, b_replicas=2, a_replicas=10)
        plan2 = _decides(sched2, [10, 11, 12])
        assert len(plan2.borrows) == 1

    def test_heterogeneous_units_are_atomic_and_keep_one_home_replica(self):
        config = _config(
            borrowing={"pairs": [{
                "home": "b", "donor": "a",
                "home_replicas_per_unit": 2,
                "guest_replicas_per_unit": 1,
            }]},
        )
        handles = {"a": _handle("a", n_replicas=2, cards=16),
                   "b": _handle("b", n_replicas=5, cards=8)}
        groups = [[handles["b"].replicas[0], handles["b"].replicas[1]],
                  [handles["b"].replicas[2], handles["b"].replicas[3]]]
        units = FakeGroupedGuestUnits({("b", "a"): groups})
        store = SignalStore(["a", "b"], max_num_seqs={"a": 32, "b": 32})
        sched = MultiPolicyInferenceScheduler(config, handles, store, units)
        _feed(store, "a", {"a-rep0": 4}, kv={"a-rep0": 0.99})
        _feed(store, "b", {f"b-rep{i}": 1 for i in range(5)},
              kv={f"b-rep{i}": 0.05 for i in range(5)})
        _saturate_kv(sched)
        plan = _decides(sched, [10, 11, 12])

        assert len(plan.borrows) == 1
        assert all(len(borrow.home_replicas) == 2 for borrow in plan.borrows)
        borrowed = {id(r) for borrow in plan.borrows for r in borrow.home_replicas}
        assert len(borrowed) == 2
        assert id(handles["b"].replicas[4]) not in borrowed


# ------------------------------------------------------------- returns
class TestReturns:
    def test_no_new_candidate_keeps_existing_lend_in_hysteresis_band(self):
        sched = _scheduler(min_lend_polls=1)
        lend = self._active_lend(sched)
        sched._settle_polls_remaining = 0
        sched.ema_kv.update(a=.7, b=.1)
        assert sched._pick_candidate() is None
        plan = sched.decide(_stats(11))
        assert sched.active_lends == [lend]
        assert plan.returns == [] and plan.borrows == []
        assert sched._pending_borrow_pair is None

    def test_return_does_not_wait_for_borrow_confirmation(self):
        sched = _scheduler(rebalance_confirm_polls=100, min_lend_polls=1)
        lend = self._active_lend(sched)
        sched._settle_polls_remaining = 0
        sched.ema_kv.update(a=.2, b=.1)
        sched._return_low_polls["a"] = sched.config.return_confirm_polls
        sched._pending_borrow_pair = ("a", "b")
        sched._pending_borrow_polls = 1
        plan = sched.decide(_stats(11))
        assert plan.returns == [lend]
        assert plan.borrows == []
        assert sched._pending_borrow_polls == 0
        assert sched._pending_borrow_pair is None

    @pytest.mark.parametrize("blocked_by", ["confirmation", "age", "capacity"])
    def test_blocked_return_does_not_block_an_unrelated_safe_borrow(self, blocked_by):
        sched = _scheduler(
            policies=("a", "b", "c", "d"),
            min_lend_polls=10,
            return_confirm_polls=10,
            borrowing={"pairs": [
                {"home": "b", "donor": "a"},
                {"home": "d", "donor": "c"},
            ]},
        )
        lend = self._active_lend(sched)
        sched._settle_polls_remaining = 0
        sched._decision_count = 0 if blocked_by == "age" else 10
        sched._return_low_polls["a"] = 9 if blocked_by == "confirmation" else 10
        sched.ema_kv.update(a=.6 if blocked_by == "capacity" else .2, b=.1, c=.95, d=.1)

        assert sched.decide(_stats(11)).borrows == []
        plan = sched.decide(_stats(12))
        assert plan.returns == []
        assert [(p.home_policy, p.donor) for p in plan.borrows] == [("d", "c")]
        assert sched.active_lends == [lend]

    def test_borrow_uses_current_capacity_not_capacity_after_pending_return(self):
        sched = _scheduler(
            policies=("a", "b", "c"), min_lend_polls=1,
            rebalance_confirm_polls=1,
            borrowing={"pairs": [
                {"home": "b", "donor": "a"},
                {"home": "b", "donor": "c"},
            ]},
        )
        sched.handles["b"].replicas.append(FakeReplica("b", 2))
        lend = self._active_lend(sched)
        sched._settle_polls_remaining = 0
        sched._return_low_polls["a"] = sched.config.return_confirm_polls
        sched.ema_kv.update(a=.6, b=.4, c=.95)
        # Returning a's guest would predict .6 * 24 / 16 = .9: unsafe.
        # b has 16 cards NOW, not its original 24. Another lend would predict
        # .4 * 16 / 8 = .8: unsafe, even though .4 * 24 / 16 = .6 looks safe.
        plan = sched.decide(_stats(11))
        assert plan.returns == [] and plan.borrows == []
        assert sched._pending_borrow_pair is None

        sched.ema_kv["b"] = .3
        plan = sched.decide(_stats(12))
        assert plan.returns == []
        assert len(plan.borrows) == 1
        assert plan.borrows[0].donor == "c"
        assert lend.home_replicas[0] not in plan.borrows[0].home_replicas
        assert sched.active_lends == [lend]

    def test_young_lend_does_not_block_return_of_another_mature_lend(self):
        sched = _scheduler(policies=("a", "b", "c"), min_lend_polls=10)
        young = self._active_lend(sched)
        mature = LendRecord(
            lend_id=2, home_policy="c", donor="a",
            home_replicas=[sched.handles["c"].replicas[0]], guests=[object()], since_step=11,
        )
        sched.note_borrow_started(mature)
        sched._settle_polls_remaining = 0
        sched._decision_count = 10
        sched._lend_start_poll[id(young)] = 10
        sched._return_low_polls["a"] = sched.config.return_confirm_polls
        sched.ema_kv.update(a=.2, b=.1, c=.1)
        plan = sched.decide(_stats(12))
        assert plan.returns == [mature]
        assert plan.borrows == []
        assert sched.active_lends == [young, mature]

    def test_ready_return_skips_borrow_selection_entirely(self):
        sched = _scheduler(min_lend_polls=1)
        lend = self._active_lend(sched)
        sched._settle_polls_remaining = 0
        sched._return_low_polls["a"] = sched.config.return_confirm_polls

        def unexpected_borrow(*args):
            pytest.fail("Borrow selection must not run when a return is selected")

        sched.quantity.select_borrow = unexpected_borrow
        plan = sched.decide(_stats(11))
        assert plan.returns == [lend]
        assert plan.borrows == []

    @pytest.mark.parametrize("action", ["borrow", "return", "early_return"])
    def test_successful_action_resets_borrow_confirmation(self, action):
        sched = _scheduler()
        lend = self._active_lend(sched)
        sched._pending_borrow_pair = ("a", "b")
        sched._pending_borrow_polls = 1
        if action == "borrow":
            sched.active_lends.remove(lend)
            sched.note_borrow_started(lend)
        else:
            sched.note_lend_returned(lend, step=11, early=action == "early_return")
        assert sched._pending_borrow_pair is None
        assert sched._pending_borrow_polls == 0
        assert sched._settle_polls_remaining == sched.config.rebalance_settle_polls

    def _active_lend(self, sched):
        lend = LendRecord(
            lend_id=1, home_policy="b", donor="a",
            home_replicas=[sched.handles["b"].replicas[0]],
            guests=[object()], since_step=10,
        )
        sched.note_borrow_started(lend)
        return lend

    def test_matching_target_keeps_active_lend_and_only_borrows_delta(self):
        sched = _scheduler()
        sched.handles["b"].replicas = [FakeReplica("b", i) for i in range(3)]
        store = sched.signals
        _feed(store, "a", {"a-rep0": 4}, kv={"a-rep0": 0.95})
        _feed(store, "b", {f"b-rep{i}": 4 for i in range(3)},
              kv={f"b-rep{i}": 0.1 for i in range(3)})
        _saturate_kv(sched)
        _decides(sched, [10, 11])
        assert sched._pick_candidate() == "a"
        lend = self._active_lend(sched)
        _saturate_kv(sched, n=sched.config.rebalance_settle_polls)
        for _ in range(sched.config.rebalance_confirm_polls):
            plan = sched.decide(_stats(step=12))
        assert plan.returns == []
        assert sched.active_lends == [lend]
        assert len(plan.borrows) == 1
        assert lend.home_replicas[0] not in plan.borrows[0].home_replicas

    def test_settle_window_holds_current_allocation(self):
        sched = _scheduler()
        lend = self._active_lend(sched)
        plan = sched.decide(_stats(step=11))
        assert sched.active_lends == [lend]
        assert plan.returns == []
        assert plan.borrows == []
        assert set(vars(plan)) == {"returns", "borrows"}

    def test_disabled_plan_returns_one_lend_without_mutating_active_lends(self):
        sched = _scheduler(policies=("a", "b", "c"))
        lends = [LendRecord(
            lend_id=i, home_policy=home, donor="a",
            home_replicas=[sched.handles[home].replicas[0]],
            guests=[object()], since_step=10,
        ) for i, home in enumerate(("b", "c"), 1)]
        for lend in lends:
            sched.note_borrow_started(lend)
        sched.disabled = True
        plan = sched.decide(_stats(11))
        assert plan.returns == lends[:1]
        assert plan.borrows == []
        assert sched.active_lends == lends

    def test_retained_lend_prevents_overlapping_new_borrow(self):
        sched = _scheduler(policies=("a", "b", "c"), rebalance_confirm_polls=1)
        lend = self._active_lend(sched)
        sched._settle_polls_remaining = 0
        sched.ema_kv.update(a=.7, b=.1, c=.95)
        # Only b may lend to c, but its remaining home replica is the seed.
        sched.pairs = [BorrowPairSpec(home="b", donor="c")]
        assert sched.quantity.select_borrow(sched, "c", 11) is None
        plan = sched.decide(_stats(11))
        assert plan.returns == [] and plan.borrows == []
        assert sched.active_lends == [lend]

    def test_settle_window_counts_complete_metric_polls(self):
        sched = _scheduler()
        self._active_lend(sched)
        _feed(sched.signals, "a", {"a-rep0": 1}, kv={"a-rep0": 0.5})
        _feed(sched.signals, "b", {"b-rep1": 1}, kv={"b-rep1": 0.2})
        remaining = sched._settle_polls_remaining
        sched.poll()
        assert sched._settle_polls_remaining == remaining - 1

    def test_invalid_metrics_do_not_consume_settle_window(self):
        sched = _scheduler()
        self._active_lend(sched)
        _feed(sched.signals, "a", {"a-rep0": 1}, kv={"a-rep0": float("nan")})
        _feed(sched.signals, "b", {"b-rep1": 1}, kv={"b-rep1": 0.2})

        remaining = sched._settle_polls_remaining
        assert sched.poll() is False

        assert sched._settle_polls_remaining == remaining
        assert "a" not in sched._ema_initialized
        assert "b" in sched._ema_initialized
        assert sched.ema_kv["b"] == pytest.approx(0.2)

    def test_return_respects_minimum_lend_age(self):
        sched = _scheduler()
        lend = self._active_lend(sched)
        sched._settle_polls_remaining = 0
        sched._return_low_polls["a"] = sched.config.return_confirm_polls
        first = sched.decide(_stats(step=10))
        assert first.returns == [] and first.borrows == []
        assert sched.active_lends == [lend]
        second = sched.decide(_stats(step=10))
        assert second.returns == [lend]

    def test_target_switch_returns_before_any_new_borrow(self):
        sched = _scheduler(policies=("a", "b", "c"), borrowing={"pairs": [
            {"home": "b", "donor": "a"},
            {"home": "b", "donor": "c"},
        ]})
        lend = LendRecord(
            lend_id=1, home_policy="b", donor="a",
            home_replicas=[sched.handles["b"].replicas[0]],
            guests=[object()], since_step=11,
        )
        sched.note_borrow_started(lend)
        sched._settle_polls_remaining = 0
        sched.ema_kv.update({"a": 0.4, "b": 0.1, "c": 0.95})
        sched._return_low_polls["a"] = sched.config.return_confirm_polls

        first = sched.decide(_stats(step=11))
        plan = sched.decide(_stats(step=11))
        assert first.returns == [] and first.borrows == []
        assert sched.active_lends == [lend]
        assert plan.returns == [lend]
        assert plan.borrows == []
        sched.note_lend_returned(lend, step=11)
        assert sched.decide(_stats(step=11)).borrows == []
        assert sched._pending_borrow_pair is None

    def test_ten_low_kv_scrapes_return_only_post_return_safe_units(self):
        sched = _scheduler(
            policies=("a", "b", "c"),
            return_confirm_polls=10,
            min_lend_polls=1,
        )
        lends = [
            LendRecord(
                lend_id=1, home_policy="b", donor="a",
                home_replicas=[sched.handles["b"].replicas[0]],
                guests=[object()], since_step=1,
            ),
            LendRecord(
                lend_id=2, home_policy="c", donor="a",
                home_replicas=[sched.handles["c"].replicas[0]],
                guests=[object()], since_step=1,
            ),
        ]
        for lend in lends:
            sched.note_borrow_started(lend)
        sched._settle_polls_remaining = 0
        _feed(sched.signals, "a", {"a-rep0": 1}, kv={"a-rep0": 0.5})
        _feed(sched.signals, "b", {"b-rep1": 1}, kv={"b-rep1": 0.1})
        _feed(sched.signals, "c", {"c-rep1": 1}, kv={"c-rep1": 0.1})

        for step in range(9):
            sched.poll()
            plan = sched.decide(_stats(step=step))
            assert plan.returns == []
        sched.poll()
        plan = sched.decide(_stats(step=9))

        # Current donor capacity is 32 cards at KV .5 (demand 16). Returning
        # one 8-card guest predicts .667 and is safe; returning both predicts
        # 1.0, so one lend remains active.
        assert plan.returns == [lends[0]]
        assert lends[1] in sched.active_lends
        assert lends[1] not in plan.returns

    def test_low_kv_does_not_return_when_shrunk_donor_would_exceed_guard(self):
        sched = _scheduler(return_confirm_polls=10, min_lend_polls=1)
        lend = self._active_lend(sched)
        sched._settle_polls_remaining = 0
        _feed(sched.signals, "a", {"a-rep0": 1}, kv={"a-rep0": 0.6})
        _feed(sched.signals, "b", {"b-rep1": 1}, kv={"b-rep1": 0.1})

        for step in range(10):
            sched.poll()
            plan = sched.decide(_stats(step=step))

        # 24 current cards * .6 demand / 16 cards after return = .9 > .7.
        assert plan.returns == []
        assert sched.active_lends == [lend]


class TestAntiJitter:
    def test_borrow_confirmation_uses_configured_threshold(self):
        sched = _prepared_scheduler(kv_a=.95, kv_b=.1)
        sched.config.rebalance_confirm_polls = 3
        assert sched.decide(_stats(10)).borrows == []
        assert sched.decide(_stats(11)).borrows == []
        assert len(sched.decide(_stats(12)).borrows) == 1

    def test_confirmation_tracks_pair_not_replica_identity(self):
        sched = _scheduler(rebalance_confirm_polls=2)
        r0, r1 = sched.handles["b"].replicas
        sched.ema_kv.update({"a": 0.95, "b": 0.1})
        targets = iter([
            BorrowPlan(home_policy="b", donor="a", home_replicas=[r0]),
            BorrowPlan(home_policy="b", donor="a", home_replicas=[r1]),
        ])
        sched.quantity.select_borrow = lambda *_: next(targets)

        assert sched.decide(_stats(step=10)).borrows == []
        plan = sched.decide(_stats(step=11))
        assert [borrow.home_replicas for borrow in plan.borrows] == [[r1]]

    def test_unsafe_active_lender_is_excluded_from_new_target(self):
        config = _config(
            borrowing={"pairs": [
                {"home": "b", "donor": "a"},
                {"home": "c", "donor": "a"},
            ]},
        )
        handles = {
            "a": _handle("a", n_replicas=100),
            "b": _handle("b", n_replicas=2),
            "c": _handle("c", n_replicas=2),
        }
        sched = MultiPolicyInferenceScheduler(
            config, handles,
            SignalStore(["a", "b", "c"], max_num_seqs={p: 32 for p in "abc"}),
            FakeGuestUnits(),
        )
        lend = LendRecord(
            lend_id=1, home_policy="b", donor="a",
            home_replicas=[handles["b"].replicas[0]], guests=[object()], since_step=1,
        )
        sched.note_borrow_started(lend)
        sched.ema_kv.update({"a": 0.99, "b": 0.95, "c": 0.1})

        borrow = sched.quantity.select_borrow(sched, "a", step=10)
        assert (borrow.home_policy, borrow.donor) == ("c", "a")


class TestBorrowBatchAndCooldown:
    def test_one_decision_adds_units_from_only_one_policy_pair(self):
        sched = _scheduler(
            policies=("a", "b", "c"),
            rebalance_confirm_polls=1,
        )
        sched.handles["b"].replicas.append(FakeReplica("b", 2))
        sched.ema_kv.update({"a": 0.95, "b": 0.05, "c": 0.1})

        plan = sched.decide(_stats(step=10))

        assert [(item.home_policy, item.donor) for item in plan.borrows] == [
            ("b", "a"),
        ]

    def test_successful_borrow_blocks_only_new_borrows_for_ten_seconds(
        self, monkeypatch,
    ):
        now = [100.0]
        monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: now[0])
        sched = _scheduler(
            policies=("a", "b", "c"),
            rebalance_confirm_polls=1,
            borrow_cooldown_s=10,
        )
        b0 = sched.handles["b"].replicas[0]
        lend = LendRecord(
            lend_id=1, home_policy="b", donor="a",
            home_replicas=[b0], guests=[object()], since_step=10,
        )
        sched.note_borrow_started(lend)
        sched._settle_polls_remaining = 0
        sched.ema_kv.update({"a": 0.95, "b": 0.1, "c": 0.1})

        cooling = sched.decide(_stats(step=10))
        assert cooling.returns == []
        assert sched.active_lends == [lend]
        assert cooling.borrows == []
        assert sched.metrics_snapshot()["borrow_cooldown_remaining_s"] == 10.0

        now[0] = 110.0
        after_cooldown = sched.decide(_stats(step=10))
        assert after_cooldown.returns == []
        assert sched.active_lends == [lend]
        assert [(item.home_policy, item.donor) for item in after_cooldown.borrows] == [
            ("c", "a"),
        ]

    def test_borrow_cooldown_does_not_block_a_return(self, monkeypatch):
        monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: 100.0)
        sched = _scheduler(
            rebalance_confirm_polls=1,
            min_lend_polls=1,
            borrow_cooldown_s=10,
        )
        lend = LendRecord(
            lend_id=1, home_policy="b", donor="a",
            home_replicas=[sched.handles["b"].replicas[0]],
            guests=[object()], since_step=10,
        )
        sched.note_borrow_started(lend)
        sched._settle_polls_remaining = 0
        sched._return_low_polls["a"] = sched.config.return_confirm_polls

        plan = sched.decide(_stats(step=10))

        assert plan.returns == [lend]
        assert plan.borrows == []


# ------------------------------------------------------------- early return
class TestEarlyReturn:
    def test_one_early_return_then_all_pairs_wait_for_settle(self):
        sched = _scheduler(policies=("a", "b", "c"))
        lends = [LendRecord(
            lend_id=i, home_policy=home, donor="a",
            home_replicas=[sched.handles[home].replicas[0]],
            guests=[object()], since_step=10,
        ) for i, home in enumerate(("b", "c"), 1)]
        for lend in lends:
            sched.note_borrow_started(lend)
        sched._settle_polls_remaining = 0
        for lend in lends:
            lend.home_kv_hot_polls = sched.config.early_return_confirm_polls
        assert sched.decide_early_returns() == [lends[0]]
        sched.note_lend_returned(lends[0], step=11, early=True)
        assert sched.decide_early_returns() == []
        plan = sched.decide(_stats(step=11))
        assert plan.returns == [] and plan.borrows == []
        assert lends[1] in sched.active_lends
        assert lends[1] not in plan.returns
        assert lends[1].home_kv_hot_polls == 0
        assert sched._settle_polls_remaining == sched.config.rebalance_settle_polls

    def test_persistent_home_kv_saturation_triggers_early_return(self):
        sched = _scheduler()
        lend = LendRecord(
            lend_id=1, home_policy="b", donor="a",
            home_replicas=[sched.handles["b"].replicas[0]],
            guests=[object()], since_step=10,
        )
        sched.note_borrow_started(lend)
        store = sched.signals
        _feed(store, "a", {"a-rep0": 4}, kv={"a-rep0": 0.5})
        _feed(store, "b", {"b-rep1": 70}, kv={"b-rep1": 0.95})
        for _ in range(sched.config.rebalance_settle_polls - 1):
            assert sched.poll() is True
            assert sched.decide_early_returns() == []
            assert lend.home_kv_hot_polls == 0
        sched.poll()
        assert sched.decide_early_returns() == [] and lend.home_kv_hot_polls == 1
        for _ in range(sched.config.early_return_confirm_polls - 1):
            sched.poll()
        assert sched.decide_early_returns() == [lend]
        assert lend in sched.active_lends  # Planning does not execute the return.
        # A complete cool KV sample resets the counter.
        lend.home_kv_hot_polls = 0
        cool = SignalStore(["a", "b"], max_num_seqs={"a": 32, "b": 32})
        _feed(cool, "a", {"a-rep0": 1}, kv={"a-rep0": 0.5})
        _feed(cool, "b", {"b-rep0": 1}, kv={"b-rep0": 0.2})
        sched.signals = cool
        sched.poll()
        assert lend.home_kv_hot_polls == 0

    def test_missing_kv_metrics_does_not_trigger_early_return(self):
        sched = _scheduler()
        lend = LendRecord(
            lend_id=1, home_policy="b", donor="a",
            home_replicas=[sched.handles["b"].replicas[0]],
            guests=[object()], since_step=10,
        )
        sched.note_borrow_started(lend)

        for _ in range(sched.config.early_return_confirm_polls + 1):
            sched.poll()
        assert sched.decide_early_returns() == []
        assert lend.home_kv_hot_polls == 0


class TestFailureDegradation:
    def test_disable_notification_is_idempotent(self):
        sched = _scheduler()

        sched.note_disabled_by_failures()
        sched.note_disabled_by_failures()

        snapshot = sched.metrics_snapshot()
        assert snapshot["disabled"] is True
        assert snapshot["disabled_by_failures"] == 1


# ------------------------------------------------------------- capacity math
class TestCapacityAccounting:
    def test_capacity_tracks_borrows_and_returns(self):
        sched = _scheduler()
        # 2 replicas x 8 cards each
        assert sched.capacity_cards("a") == 16
        assert sched.capacity_cards("b") == 16
        lend = LendRecord(
            lend_id=1, home_policy="b", donor="a",
            home_replicas=[sched.handles["b"].replicas[0]],
            guests=[object()], since_step=10,
        )
        sched.note_borrow_started(lend)
        # b lost one home replica (8 cards), a gained one guest (8 cards)
        assert sched.capacity_cards("b") == 8
        assert sched.capacity_cards("a") == 24
        sched.note_lend_returned(lend, step=11)
        assert sched.capacity_cards("a") == 16
        assert sched.capacity_cards("b") == 16

    def test_borrow_then_reborrow_skips_lent_replica(self):
        sched = _scheduler()
        home_replicas = sched.handles["b"].replicas
        lend = LendRecord(lend_id=1, home_policy="b", donor="a",
                          home_replicas=[home_replicas[0]], guests=[object()],
                          since_step=10)
        sched.note_borrow_started(lend)
        avail = sched._available_home_replicas("b")
        assert [r.replica_rank for r in avail] == [1]

    def test_units_ordered_by_migration_cost(self):
        sched = _scheduler()
        sched.handles["b"].replicas = [FakeReplica("b", i) for i in range(3)]
        store = sched.signals
        _feed(store, "b", {"b-rep0": 30, "b-rep1": 1, "b-rep2": 10})
        units = sched._available_home_units("b", "a")
        # consecutive-slice fallback groups [0],[1],[2]; ordered by inflight
        assert [u[0].replica_rank for u in units] == [1, 2, 0]


class TestMetricsSnapshot:
    def test_snapshot_shape(self):
        sched = _scheduler()
        snap = sched.metrics_snapshot()
        assert snap["pending_borrow_pair"] is None
        assert snap["borrow_confirmations"] == 0
        assert snap["disabled"] is False
        assert snap["active_lends"] == 0
        assert set(snap["kv_util"]) == {"a", "b"}
        assert snap["kv_util/a"] == pytest.approx(sched.ema_kv["a"])
        assert snap["kv_util/b"] == pytest.approx(sched.ema_kv["b"])
        assert "swap_episodes" in snap and "early_returns" in snap
        assert "renewals" not in snap
