"""Tests for the dynamic-inference configuration types and validation."""

import pytest

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

from uni_agent.trainer.dynamic_inference.types import (
    BorrowPairSpec,
    SchedulingConfig,
    expand_pairs,
    required_slots_by_home,
    parse_scheduling_config,
    validate_against_handles,
    validate_policy_prerequisites,
)


def _cfg(**overrides):
    base = {"enable": True, "mode": "resource"}
    base.update(overrides)
    return OmegaConf.create(base) if OmegaConf is not None else base


class TestParseSchedulingConfig:
    def test_absent_or_disabled_returns_none(self):
        assert parse_scheduling_config(None) is None
        assert parse_scheduling_config({}) is None
        assert parse_scheduling_config({"enable": False, "mode": "resource"}) is None

    def test_defaults(self):
        cfg = parse_scheduling_config(_cfg())
        assert isinstance(cfg, SchedulingConfig)
        assert cfg.mode == "resource"
        assert cfg.resource_usage.kv_enter == 0.85
        assert cfg.resource_usage.kv_exit == 0.6
        assert cfg.resource_usage.kv_post_lend_max == 0.7
        assert cfg.resource_usage.kv_metric_names == [
            "kv_cache_usage_perc", "gpu_cache_usage_perc", "kv_cache_usage_ratio"]
        assert cfg.resource_usage.ema_alpha == 0.3
        assert cfg.metrics_scrape_interval_s == 1.0
        assert cfg.bottleneck_confirm_polls == 10
        assert cfg.rebalance_confirm_polls == 2
        assert cfg.rebalance_settle_polls == 3
        assert cfg.min_lend_polls == 2
        assert cfg.borrow_cooldown_s == 10.0
        assert cfg.return_confirm_polls == 10
        assert cfg.early_return_confirm_polls == 10
        assert cfg.borrowing.guest_replica_rank_offset == 10000

    def test_plain_dict_and_omegaconf_agree(self):
        if OmegaConf is None:
            pytest.skip("omegaconf is not installed")
        from_dict = parse_scheduling_config({"enable": True, "mode": "static"})
        from_oc = parse_scheduling_config(_cfg(mode="static"))
        assert from_dict.mode == from_oc.mode == "static"

    def test_unsupported_modes_rejected(self):
        for mode, message in (
            ("coverage", "removed"),
            ("fused", "removed"),
            ("oracle", "unknown"),
        ):
            with pytest.raises(ValueError, match=message):
                parse_scheduling_config(_cfg(mode=mode))

    @pytest.mark.parametrize(
        "removed",
        [
            {"quantity_strategy": "equalisation"},
            {"min_rebalance_gain": 0.03},
            {"policy_specs": {"a": {"num_params": 7e9}}},
            {"borrowing": {"precreate_guest_replicas": True}},
            {"borrowing": {"guest_load_format": "dummy"}},
            {"borrowing": {"guest_max_colocate_count": 2}},
            {"borrowing": {"rebalance_backlog": "off"}},
        ],
    )
    def test_removed_configuration_is_rejected(self, removed):
        with pytest.raises(ValueError, match="unknown .* configuration keys"):
            parse_scheduling_config(_cfg(**removed))

    def test_heterogeneous_folding_parsed(self):
        pairs = [{"home": "a", "donor": "b", "home_replicas_per_unit": 2,
                  "guest_replicas_per_unit": 1}]
        cfg = parse_scheduling_config(_cfg(borrowing={"pairs": pairs}))
        assert cfg.borrowing.pairs[0].home_replicas_per_unit == 2
        assert cfg.borrowing.pairs[0].guest_replicas_per_unit == 1

    def test_invalid_borrowing_integers_rejected(self):
        invalid = (
            ({"pairs": [{
                "home": "a", "donor": "b", "home_replicas_per_unit": 0,
            }]}, "must both be positive"),
            ({"pairs": [{
                "home": "a", "donor": "b", "home_replicas_per_unit": 1.5,
            }]}, "positive integers"),
            ({"guest_replica_rank_offset": 1.5}, "non-negative integer"),
        )
        for borrowing, message in invalid:
            with pytest.raises(ValueError, match=message):
                parse_scheduling_config(_cfg(borrowing=borrowing))

    def test_self_and_duplicate_pairs_rejected(self):
        with pytest.raises(ValueError, match="self-pair"):
            parse_scheduling_config(_cfg(borrowing={"pairs": [{"home": "a", "donor": "a"}]}))
        dup = [{"home": "a", "donor": "b"}, {"home": "a", "donor": "b"}]
        with pytest.raises(ValueError, match="duplicate"):
            parse_scheduling_config(_cfg(borrowing={"pairs": dup}))

    def test_kv_thresholds_parsed(self):
        cfg = parse_scheduling_config(_cfg(resource_usage={
            "kv_enter": 0.9, "kv_exit": 0.5, "kv_post_lend_max": 0.8,
            "kv_metric_names": ["kv_cache_usage_ratio"],
        }))
        assert cfg.resource_usage.kv_enter == 0.9
        assert cfg.resource_usage.kv_exit == 0.5
        assert cfg.resource_usage.kv_post_lend_max == 0.8
        assert cfg.resource_usage.kv_metric_names == ["kv_cache_usage_ratio"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rebalance_confirm_polls", 0),
            ("rebalance_confirm_polls", 1.5),
            ("rebalance_settle_polls", 0),
            ("min_lend_polls", 0),
            ("borrow_cooldown_s", -0.1),
            ("return_confirm_polls", 0),
            ("poll_interval_s", 0),
            ("metrics_scrape_interval_s", 0),
            ("borrow_drain_timeout_s", 0),
        ],
    )
    def test_invalid_anti_jitter_values_rejected(self, field, value):
        with pytest.raises(ValueError):
            parse_scheduling_config(_cfg(**{field: value}))

    def test_legacy_step_confirmation_names_are_mapped_to_polls(self):
        cfg = parse_scheduling_config(_cfg(
            bottleneck_confirm_steps=3,
            rebalance_confirm_steps=4,
            min_lend_steps=5,
        ))
        assert cfg.bottleneck_confirm_polls == 3
        assert cfg.rebalance_confirm_polls == 4
        assert cfg.min_lend_polls == 5

    def test_legacy_lb_early_return_threshold_is_ignored(self):
        cfg = parse_scheduling_config(_cfg(early_return_waiting_ratio=0))
        assert isinstance(cfg, SchedulingConfig)

    def test_legacy_and_poll_confirmation_names_cannot_be_mixed(self):
        with pytest.raises(ValueError, match="only one"):
            parse_scheduling_config(_cfg(
                bottleneck_confirm_steps=2,
                bottleneck_confirm_polls=2,
            ))

    @pytest.mark.parametrize(
        "resource_usage",
        [
            {"kv_exit": 0.9, "kv_enter": 0.8},
            {"kv_post_lend_max": 1.1},
            {"kv_post_lend_max": 0.85},
            {"kv_post_lend_max": 0.6},
            {"ema_alpha": 0.0},
            {"kv_enter": "0.9"},
            {"kv_metric_names": []},
        ],
    )
    def test_invalid_resource_thresholds_rejected(self, resource_usage):
        with pytest.raises(ValueError):
            parse_scheduling_config(_cfg(resource_usage=resource_usage))

class TestExpandPairs:
    def test_auto_two_policies(self):
        cfg = parse_scheduling_config(_cfg())
        pairs = expand_pairs(cfg, ["a", "b"])
        assert [(p.home, p.donor) for p in pairs] == [("a", "b"), ("b", "a")]
        assert all(p.home_replicas_per_unit == 1 and p.guest_replicas_per_unit == 1 for p in pairs)

    def test_auto_three_policies_builds_complete_directed_graph(self):
        cfg = parse_scheduling_config(_cfg())
        pairs = expand_pairs(cfg, ["a", "b", "c"])
        assert len(pairs) == 6
        assert required_slots_by_home(pairs, ["a", "b", "c"]) == {
            "a": 3, "b": 3, "c": 3,
        }

    def test_explicit_pairs_kept(self):
        pairs = [{"home": "a", "donor": "b"}]
        cfg = parse_scheduling_config(_cfg(borrowing={"pairs": pairs}))
        result = expand_pairs(cfg, ["a", "b"])
        assert [(p.home, p.donor) for p in result] == [("a", "b")]

    def test_explicit_pair_unknown_policy_rejected(self):
        pairs = [{"home": "a", "donor": "zzz"}]
        cfg = parse_scheduling_config(_cfg(borrowing={"pairs": pairs}))
        with pytest.raises(ValueError, match="unknown policy"):
            expand_pairs(cfg, ["a", "b"])


def _rollout(name="vllm", backend="nccl", sleep=True, free=True, nnodes=1, gpus=8):
    data = {
        "name": name,
        "nnodes": nnodes,
        "n_gpus_per_node": gpus,
        "enable_sleep_mode": sleep,
        "free_cache_engine": free,
        "checkpoint_engine": {"backend": backend},
    }
    return OmegaConf.create(data) if OmegaConf is not None else data


class TestValidatePolicyPrerequisites:
    def test_valid(self):
        cfg = parse_scheduling_config(_cfg())
        validate_policy_prerequisites(cfg, {"a": _rollout(), "b": _rollout()})

    def test_naive_backend_rejected(self):
        cfg = parse_scheduling_config(_cfg())
        with pytest.raises(ValueError, match="forbids naive"):
            validate_policy_prerequisites(cfg, {"a": _rollout(backend="naive")})

    def test_non_vllm_rejected(self):
        cfg = parse_scheduling_config(_cfg())
        with pytest.raises(ValueError, match="must be 'vllm'"):
            validate_policy_prerequisites(cfg, {"a": _rollout(name="sglang")})

    def test_sleep_mode_required(self):
        cfg = parse_scheduling_config(_cfg())
        with pytest.raises(ValueError, match="enable_sleep_mode"):
            validate_policy_prerequisites(cfg, {"a": _rollout(sleep=False)})

    def test_all_problems_reported_at_once(self):
        cfg = parse_scheduling_config(_cfg())
        with pytest.raises(ValueError) as err:
            validate_policy_prerequisites(cfg, {
                "a": _rollout(backend="naive"),
                "b": _rollout(sleep=False),
            })
        assert "policy 'a'" in str(err.value)
        assert "policy 'b'" in str(err.value)


class TestValidateAgainstHandles:
    def _handles(self, n_replicas=2, cards=8, nnodes=1):
        from uni_agent.trainer.dynamic_inference.types import PolicyInferenceHandles

        handles = {}
        for name in ("a", "b"):
            handles[name] = PolicyInferenceHandles(
                actor_rollout_wg=None,
                standalone_checkpoint_manager=None,
                lb_handle=None, rollout_config=None, model_config=None,
                replicas=[object() for _ in range(n_replicas)],
                cards_per_replica=cards, nnodes=nnodes,
            )
        return handles

    def _pairs(self):
        return [BorrowPairSpec(home="a", donor="b"), BorrowPairSpec(home="b", donor="a")]

    def test_valid(self):
        cfg = parse_scheduling_config(_cfg())
        validate_against_handles(cfg, self._handles(), self._pairs())

    def test_too_few_replicas(self):
        cfg = parse_scheduling_config(_cfg())
        with pytest.raises(ValueError, match="N\\+1=2"):
            validate_against_handles(cfg, self._handles(n_replicas=1), self._pairs())

    def test_unequal_cards_rejected(self):
        cfg = parse_scheduling_config(_cfg())
        handles = self._handles()
        handles["b"].cards_per_replica = 16
        with pytest.raises(ValueError, match="card equation"):
            validate_against_handles(cfg, handles, self._pairs())

    def test_heterogeneous_aggregate_and_split_are_valid(self):
        cfg = parse_scheduling_config(_cfg())
        handles = self._handles(n_replicas=3)
        handles["b"].cards_per_replica = 16
        handles["b"].nnodes = 2
        pairs = [
            BorrowPairSpec(home="a", donor="b", home_replicas_per_unit=2,
                           guest_replicas_per_unit=1),
            BorrowPairSpec(home="b", donor="a", home_replicas_per_unit=1,
                           guest_replicas_per_unit=2),
        ]
        validate_against_handles(cfg, handles, pairs)

    def test_heterogeneous_per_node_layout_can_be_folded_at_runtime(self):
        cfg = parse_scheduling_config(_cfg())
        handles = self._handles(n_replicas=3)
        handles["b"].cards_per_replica = 16
        handles["b"].nnodes = 1
        pair = BorrowPairSpec(home="a", donor="b", home_replicas_per_unit=2)
        validate_against_handles(cfg, handles, [pair])

    def test_multiple_outgoing_pairs_are_valid(self):
        cfg = parse_scheduling_config(_cfg())
        pairs = [
            BorrowPairSpec(home="a", donor="b"),
            BorrowPairSpec(home="a", donor="c"),
        ]
        validate_against_handles(cfg, self._handles(), pairs)

    def test_home_pool_must_have_one_slot_per_outgoing_edge(self):
        from types import SimpleNamespace

        cfg = parse_scheduling_config(_cfg())
        handles = self._handles()
        handles["a"].replicas = [
            SimpleNamespace(resource_pool=SimpleNamespace(max_colocate_count=2))
            for _ in range(2)
        ]
        pairs = [
            BorrowPairSpec(home="a", donor="b"),
            BorrowPairSpec(home="a", donor="c"),
        ]
        with pytest.raises(ValueError, match="needs at least 3"):
            validate_against_handles(cfg, handles, pairs)
