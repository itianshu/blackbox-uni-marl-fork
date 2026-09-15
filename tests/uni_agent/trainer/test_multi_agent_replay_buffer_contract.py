import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class FakeKVBatchMeta:
    partition_id: str
    keys: list
    tags: list


class FakeTransferQueueModule(types.ModuleType):
    def __init__(self):
        super().__init__("transfer_queue")
        self.KVBatchMeta = FakeKVBatchMeta
        self.partitions = {}
        self.cleared = []

    def kv_list(self, partition_id=None):
        if partition_id is None:
            return {
                partition: {
                    key: dict(record["tag"])
                    for key, record in items.items()
                }
                for partition, items in self.partitions.items()
            }
        return {
            partition_id: {
                key: dict(record["tag"])
                for key, record in self.partitions.get(partition_id, {}).items()
            }
        }

    def kv_put(self, *, key, partition_id, tag, fields=None):
        self.partitions.setdefault(partition_id, {})[key] = {
            "tag": dict(tag),
            "fields": fields,
        }

    def kv_clear(self, *, partition_id, keys):
        self.cleared.append({"partition_id": partition_id, "keys": list(keys)})
        partition = self.partitions.setdefault(partition_id, {})
        for key in keys:
            partition.pop(key, None)


def _load_v1_replay_buffer_with_fake_tq(fake_tq):
    import verl

    original_modules = {
        name: sys.modules.get(name)
        for name in ("transfer_queue", "omegaconf", "verl.utils.skip")
    }
    try:
        sys.modules["transfer_queue"] = fake_tq

        omegaconf_mod = types.ModuleType("omegaconf")
        omegaconf_mod.DictConfig = dict
        sys.modules["omegaconf"] = omegaconf_mod

        skip_mod = types.ModuleType("verl.utils.skip")

        class FakeSkipManager:
            @staticmethod
            def annotate_tq(**_kwargs):
                def decorator(fn):
                    return fn

                return decorator

        skip_mod.SkipManager = FakeSkipManager
        sys.modules["verl.utils.skip"] = skip_mod

        module_path = Path(verl.__file__).resolve().parent / "trainer" / "ppo" / "v1" / "replay_buffer.py"
        if not module_path.is_file():
            pytest.skip(f"Installed verl package does not expose replay_buffer.py: {module_path}")
        spec = importlib.util.spec_from_file_location("_uni_agent_test_v1_replay_buffer", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.ReplayBuffer
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_v1_replay_buffer_samples_multi_agent_role_records_by_prompt_uid():
    fake_tq = FakeTransferQueueModule()
    ReplayBuffer = _load_v1_replay_buffer_with_fake_tq(fake_tq)
    partition_id = "train"
    uid = "promptuid"
    role_policy_mapping = {
        "agent_1": "policy_1",
        "agent_2": "policy_1",
        "agent_3": "policy_2",
    }
    expected_keys = []
    expected_tags = []
    for sample_idx in range(2):
        rollout_id = f"rollout-{sample_idx}"
        for record_idx, (role, policy_name) in enumerate(role_policy_mapping.items()):
            key = f"{uid}_{sample_idx}_{record_idx}"
            tag = {
                "is_prompt": False,
                "global_steps": 0,
                "seq_len": 3,
                "uid": uid,
                "rollout_id": rollout_id,
                "sample_idx": sample_idx,
                "record_idx": record_idx,
                "role": role,
                "policy_name": policy_name,
            }
            fake_tq.kv_put(
                key=key,
                partition_id=partition_id,
                tag=tag,
                fields={"input_ids": [1, 2, 3]},
            )
            expected_keys.append(key)
            expected_tags.append(tag)
    fake_tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": "finished", "global_steps": 0},
    )

    replay_buffer = ReplayBuffer(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=8,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0.0,
    )

    batch, metrics = replay_buffer.sample(global_steps=0, partition_id=partition_id, batch_size=1)

    assert metrics == {}
    assert batch.partition_id == partition_id
    assert batch.keys == expected_keys
    assert batch.tags == expected_tags
    assert fake_tq.cleared == [{"partition_id": partition_id, "keys": [uid]}]
