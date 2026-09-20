import importlib.util
import sys
import types
from pathlib import Path


def _load_single_ppo_module(monkeypatch):
    class FakePPOTrainerSync:
        def _setup(self):
            self.setup_calls += 1

    sync_module = types.ModuleType("verl.trainer.ppo.v1.trainer_sync")
    sync_module.PPOTrainerSync = FakePPOTrainerSync
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1.trainer_sync", sync_module)

    module_path = Path(__file__).resolve().parents[3] / "uni_agent" / "trainer" / "single_ppo_trainer.py"
    spec = importlib.util.spec_from_file_location("single_ppo_trainer_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sync_trainer_exposes_only_training_runtime_initialization(monkeypatch):
    trainer_module = _load_single_ppo_module(monkeypatch)
    trainer = object.__new__(trainer_module.SinglePPOTrainer)
    trainer.setup_calls = 0

    trainer.init_training_runtime()

    assert trainer.setup_calls == 1
    assert not hasattr(trainer_module.SinglePPOTrainer, "init_runtime")
