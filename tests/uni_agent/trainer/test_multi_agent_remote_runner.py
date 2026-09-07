from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from functools import partial
from types import SimpleNamespace

import pytest


def test_remote_runner_invocation_loads_fqn_and_forwards_serializable_inputs(monkeypatch):
    from examples.multi_agent_blackbox import remote_runner

    captured = {}

    async def runner(**kwargs):
        captured.update(kwargs)
        return {"reward_info": {"score": 1.0}}

    monkeypatch.setattr(remote_runner, "load_class_from_fqn", lambda fqn, **_kwargs: runner)

    result = asyncio.run(
        remote_runner.run_multi_agent_runner(
            runner_fqn="tests.fake.runner",
            raw_prompt="task",
            rollout=SimpleNamespace(base_url="http://gateway", sessions={"agent": "session"}),
            sample_index=3,
            role_policy_mapping={"agent": "policy_1"},
            runner_kwargs={"max_tokens": 128},
        )
    )

    assert result == {"reward_info": {"score": 1.0}}
    assert captured["raw_prompt"] == "task"
    assert captured["sample_index"] == 3
    assert captured["role_policy_mapping"] == {"agent": "policy_1"}
    assert captured["max_tokens"] == 128
    assert captured["session_runtime"] is not None


def test_multi_agent_framework_submits_runner_to_remote_task(monkeypatch):
    from examples.multi_agent_blackbox import framework as framework_module

    calls = []

    class FakeRuntime:
        async def create_multi_agent_rollout(self, rollout_id, **kwargs):
            return SimpleNamespace(
                rollout_id=rollout_id,
                base_url="http://gateway/rollouts/x/v1",
                sessions={"agent": "session"},
            )

        async def complete_multi_agent_rollout(self, rollout_id, reward_info=None):
            calls.append(("complete", rollout_id, reward_info))

        async def finalize_multi_agent_rollout(self, rollout_id):
            calls.append(("finalize", rollout_id))
            return SimpleNamespace(rollout_id=rollout_id, trajectories=[])

        async def abort_multi_agent_rollout(self, rollout_id):
            calls.append(("abort", rollout_id))

    class FakeRemote:
        def remote(self, **kwargs):
            calls.append(("remote", kwargs))
            result = concurrent.futures.Future()
            result.set_result({"reward_info": {"score": 2.0}})
            return SimpleNamespace(future=lambda: result)

    monkeypatch.setattr(framework_module, "remote_multi_agent_run", FakeRemote())

    async def runner(**_kwargs):
        return None

    framework = object.__new__(framework_module.RemoteMultiAgentFramework)
    framework.session_runtime = FakeRuntime()
    framework.multi_agent_runner = partial(runner, bound_value=7)
    framework.role_policy_mapping = {"agent": "policy_1"}
    framework._remote_tasks = {}
    framework.wait_for_completion_after_runner = False

    result = asyncio.run(
        framework.run_rollout(
            raw_prompt="task",
            rollout_id="rollout-1",
            sample_index=4,
            runner_kwargs={"max_tokens": 64},
        )
    )

    assert result.rollout_id == "rollout-1"
    remote_call = next(call for call in calls if call[0] == "remote")[1]
    assert remote_call["runner_fqn"].endswith(".runner")
    assert remote_call["raw_prompt"] == "task"
    assert remote_call["sample_index"] == 4
    assert remote_call["runner_kwargs"]["bound_value"] == 7
    assert remote_call["runner_kwargs"]["max_tokens"] == 64
    assert "session_runtime" not in remote_call
    assert any(call[0] == "complete" for call in calls)
    assert any(call[0] == "finalize" for call in calls)
    assert framework._remote_tasks == {}


def test_multi_agent_framework_aborts_gateway_rollout_when_remote_task_is_cancelled(monkeypatch):
    from examples.multi_agent_blackbox import framework as framework_module

    calls = []

    class FakeRuntime:
        async def create_multi_agent_rollout(self, rollout_id, **kwargs):
            return SimpleNamespace(rollout_id=rollout_id, base_url="http://gateway", sessions={})

        async def abort_multi_agent_rollout(self, rollout_id):
            calls.append(("abort", rollout_id))

    class FakeRemote:
        def remote(self, **kwargs):
            result = concurrent.futures.Future()
            result.cancel()
            return SimpleNamespace(future=lambda: result)

    async def runner(**_kwargs):
        return None

    monkeypatch.setattr(framework_module, "remote_multi_agent_run", FakeRemote())

    framework = object.__new__(framework_module.RemoteMultiAgentFramework)
    framework.session_runtime = FakeRuntime()
    framework.multi_agent_runner = runner
    framework.role_policy_mapping = {"agent": "policy_1"}
    framework._remote_tasks = {}
    framework.wait_for_completion_after_runner = False

    try:
        asyncio.run(framework.run_rollout(raw_prompt="task", rollout_id="rollout-cancelled"))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("expected CancelledError")

    assert ("abort", "rollout-cancelled") in calls
    assert framework._remote_tasks == {}


def test_multi_agent_framework_aborts_if_cancelled_while_gateway_rollout_is_created():
    from uni_agent.trainer.framework import framework as framework_module

    create_started = asyncio.Event()
    allow_create_to_finish = asyncio.Event()
    calls = []

    class FakeRuntime:
        async def create_multi_agent_rollout(self, rollout_id, **_kwargs):
            create_started.set()
            await allow_create_to_finish.wait()
            calls.append(("create", rollout_id))
            return SimpleNamespace(rollout_id=rollout_id, base_url="http://gateway", sessions={})

        async def abort_multi_agent_rollout(self, rollout_id):
            calls.append(("abort", rollout_id))

    async def runner(**_kwargs):
        return None

    framework = object.__new__(framework_module.MultiAgentFramework)
    framework.session_runtime = FakeRuntime()
    framework.multi_agent_runner = runner
    framework.role_policy_mapping = {"agent": "policy_1"}
    framework.wait_for_completion_after_runner = False

    async def cancel_during_create():
        task = asyncio.create_task(
            framework.run_rollout(raw_prompt="task", rollout_id="rollout-create")
        )
        await create_started.wait()
        task.cancel()
        allow_create_to_finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_during_create())

    assert calls == [
        ("create", "rollout-create"),
        ("abort", "rollout-create"),
    ]


def test_framework_shutdown_aborts_rollout_created_during_cancellation():
    from uni_agent.trainer.framework import framework as framework_module

    create_started = threading.Event()
    calls = []

    class FakeRuntime:
        allow_create_to_finish = None

        async def create_multi_agent_rollout(self, rollout_id, **_kwargs):
            self.allow_create_to_finish = asyncio.Event()
            create_started.set()
            await self.allow_create_to_finish.wait()
            calls.append(("create", rollout_id))
            return SimpleNamespace(rollout_id=rollout_id, base_url="http://gateway", sessions={})

        async def abort_multi_agent_rollout(self, rollout_id):
            calls.append(("abort", rollout_id))

    async def runner(**_kwargs):
        raise AssertionError("runner must not start after shutdown cancellation")

    runtime = FakeRuntime()
    framework = object.__new__(framework_module.MultiAgentFramework)
    framework.session_runtime = runtime
    framework.multi_agent_runner = runner
    framework.role_policy_mapping = {"agent": "policy_1"}
    framework.wait_for_completion_after_runner = False
    framework._bg_loop = None
    framework._bg_thread = None
    framework._bg_tasks = set()
    framework._bg_drain_future = None

    loop = framework._ensure_background_loop()
    task = asyncio.run_coroutine_threadsafe(
        framework.run_rollout(raw_prompt="task", rollout_id="rollout-shutdown-create"),
        loop,
    )
    framework._bg_tasks.add(task)
    assert create_started.wait(timeout=2.0)

    release_timer = threading.Timer(
        0.05,
        lambda: loop.call_soon_threadsafe(runtime.allow_create_to_finish.set),
    )
    release_timer.start()
    framework.shutdown(timeout=2.0)
    release_timer.join(timeout=2.0)

    assert calls == [
        ("create", "rollout-shutdown-create"),
        ("abort", "rollout-shutdown-create"),
    ]


def test_framework_shutdown_waits_for_cancelled_coroutine_cleanup():
    from uni_agent.trainer.framework import framework as framework_module

    started = threading.Event()
    cleaned_up = threading.Event()

    async def background_work():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned_up.set()

    framework = object.__new__(framework_module.MultiAgentFramework)
    framework._bg_loop = None
    framework._bg_thread = None
    framework._bg_tasks = set()

    loop = framework._ensure_background_loop()
    task = asyncio.run_coroutine_threadsafe(background_work(), loop)
    framework._bg_tasks.add(task)
    assert started.wait(timeout=2.0)

    framework.shutdown(timeout=2.0)

    assert cleaned_up.is_set()
    assert task.done()
    assert loop.is_closed()


def test_framework_shutdown_timeout_preserves_loop_for_a_later_retry():
    from uni_agent.trainer.framework import framework as framework_module

    started = threading.Event()
    release_cleanup = None

    async def background_work():
        nonlocal release_cleanup
        release_cleanup = asyncio.Event()
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await release_cleanup.wait()

    framework = object.__new__(framework_module.MultiAgentFramework)
    framework._bg_loop = None
    framework._bg_thread = None
    framework._bg_tasks = set()

    loop = framework._ensure_background_loop()
    task = asyncio.run_coroutine_threadsafe(background_work(), loop)
    framework._bg_tasks.add(task)
    assert started.wait(timeout=2.0)

    with pytest.raises(TimeoutError, match="draining"):
        framework.shutdown(timeout=0.05)

    assert framework._bg_loop is loop
    assert framework._bg_thread is not None and framework._bg_thread.is_alive()
    assert not loop.is_closed()

    loop.call_soon_threadsafe(release_cleanup.set)
    framework.shutdown(timeout=2.0)
    assert task.done()
    assert loop.is_closed()


def test_remote_framework_cancels_ray_task_when_local_wait_is_cancelled(monkeypatch):
    from examples.multi_agent_blackbox import framework as framework_module

    remote_future = concurrent.futures.Future()
    remote_ref = SimpleNamespace(future=lambda: remote_future)
    cancelled_refs = []

    class FakeRemote:
        def remote(self, **_kwargs):
            return remote_ref

    monkeypatch.setattr(framework_module, "remote_multi_agent_run", FakeRemote())
    monkeypatch.setattr(
        framework_module.ray,
        "cancel",
        lambda ref, force=False: cancelled_refs.append((ref, force)),
    )
    monkeypatch.setattr(
        framework_module.ray,
        "wait",
        lambda refs, **_kwargs: ([], list(refs)),
    )

    async def runner(**_kwargs):
        return None

    framework = object.__new__(framework_module.RemoteMultiAgentFramework)
    framework.multi_agent_runner = runner
    framework.role_policy_mapping = {"agent": "policy_1"}
    framework._remote_tasks = {}

    async def cancel_local_wait():
        task = asyncio.create_task(
            framework._execute_multi_agent_runner(
                raw_prompt="task",
                rollout=SimpleNamespace(base_url="http://gateway", sessions={}),
                rollout_id="rollout-cancel",
                sample_index=0,
                runner_kwargs=None,
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(cancel_local_wait())

    assert cancelled_refs == [(remote_ref, False), (remote_ref, True)]
    assert framework._remote_tasks == {}


def test_remote_framework_applies_configured_ray_task_options(monkeypatch):
    from examples.multi_agent_blackbox import framework as framework_module

    calls = []
    result = concurrent.futures.Future()
    result.set_result({"reward_info": {}})

    class FakeRemote:
        def options(self, **kwargs):
            calls.append(("options", kwargs))
            return self

        def remote(self, **kwargs):
            calls.append(("remote", kwargs))
            return SimpleNamespace(future=lambda: result)

    monkeypatch.setattr(framework_module, "remote_multi_agent_run", FakeRemote())

    async def runner(**_kwargs):
        return None

    framework = object.__new__(framework_module.RemoteMultiAgentFramework)
    framework.multi_agent_runner = partial(
        runner,
        execution={
            "backend": "local_process",
            "ray": {
                "num_cpus": 1,
                "resources": {"mas_runner_slot": 1},
                "scheduling_strategy": "SPREAD",
                "cancel_grace_seconds": 3,
            },
        },
    )
    framework.role_policy_mapping = {"agent": "policy_1"}
    framework._remote_tasks = {}

    asyncio.run(
        framework._execute_multi_agent_runner(
            raw_prompt="task",
            rollout=SimpleNamespace(base_url="http://gateway", sessions={}),
            rollout_id="rollout-options",
            sample_index=0,
            runner_kwargs=None,
        )
    )

    assert calls[0] == (
        "options",
        {
            "num_cpus": 1,
            "resources": {"mas_runner_slot": 1},
            "scheduling_strategy": "SPREAD",
        },
    )
    remote_kwargs = calls[1][1]
    assert remote_kwargs["runner_kwargs"]["execution"] == {"backend": "local_process"}


def test_remote_framework_shutdown_waits_for_cancelled_ray_tasks(monkeypatch):
    from examples.multi_agent_blackbox import framework as framework_module

    refs = [object(), object()]
    events = []
    monkeypatch.setattr(
        framework_module.ray,
        "cancel",
        lambda ref, force=False: events.append(("cancel", ref, force)),
    )

    def wait(remote_refs, *, num_returns, timeout):
        events.append(("wait", list(remote_refs), num_returns, timeout))
        return list(remote_refs), []

    monkeypatch.setattr(framework_module.ray, "wait", wait)

    framework = object.__new__(framework_module.RemoteMultiAgentFramework)
    framework._remote_tasks = {"rollout-1": refs[0], "rollout-2": refs[1]}
    framework._bg_loop = None
    framework._bg_thread = None
    framework._bg_tasks = set()

    framework.shutdown(timeout=2.0)

    assert events[:2] == [
        ("cancel", refs[0], False),
        ("cancel", refs[1], False),
    ]
    assert events[2][0:3] == ("wait", refs, 2)
    assert framework._remote_tasks == {}


def test_remote_framework_shutdown_keeps_refs_when_ray_tasks_do_not_stop(monkeypatch):
    from examples.multi_agent_blackbox import framework as framework_module

    remote_ref = object()
    monkeypatch.setattr(framework_module.ray, "cancel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        framework_module.ray,
        "wait",
        lambda refs, **_kwargs: ([], list(refs)),
    )

    framework = object.__new__(framework_module.RemoteMultiAgentFramework)
    framework._remote_tasks = {"rollout-1": remote_ref}
    framework._bg_loop = None
    framework._bg_thread = None
    framework._bg_tasks = set()

    with pytest.raises(TimeoutError, match="remote MAS"):
        framework.shutdown(timeout=0.01)

    assert framework._remote_tasks == {"rollout-1": remote_ref}


def test_trainer_cleanup_stops_framework_before_releasing_ray_resources():
    from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

    events = []
    trainer = object.__new__(MultiAgentsPPOTrainer)
    trainer.agent_loop_manager = SimpleNamespace(
        framework=SimpleNamespace(shutdown=lambda: events.append("framework"))
    )
    trainer.policy_trainers = {
        "policy_1": SimpleNamespace(cleanup=lambda: events.append("policy")),
    }
    trainer._collect_placement_groups = lambda: ["policy-pg"]
    trainer._remove_placement_groups = lambda groups: events.append(("placement_groups", groups))
    trainer._shutdown_gateway_actors = lambda: events.append("gateways")
    trainer._policy_pool = SimpleNamespace(
        shutdown=lambda *, wait: events.append(("policy_pool", wait))
    )

    trainer.cleanup()

    assert events == [
        "framework",
        "policy",
        ("placement_groups", ["policy-pg"]),
        "gateways",
        ("policy_pool", True),
    ]


def test_trainer_cleanup_does_not_release_resources_if_framework_shutdown_fails():
    from uni_agent.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer

    events = []

    def fail_shutdown():
        events.append("framework")
        raise TimeoutError("rollouts still active")

    trainer = object.__new__(MultiAgentsPPOTrainer)
    trainer.agent_loop_manager = SimpleNamespace(
        framework=SimpleNamespace(shutdown=fail_shutdown)
    )
    trainer.policy_trainers = {
        "policy_1": SimpleNamespace(cleanup=lambda: events.append("policy")),
    }
    trainer._collect_placement_groups = lambda: []
    trainer._remove_placement_groups = lambda _groups: events.append("placement_groups")
    trainer._shutdown_gateway_actors = lambda: events.append("gateways")
    trainer._policy_pool = SimpleNamespace(shutdown=lambda **_kwargs: events.append("policy_pool"))

    with pytest.raises(TimeoutError, match="rollouts still active"):
        trainer.cleanup()

    assert events == ["framework"]
