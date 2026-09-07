from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def test_three_agent_external_mas_runs_through_command_runner():
    from examples.multi_agent_blackbox.external_mas_runner import external_mas_runner

    requests = []

    class GatewayHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(content_length))
            requests.append((self.path, payload))
            response = json.dumps(
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": f"{payload['model']}-output"}}
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), GatewayHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    gateway_url = f"http://127.0.0.1:{server.server_port}/v1"
    template_path = Path("examples/multi_agent_blackbox/config/external_mas_template.yaml").resolve()

    try:
        result = asyncio.run(
            external_mas_runner(
                raw_prompt="solve this task",
                rollout=SimpleNamespace(base_url=gateway_url, rollout_id="integration-rollout"),
                sample_index=0,
                session_runtime=None,
                role_policy_mapping={
                    "agent_1": "policy_1",
                    "agent_2": "policy_1",
                    "agent_3": "policy_2",
                },
                execution={"backend": "local_process"},
                command={
                    "argv": [
                        sys.executable,
                        "-m",
                        "examples.multi_agent_blackbox.scripts.three_agent_external_mas",
                        "--config",
                        "{config_path}",
                        "--prompt",
                        "{prompt}",
                    ],
                },
                config={"template_path": str(template_path)},
                result={"mode": "stdout_json"},
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    assert [payload["model"] for _, payload in requests] == ["agent_1", "agent_2", "agent_3"]
    assert all(path == "/v1/chat/completions" for path, _ in requests)
    assert all("temperature" not in payload for _, payload in requests)
    assert [payload["max_tokens"] for _, payload in requests] == [1024, 2048, 1024]
    assert requests[0][1]["messages"][-1]["content"] == "solve this task"
    assert "agent_1: agent_1-output" in requests[1][1]["messages"][-1]["content"]
    assert "agent_2: agent_2-output" in requests[2][1]["messages"][-1]["content"]
    assert result == {
        "final_result": "agent_3-output",
        "agent_outputs": {
            "agent_1": "agent_1-output",
            "agent_2": "agent_2-output",
            "agent_3": "agent_3-output",
        },
        "reward_info": {
            "final_result": "agent_3-output",
            "agent_outputs": {
                "agent_1": "agent_1-output",
                "agent_2": "agent_2-output",
                "agent_3": "agent_3-output",
            },
        },
    }


def test_inject_gateway_config_sets_role_models_without_mutating_template():
    from examples.multi_agent_blackbox.external_mas_runner import inject_gateway_config

    template = {
        "llm": {"base_url": "http://old-global", "api_key": "EMPTY"},
        "agents": {
            "planner": {"llm": {"base_url": "http://old-planner"}},
            "reviewer": {"system_prompt": "review"},
        },
    }

    injected = inject_gateway_config(
        template,
        gateway_url="http://gateway/rollouts/r-1/v1",
        roles=["planner", "reviewer"],
    )

    assert template["llm"]["base_url"] == "http://old-global"
    assert template["agents"]["planner"]["llm"]["base_url"] == "http://old-planner"
    assert injected["llm"]["base_url"] == "http://gateway/rollouts/r-1/v1"
    assert injected["agents"]["planner"] == {
        "llm": {"base_url": "http://gateway/rollouts/r-1/v1"},
        "model": "planner",
    }
    assert injected["agents"]["reviewer"] == {
        "system_prompt": "review",
        "llm": {"base_url": "http://gateway/rollouts/r-1/v1"},
        "model": "reviewer",
    }


def test_external_runner_executes_command_and_removes_temporary_config(tmp_path):
    from examples.multi_agent_blackbox.external_mas_runner import external_mas_runner

    template_path = tmp_path / "mas-template.yaml"
    template_path.write_text(
        yaml.safe_dump({"llm": {}, "agents": {"planner": {}, "reviewer": {}}}),
        encoding="utf-8",
    )
    script_path = tmp_path / "inspect_config.py"
    script_path.write_text(
        """
import json
import pathlib
import sys
import yaml

path = pathlib.Path(sys.argv[1])
config = yaml.safe_load(path.read_text(encoding="utf-8"))
print(json.dumps({
    "final_result": sys.argv[2],
    "reward_info": {
        "gateway": config["llm"]["base_url"],
        "models": [config["agents"]["planner"]["model"], config["agents"]["reviewer"]["model"]],
        "config_path": str(path),
    },
}))
""".strip(),
        encoding="utf-8",
    )

    result = asyncio.run(
        external_mas_runner(
            raw_prompt="task with spaces & punctuation",
            rollout=SimpleNamespace(base_url="http://gateway/rollouts/r-1/v1"),
            sample_index=0,
            session_runtime=None,
            role_policy_mapping={"planner": "policy_1", "reviewer": "policy_2"},
            execution={"backend": "local_process"},
            command={
                "argv": [sys.executable, str(script_path), "{config_path}", "{prompt}"],
                "work_dir": str(tmp_path),
            },
            config={"template_path": str(template_path)},
            result={"mode": "stdout_json"},
        )
    )

    assert result["final_result"] == "task with spaces & punctuation"
    assert result["reward_info"]["gateway"] == "http://gateway/rollouts/r-1/v1"
    assert result["reward_info"]["models"] == ["planner", "reviewer"]
    assert not Path(result["reward_info"]["config_path"]).exists()


def test_local_process_backend_reports_nonzero_exit():
    from examples.multi_agent_blackbox.process_backend import LocalProcessError, run_local_process

    with pytest.raises(LocalProcessError, match="exit code 7") as exc_info:
        asyncio.run(
            run_local_process(
                [sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(7)"],
            )
        )

    assert exc_info.value.returncode == 7
    assert "bad" in exc_info.value.stderr


def test_local_process_backend_times_out_and_terminates_process():
    from examples.multi_agent_blackbox.process_backend import LocalProcessTimeout, run_local_process

    with pytest.raises(LocalProcessTimeout, match="timed out"):
        asyncio.run(
            run_local_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout_seconds=0.05,
            )
        )


def test_local_process_backend_terminates_process_when_cancelled():
    from examples.multi_agent_blackbox.process_backend import run_local_process

    async def cancel_process() -> float:
        task = asyncio.create_task(
            run_local_process([sys.executable, "-c", "import time; time.sleep(30)"])
        )
        await asyncio.sleep(0.1)
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - started

    assert asyncio.run(cancel_process()) < 5


def test_local_process_backend_terminates_descendant_processes_when_cancelled(tmp_path):
    from examples.multi_agent_blackbox.process_backend import run_local_process

    marker_path = tmp_path / "descendant-finished"
    descendant_code = (
        "import pathlib, time; "
        "time.sleep(1); "
        f"pathlib.Path({str(marker_path)!r}).write_text('alive', encoding='utf-8')"
    )
    parent_code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {descendant_code!r}]); "
        "time.sleep(30)"
    )

    async def cancel_process_tree() -> None:
        task = asyncio.create_task(run_local_process([sys.executable, "-c", parent_code]))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(1.1)

    asyncio.run(cancel_process_tree())

    assert not marker_path.exists()


def test_external_runner_sanitizes_rollout_id_in_log_filenames(tmp_path):
    from examples.multi_agent_blackbox.external_mas_runner import external_mas_runner

    template_path = tmp_path / "mas.yaml"
    template_path.write_text("agents: {planner: {}}\n", encoding="utf-8")
    log_dir = tmp_path / "logs"

    asyncio.run(
        external_mas_runner(
            raw_prompt="task",
            rollout=SimpleNamespace(
                base_url="http://gateway/v1",
                rollout_id=r"group/rollout\\one",
            ),
            sample_index=0,
            session_runtime=None,
            role_policy_mapping={"planner": "policy_1"},
            execution={"backend": "local_process"},
            command={"argv": [sys.executable, "-c", "print('result')"]},
            config={"template_path": str(template_path)},
            result={"mode": "stdout_text"},
            logs={"directory": str(log_dir)},
        )
    )

    assert (log_dir / "group_rollout_one.stdout.log").read_text(encoding="utf-8") == "result\n"
    assert (log_dir / "group_rollout_one.stderr.log").read_text(encoding="utf-8") == ""


def test_external_runner_removes_temporary_config_when_command_is_invalid(tmp_path):
    from examples.multi_agent_blackbox.external_mas_runner import external_mas_runner

    template_path = tmp_path / "mas.yaml"
    template_path.write_text("agents: {planner: {}}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="command.argv or command.template"):
        asyncio.run(
            external_mas_runner(
                raw_prompt="task",
                rollout=SimpleNamespace(base_url="http://gateway/v1"),
                sample_index=0,
                session_runtime=None,
                role_policy_mapping={"planner": "policy_1"},
                execution={"backend": "local_process"},
                command={},
                config={"template_path": str(template_path), "temp_dir": str(tmp_path)},
            )
        )

    assert list(tmp_path.glob("uni_mas_*.yaml")) == []


def test_external_runner_rejects_unimplemented_sandbox_backend(tmp_path):
    from examples.multi_agent_blackbox.external_mas_runner import external_mas_runner

    template_path = tmp_path / "mas.yaml"
    template_path.write_text("agents: {planner: {}}\n", encoding="utf-8")

    with pytest.raises(NotImplementedError, match="sandbox"):
        asyncio.run(
            external_mas_runner(
                raw_prompt="task",
                rollout=SimpleNamespace(base_url="http://gateway/v1"),
                sample_index=0,
                session_runtime=None,
                role_policy_mapping={"planner": "policy_1"},
                execution={"backend": "sandbox"},
                command={"argv": [sys.executable, "-c", "print('unused')"]},
                config={"template_path": str(template_path)},
            )
        )
