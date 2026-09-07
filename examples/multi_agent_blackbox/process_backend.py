"""Execution backends for command-driven external MAS rollouts."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LocalProcessResult:
    returncode: int
    stdout: str
    stderr: str


class LocalProcessError(RuntimeError):
    def __init__(self, returncode: int, stdout: str, stderr: str):
        super().__init__(f"External MAS process exited with exit code {returncode}: {stderr.strip()}")
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class LocalProcessTimeout(TimeoutError):
    pass


async def _terminate_windows_process_tree(process: asyncio.subprocess.Process) -> bool:
    try:
        os.kill(process.pid, signal.CTRL_BREAK_EVENT)
        return True
    except (OSError, ValueError):
        pass

    try:
        taskkill = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await taskkill.wait()
    except OSError:
        return False
    return taskkill.returncode == 0


async def _force_terminate_windows_process_tree(process: asyncio.subprocess.Process) -> None:
    try:
        taskkill = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await taskkill.wait()
        if taskkill.returncode == 0:
            return
    except OSError:
        pass
    if process.returncode is None:
        process.kill()


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        terminated_tree = await _terminate_windows_process_tree(process)
        if not terminated_tree and process.returncode is None:
            process.terminate()

    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
        return
    except asyncio.TimeoutError:
        pass

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        await _force_terminate_windows_process_tree(process)
    await process.wait()


async def run_local_process(
    command: str | Sequence[str],
    *,
    work_dir: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    shell: bool = False,
) -> LocalProcessResult:
    """Run one MAS command and terminate its process group on timeout/cancellation."""
    process_env = os.environ.copy()
    if env:
        process_env.update({str(key): str(value) for key, value in env.items()})

    process_kwargs = {
        "cwd": work_dir,
        "env": process_env,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
    }
    if os.name == "posix":
        process_kwargs["start_new_session"] = True
    elif os.name == "nt":
        process_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    if shell:
        if not isinstance(command, str):
            raise TypeError("shell commands must be strings")
        process = await asyncio.create_subprocess_shell(command, **process_kwargs)
    else:
        if isinstance(command, str):
            raise TypeError("non-shell commands must be an argv sequence")
        process = await asyncio.create_subprocess_exec(*[str(part) for part in command], **process_kwargs)

    communicate_task = asyncio.create_task(process.communicate())
    try:
        if timeout_seconds is None:
            stdout_bytes, stderr_bytes = await communicate_task
        else:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=float(timeout_seconds),
            )
    except asyncio.TimeoutError as exc:
        await _terminate_process(process)
        await communicate_task
        raise LocalProcessTimeout(
            f"External MAS process timed out after {timeout_seconds} seconds"
        ) from exc
    except asyncio.CancelledError:
        await _terminate_process(process)
        await asyncio.shield(communicate_task)
        raise

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    result = LocalProcessResult(returncode=int(process.returncode or 0), stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        raise LocalProcessError(result.returncode, result.stdout, result.stderr)
    return result
