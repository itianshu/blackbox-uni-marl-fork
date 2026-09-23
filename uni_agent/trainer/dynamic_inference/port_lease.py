"""Process-lifetime port blocks for local vLLM engine subprocesses."""
import errno
import fcntl
import os
import socket
from pathlib import Path


def lease_port_block(directory=None, start=None, stop=None, stride=None):
    """Lease a contiguous local port block for one vLLM process lifetime.

    Defaults avoid the usual Linux ephemeral range, while environment
    overrides make the allocator portable to clusters with different network
    policy: ``UNI_AGENT_VLLM_PORT_RANGE=start:stop``,
    ``UNI_AGENT_VLLM_PORT_BLOCK_SIZE`` and ``UNI_AGENT_VLLM_PORT_LOCK_DIR``.
    """
    directory = directory or os.environ.get(
        'UNI_AGENT_VLLM_PORT_LOCK_DIR', '/tmp/uni_agent_vllm_ports')
    if start is None or stop is None:
        configured_range = os.environ.get('UNI_AGENT_VLLM_PORT_RANGE', '24000:32000')
        try:
            configured_start, configured_stop = (
                int(value) for value in configured_range.split(':', 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'UNI_AGENT_VLLM_PORT_RANGE must use the form start:stop') from exc
        start = configured_start if start is None else start
        stop = configured_stop if stop is None else stop
    stride = int(os.environ.get('UNI_AGENT_VLLM_PORT_BLOCK_SIZE', '32')) if stride is None else stride
    if not (0 < start < stop <= 65536 and stride > 0 and start + stride <= stop):
        raise ValueError(
            f'invalid vLLM port allocation: start={start}, stop={stop}, stride={stride}')

    # Per-block flock coordinates concurrent Ray actors. Socket binds also
    # detect listeners that did not acquire one of our leases.
    Path(directory).mkdir(parents=True, exist_ok=True)
    for base in range(start, stop - stride + 1, stride):
        handle = open(Path(directory) / f'{base}.lock', 'a+')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                continue
            raise
        sockets = []
        try:
            # Check existing external listeners too, not only our own leases.
            for port in range(base, base + stride):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(sock)
                sock.bind(('', port))
        except OSError as exc:
            handle.close()
            if exc.errno in (errno.EADDRINUSE, errno.EACCES):
                continue
            raise
        finally:
            for sock in sockets:
                sock.close()
        handle.seek(0)
        handle.truncate()
        handle.write(f'{os.getpid()} {base} {base + stride - 1}\n')
        handle.flush()
        return base, handle
    raise RuntimeError('No free vLLM port block available')
