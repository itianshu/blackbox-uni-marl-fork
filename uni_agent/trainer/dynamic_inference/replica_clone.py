"""Same-layout vLLM tensor cloning, independent of the training actor.

The controller must hold the topology/publish gate for this entire operation.
Worker RPCs run with source generation paused. Unknown quantization/draft/EP
layouts fail closed; they require a verified runtime-state adapter first.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid

logger = logging.getLogger(__name__)


def check_abort_result(result):
    """verl can encode remote pause failures in a successful RPC response."""
    if isinstance(result, dict):
        if result.get('error'):
            raise RuntimeError(f"vLLM abort failed: {result['error']}")
        for nested in result.get('server_results', []):
            check_abort_result(nested)


async def abort_replica(replica):
    check_abort_result(await replica.abort_all_requests())


def runtime_tensors(model):
    """Enumerate registered state without replacing storage or tied parameters."""
    import torch
    tensors = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))
    for name, tensor in tensors.items():
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError(f"Unsupported clone tensor layout: {name}")
    # Unregistered tensor attributes may carry weight-derived state. Reject them
    # rather than claiming that a matching manifest proves completeness.
    def has_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, (list, tuple)):
            return any(has_tensor(v) for v in value)
        if isinstance(value, dict):
            return any(has_tensor(v) for v in value.values())
        return False
    for module_name, module in model.named_modules():
        for key, value in vars(module).items():
            if type(module).__module__.startswith('vllm.model_executor.layers.attention'):
                if key == 'kv_cache':
                    continue  # request-local cache is cleared, never cloned
                if key in ('q_range', 'k_range', 'v_range') and isinstance(value, torch.Tensor):
                    # vLLM 0.19 keeps these scale constants outside registered
                    # buffers, even for unquantized attention. Copy and hash
                    # them too; do not silently omit runtime tensor state.
                    if value.numel() != 1 or value.dtype != torch.float32:
                        raise ValueError(f"Unsupported attention range: {module_name}.{key}")
                    tensors[f'{module_name}.{key}' if module_name else key] = value
                    continue
            if key not in ('_parameters', '_buffers', '_modules') and has_tensor(value):
                raise ValueError(f"Unregistered runtime tensor: {module_name}.{key}")
    return dict(sorted(tensors.items()))


def tensor_manifest(tensors):
    return [(name, list(t.shape), str(t.dtype), list(t.stride()), t.numel() * t.element_size())
            for name, t in tensors.items()]


def clone_worker(worker, operation, payload):
    """Callable vLLM collective_rpc; weights never pass through the driver."""
    import ctypes
    import torch
    import vllm
    from vllm.distributed import get_world_group
    from vllm.distributed.device_communicators.pynccl_wrapper import (
        NCCLLibrary, buffer_type, cudaStream_t, ncclDataTypeEnum,
    )
    model = worker.model_runner.model
    cfg = worker.vllm_config
    if cfg.model_config.quantization or cfg.speculative_config is not None or cfg.lora_config is not None:
        raise ValueError('Clone requires a verified adapter for quantization/speculative models')
    if str(cfg.cache_config.cache_dtype).startswith('fp8'):
        raise ValueError('FP8 KV scale state requires a verified adapter')
    if getattr(cfg.parallel_config, 'enable_expert_parallel', False):
        raise ValueError('Clone EP runtime state is not yet verified')
    tensors = runtime_tensors(model)
    rank = get_world_group().rank_in_group
    if not hasattr(worker, '_replica_clone_channels'):
        worker._replica_clone_channels = {}
    channels = worker._replica_clone_channels
    if operation == 'inspect':
        fingerprint = {
            'model_class': type(model).__module__ + '.' + type(model).__qualname__,
            'model': cfg.model_config.model,
            'hf_config': cfg.model_config.hf_config.to_dict(),
            'vllm': vllm.__version__,
            'tp': cfg.parallel_config.tensor_parallel_size,
            'pp': cfg.parallel_config.pipeline_parallel_size,
            'dp': cfg.parallel_config.data_parallel_size,
            'dtype': str(cfg.model_config.dtype),
        }
        if fingerprint['dp'] != 1:
            raise ValueError('Clone currently requires rollout DP=1')
        import socket
        return {'rank': rank, 'node': socket.gethostname(), 'manifest': tensor_manifest(tensors),
                'fingerprint': hashlib.sha256(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()}
    key = payload['channel']
    if operation == 'prepare':
        if key not in channels:
            if len(channels) >= 16:
                raise RuntimeError('Clone communicator cache limit reached; restart required')
            lib = NCCLLibrary()
            uid = lib.ncclGetUniqueId()
            channels[key] = {'lib': lib, 'uid': ctypes.string_at(ctypes.addressof(uid), ctypes.sizeof(uid)), 'comm': None}
        return {'rank': rank, 'uid': channels[key]['uid']}
    if operation != 'transfer':
        raise ValueError(operation)
    is_source = payload['source']
    if key not in channels:
        channels[key] = {'lib': NCCLLibrary(), 'uid': payload['ids'][rank], 'comm': None}
    channel = channels[key]
    if channel.get('failed'):
        raise RuntimeError('Clone communicator previously failed')
    lib = channel['lib']
    device = torch.device('cuda', torch.cuda.current_device())
    stream = torch.cuda.current_stream(device)
    started = time.monotonic()
    try:
        torch.cuda.synchronize(device)
        if channel['comm'] is None:
            channel['comm'] = lib.ncclCommInitRank(
                2, lib.unique_id_from_bytes(channel['uid']), 0 if is_source else 1)
        digest = hashlib.sha256()
        byte_count = 0
        # One bounded staging buffer; no full model copy, CPU serialization, or
        # use of the model's TP/PP communicators.
        capacity = min(payload['bucket_bytes'], max(t.numel() * t.element_size() for t in tensors.values()))
        bucket = torch.empty(capacity, dtype=torch.uint8, device=device)
        for name, tensor in tensors.items():
            raw = tensor.detach().reshape(-1).view(torch.uint8)
            for offset in range(0, raw.numel(), capacity):
                count = min(capacity, raw.numel() - offset)
                part = bucket[:count]
                if is_source:
                    part.copy_(raw[offset:offset + count])
                    lib.ncclSend(buffer_type(part.data_ptr()), count,
                                 ncclDataTypeEnum.from_torch(torch.uint8), 1,
                                 channel['comm'], cudaStream_t(stream.cuda_stream))
                else:
                    lib.ncclRecv(buffer_type(part.data_ptr()), count,
                                 ncclDataTypeEnum.from_torch(torch.uint8), 0,
                                 channel['comm'], cudaStream_t(stream.cuda_stream))
                    raw[offset:offset + count].copy_(part)
                stream.synchronize()
                # Check target storage, not just the receive buffer.
                digest.update(raw[offset:offset + count].cpu().numpy().tobytes())
                byte_count += count
        del part, bucket
        torch.cuda.empty_cache()
        return {'rank': rank, 'sha256': digest.hexdigest(), 'bytes': byte_count,
                'version': payload['version'], 'transaction_id': payload['transaction_id'],
                'seconds': time.monotonic() - started}
    except BaseException:
        channel['failed'] = True
        raise


async def clone_rpc(server, operation, payload, timeout):
    return await asyncio.wait_for(
        server.dynamic_inference_clone_rpc.remote(operation, payload, timeout), timeout)


def validate_manifests(source, destination):
    by_rank = lambda entries: {e['rank']: (e['fingerprint'], e['manifest']) for e in entries}
    if len(source) != len(destination) or len(by_rank(source)) != len(source) or by_rank(source) != by_rank(destination):
        raise ValueError('Source/destination clone layout mismatch')


def verify_transfer(source, destination):
    fields = ('sha256', 'bytes', 'version', 'transaction_id')
    by_rank = lambda entries: {e['rank']: tuple(e[f] for f in fields) for e in entries}
    if not source or len(source) != len(destination) or len(by_rank(source)) != len(source) or by_rank(source) != by_rank(destination):
        raise RuntimeError('Replica clone verification failed')


async def clone_replica(source, destination, *, policy, version, timeout, bucket_bytes=64 << 20):
    transaction_id = uuid.uuid4().hex
    channel = f'{source.server_address}->{destination.server_address}'
    started = time.monotonic()
    logger.warning('REPLICA_CLONE_START tx=%s policy=%s version=%s source=%s target=%s',
                   transaction_id, policy, version, source.server_address, destination.server_address)
    # Pause generation, including the source, to avoid running a blocking worker
    # RPC while TP/PP forward collectives are in flight. Other replicas serve.
    await abort_replica(source)
    await abort_replica(destination)
    source_info, dest_info = await asyncio.gather(
        clone_rpc(source.server_handle, 'inspect', {}, timeout),
        clone_rpc(destination.server_handle, 'inspect', {}, timeout))
    validate_manifests(source_info, dest_info)
    prepared = await clone_rpc(source.server_handle, 'prepare', {'channel': channel}, timeout)
    payload = {'channel': channel, 'ids': {e['rank']: e['uid'] for e in prepared},
               'version': version, 'transaction_id': transaction_id, 'bucket_bytes': bucket_bytes}
    sent, received = await asyncio.gather(
        clone_rpc(source.server_handle, 'transfer', {**payload, 'source': True}, timeout),
        clone_rpc(destination.server_handle, 'transfer', {**payload, 'source': False}, timeout))
    verify_transfer(sent, received)
    # All ranks have completed and hashes agree. Only now may either engine resume.
    if version[0] is not None:
        await asyncio.gather(*(server.set_global_steps.remote(version[0]) for server in destination.servers))
    destination._dynamic_weight_version = version
    await source.resume_generation()
    await destination.resume_generation()
    result = {'transaction_id': transaction_id, 'policy': policy, 'version': version,
              'source': source.server_address, 'target': destination.server_address,
              'bytes': sum(r['bytes'] for r in received), 'seconds': time.monotonic() - started,
              'ranks': received, 'source_nodes': [e['node'] for e in source_info],
              'target_nodes': [e['node'] for e in dest_info]}
    logger.warning('REPLICA_CLONE_COMPLETE %s', json.dumps(result))
    return result
