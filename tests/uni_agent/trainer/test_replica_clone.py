import asyncio
from types import SimpleNamespace as NS

import pytest

from uni_agent.trainer.dynamic_inference.replica_clone import (
    runtime_tensors, tensor_manifest, validate_manifests, verify_transfer, clone_replica,
)


def test_tensor_inventory_includes_buffers_and_rejects_hidden_state():
    import torch
    model = torch.nn.Linear(2, 2)
    model.register_buffer('scale', torch.ones(2))
    assert set(runtime_tensors(model)) == {'weight', 'bias', 'scale'}
    assert len(tensor_manifest(runtime_tensors(model))) == 3
    model.hidden = {'tensor': torch.ones(2)}
    with pytest.raises(ValueError, match='Unregistered'):
        runtime_tensors(model)


def test_layout_mismatch_and_duplicate_rank_rejected():
    row = {'rank': 0, 'fingerprint': 'a', 'manifest': []}
    validate_manifests([row], [row])
    with pytest.raises(ValueError):
        validate_manifests([row], [{**row, 'fingerprint': 'b'}])
    with pytest.raises(ValueError):
        validate_manifests([row, row], [row, row])


@pytest.mark.parametrize('field', ['sha256', 'bytes', 'version', 'transaction_id'])
def test_transfer_mismatch_is_fatal(field):
    row = {'rank': 0, 'sha256': 'abc', 'bytes': 4, 'version': (1, 2), 'transaction_id': 't'}
    verify_transfer([row], [row])
    with pytest.raises(RuntimeError):
        verify_transfer([row], [{**row, field: 'bad'}])


def test_failed_clone_never_resumes_or_marks_version(monkeypatch):
    from uni_agent.trainer.dynamic_inference import replica_clone
    calls = []
    async def abort(): calls.append('abort')
    async def resume(): calls.append('resume')
    source = NS(server_address='source', server_handle='s', abort_all_requests=abort, resume_generation=resume)
    target = NS(server_address='target', server_handle='t', abort_all_requests=abort, resume_generation=resume)
    async def rpc(*args): raise TimeoutError('injected')
    monkeypatch.setattr(replica_clone, 'clone_rpc', rpc)
    with pytest.raises(TimeoutError):
        asyncio.run(clone_replica(source, target, policy='p', version=(1, 2), timeout=1))
    assert calls == ['abort', 'abort']
    assert not hasattr(target, '_dynamic_weight_version')


def test_abort_error_return_is_not_success():
    from uni_agent.trainer.dynamic_inference.replica_clone import check_abort_result
    check_abort_result({'server_results': [{'aborted_count': 0}]})
    with pytest.raises(RuntimeError, match='pause failed'):
        check_abort_result({'server_results': [{'error': 'pause failed'}]})


def test_success_updates_serving_version_after_hash_verification(monkeypatch):
    from uni_agent.trainer.dynamic_inference import replica_clone
    calls = []
    async def abort(): calls.append('pause')
    async def resume(): calls.append('resume')
    async def set_version(version): calls.append(('version', version))
    source = NS(server_address='s:1', server_handle='s', abort_all_requests=abort, resume_generation=resume)
    target = NS(server_address='t:1', server_handle='t', abort_all_requests=abort, resume_generation=resume,
                servers=[NS(set_global_steps=NS(remote=set_version))])
    async def rpc(server, operation, payload, timeout):
        if operation == 'inspect':
            return [{'rank': 0, 'fingerprint': 'model', 'manifest': [], 'node': server}]
        if operation == 'prepare':
            return [{'rank': 0, 'uid': b'id'}]
        return [{'rank': 0, 'sha256': 'same', 'bytes': 42, 'version': payload['version'],
                 'transaction_id': payload['transaction_id']}]
    monkeypatch.setattr(replica_clone, 'clone_rpc', rpc)
    result = asyncio.run(clone_replica(source, target, policy='p', version=(8, 2), timeout=1))
    assert result['bytes'] == 42
    assert target._dynamic_weight_version == (8, 2)
    assert calls == ['pause', 'pause', ('version', 8), 'resume', 'resume']


def test_attention_range_constants_are_copied_but_kv_is_excluded():
    import torch
    Attention = type('Attention', (torch.nn.Module,), {
        '__module__': 'vllm.model_executor.layers.attention.attention'})
    model = torch.nn.Module()
    model.attn = Attention()
    for key in ('q_range', 'k_range', 'v_range'):
        setattr(model.attn, key, torch.tensor(1.0))
    model.attn.kv_cache = torch.ones(3)
    tensors = runtime_tensors(model)
    assert set(tensors) == {'attn.q_range', 'attn.k_range', 'attn.v_range'}
    assert tensors['attn.q_range'] is model.attn.q_range
    model.attn.q_range = torch.ones(2)
    with pytest.raises(ValueError, match='Unsupported attention range'):
        runtime_tensors(model)
