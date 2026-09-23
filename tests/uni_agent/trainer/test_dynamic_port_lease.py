import socket
import pytest
from uni_agent.trainer.dynamic_inference.port_lease import lease_port_block


def test_live_leases_do_not_overlap_and_closed_lease_is_reusable(tmp_path):
    first, a = lease_port_block(tmp_path)
    second, b = lease_port_block(tmp_path)
    try:
        assert second >= first + 32
        a.close()
        third, c = lease_port_block(tmp_path)
        try:
            assert third == first
        finally:
            c.close()
    finally:
        a.close()
        b.close()


def test_existing_listener_is_skipped_and_exhaustion_is_explicit(tmp_path):
    base, lease = lease_port_block(tmp_path)
    lease.close()
    with socket.socket() as listener:
        listener.bind(('', base + 3))
        with pytest.raises(RuntimeError, match='No free'):
            lease_port_block(tmp_path, start=base, stop=base+32)
        other, handle = lease_port_block(tmp_path, start=base, stop=base+64)
        try:
            assert other == base + 32
        finally:
            handle.close()


def test_environment_can_override_port_range_and_block_size(tmp_path, monkeypatch):
    monkeypatch.setenv('UNI_AGENT_VLLM_PORT_RANGE', '23000:23008')
    monkeypatch.setenv('UNI_AGENT_VLLM_PORT_BLOCK_SIZE', '4')

    first, first_handle = lease_port_block(tmp_path)
    second, second_handle = lease_port_block(tmp_path)
    try:
        assert (first, second) == (23000, 23004)
        with pytest.raises(RuntimeError, match='No free'):
            lease_port_block(tmp_path)
    finally:
        first_handle.close()
        second_handle.close()


def test_invalid_environment_range_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv('UNI_AGENT_VLLM_PORT_RANGE', 'not-a-range')

    with pytest.raises(ValueError, match='start:stop'):
        lease_port_block(tmp_path)
