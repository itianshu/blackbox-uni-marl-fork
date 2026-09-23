from concurrent.futures import ThreadPoolExecutor

from examples.multi_agent_blackbox import port_allocator


def test_parallel_worker_groups_use_disjoint_ranges_and_preserve_explicit_ranges(monkeypatch):
    class Group:
        _master_port = None
        def _get_master_addr_port(self, pg, bundle_index=0, master_port_range=None):
            return master_port_range
    original = Group._get_master_addr_port
    monkeypatch.setenv('UNI_AGENT_MASTER_PORT_RANGE', '21000:22000')
    try:
        port_allocator.install(Group)
        groups = [Group() for _ in range(20)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            ranges = list(pool.map(lambda group: group._get_master_addr_port(None), groups))
        ports = [port for start, stop in ranges for port in range(start, stop)]
        assert len(ports) == len(set(ports))
        assert groups[0]._get_master_addr_port(None) == ranges[0]
        assert Group()._get_master_addr_port(None, master_port_range=[23000, 23004]) == [23000, 23004]
    finally:
        port_allocator.restore(Group)
    assert Group._get_master_addr_port is original
