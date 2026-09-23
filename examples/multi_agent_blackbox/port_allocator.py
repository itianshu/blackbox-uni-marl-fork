"""Opt-in disjoint rendezvous ranges for worker groups created by one driver."""
import os
import threading

_lock = threading.Lock()
_original = None
_next_port = None


def is_installed():
    return _original is not None


def install(worker_group_class):
    global _original, _next_port
    value = os.environ.get('UNI_AGENT_MASTER_PORT_RANGE')
    if not value or _original is not None:
        return
    start, stop = map(int, value.split(':'))
    if not 1024 <= start < stop <= 65536:
        raise ValueError('UNI_AGENT_MASTER_PORT_RANGE must be start:stop, end exclusive')
    _next_port = start
    _original = worker_group_class._get_master_addr_port

    def get_master(self, pg, bundle_index=0, master_port_range=None):
        global _next_port
        if master_port_range is None and getattr(self, '_master_port', None) is None:
            with _lock:
                reserved = getattr(self, '_example_master_port_range', None)
                if reserved is None:
                    if _next_port + 8 > stop:
                        raise RuntimeError('UNI_AGENT_MASTER_PORT_RANGE exhausted')
                    reserved = [_next_port, _next_port + 8]
                    _next_port += 8
                    self._example_master_port_range = reserved
            master_port_range = reserved
        return _original(self, pg, bundle_index=bundle_index, master_port_range=master_port_range)

    worker_group_class._get_master_addr_port = get_master


def restore(worker_group_class):
    global _original, _next_port
    if _original is not None:
        worker_group_class._get_master_addr_port = _original
    _original = _next_port = None
