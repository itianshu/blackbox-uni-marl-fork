import json
import time
from types import SimpleNamespace

from uni_agent.trainer.dynamic_inference.observability import SchedulingTrace
from uni_agent.trainer.dynamic_inference.signals import PolicySample, SignalStore
from uni_agent.trainer.dynamic_inference.types import SchedulingConfig


def test_trace_preserves_event_load_and_marks_stale_measurements(tmp_path):
    path = tmp_path / 'trace.jsonl'
    trace = SchedulingTrace(SchedulingConfig(trace_path=str(path), trace_interval_s=60))
    store = SignalStore(['p1', 'p2'])
    for policy, kv in [('p1', .12), ('p2', .01)]:
        store.record(policy, PolicySample(t=time.monotonic() - 2, total_inflight=8,
            kv_cache_usage={policy + ':8000': kv}, metrics_fresh=True))
    scheduler = SimpleNamespace(signals=store, policies=['p1', 'p2'],
        capacity_cards=lambda p: 4, ema_kv={'p1': .1, 'p2': .02}, active_lends=[])
    trace.emit(scheduler, 'sample', 1)
    trace.emit(scheduler, 'sample', 1)  # throttled; events must never be throttled
    trace.emit(scheduler, 'borrow_start', 1, lender='p2', borrower='p1')
    store.record('p1', PolicySample(t=time.monotonic(), kv_cache_usage={'p1:8000': .06}))
    trace.emit(scheduler, 'borrow_complete', 1, execution_s=3)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['kind'] for r in records] == ['sample', 'borrow_start', 'borrow_complete']
    assert records[1]['policies']['p1']['replica_kv'] == {'p1:8000': .12}
    assert records[1]['policies']['p1']['sample_age_s'] >= 2
    assert records[2]['policies']['p1']['replica_kv'] == {'p1:8000': .06}


def test_trace_io_failure_does_not_interrupt_scheduling(tmp_path, caplog):
    trace = SchedulingTrace(SchedulingConfig(trace_path=str(tmp_path)))
    scheduler = SimpleNamespace(policies=[], active_lends=[], signals=None)
    trace.emit(scheduler, 'borrow', 1)
    assert 'failed to write trace' in caplog.text
