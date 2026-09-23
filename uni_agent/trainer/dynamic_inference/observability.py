"""Durable scheduling events and serving measurements, with explicit sample ages."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)


class SchedulingTrace:
    def __init__(self, config):
        self.path = config.trace_path
        self.interval = config.trace_interval_s
        self.last_sample = float('-inf')
        self.started = {}
        self.config = asdict(config)

    def snapshot(self, scheduler):
        result = {}
        store = scheduler.signals
        now = time.monotonic()
        for policy in scheduler.policies:
            samples = getattr(store, 'samples', {}).get(policy, [])
            latest = samples[-1] if samples else None
            result[policy] = {
                'serving_cards': scheduler.capacity_cards(policy),
                'kv_decision': scheduler.ema_kv[policy],
                'waiting_per_replica_decision': getattr(
                    scheduler, 'waiting_per_replica', {}).get(policy, 0.0),
                'queue_time_s_decision': getattr(
                    scheduler, 'queue_time_s', {}).get(policy, 0.0),
                'composite_load_score': (
                    scheduler._load_score(policy)
                    if hasattr(scheduler, '_load_score') else None
                ),
                'sample_age_s': now - latest.t if latest else None,
                'metrics_fresh': latest.metrics_fresh if latest else False,
                'inflight': latest.total_inflight if latest else None,
                'replica_inflight': latest.inflight_per_replica if latest else {},
                'replica_kv': latest.kv_cache_usage if latest else {},
                'vllm': store.vllm_observability(policy) if latest else {},
                'replica_metrics': {
                    address: {name: snapshot.aggregate((name,)) for name in (
                        'num_requests_running', 'num_requests_waiting',
                        'generation_tokens_total', 'prompt_tokens_total',
                        'num_preemptions_total', 'request_success_total')}
                    for address, snapshot in (latest.vllm_metrics.items() if latest else [])
                },
            }
        return result

    def emit(self, scheduler, kind, step, **detail):
        if not self.path:
            return
        now = time.monotonic()
        if kind == 'sample' and now - self.last_sample < self.interval:
            return
        try:
            record = {
                'time_unix_s': time.time(), 'monotonic_s': now,
                'kind': kind, 'step': step, 'detail': detail,
                'decision_state': {
                    'bottleneck': getattr(scheduler, 'bottleneck', None),
                    'bottleneck_confirmations': getattr(scheduler, 'confirm', {}),
                    'target_confirmations': getattr(scheduler, '_pending_target_polls', 0),
                    'settle_polls_remaining': getattr(scheduler, '_settle_polls_remaining', 0),
                    'return_low_polls': getattr(scheduler, '_return_low_polls', {}),
                    'borrow_cooldown_remaining_s': max(0, getattr(scheduler, '_borrow_cooldown_until', 0) - now),
                    'disabled': getattr(scheduler, 'disabled', False),
                },
                'config': self.config,
                'policies': self.snapshot(scheduler),
                'active_lends': [{
                    'lend_id': lend.lend_id, 'lender': lend.home_policy,
                    'borrower': lend.donor,
                    'home_servers': [getattr(r, 'server_address', None) for r in lend.home_replicas],
                    'guest_servers': [getattr(r, 'server_address', None) for r in lend.guests],
                    'age_s': now - self.started[lend.lend_id] if lend.lend_id in self.started else None,
                } for lend in scheduler.active_lends],
            }
            path = Path(self.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            if kind == 'sample':
                self.last_sample = now
            else:
                print('DYNAMIC_EVENT ' + json.dumps({k: v for k, v in record.items()
                      if k not in ('policies', 'active_lends', 'config', 'decision_state')}), flush=True)
        except Exception:
            logger.warning('dynamic_inference: failed to write trace', exc_info=True)
