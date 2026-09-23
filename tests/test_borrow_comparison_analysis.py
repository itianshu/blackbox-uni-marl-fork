import json

from examples.multi_agent_blackbox.scripts.analyze_borrow_comparison import summarize_run, histogram_summary


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_comparison_excludes_warmup_and_integrates_only_observed_energy(tmp_path):
    write_rows(tmp_path / 'steps.jsonl', [
        {'step': step, 'observed_unix_s': step * 10,
         'metrics': {'perf/time_per_step': 100 if step <= 2 else 10}}
        for step in range(1, 21)])
    write_rows(tmp_path / 'gpu.jsonl', [
        {'time_unix_s': t, 'node_ip': 'one', 'gpus': [
            {'power.draw': '100', 'utilization.gpu': '50',
             'utilization.memory': '20', 'memory.used': '1000'}]}
        for t in range(0, 211, 15)])
    result = summarize_run(tmp_path)
    assert result['steady_step_s']['n'] == 18
    assert result['steady_step_s']['mean'] == 10
    assert result['warmup_step_s']['mean'] == 100
    assert abs(result['observed_energy_kwh'] - .005) < 1e-9
    assert abs(result['energy_monitor_coverage'] - 1 / 32) < 1e-9


def test_missing_monitor_data_is_not_reported_as_measured_zero_utilization(tmp_path):
    result = summarize_run(tmp_path)
    assert result['completed_steps'] == 0
    assert result['steady_step_s'] == {}
    assert 'gpu_utilization.gpu' not in result


def test_sleeping_guest_cached_gauge_is_excluded_from_active_load(tmp_path):
    write_rows(tmp_path / 'steps.jsonl', [
        {'step': step, 'observed_unix_s': step * 10,
         'metrics': {'perf/time_per_step': 10}} for step in range(1, 21)])
    (tmp_path / 'checkpoints').mkdir()
    write_rows(tmp_path / 'checkpoints/dynamic_inference.jsonl', [{
        'kind': 'sample', 'time_unix_s': 25, 'policies': {
            f'policy_{i}': {'replica_inflight': {'home': 2}} for i in range(1, 4)}}])
    write_rows(tmp_path / 'serving.jsonl', [{
        'time_unix_s': 30, 'servers': [
            {'policy': 'policy_1', 'actor_id': name, 'address': name,
             'metrics': {'gpu_cache_usage_max': kv, 'requests_running': requests}}
            for name, kv, requests in [('home', .1, 2), ('sleeping', .9, 5)]]}])
    result = summarize_run(tmp_path)['serving']['policy_1']
    assert result['kv_replica_max']['mean'] == .1
    assert result['requests_running']['mean'] == 2


def test_latency_quantile_uses_merged_bucket_counts():
    result = histogram_summary({'count': 100, 'sum': 40,
        'buckets': {'0.1': 50, '1.0': 90, 'inf': 100}})
    assert result['mean'] == .4
    assert result['p50'] == .1
    assert result['p90'] == 1


def test_event_weighting_clips_pre_warmup_and_shutdown_transfers(tmp_path):
    from examples.multi_agent_blackbox.scripts.analyze_borrow_comparison import summarize_events
    run = tmp_path / 'dynamic'
    (run / 'checkpoints').mkdir(parents=True)
    write_rows(run / 'steps.jsonl', [
        {'step': 2, 'observed_unix_s': 20}, {'step': 20, 'observed_unix_s': 200}])
    write_rows(run / 'checkpoints/dynamic_inference.jsonl', [
        {'kind': kind, 'time_unix_s': t, 'detail': {'lender': 'policy_2', 'borrower': borrower}}
        for kind, t, borrower in [('borrow', 10, 'policy_1'), ('early_return', 50, 'policy_1'),
                                  ('borrow', 60, 'policy_3'), ('return', 220, 'policy_3')]])
    result = summarize_events(tmp_path)
    cards = result['steady_time_weighted_serving_cards']
    assert abs(cards['policy_1'] - (4 + 2 * 30 / 180)) < 1e-9
    assert abs(cards['policy_3'] - (4 + 2 * 140 / 180)) < 1e-9
    assert abs(sum(cards.values()) - 20) < 1e-9
    assert sum(result['steady_allocation_seconds'].values()) == 180


def test_completed_borrow_registry_overrides_stale_load_snapshot(tmp_path):
    write_rows(tmp_path / 'steps.jsonl', [
        {'step': i, 'observed_unix_s': i * 10, 'metrics': {'perf/time_per_step': 10}}
        for i in range(1, 21)])
    (tmp_path / 'checkpoints').mkdir()
    write_rows(tmp_path / 'checkpoints/dynamic_inference.jsonl', [{
        'kind': 'borrow_complete', 'time_unix_s': 25,
        'policies': {f'policy_{i}': {'replica_inflight': {'stale': 0}} for i in range(1, 4)},
        'active_lends': [{'lender': 'policy_2', 'borrower': 'policy_1',
                          'home_servers': ['home2'], 'guest_servers': ['guest1']}]}])
    write_rows(tmp_path / 'serving.jsonl', [{'time_unix_s': 30, 'servers': [
        {'policy': p, 'actor_id': address, 'address': address, 'rank': rank,
         'metrics': {'gpu_cache_usage_max': kv, 'requests_running': 1}}
        for p, address, rank, kv in [('policy_1', 'home1', 2, .1),
            ('policy_1', 'guest1', 10000, .2), ('policy_1', 'sleeping1', 10001, .9),
            ('policy_2', 'home2', 4, .9), ('policy_2', 'other2', 5, .05)]]}])
    serving = summarize_run(tmp_path)['serving']
    assert abs(serving['policy_1']['kv_replica_mean']['mean'] - .15) < 1e-9
    assert serving['policy_1']['requests_running']['mean'] == 2
    assert serving['policy_2']['kv_replica_max']['mean'] == .05
