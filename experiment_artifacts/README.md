# Experiment artifacts

`two_policy_2b_bidir4_retry1_20260923_234357.tar.zst` is a lossless snapshot of
the first retry only:

`examples/multi_agent_blackbox/logs/two_policy_2b_bidir4_8step_comparison_retry1_20260923_234357/`

It contains all raw collection data, launcher and training logs, GPU and vLLM
serving samples, checkpoint metrics, monitoring output, manifests, and the
generated seven-step pre-OOM report. No data from other experiment retries is
included in this archive.

Restore the snapshot from the repository root with:

```bash
tar --zstd -xf experiment_artifacts/two_policy_2b_bidir4_retry1_20260923_234357.tar.zst
```

Verify the downloaded archive with the SHA-256 value stored in
`two_policy_2b_bidir4_retry1_20260923_234357.tar.zst.sha256` before extracting
it. This archive is committed as a regular Git object, so Git LFS is not
required on another machine.
