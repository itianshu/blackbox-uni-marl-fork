#!/usr/bin/env python
"""Generate a small mock dataset satisfying verl's RLHFDataset format.

The multi_agent_blackbox example trains with ``return_raw_chat: true``, so the
``prompt`` column must be an OpenAI-style chat message list. verl's
NaiveRewardManager reads ``reward_model["ground_truth"]``, so each row carries a
``reward_model`` dict with a ``ground_truth`` list (verl standard data format).

Output:
    examples/multi_agent_blackbox/scripts/mock_data/mock_mas_train.parquet
    examples/multi_agent_blackbox/scripts/mock_data/mock_mas_val.parquet

Usage:
    python examples/multi_agent_blackbox/scripts/generate_mock_mas_data.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


_SAMPLES = [
    # Simple questions (agent often answers correctly -> reward=1.0)
    {
        "question": "What is 2 + 2? Answer with just the number.",
        "answer": "4",
    },
    {
        "question": "What is the capital of France? Answer with one word.",
        "answer": "Paris",
    },
    # Harder questions: small models often get these wrong, so different
    # rollouts of the same prompt yield a 0/1 reward mix -> GRPO advantage is
    # non-zero -> PPO loss becomes informative.
    {
        "question": "What is 17 times 23? Answer with just the number.",
        "answer": "391",
    },
    {
        "question": "What is 7 cubed? Answer with just the number.",
        "answer": "343",
    },
    {
        "question": "What is 13 times 17? Answer with just the number.",
        "answer": "221",
    },
    {
        "question": "How many amendments are there in the US Constitution? Answer with just the number.",
        "answer": "27",
    },
    {
        "question": "Which element has atomic number 79? Answer with one word.",
        "answer": "Gold",
    },
    {
        "question": "In what year did World War II end? Answer with just the year.",
        "answer": "1945",
    },
]


def _row(question: str, answer: str, index: int) -> dict:
    return {
        "data_source": "mock_mas",
        "prompt": [{"role": "user", "content": question}],
        "ability": "knowledge",
        "reward_model": {"ground_truth": [answer]},
        "extra_info": {"index": index},
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "mock_data")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [_row(s["question"], s["answer"], i) for i, s in enumerate(_SAMPLES * args.repeat)]
    table = pa.Table.from_pylist(rows)

    train_path = out_dir / "mock_mas_train.parquet"
    val_path = out_dir / "mock_mas_val.parquet"
    pq.write_table(table, train_path)
    pq.write_table(table, val_path)

    print(f"wrote {len(rows)} rows -> {train_path}")
    print(f"wrote {len(rows)} rows -> {val_path}")
    print("\nschema:")
    for field in pq.read_schema(train_path):
        print(f"  {field.name}: {field.type}")


if __name__ == "__main__":
    main()
