#!/usr/bin/env python3
"""Prepare python-code Alpaca data for LLaMA-Factory SFT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "python_code_instructions_18k_alpaca"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sft" / "data"


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    except Exception as pandas_error:
        try:
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pylist()
        except Exception as arrow_error:
            try:
                from datasets import load_dataset

                dataset = load_dataset("parquet", data_files=str(path), split="train")
                return [dict(row) for row in dataset]
            except Exception as datasets_error:
                raise RuntimeError(
                    "Failed to read parquet. Install pandas, pyarrow, or datasets with parquet support "
                    "in the active environment."
                ) from datasets_error


def find_parquet_files(source_dir: Path) -> list[Path]:
    data_dir = source_dir / "data"
    candidates = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []
    if not candidates:
        candidates = sorted(source_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {source_dir}")
    return candidates


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def convert_row(row: dict[str, Any]) -> dict[str, str] | None:
    instruction = clean_text(row.get("instruction"))
    input_text = clean_text(row.get("input"))
    output = clean_text(row.get("output"))

    if not instruction:
        instruction = clean_text(row.get("prompt"))
    if not output:
        return None

    if not instruction and input_text:
        instruction, input_text = input_text, ""
    if not instruction:
        return None

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    unique_rows = []
    for row in rows:
        key = (row["instruction"], row["input"], row["output"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def compute_statistics(rows: list[dict[str, str]], original_count: int) -> dict[str, Any]:
    """计算数据集的统计信息"""
    if not rows:
        return {}

    # 提取各字段长度
    instruction_lens = [len(row["instruction"]) for row in rows]
    input_lens = [len(row["input"]) for row in rows]
    output_lens = [len(row["output"]) for row in rows]

    # 计算空值数量
    empty_instruction = sum(1 for row in rows if not row["instruction"])
    empty_input = sum(1 for row in rows if not row["input"])
    empty_output = sum(1 for row in rows if not row["output"])

    total = len(rows)

    def get_length_stats(lengths: list[int]) -> dict[str, Any]:
        """计算长度的统计指标"""
        sorted_lens = sorted(lengths)
        return {
            "mean": sum(lengths) / len(lengths) if lengths else 0,
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "median": sorted_lens[len(sorted_lens) // 2] if sorted_lens else 0,
            "p25": sorted_lens[len(sorted_lens) // 4] if sorted_lens else 0,
            "p75": sorted_lens[len(sorted_lens) * 3 // 4] if sorted_lens else 0,
        }

    return {
        "total_samples": total,
        "original_samples": original_count,
        "duplicate_count": original_count - total,
        "duplicate_rate": (original_count - total) / original_count if original_count > 0 else 0,
        "instruction_stats": {
            **get_length_stats(instruction_lens),
            "empty_count": empty_instruction,
            "empty_rate": empty_instruction / total if total > 0 else 0,
        },
        "input_stats": {
            **get_length_stats(input_lens),
            "empty_count": empty_input,
            "empty_rate": empty_input / total if total > 0 else 0,
        },
        "output_stats": {
            **get_length_stats(output_lens),
            "empty_count": empty_output,
            "empty_rate": empty_output / total if total > 0 else 0,
        },
    }


def split_rows(
    rows: list[dict[str, str]], train_ratio: float, valid_ratio: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train_ratio must be between 0 and 1.")
    if not 0 <= valid_ratio < 1:
        raise ValueError("--valid_ratio must be between 0 and 1.")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("--train_ratio + --valid_ratio must be less than 1.")

    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_rows = shuffled[:train_end]
    valid_rows = shuffled[train_end:valid_end]
    test_rows = shuffled[valid_end:]
    if not train_rows or not valid_rows or not test_rows:
        raise ValueError(
            f"Split produced an empty subset: train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)}"
        )
    return train_rows, valid_rows, test_rows


def save_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all examples.")
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    parquet_files = find_parquet_files(args.source_dir)
    raw_rows: list[dict[str, Any]] = []
    for parquet_file in parquet_files:
        raw_rows.extend(read_parquet(parquet_file))

    converted = [row for row in (convert_row(raw) for raw in raw_rows) if row is not None]
    converted_before_dedup = len(converted)
    converted = deduplicate(converted)

    # 计算统计信息（在 limit 之前，获取完整数据的统计）
    statistics = compute_statistics(converted, converted_before_dedup)

    if args.limit > 0:
        converted = converted[: args.limit]
    if len(converted) < 3:
        raise ValueError(f"Need at least 3 valid examples, got {len(converted)}")

    train_rows, valid_rows, test_rows = split_rows(
        converted, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio, seed=args.seed
    )

    save_json(args.output_dir / "code_sft_train.json", train_rows)
    save_json(args.output_dir / "code_sft_valid.json", valid_rows)
    save_json(args.output_dir / "code_sft_test.json", test_rows)
    save_json(
        args.output_dir / "dataset_info.json",
        {
            "code_sft_train": {"file_name": "code_sft_train.json"},
            "code_sft_valid": {"file_name": "code_sft_valid.json"},
            "code_sft_test": {"file_name": "code_sft_test.json"},
        },
    )

    # 保存统计信息到 JSON 文件
    save_json(args.output_dir / "data_statistics.json", statistics)

    # 打印基本信息
    print(f"Read {len(raw_rows)} raw rows from {len(parquet_files)} parquet file(s).")
    print(f"Kept {len(converted)} valid unique rows.")
    print(f"Wrote train: {len(train_rows)} -> {args.output_dir / 'code_sft_train.json'}")
    print(f"Wrote valid: {len(valid_rows)} -> {args.output_dir / 'code_sft_valid.json'}")
    print(f"Wrote test : {len(test_rows)} -> {args.output_dir / 'code_sft_test.json'}")
    print(f"Wrote registry -> {args.output_dir / 'dataset_info.json'}")

    # 打印统计摘要
    print("\n========== 数据质量统计 ==========")
    print(f"原始有效样本数 : {statistics['original_samples']}")
    print(f"去重后样本数   : {statistics['total_samples']}")
    print(f"重复样本数     : {statistics['duplicate_count']}  (重复率: {statistics['duplicate_rate']:.2%})")

    for field in ("instruction", "input", "output"):
        s = statistics[f"{field}_stats"]
        print(f"\n[{field}]")
        print(f"  空值数量 : {s['empty_count']}  (空值率: {s['empty_rate']:.2%})")
        print(f"  平均长度 : {s['mean']:.1f} 字符")
        print(f"  最短/最长: {s['min']} / {s['max']} 字符")
        print(f"  中位数   : {s['median']} 字符")
        print(f"  25%/75%  : {s['p25']} / {s['p75']} 字符")

    print(f"\n统计文件已保存 -> {args.output_dir / 'data_statistics.json'}")
    print("===================================")


if __name__ == "__main__":
    main()
