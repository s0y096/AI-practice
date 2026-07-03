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


def filter_rows(
    rows: list[dict[str, str]],
    min_instruction_len: int = 0,
    max_instruction_len: int = 0,
    min_output_len: int = 0,
    max_output_len: int = 0,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """过滤不符合质量要求的样本，返回 (合格样本列表, 被过滤样本列表)。

    被过滤样本附带 reason 字段，说明过滤原因，方便后续写入 bad_cases.json。
    """
    passed: list[dict[str, str]] = []
    bad_cases: list[dict[str, Any]] = []

    for row in rows:
        reasons: list[str] = []

        inst_len = len(row["instruction"])
        out_len = len(row["output"])

        if min_instruction_len > 0 and inst_len < min_instruction_len:
            reasons.append(f"instruction 过短: {inst_len} < {min_instruction_len}")
        if max_instruction_len > 0 and inst_len > max_instruction_len:
            reasons.append(f"instruction 过长: {inst_len} > {max_instruction_len}")
        if min_output_len > 0 and out_len < min_output_len:
            reasons.append(f"output 过短: {out_len} < {min_output_len}")
        if max_output_len > 0 and out_len > max_output_len:
            reasons.append(f"output 过长: {out_len} > {max_output_len}")

        if reasons:
            bad_cases.append({**row, "filter_reasons": reasons})
        else:
            passed.append(row)

    return passed, bad_cases


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
    # 数据质量过滤参数
    parser.add_argument("--min_output_len", type=int, default=10,
                        help="output 最短字符数，低于此值的样本将被过滤（0 表示不限制）。")
    parser.add_argument("--max_output_len", type=int, default=4096,
                        help="output 最长字符数，超过此值的样本将被过滤（0 表示不限制）。")
    parser.add_argument("--min_instruction_len", type=int, default=5,
                        help="instruction 最短字符数（0 表示不限制）。")
    parser.add_argument("--max_instruction_len", type=int, default=0,
                        help="instruction 最长字符数（0 表示不限制）。")
    parser.add_argument("--remove_duplicates", type=lambda x: x.lower() != "false",
                        default=True, help="是否去重，默认 true；传入 false 可跳过去重步骤。")
    parser.add_argument("--preview_count", type=int, default=10,
                        help="保存到 sample_preview.json 的样本数量，默认 10（0 表示不生成）。")
    args = parser.parse_args()

    parquet_files = find_parquet_files(args.source_dir)
    raw_rows: list[dict[str, Any]] = []
    for parquet_file in parquet_files:
        raw_rows.extend(read_parquet(parquet_file))

    converted = [row for row in (convert_row(raw) for raw in raw_rows) if row is not None]
    converted_before_dedup = len(converted)

    # 可选的去重步骤
    if args.remove_duplicates:
        converted = deduplicate(converted)
    converted_after_dedup = len(converted)  # 去重后、过滤前的数量

    # 应用质量过滤
    converted, bad_cases = filter_rows(
        converted,
        min_instruction_len=args.min_instruction_len,
        max_instruction_len=args.max_instruction_len,
        min_output_len=args.min_output_len,
        max_output_len=args.max_output_len,
    )

    # 计算统计信息（在 limit 之前，获取完整数据的统计）
    # 传入 converted_after_dedup 而非 converted_before_dedup，避免把过滤样本误算为重复
    statistics = compute_statistics(converted, converted_after_dedup)

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

    # 保存过滤后的样本预览（用于人工检查数据格式）
    if args.preview_count > 0:
        preview_samples = converted[: args.preview_count]
        preview_output = [
            {
                "index": i,
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
                "instruction_len": len(row["instruction"]),
                "input_len": len(row["input"]),
                "output_len": len(row["output"]),
            }
            for i, row in enumerate(preview_samples)
        ]
        save_json(args.output_dir / "sample_preview.json", preview_output)

    # 保存被过滤的样本（附带过滤原因，用于分析过滤策略是否合理）
    if bad_cases:
        bad_cases_output = [
            {
                "index": i,
                "filter_reasons": case["filter_reasons"],
                "instruction": case["instruction"],
                "input": case["input"],
                "output": case["output"],
                "instruction_len": len(case["instruction"]),
                "input_len": len(case["input"]),
                "output_len": len(case["output"]),
            }
            for i, case in enumerate(bad_cases)
        ]
        save_json(args.output_dir / "bad_cases.json", bad_cases_output)

    # 打印基本信息
    print(f"Read {len(raw_rows)} raw rows from {len(parquet_files)} parquet file(s).")
    print(f"Kept {len(converted)} valid unique rows.")
    print(f"Wrote train: {len(train_rows)} -> {args.output_dir / 'code_sft_train.json'}")
    print(f"Wrote valid: {len(valid_rows)} -> {args.output_dir / 'code_sft_valid.json'}")
    print(f"Wrote test : {len(test_rows)} -> {args.output_dir / 'code_sft_test.json'}")
    print(f"Wrote registry -> {args.output_dir / 'dataset_info.json'}")

    # 打印过滤参数和结果
    print("\n========== 数据质量过滤 ==========")
    print(f"去重开关            : {'开启' if args.remove_duplicates else '关闭'}")
    print(f"instruction 长度限制: [{args.min_instruction_len or '不限'}, {args.max_instruction_len or '不限'}]")
    print(f"output 长度限制     : [{args.min_output_len or '不限'}, {args.max_output_len or '不限'}]")
    print(f"原始有效样本数      : {converted_before_dedup}")
    dedup_removed = converted_before_dedup - converted_after_dedup
    print(f"去重删除样本数      : {dedup_removed}  (去重后剩余: {converted_after_dedup})")
    print(f"质量过滤删除样本数  : {len(bad_cases)}")
    print(f"最终样本数          : {len(converted)}")
    if bad_cases:
        print(f"bad_cases 已保存 -> {args.output_dir / 'bad_cases.json'}  ({len(bad_cases)} 条)")
    if args.preview_count > 0:
        preview_count_actual = min(args.preview_count, len(converted))
        print(f"sample_preview 已保存 -> {args.output_dir / 'sample_preview.json'}  (前 {preview_count_actual} 条)")

    # 打印统计摘要
    print("\n========== 数据质量统计 ==========")
    print(f"参与统计样本数 : {statistics['total_samples']}  (去重+过滤后)")

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
