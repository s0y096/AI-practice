#!/usr/bin/env python3
"""Prepare python-code Alpaca data for LLaMA-Factory SFT."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter
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
    """过滤不符合质量（长度）要求的样本，返回 (合格样本列表, 被过滤样本列表)。

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


# 数据高质量筛选部分

def is_python_parsable(code: str) -> bool:
    """判断 output 是否为语法可解析的 Python 代码。

    使用 ast.parse 尝试解析。能解析成功说明代码语法完整，这类样本对训练更有价值。
    """
    stripped = code.strip()
    if not stripped:
        return False
    try:
        ast.parse(stripped)
        return True
    except (SyntaxError, ValueError):
        return False


# 判断任务完整性时使用：output 中若出现这些占位/省略标记，
# 说明代码不完整，训练价值低。
_INCOMPLETE_MARKERS = (  # 代码不完整key_words标记
    "# your code here",
    "# todo",
    "pass  # implement",
    "...",
    "raise notimplementederror",
)


def is_task_complete(row: dict[str, str]) -> bool:
    """判断样本任务是否完整。

    完整性要求: instruction非空 或 output非空，且 output 不包含明显的
    占位符 / 省略标记（说明答案被截断或留空待填）。
    答案里如果出现 # TODO、# your code here、... 这种，说明这段代码只是个框架、没真正写完，训练价值低，剔除。
    """
    instruction = row["instruction"].strip()
    output = row["output"].strip()

    if not instruction or not output:
        return False

    lowered = output.lower()
    for marker in _INCOMPLETE_MARKERS:
        if marker in lowered:
            return False

    # output 以省略号或未闭合的续行符结尾，通常是被截断的代码
    if output.endswith(("\\", "…")):
        return False

    return True


# 任务类型分类规则：按 instruction 的关键词归类，用于统计任务类型分布
# 并做均衡采样。规则从上到下匹配，命中即归类。
_TASK_TYPE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("class_oop", ("class", "object-oriented", "inheritance", "method")),
    ("data_structure", ("list", "dictionary", "array", "stack", "queue", "tree", "linked list", "hash")),
    ("algorithm", ("sort", "search", "algorithm", "fibonacci", "recursion", "dynamic programming", "prime")),
    ("string_processing", ("string", "text", "regex", "regular expression", "substring", "palindrome")),
    ("math", ("calculate", "sum", "average", "factorial", "math", "number", "matrix", "equation")),
    ("file_io", ("file", "read", "write", "csv", "json", "parse", "load", "save")),
    ("web_api", ("api", "request", "http", "url", "web", "scrape", "flask", "django")),
    ("data_science", ("dataframe", "pandas", "numpy", "plot", "model", "train", "predict", "sklearn")),
    ("function_general", ("function", "program", "script", "generate", "create")),
]


def classify_task_type(instruction: str) -> str:
    """根据 instruction 内容将样本归类到一种任务类型。

    用于统计任务类型分布并支撑均衡采样。未命中任何规则归为 other。
    """
    lowered = instruction.lower()
    for task_type, keywords in _TASK_TYPE_RULES:
        for kw in keywords:
            if kw in lowered:
                return task_type
    return "other"


# 高质量筛选器

def high_quality_filter(
    rows: list[dict[str, str]],
    require_parsable: bool = True,
    require_complete: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """高质量筛选：按代码语法可解析性 + 任务完整性过滤样本。

    返回 (合格样本列表, 被过滤样本列表)。
    合格样本会附带 task_type 字段，供后续均衡采样使用。
    """
    passed: list[dict[str, str]] = []
    bad_cases: list[dict[str, Any]] = []

    for row in rows:
        reasons: list[str] = []

        if require_complete and not is_task_complete(row):
            reasons.append("任务不完整: instruction/output 为空或含占位符/截断标记")
        if require_parsable and not is_python_parsable(row["output"]):
            reasons.append("代码语法不可解析: ast.parse 失败")

        if reasons:
            bad_cases.append({**row, "filter_reasons": reasons})
        else:
            passed.append({**row, "task_type": classify_task_type(row["instruction"])})

    return passed, bad_cases


def balance_task_types(
    rows: list[dict[str, str]],
    max_per_type: int = 0,
    seed: int = 42,
) -> tuple[list[dict[str, str]], dict[str, int], dict[str, int]]:
    """按任务类型做均衡采样，避免某一类任务样本过多导致模型偏斜。

    每种任务类型最多保留 max_per_type 条（0 表示不限制）。
    返回 (采样后样本, 采样前各类型数量, 采样后各类型数量)。
    """
    before_dist = Counter(row.get("task_type", "other") for row in rows)

    if max_per_type <= 0:
        return rows[:], dict(before_dist), dict(before_dist)

    # 按类型分桶
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        buckets.setdefault(row.get("task_type", "other"), []).append(row)

    rng = random.Random(seed)
    balanced: list[dict[str, str]] = []
    for task_type, bucket in buckets.items():
        if len(bucket) > max_per_type:
            balanced.extend(rng.sample(bucket, max_per_type))
        else:
            balanced.extend(bucket)

    rng.shuffle(balanced)
    after_dist = Counter(row.get("task_type", "other") for row in balanced)
    return balanced, dict(before_dist), dict(after_dist)


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

def convert_mbpp_split(split):
    split_path = (
        PROJECT_ROOT.parent
        / "mbpp"
        / "sanitized"
        / f"{split}-00000-of-00001.parquet"
    )

    rows = read_parquet_rows(split_path)

    dataset = []

    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        code = str(row.get("code", "")).strip()

        if not prompt or not code:
            continue

        dataset.append(
            {
                "instruction":
                    "Write a Python function according to the following description. "
                    "Return only executable Python code without explanation.\n\n"
                    + prompt,
                "input": "",
                "output": code,
                "task_id": int(row["task_id"])
            }
        )

    return dataset


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
    # 第四个要求：高质量数据筛选策略
    parser.add_argument("--high_quality", type=lambda x: x.lower() != "false",
                        default=True,
                        help="是否启用高质量筛选（语法可解析性+任务完整性），默认 true。")
    parser.add_argument("--require_parsable", type=lambda x: x.lower() != "false",
                        default=True,
                        help="是否要求 output 为语法可解析的 Python 代码，默认 true。")
    parser.add_argument("--require_complete", type=lambda x: x.lower() != "false",
                        default=True,
                        help="是否要求任务完整（无占位符/截断），默认 true。")
    parser.add_argument("--max_per_type", type=int, default=0,
                        help="每种任务类型最多保留的样本数，用于均衡采样（0 表示不限制）。")
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

    # 应用质量过滤（长度过滤）
    converted, bad_cases = filter_rows(
        converted,
        min_instruction_len=args.min_instruction_len,
        max_instruction_len=args.max_instruction_len,
        min_output_len=args.min_output_len,
        max_output_len=args.max_output_len,
    )
    converted_after_length = len(converted)

    # 高质量筛选（代码语法可解析性 + 任务完整性）
    hq_removed = 0
    if args.high_quality:
        converted, hq_bad_cases = high_quality_filter(
            converted,
            require_parsable=args.require_parsable,
            require_complete=args.require_complete,
        )
        hq_removed = len(hq_bad_cases)
        bad_cases.extend(hq_bad_cases)
    else:
        # 未启用高质量筛选时也补上 task_type，供任务类型统计使用
        converted = [{**row, "task_type": classify_task_type(row["instruction"])} for row in converted]
    converted_after_hq = len(converted)

    # 任务类型均衡采样
    converted, type_dist_before, type_dist_after = balance_task_types(
        converted, max_per_type=args.max_per_type, seed=args.seed
    )
    balance_removed = converted_after_hq - len(converted)

    # 计算统计信息（在 limit 之前，获取完整数据的统计）
    statistics = compute_statistics(converted, converted_after_dedup)
    # 附加任务类型分布到统计信息
    statistics["task_type_distribution"] = {
        "before_balance": type_dist_before,
        "after_balance": type_dist_after,
    }

    if args.limit > 0:
        converted = converted[: args.limit]
    if len(converted) < 3:
        raise ValueError(f"Need at least 3 valid examples, got {len(converted)}")

    train_rows, valid_rows, test_rows = split_rows(
        converted, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio, seed=args.seed
    )

    # 保存前剥离辅助字段 task_type，只保留 LLaMA-Factory 需要的三个字段
    def strip_aux(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        return [
            {"instruction": r["instruction"], "input": r["input"], "output": r["output"]}
            for r in rows
        ]

    save_json(args.output_dir / "code_sft_train.json", strip_aux(train_rows))
    save_json(args.output_dir / "code_sft_valid.json", strip_aux(valid_rows))
    save_json(args.output_dir / "code_sft_test.json", strip_aux(test_rows))
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
    length_removed = converted_after_dedup - converted_after_length
    print(f"长度过滤删除样本数  : {length_removed}  (剩余: {converted_after_length})")
    print(f"高质量筛选删除样本数: {hq_removed}  (语法+完整性，剩余: {converted_after_hq})")
    print(f"均衡采样删除样本数  : {balance_removed}  (max_per_type={args.max_per_type or '不限'})")
    print(f"质量问题样本总数    : {len(bad_cases)}  (写入 bad_cases.json)")
    print(f"最终样本数          : {len(converted)}")

    # 打印任务类型分布
    print("\n========== 任务类型分布 ==========")
    print(f"{'任务类型':<20}{'均衡前':>8}{'均衡后':>8}")
    for task_type in sorted(type_dist_before, key=lambda k: type_dist_before[k], reverse=True):
        before = type_dist_before.get(task_type, 0)
        after = type_dist_after.get(task_type, 0)
        print(f"{task_type:<20}{before:>8}{after:>8}")
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
