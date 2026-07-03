# AI-practice

## 第一部分：导入与常量（第1-16行）

```
#!/usr/bin/env python3"""Prepare python-code Alpaca data for LLaMA-Factory SFT."""
from __future__ import annotations
import argparse   # 处理命令行参数import json       # 读写JSON 文件import random# 打乱数据顺序from pathlib import Path   # 跨平台路径操作from typing import Any     # 类型标注用
PROJECT_ROOT = Path(__file__).resolve().parents[2]DEFAULT_SOURCE_DIR = PROJECT_ROOT / "python_code_instructions_18k_alpaca"DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sft" / "data"
```

**解释：**

- `Path(__file__).resolve().parents[2]`：`__file__` 是当前 `.py` 文件自身，`.parents[2]` 表示"往上两级目录"。脚本在`sft/scripts/` 下，所以 `parents[2]` 就是项目根目录。
- `DEFAULT_SOURCE_DIR` 和 `DEFAULT_OUTPUT_DIR`：定义原始数据和输出数据的默认路径，后面可以被命令行参数覆盖。

------

## 第二部分：`read_parquet` 函数（第18-38行）

```
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
                raise RuntimeError("Failed to read parquet...") from datasets_error
```

**解释：**

这个函数负责**读取 `.parquet` 格式的数据文件**，返回一个列表，每条数据是一个字典。

Parquet 是一种列式存储文件格式，Python 有三个库可以读它，代码用了**三层try/except 容错**：

| 优先级   | 库         | 如果失败…      |
| :------- | :--------- | :------------- |
| 第一选择 | `pandas`   | 尝试第二个     |
| 第二选择 | `pyarrow`  | 尝试第三个     |
| 第三选择 | `datasets` | 全部失败则报错 |

`.to_dict("records")` 的意思是：把 pandas 的表格转成 `[{列名: 值, ...}, ...]` 的列表形式，每行是一个字典。

------

## 第三部分：`find_parquet_files` 函数（第41-48行）

```
def find_parquet_files(source_dir: Path) -> list[Path]:
    data_dir = source_dir / "data"
    candidates = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []
    if not candidates:
        candidates = sorted(source_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {source_dir}")
    return candidates
```

**解释：**

这个函数在指定目录下**查找所有 `.parquet` 文件**，有两个查找位置，先找子目录 `data/`，再找根目录：

```
python_code_instructions_18k_alpaca/
├── data/
│   └── train-00000-of-00001.parquet  ← 优先在这里找
└── train-00000-of-00001.parquet      ← 找不到再在这里找
```

`glob("*.parquet")` 是通配符匹配，`sorted()` 保证多个文件时顺序固定（可复现）。

------

## 第四部分：`clean_text` 和 `convert_row`（第51-76行）

```
def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
```

**`clean_text`**：把任意类型转成字符串，去掉首尾空白。`None` 直接返回空字符串，防止后续操作报错。

------

```
def convert_row(row: dict[str, Any]) -> dict[str, str] | None:    instruction = clean_text(row.get("instruction"))    input_text  = clean_text(row.get("input"))    output      = clean_text(row.get("output"))
    if not instruction:        instruction = clean_text(row.get("prompt"))  # 备用字段    if not output:        return None   # output 为空 → 丢弃
    if not instruction and input_text:        instruction, input_text = input_text, ""# 把 input 提升为 instruction    if not instruction:        return None   # instruction 还是空 → 丢弃
    return {"instruction": instruction, "input": input_text, "output": output}
```

**`convert_row`**：把原始数据的一行转换成 LLaMA-Factory 要求的标准格式。

丢弃逻辑如下：

```
output为空？→ 丢弃（没有答案的数据没法训练）
instruction 为空且 input 也为空？→ 丢弃（没有问题的数据没法训练）
instruction 为空但 input 非空？→ 把 input 当作 instruction 使用
```

------

## 第五部分：`deduplicate` 函数（第79-88行）

```
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
```

**解释：**

去除完全重复的数据条目。

用 `set`（集合）来记录已经见过的数据，集合的查找是O(1) 的，非常高效。`key` 是三个字段拼成的元组——只有三个字段完全相同才算重复。

------

## 第六部分：`split_rows` 函数（第91-114行）

```
def split_rows(rows, train_ratio, valid_ratio, seed):    # 参数校验    if not0 < train_ratio < 1: raise ValueError(...)    if not 0 <= valid_ratio < 1: raise ValueError(...)    if train_ratio + valid_ratio >= 1: raise ValueError(...)
    shuffled = rows[:]                    # 复制一份，不改原列表    random.Random(seed).shuffle(shuffled) # 用固定 seed 打乱    total = len(shuffled)
    train_end = int(total * train_ratio)    valid_end = train_end + int(total * valid_ratio)
    train_rows = shuffled[:train_end]    valid_rows = shuffled[train_end:valid_end]    test_rows  = shuffled[valid_end:]# 剩余全部作为测试集
    if not train_rows or not valid_rows or not test_rows:        raise ValueError(f"Split produced an empty subset: ...")    return train_rows, valid_rows, test_rows
```

**解释：**

按比例把数据切成三份，以默认参数（90% / 5% / 5%）为例：

```
假设共有 18000 条数据：train_end = 18000 * 0.90 = 16200
  valid_end = 16200 + 18000 * 0.05 = 17100train_rows = [0    : 16200]  → 16200 条
  valid_rows = [16200: 17100]  →900 条
  test_rows  = [17100: 18000]  →   900 条
```

`random.Random(seed)` 创建一个**独立的随机数生成器**，用固定 seed 保证每次运行结果相同（可复现）。

------

## 第七部分：`main` 函数（第124-171行）

```
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", ...)
    parser.add_argument("--output_dir", ...)
    parser.add_argument("--limit",type=int,default=0)
    parser.add_argument("--train_ratio",type=float, default=0.90)
    parser.add_argument("--valid_ratio",type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
```

**解释：**`argparse` 用于解析命令行参数，`default` 是不传参时的默认值，`type` 负责自动转换类型。

------

```
    # 1. 读取所有 parquet 文件    parquet_files = find_parquet_files(args.source_dir)    raw_rows = []    for parquet_file in parquet_files:        raw_rows.extend(read_parquet(parquet_file))
    # 2. 清洗转换 + 去重    converted = [row for row in (convert_row(raw) for raw in raw_rows) if row is not None]    converted = deduplicate(converted)
    # 3. 可选：只取前 N 条（用于调试）    if args.limit > 0:        converted = converted[: args.limit]
    # 4. 划分数据集    train_rows, valid_rows, test_rows = split_rows(converted, ...)
    # 5. 保存 JSON 文件    save_json(args.output_dir / "code_sft_train.json", train_rows)    save_json(args.output_dir / "code_sft_valid.json", valid_rows)    save_json(args.output_dir / "code_sft_test.json",  test_rows)    save_json(args.output_dir / "dataset_info.json",   {...})
```

------

## 整体流程总结

```
parquet文件↓  read_parquet()读取原始数据（三层容错）
raw_rows (列表)
    ↓  convert_row()        字段映射 + 清洗 + 丢弃无效行↓  deduplicate()        去重
    ↓  limit（可选）         限制数量（调试用）
converted (列表)
    ↓  split_rows()         按比例划分
train / valid / test
    ↓  save_json()          写入 JSON 文件
sft/data/
├── code_sft_train.json
├── code_sft_valid.json
├── code_sft_test.json
└── dataset_info.json
```
