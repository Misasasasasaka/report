#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Stanford 的 IMDB 数据集（aclImdb）转换为两个 CSV：
- ./data/train.csv
- ./data/test.csv
CSV 列：text,label   （pos→1, neg→0）

使用方法：
    python convert_imdb_to_csv.py /path/to/aclImdb --out ./data

注意：会忽略 train/unsup 等无标签目录。
"""
import argparse
import csv
from pathlib import Path

LABEL_MAP = {"pos": 1, "neg": 0}

def convert_split(root: Path, split: str, out_dir: Path):
    """将某个 split（train/test）写成 CSV"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{split}.csv"
    total = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for label_name, label_id in LABEL_MAP.items():
            src_dir = root / split / label_name
            if not src_dir.exists():
                continue
            # 遍历所有 .txt
            for txt_path in sorted(src_dir.glob("*.txt")):
                try:
                    text = txt_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    # 兜底：有极少数文件可能不是 utf-8，尝试 latin-1
                    text = txt_path.read_text(encoding="latin-1")
                # csv.writer 会自动加引号，允许换行
                writer.writerow([text, label_id])
                total += 1
    print(f"[{split}] 写入 {total} 行 -> {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Convert aclImdb to train.csv/test.csv")
    parser.add_argument("aclImdb_root", type=str, help="aclImdb 根目录（包含 train/ 和 test/）")
    parser.add_argument("--out", type=str, default="./data", help="CSV 输出目录（默认 ./data）")
    args = parser.parse_args()

    root = Path(args.aclImdb_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not (root / "train").exists() or not (root / "test").exists():
        raise SystemExit(f"未找到 {root}/train 或 {root}/test，请确认路径是否为 aclImdb 根目录。")

    # 生成 train.csv / test.csv
    for split in ("train", "test"):
        convert_split(root, split, out_dir)

    print("完成。")

if __name__ == "__main__":
    main()
