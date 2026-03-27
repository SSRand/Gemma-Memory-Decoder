#!/usr/bin/env python3
"""
Sample the first n% of a HuggingFace dataset to create a smaller version.
This takes the first rows (prefix), not random sampling.

用法示例:
    # 直接指定目标train大小
    python sample_dataset.py --input_path /path/to/dataset-hf --output_path /path/to/dataset-hf-small --target_train_size 1454
    
    # 指定采样比例
    python sample_dataset.py --input_path /path/to/dataset-hf --output_path /path/to/dataset-hf-small --ratio 0.05
    
    # 指定预处理后的目标大小（会根据转换率自动计算）
    python sample_dataset.py --input_path /path/to/dataset-hf --output_path /path/to/dataset-hf-small --target_processed_size 12206 --conversion_rate 8.4

转换率参考（-hf → -qwen3）:
    - asylex: 8.40x (长文本被分成多个chunk)
    - mimic: 4.48x
    - wikitext: 0.134x (大量空行被过滤)
"""

import argparse
import os
from datasets import load_from_disk, DatasetDict
from loguru import logger


def sample_dataset(
    input_path: str, 
    output_path: str, 
    target_train_size: int = None, 
    ratio: float = None,
    target_processed_size: int = None,
    conversion_rate: float = None,
):
    """
    Sample the first n rows or n% of a dataset.
    
    Args:
        input_path: Path to the input HF dataset
        output_path: Path to save the sampled dataset
        target_train_size: Target number of rows for train split (direct)
        ratio: Ratio to sample (e.g., 0.05 for 5%)
        target_processed_size: Target size after preprocessing (requires conversion_rate)
        conversion_rate: Conversion rate from -hf to -qwen3 format
    """
    logger.info(f"Loading dataset from {input_path}...")
    ds = load_from_disk(input_path)
    logger.info(f"Original dataset: {ds}")
    
    original_train_size = len(ds['train'])
    
    # Calculate target_train_size from processed size and conversion rate
    if target_processed_size is not None and conversion_rate is not None:
        target_train_size = int(target_processed_size / conversion_rate)
        logger.info(f"Target processed size: {target_processed_size}, conversion rate: {conversion_rate}x")
        logger.info(f"Calculated target train size: {target_train_size}")
    
    # Calculate ratio from target size if provided
    if target_train_size is not None:
        ratio = target_train_size / original_train_size
        logger.info(f"Target train size: {target_train_size}, calculated ratio: {ratio:.4f}")
    
    if ratio is None:
        raise ValueError("Must provide one of: ratio, target_train_size, or (target_processed_size + conversion_rate)")
    
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"Ratio must be between 0 and 1, got {ratio}")
    
    # Sample each split
    sampled_splits = {}
    for split_name, split_data in ds.items():
        original_size = len(split_data)
        new_size = max(1, int(original_size * ratio))  # At least 1 row
        
        # Take the first n rows (prefix sampling)
        sampled_data = split_data.select(range(new_size))
        sampled_splits[split_name] = sampled_data
        
        logger.info(f"  {split_name}: {original_size} -> {new_size} rows ({ratio*100:.2f}%)")
    
    # Create new DatasetDict
    sampled_ds = DatasetDict(sampled_splits)
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    sampled_ds.save_to_disk(output_path)
    logger.info(f"Saved sampled dataset to {output_path}")
    logger.info(f"Final dataset: {sampled_ds}")
    
    # Estimate processed size
    if conversion_rate is not None:
        estimated_processed = int(len(sampled_ds['train']) * conversion_rate)
        logger.info(f"Estimated processed train size: ~{estimated_processed} rows")
    
    return sampled_ds


def main():
    parser = argparse.ArgumentParser(
        description="Sample first n% of a HuggingFace dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
转换率参考（-hf → -qwen3）:
    - asylex: 8.40x (长文本被分成多个chunk)
    - mimic: 4.48x
    - wikitext: 0.134x (大量空行被过滤)

示例:
    # 创建 asylex-hf-small，预处理后约12206条
    python sample_dataset.py \\
        --input_path <dataset_root>/asylex-hf \\
        --output_path <dataset_root>/asylex-hf-small \\
        --target_processed_size 12206 \\
        --conversion_rate 8.4
        
    # 创建 mimic-hf-small，预处理后约12206条
    python sample_dataset.py \\
        --input_path <dataset_root>/mimic-hf \\
        --output_path <dataset_root>/mimic-hf-small \\
        --target_processed_size 12206 \\
        --conversion_rate 4.48
"""
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input HF dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the sampled dataset",
    )
    parser.add_argument(
        "--target_train_size",
        type=int,
        default=None,
        help="Target number of rows for train split (direct)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Ratio to sample (e.g., 0.05 for 5%%)",
    )
    parser.add_argument(
        "--target_processed_size",
        type=int,
        default=None,
        help="Target size after preprocessing (requires --conversion_rate)",
    )
    parser.add_argument(
        "--conversion_rate",
        type=float,
        default=None,
        help="Conversion rate from -hf to -qwen3 format (e.g., 8.4 for asylex, 4.48 for mimic)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    has_direct = args.target_train_size is not None or args.ratio is not None
    has_processed = args.target_processed_size is not None and args.conversion_rate is not None
    
    if not has_direct and not has_processed:
        parser.error("Must provide one of: --ratio, --target_train_size, or (--target_processed_size and --conversion_rate)")
    
    if args.target_processed_size is not None and args.conversion_rate is None:
        parser.error("--target_processed_size requires --conversion_rate")
    
    sample_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        target_train_size=args.target_train_size,
        ratio=args.ratio,
        target_processed_size=args.target_processed_size,
        conversion_rate=args.conversion_rate,
    )


if __name__ == "__main__":
    main()
