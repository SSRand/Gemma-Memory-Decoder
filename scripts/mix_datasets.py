#!/usr/bin/env python3
"""
Mix multiple HuggingFace datasets with balanced token/char contribution.
Supports:
  - Token-balanced sampling (each domain contributes similar total chars/tokens)
  - Complete shuffle for better generalization
  - Separate + mixed test sets
"""

import argparse
import os
import random
import logging
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_hf_split(path: str, split: str, filter_empty: bool = True) -> List[str]:
    """Load texts from a HuggingFace dataset saved with save_to_disk.

    Handles multi-shard and cache-based arrow layouts via load_from_disk.
    """
    split_dir = os.path.join(path, split)
    if not os.path.isdir(split_dir):
        logger.warning(f"Split directory not found: {split_dir}")
        return []

    ds = load_from_disk(path)
    if split not in ds:
        logger.warning(f"Split '{split}' not in dataset at {path}")
        return []

    texts = ds[split]['text']
    if filter_empty:
        before = len(texts)
        texts = [t for t in texts if t and t.strip()]
        removed = before - len(texts)
        if removed > 0:
            logger.info(f"  Filtered {removed} empty texts from {path}/{split}")
    return texts


def get_dataset_stats(texts: List[str]) -> Dict:
    """Calculate statistics for a dataset."""
    lengths = [len(t) for t in texts]
    return {
        'n_samples': len(texts),
        'total_chars': sum(lengths),
        'avg_length': np.mean(lengths) if lengths else 0,
        'median_length': np.median(lengths) if lengths else 0,
    }


def token_balanced_sample(
    datasets: Dict[str, List[str]],
    target_chars_per_domain: int = None,
    seed: int = 42
) -> Tuple[Dict[str, List[str]], Dict]:
    """
    Sample from each dataset to achieve balanced total chars.
    
    Args:
        datasets: Dict of {name: texts}
        target_chars_per_domain: Target chars per domain. If None, uses min domain.
        seed: Random seed
    
    Returns:
        sampled_datasets: Dict of sampled texts per domain
        stats: Sampling statistics
    """
    random.seed(seed)
    np.random.seed(seed)
    
    stats = {}
    for name, texts in datasets.items():
        ds_stats = get_dataset_stats(texts)
        stats[name] = ds_stats
        logger.info(f"{name}: {ds_stats['n_samples']} samples, "
                   f"{ds_stats['total_chars']:,} chars, "
                   f"avg {ds_stats['avg_length']:.0f} chars")
    
    if target_chars_per_domain is None:
        target_chars_per_domain = min(s['total_chars'] for s in stats.values())
    
    logger.info(f"\nTarget chars per domain: {target_chars_per_domain:,}")
    
    sampled = {}
    for name, texts in datasets.items():
        current_stats = stats[name]
        
        if current_stats['total_chars'] <= target_chars_per_domain:
            sampled[name] = texts.copy()
            logger.info(f"{name}: using all {len(texts)} samples "
                       f"({current_stats['total_chars']:,} chars)")
        else:
            texts_with_len = [(t, len(t)) for t in texts]
            random.shuffle(texts_with_len)
            
            selected = []
            total = 0
            for t, length in texts_with_len:
                if total + length > target_chars_per_domain:
                    if not selected:
                        selected.append(t)
                        total += length
                    break
                selected.append(t)
                total += length
            
            sampled[name] = selected
            logger.info(f"{name}: sampled {len(selected)} samples "
                       f"({total:,} chars from {current_stats['total_chars']:,})")
    
    return sampled, stats


def mix_and_shuffle(
    sampled_datasets: Dict[str, List[str]],
    seed: int = 42,
    add_domain_tag: bool = False
) -> List[Dict]:
    """Mix and shuffle samples from all domains."""
    random.seed(seed)
    
    all_samples = []
    for domain, texts in sampled_datasets.items():
        for text in texts:
            sample = {'text': text}
            if add_domain_tag:
                sample['domain'] = domain
            all_samples.append(sample)
    
    random.shuffle(all_samples)
    logger.info(f"Mixed dataset: {len(all_samples)} total samples")
    
    return all_samples


def _build_split_samples(
    datasets_by_domain: Dict[str, List[str]],
    seed: int,
    add_domain_tag: bool
) -> List[Dict]:
    """Collect all texts from multiple domains, tag them, and shuffle."""
    samples = []
    for domain, texts in datasets_by_domain.items():
        for text in texts:
            sample = {'text': text}
            if add_domain_tag:
                sample['domain'] = domain
            samples.append(sample)
    random.seed(seed)
    random.shuffle(samples)
    return samples


def _samples_to_dataset(samples: List[Dict], add_domain_tag: bool) -> Dataset:
    """Convert list-of-dicts samples into a HF Dataset."""
    if add_domain_tag:
        return Dataset.from_dict({
            'text': [s['text'] for s in samples],
            'domain': [s['domain'] for s in samples],
        })
    return Dataset.from_dict({'text': [s['text'] for s in samples]})


def create_mixed_dataset(
    dataset_paths: Dict[str, str],
    output_path: str,
    target_chars: int = None,
    seed: int = 42,
    add_domain_tag: bool = True,
    include_test: bool = True,
    balance: bool = True,
):
    """
    Create a mixed dataset from multiple domains.

    Args:
        dataset_paths: Dict of {name: path}
        output_path: Output directory
        target_chars: Target chars per domain (None = use minimum)
        seed: Random seed
        add_domain_tag: Whether to add 'domain' column
        include_test: Whether to create a test split
        balance: If True, apply token-balanced sampling; if False, use all data
    """
    logger.info("Loading train sets...")
    train_datasets = {}
    for name, path in dataset_paths.items():
        texts = load_hf_split(path, 'train')
        if texts:
            train_datasets[name] = texts
            logger.info(f"  {name}: {len(texts)} samples")

    logger.info("\nLoading validation sets...")
    val_datasets = {}
    for name, path in dataset_paths.items():
        texts = load_hf_split(path, 'validation')
        if texts:
            val_datasets[name] = texts
            logger.info(f"  {name}: {len(texts)} samples")

    test_datasets = {}
    if include_test:
        logger.info("\nLoading test sets...")
        for name, path in dataset_paths.items():
            texts = load_hf_split(path, 'test')
            if texts:
                test_datasets[name] = texts
                logger.info(f"  {name}: {len(texts)} samples")

    # --- Train: optionally balance, then shuffle ---
    if balance:
        logger.info("\n" + "="*50)
        logger.info("Token-balanced sampling for train set")
        logger.info("="*50)
        sampled_train, train_stats = token_balanced_sample(
            train_datasets, target_chars, seed
        )
    else:
        logger.info("\n" + "="*50)
        logger.info("Using all train data (no balancing)")
        logger.info("="*50)
        sampled_train = train_datasets
        for name, texts in sampled_train.items():
            stats = get_dataset_stats(texts)
            logger.info(f"{name}: {stats['n_samples']} samples, "
                       f"{stats['total_chars']:,} chars")

    logger.info("\n" + "="*50)
    logger.info("Mixing and shuffling train set")
    logger.info("="*50)
    mixed_train = mix_and_shuffle(sampled_train, seed, add_domain_tag)

    # --- Validation: merge all domains + shuffle ---
    logger.info("\n" + "="*50)
    logger.info("Creating validation set")
    logger.info("="*50)
    mixed_val = _build_split_samples(val_datasets, seed, add_domain_tag)
    logger.info(f"Mixed validation: {len(mixed_val)} samples")

    # --- Test (optional): merge test sets, fall back to val for missing domains ---
    mixed_test = []
    if include_test:
        logger.info("\n" + "="*50)
        logger.info("Creating test set")
        logger.info("="*50)

        test_merged = {}
        for domain, texts in test_datasets.items():
            test_merged[domain] = texts
        for domain, texts in val_datasets.items():
            if domain not in test_merged:
                logger.info(f"  {domain}: using validation as test (no test split)")
                test_merged[domain] = texts

        mixed_test = _build_split_samples(test_merged, seed, add_domain_tag)
        logger.info(f"Mixed test: {len(mixed_test)} samples")

    # --- Save ---
    logger.info("\n" + "="*50)
    logger.info("Saving datasets")
    logger.info("="*50)

    os.makedirs(output_path, exist_ok=True)

    splits = {
        'train': _samples_to_dataset(mixed_train, add_domain_tag),
        'validation': _samples_to_dataset(mixed_val, add_domain_tag),
    }
    if include_test and mixed_test:
        splits['test'] = _samples_to_dataset(mixed_test, add_domain_tag)

    dataset_dict = DatasetDict(splits)
    dataset_dict.save_to_disk(output_path)
    logger.info(f"Saved to {output_path}")

    if include_test:
        for name, path in dataset_paths.items():
            domain_output = os.path.join(output_path, f"test_{name}")
            if name in test_datasets:
                texts = test_datasets[name]
            elif name in val_datasets:
                texts = val_datasets[name]
                logger.info(f"  {name}: using validation as test")
            else:
                continue
            domain_ds = Dataset.from_dict({'text': texts})
            domain_ds.save_to_disk(domain_output)
            logger.info(f"  Saved {name} test set ({len(texts)} samples) to {domain_output}")

    # --- Statistics ---
    logger.info("\n" + "="*50)
    logger.info("Final Statistics")
    logger.info("="*50)
    logger.info(f"Train: {len(mixed_train)} samples")
    logger.info(f"Validation: {len(mixed_val)} samples")
    if include_test:
        logger.info(f"Test (mixed): {len(mixed_test)} samples")

    if add_domain_tag:
        from collections import Counter
        train_domains = Counter(s['domain'] for s in mixed_train)
        logger.info(f"\nTrain domain distribution:")
        for domain, count in sorted(train_domains.items()):
            pct = count / len(mixed_train) * 100
            logger.info(f"  {domain}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Mix datasets with token-balanced sampling"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        required=True,
        help="Dataset paths in format name:path (e.g., asylex:/path/to/asylex)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for mixed dataset"
    )
    parser.add_argument(
        "--target_chars",
        type=int,
        default=None,
        help="Target chars per domain (default: use minimum domain)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no_domain_tag",
        action="store_true",
        help="Don't add domain column to dataset"
    )
    parser.add_argument(
        "--no_test",
        action="store_true",
        help="Skip test split creation (only produce train + validation)"
    )
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="Skip token-balanced sampling; use all data from every domain"
    )
    
    args = parser.parse_args()
    
    dataset_paths = {}
    for ds_spec in args.datasets:
        if ':' not in ds_spec:
            raise ValueError(f"Invalid dataset spec: {ds_spec}. Use name:path format.")
        name, path = ds_spec.split(':', 1)
        dataset_paths[name] = path
    
    create_mixed_dataset(
        dataset_paths=dataset_paths,
        output_path=args.output_path,
        target_chars=args.target_chars,
        seed=args.seed,
        add_domain_tag=not args.no_domain_tag,
        include_test=not args.no_test,
        balance=not args.no_balance,
    )


if __name__ == "__main__":
    main()
