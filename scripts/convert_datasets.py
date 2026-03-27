#!/usr/bin/env python3
"""
Convert raw datasets to HuggingFace format with 'text' column.
Supports:
  - mimic_iii_diagnosis_anonymous (txt file, one record per line)
  - AsyLex (CSV files or full case texts from tar.gz)
"""

import argparse
import os
import tarfile
import pandas as pd
from datasets import Dataset, DatasetDict
from loguru import logger
from tqdm import tqdm


def convert_mimic(input_path: str, output_path: str):
    """
    Convert MIMIC-III txt file to HuggingFace dataset format.
    Each line in the txt file is treated as a separate document.
    """
    txt_file = os.path.join(input_path, "anonymized_patient_notes.txt")
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"MIMIC file not found: {txt_file}")
    
    logger.info(f"Reading MIMIC data from {txt_file}...")
    
    texts = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                texts.append(line)
    
    logger.info(f"Loaded {len(texts)} records from MIMIC")
    
    # Create train/validation split (90/10)
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts}),
        "validation": Dataset.from_dict({"text": val_texts}),
    })
    
    logger.info(f"Train: {len(train_texts)}, Validation: {len(val_texts)}")
    
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)
    logger.info(f"Saved MIMIC dataset to {output_path}")
    
    return dataset


def convert_asylex(input_path: str, output_path: str, text_column: str = None):
    """
    Convert AsyLex data to HuggingFace dataset format.
    
    Args:
        input_path: Path to AsyLex directory
        output_path: Output path for HF dataset
        text_column: Which data source to use. Options:
            - 'cases': cases_anonymized_txt_raw.tar.gz (RECOMMENDED, 59K full case texts)
            - 'determination': determination_label_extracted_sentences.csv (short sentences only)
            - 'main': main_and_case_cover_all_entities_inferred.csv (large, 3.7GB, NER data)
    """
    # Default to 'cases' (full text) instead of 'determination' (short sentences)
    if text_column is None:
        text_column = "cases"
    
    if text_column == "cases":
        # Use full case texts from tar.gz (RECOMMENDED)
        tar_file = os.path.join(input_path, "cases_anonymized_txt_raw.tar.gz")
        if not os.path.exists(tar_file):
            raise FileNotFoundError(f"AsyLex tar file not found: {tar_file}")
        
        logger.info(f"Reading AsyLex full case texts from {tar_file}...")
        
        texts = []
        with tarfile.open(tar_file, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.txt')]
            logger.info(f"Found {len(members)} case files in archive")
            
            for member in tqdm(members, desc="Extracting case texts"):
                try:
                    f = tar.extractfile(member)
                    if f is not None:
                        content = f.read().decode('utf-8', errors='ignore')
                        # Clean up the text: remove excessive whitespace
                        content = ' '.join(content.split())
                        if content.strip() and len(content) > 100:  # Skip very short files
                            texts.append(content.strip())
                except Exception as e:
                    logger.warning(f"Failed to read {member.name}: {e}")
                    continue
        
        logger.info(f"Successfully extracted {len(texts)} case texts")
        
    elif text_column == "determination":
        # Use short determination sentences (NOT recommended for LM training)
        csv_file = os.path.join(input_path, "determination_label_extracted_sentences.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"AsyLex file not found: {csv_file}")
        
        logger.info(f"Reading AsyLex determination sentences from {csv_file}...")
        logger.warning("WARNING: 'determination' contains very short sentences (~8 tokens avg). "
                      "Consider using --text_column cases for full case texts.")
        
        df = pd.read_csv(csv_file, sep=";", on_bad_lines="skip")
        col_name = "extracted_sentences_determination"
        
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in {df.columns.tolist()}")
        
        raw_texts = df[col_name].dropna().astype(str).tolist()
        
        # Clean texts (remove list formatting)
        texts = []
        for t in raw_texts:
            if t.startswith("[") and t.endswith("]"):
                try:
                    import ast
                    parsed = ast.literal_eval(t)
                    if isinstance(parsed, list):
                        t = " ".join(str(x) for x in parsed)
                except:
                    pass
            if t.strip():
                texts.append(t.strip())
        
        logger.info(f"After cleaning: {len(texts)} texts")
        
    elif text_column == "main":
        # Use main CSV (large, 3.7GB, primarily for NER)
        csv_file = os.path.join(input_path, "main_and_case_cover_all_entities_inferred.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"AsyLex file not found: {csv_file}")
        
        logger.info(f"Reading AsyLex main CSV from {csv_file}...")
        logger.warning("WARNING: This file is 3.7GB and primarily contains NER annotations.")
        
        df = pd.read_csv(csv_file, sep=";", on_bad_lines="skip")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Try to find a text column
        texts = []
        for col in df.columns:
            if 'text' in col.lower() or 'sentence' in col.lower():
                texts = df[col].dropna().astype(str).tolist()
                logger.info(f"Using column '{col}'")
                break
        
        if not texts:
            raise ValueError(f"Could not find text column in {df.columns.tolist()}")
    else:
        raise ValueError(f"Unknown text_column: {text_column}. Use 'cases', 'determination', or 'main'.")
    
    logger.info(f"Total texts collected: {len(texts)}")
    
    # Compute some statistics
    total_chars = sum(len(t) for t in texts)
    avg_chars = total_chars / len(texts) if texts else 0
    logger.info(f"Average text length: {avg_chars:.0f} characters")
    
    # Create train/validation split (90/10)
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts}),
        "validation": Dataset.from_dict({"text": val_texts}),
    })
    
    logger.info(f"Train: {len(train_texts)}, Validation: {len(val_texts)}")
    
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)
    logger.info(f"Saved AsyLex dataset to {output_path}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert datasets to HuggingFace format")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mimic", "asylex"],
        help="Dataset to convert",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the raw dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the converted dataset",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="For AsyLex: 'cases' (default, full texts), 'determination' (short sentences), or 'main' (NER data)",
    )
    
    args = parser.parse_args()
    
    if args.dataset == "mimic":
        convert_mimic(args.input_path, args.output_path)
    elif args.dataset == "asylex":
        convert_asylex(args.input_path, args.output_path, args.text_column)


if __name__ == "__main__":
    main()
