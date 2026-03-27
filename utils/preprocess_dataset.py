import datasets
import json
import argparse
import torch
import pickle
import os
import glob
from datasets import load_dataset,load_from_disk,DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    GPT2LMHeadModel,
    GPT2Config,
)

from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Path to dataset files",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Path to dataset files",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--padding_index",
        type=int,
        default=-100,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    
    args = parser.parse_args()
    return args

# Input should be a dataset dict object , with text column name being 'text'
def tokenize_and_group_text(raw_datasets, tokenizer, block_size, stride, padding_index, num_process):
    # Drop empty/blank text rows to avoid null feature shards after tokenization.
    cleaned_raw = {}
    for split, split_dataset in raw_datasets.items():
        cur_num_proc = min(num_process, len(split_dataset)) if num_process is not None else None
        cleaned_raw[split] = split_dataset.filter(
            lambda x: x.get("text") is not None and x["text"].strip() != "",
            num_proc=cur_num_proc,
            desc=f"Filtering empty text for split {split} (num_proc={cur_num_proc})",
        )
    raw_datasets = datasets.DatasetDict(cleaned_raw)

    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        return output
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        input_ids = []
        attention_mask = []
        labels = []
        # We implement a sliding window, so all tokens have a non-zero context in their prediction.
        # We then mask the duplicate tokens' labels, to not count any token twice in the loss.
        for i in tqdm(range(0, total_length, stride), total=total_length):
            begin_loc = max(i + stride - block_size, 0)
            end_loc = min(i + stride, total_length)
            trg_len = end_loc - i
            cur_input_ids = concatenated_examples['input_ids'][begin_loc:end_loc]
            cur_labels = list(cur_input_ids)
            cur_labels[:-trg_len] = [padding_index] * (len(cur_labels) - trg_len)

            if len(cur_input_ids) < block_size:
                padding_size = block_size - len(cur_input_ids)
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                cur_input_ids += [pad_token_id] * padding_size
                cur_labels += [padding_index] * padding_size
            input_ids.append(cur_input_ids)
            attention_mask.append([1] * len(cur_labels))
            labels.append(cur_labels)

        result = {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
        return result

    tokenized_datasets = {}
    for split, split_dataset in raw_datasets.items():
        cur_num_proc = min(num_process, len(split_dataset)) if num_process is not None else None
        tokenized_datasets[split] = split_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=cur_num_proc,
            remove_columns=split_dataset.column_names,
            load_from_cache_file=False,  # rebuild to avoid stale/empty shards
            desc=f"Running tokenizer on dataset split {split} (num_proc={cur_num_proc})",
        )
    
    # calculate total tokens
    for split in tokenized_datasets:
        total_tokens = 0
        for example in tokenized_datasets[split]:
            total_tokens += len(example['input_ids'])
        logger.info(f"Total tokens in dataset split {split}: {total_tokens}")
    
    lm_datasets = {}
    for split, split_dataset in tokenized_datasets.items():
        cur_num_proc = min(num_process, len(split_dataset)) if num_process is not None else None
        lm_datasets[split] = split_dataset.map(
            group_texts,
            batched=True,
            num_proc=cur_num_proc,
            load_from_cache_file=False,  # rebuild to avoid stale/empty shards
            desc=f"Grouping texts in chunks of {block_size} for split {split} (num_proc={cur_num_proc})",
        )
    
    final_dataset = DatasetDict(lm_datasets)
    print(f"Final dataset: {final_dataset}")
    
    return final_dataset

def main():
    args = parse_args()

    if args.dataset_name is None:
        raise ValueError("`--dataset_name` must be provided (either a HF name or local path).")

    # Prefer local path if it exists; otherwise fall back to HF hub.
    if os.path.exists(args.dataset_name):
        logger.info(f"Loading dataset from local disk at {args.dataset_name}")
        dataset_info_path = os.path.join(args.dataset_name, "dataset_info.json")
        dataset_dict_path = os.path.join(args.dataset_name, "dataset_dict.json")

        # Case 1: already a saved HF dataset (Dataset or DatasetDict)
        if os.path.exists(dataset_info_path) or os.path.exists(dataset_dict_path):
            raw_datasets = load_from_disk(args.dataset_name)
        else:
            # Case 2: directory of parquet files (e.g., HF cached download)
            data_files = {}
            for split in ["train", "validation", "test"]:
                matched = sorted(glob.glob(os.path.join(args.dataset_name, f"{split}*.parquet")))
                if matched:
                    data_files[split] = matched

            # If no split-specific files, try any parquet as train
            if not data_files:
                any_parquet = sorted(glob.glob(os.path.join(args.dataset_name, "*.parquet")))
                if any_parquet:
                    data_files["train"] = any_parquet

            if data_files:
                logger.info(f"Loading parquet dataset from {args.dataset_name} with splits {list(data_files.keys())}")
                raw_datasets = datasets.load_dataset("parquet", data_files=data_files)
            else:
                raise FileNotFoundError(
                    f"No dataset found at {args.dataset_name}: expected HF saved dataset or parquet files."
                )
    else:
        logger.info(f"Loading dataset from hub: name={args.dataset_name}, config={args.dataset_config_name}")
        raw_datasets = datasets.load_dataset(args.dataset_name, args.dataset_config_name)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    lm_datasets = tokenize_and_group_text(raw_datasets, tokenizer, args.block_size, args.stride, args.padding_index, args.num_proc)
    
    # Dictionary to hold the summary info for each split
    dstore_summary = {}

    # Iterate over all splits in the lm_datasets (each key is a split name, e.g., "train", "validation", etc.)
    for split_name, split_dataset in lm_datasets.items():
        dstore_size = 0
        dataset_cnt = []
        
        # Compute dataset count and dstore_size for the current split.
        for chunk in split_dataset['labels']:
            cur_len = len([x for x in chunk[1:] if x != args.padding_index])
            dstore_size += cur_len
            dataset_cnt.append(cur_len)
        
        # Log and print the computed dstore_size for this split.
        logger.info(f"Split '{split_name}': Setting dstore size to {dstore_size}!")
        print(f"Split '{split_name}': dstore size = {dstore_size}")
        
        # Store the results in our summary dictionary.
        dstore_summary[split_name] = {
            "dstore_size": dstore_size,
            "dataset_cnt_len": len(dataset_cnt)  
        }
        
        # Compute the dstore_range for each example in the split.
        # The dstore_range is a list of (start, end) indices for each chunk.
        idx = 0
        dstore_range = []
        for cnt in dataset_cnt:
            dstore_range.append((idx, idx + cnt))
            idx += cnt

        # Add the computed dstore_range column to the current dataset split.
        lm_datasets[split_name] = split_dataset.add_column("dstore_range", dstore_range)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Finally, save the lm_datasets (with the new 'dstore_range' column added to each split) to disk.
    lm_datasets.save_to_disk(args.output_dir)
    logger.info(f"lm_datasets saved to {args.output_dir}")

    # Save the summary information (which includes dstore_size for each split) to a JSON file.
    json_file_path = os.path.join(args.output_dir, "dstore_summary.json")
    with open(json_file_path, "w") as f:
        json.dump(dstore_summary, f, indent=4)
    logger.info(f"Saved dstore summary to {json_file_path}")
    
if __name__ == "__main__":
    main()