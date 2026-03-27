#!/bin/bash
# Preprocess dataset for Gemma MemoryDecoder training

TOKENIZER="/nfs-stor/changjiang.han/models/gemma-3-4b-it"
DATASET_NAME="/nfs-stor/changjiang.han/dataset/mixed-hf"
OUTPUT_DIR="/nfs-stor/changjiang.han/dataset/mixed-gemma3"
NUM_PROC=32

CMD="python utils/preprocess_dataset.py \
    --dataset_name ${DATASET_NAME} \
    --tokenizer_path ${TOKENIZER} \
    --output_dir ${OUTPUT_DIR} \
    --num_proc ${NUM_PROC}"

if [ -n "${DATASET_CONFIG:-}" ]; then
    CMD="$CMD --dataset_config_name ${DATASET_CONFIG}"
fi

echo "=========================================="
echo "Preprocessing dataset: ${DATASET_NAME}"
echo "Tokenizer: ${TOKENIZER}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
eval $CMD
echo "Dataset preprocessing complete!"
