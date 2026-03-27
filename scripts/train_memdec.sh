#!/bin/bash
# Train Gemma MemoryDecoder

MODEL_FAMILY="gemma3_text"
MODEL_SIZE="1b"
ACCELERATE_CONFIG=./accelerate_config/gemma.yaml
MODEL=/path/to/models/gemma-3-1b-it

KNN_MODEL_FAMILY="gemma3"
KNN_MODEL_SIZE="4b"
KNN_DIMENSION=2560

DATASET_NAME="mixed-small"
DATASET=/path/to/dataset/mixed-gemma3
KNN_DSTORE_PATH=/path/to/dstore/${KNN_MODEL_FAMILY}-${KNN_MODEL_SIZE}/${DATASET_NAME}/knn_${KNN_MODEL_FAMILY}_train_${KNN_DIMENSION}.arrow
OUTPUT_DIR=/path/to/checkpoints/memdec-${MODEL_FAMILY}-${MODEL_SIZE}-mixed-small

RESUME_CKPT=${RESUME_CKPT:-""}

LEARNING_RATE=1e-3
GRADIENT_ACCUMULATION_STEPS=2
BATCH_SIZE=4
NUM_EPOCHS=30

export TMPDIR=/tmp
mkdir -p $TMPDIR

echo "=========================================="
echo "Training Gemma MemoryDecoder"
echo "=========================================="
echo "Model: ${MODEL}"
echo "KNN Model: ${KNN_MODEL_FAMILY}-${KNN_MODEL_SIZE} (dim=${KNN_DIMENSION})"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "KNN Datastore: ${KNN_DSTORE_PATH}"
echo "=========================================="

accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m train_memdec \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --dataset_split_name train \
    --knn_save_path ${KNN_DSTORE_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type linear \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --seed 42 \
    --checkpointing_steps "epoch" \
    --report_to none \
    ${RESUME_CKPT:+--resume_from_checkpoint ${RESUME_CKPT}}
