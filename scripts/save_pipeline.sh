#!/bin/bash
# KNN datastore generation pipeline for Gemma MemoryDecoder

MODEL_FAMILY="gemma3"
MODEL_SIZE="4b"
DIMENSION=2560
ACCELERATE_CONFIG="./accelerate_config/gemma.yaml"
MODEL_TO_SAVE="/nfs-stor/changjiang.han/models/gemma-3-4b-it"

SKIP_STEP1=${SKIP_STEP1:-false}
SKIP_STEP2=${SKIP_STEP2:-false}

DATASET_NAME="mixed-small"
SUBSET="train"
DATASET="/nfs-stor/changjiang.han/dataset/mixed-gemma3"

DSTORE_DIR="/nfs-stor/changjiang.han/dstore/${MODEL_FAMILY}-${MODEL_SIZE}/${DATASET_NAME}"
OUTPUT_DIR="/nfs-stor/changjiang.han/results/tmp/${MODEL_FAMILY}-${MODEL_SIZE}-${DATASET_NAME}-ppl"

BATCH_SIZE_EVAL=16
BATCH_SIZE_KNN=4500

K=1024
KNN_TEMP=16.0
PROBE=32
NCENTROIDS=4096
CODE_SIZE=64
NUM_KEYS_TO_ADD=1000000

DSTORE_PATH="${DSTORE_DIR}/dstore_${MODEL_FAMILY}_${SUBSET}_${DIMENSION}.arrow"
VAL_PATH="${DSTORE_DIR}/${SUBSET}_vals.pkl"
INDEX_PATH="${DSTORE_DIR}/${SUBSET}_${DIMENSION}.index"
OUTPUT_PATH="${DSTORE_DIR}/knn_${MODEL_FAMILY}_${SUBSET}_${DIMENSION}.arrow"

echo "=========================================="
echo "Gemma MemoryDecoder - KNN Pipeline"
echo "=========================================="
echo "Model: ${MODEL_FAMILY}-${MODEL_SIZE}"
echo "Model path: ${MODEL_TO_SAVE}"
echo "Dataset: ${DATASET_NAME}"
echo "Dimension: ${DIMENSION}"
echo "=========================================="

# Step 1: Generate Datastore
if [ "${SKIP_STEP1}" != "true" ]; then
echo "[Step 1/3] Generating datastore..."

WANDB_PROJECT="neuralKNN" accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m train_base \
    --model_name_or_path ${MODEL_TO_SAVE} \
    --dataset_name ${DATASET} \
    --do_eval --eval_subset ${SUBSET} \
    --per_device_eval_batch_size ${BATCH_SIZE_EVAL} \
    --output_dir ${OUTPUT_DIR} \
    --dstore_dir ${DSTORE_DIR} \
    --save_knnlm_dstore \
    --report_to none

if [ $? -ne 0 ]; then echo "Error: Datastore generation failed!"; exit 1; fi
echo "[Step 1/3] Done"
else
    echo "[Step 1/3] Skipped"
fi

# Step 2: Build Index
if [ "${SKIP_STEP2}" != "true" ]; then
echo "[Step 2/3] Building FAISS index..."

PYTHONUNBUFFERED=1 python -u -m knn_utils.build_index \
    --dstore_path ${DSTORE_PATH} \
    --num_keys_to_add_at_a_time ${NUM_KEYS_TO_ADD} \
    --ncentroids ${NCENTROIDS} \
    --code_size ${CODE_SIZE} \
    --probe ${PROBE}

if [ $? -ne 0 ]; then echo "Error: Index building failed!"; exit 1; fi
echo "[Step 2/3] Done"
else
    echo "[Step 2/3] Skipped"
fi

# Step 3: Save KNN Results
echo "[Step 3/3] Saving KNN results..."

accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m knn_utils.saveKNNMulti \
    --model_path ${MODEL_TO_SAVE} \
    --dstore_path ${DSTORE_PATH} \
    --val_path ${VAL_PATH} \
    --index_path ${INDEX_PATH} \
    --output_path ${OUTPUT_PATH} \
    --k ${K} \
    --knn_temp ${KNN_TEMP} \
    --probe ${PROBE} \
    --batch_size ${BATCH_SIZE_KNN} \
    --ignore_first True \
    --knn_gpu

if [ $? -ne 0 ]; then echo "Error: KNN saving failed!"; exit 1; fi
echo "[Step 3/3] Done"

echo "=========================================="
echo "Pipeline completed!"
echo "  Datastore: ${DSTORE_PATH}"
echo "  Index: ${INDEX_PATH}"
echo "  KNN results: ${OUTPUT_PATH}"
echo "=========================================="
