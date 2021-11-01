#!/bin/bash
set -u
export PYTHONPATH=".:../:.:src:../:../../"
export TOKENIZERS_PARALLELISM=false

MODEL_TYPE="$1"
DATA_DIR="$2"
GRAPH_NAME="$3"
GPU="$4"

echo "Model type: $MODEL_TYPE"
echo "Data dir: $DATA_DIR"
echo "Graph name: $GRAPH_NAME"
echo "GPU: $GPU"

BASE_ARGS=(--dataset_basedir ${DATA_DIR}
    --graphs_file_name ${GRAPH_NAME}
    --lr 2e-5 --gpus 1 --max_epochs 2 --accelerator ddp
    --accumulate_grad_batches 2
    --model_type "${MODEL_TYPE}"
    --model_name roberta-base --batch_size 16 
    --n_class 2 --warmup_prop 0.10)

STR_ARGS=(--embedding_op "concat"
    --merge_nodes_op "join"
    --format phyu)


MOE_ARGS=(--embedding_op "concat" --n_heads 1)

GCN_MOE_ARGS=(--node_op all)

if [ "${MODEL_TYPE}" == "str" ]; then
    ARGS=("${BASE_ARGS[@]}" "${STR_ARGS[@]}")
elif [ "${MODEL_TYPE}" == "moe" ]; then
    ARGS=("${BASE_ARGS[@]}" "${MOE_ARGS[@]}")
elif [ "${MODEL_TYPE}" == "gcn_moe" ]; then
    ARGS=("${BASE_ARGS[@]}" "${MOE_ARGS[@]}" "${GCN_MOE_ARGS[@]}")
else
    ARGS=("${BASE_ARGS[@]}")
fi

export CUDA_VISIBLE_DEVICES="${GPU}" && python -u src/run.py "${ARGS[@]}"
