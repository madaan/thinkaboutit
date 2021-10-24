#!/bin/bash
set -u
export PYTHONPATH=".:../:.:src:../:../../"
export TOKENIZERS_PARALLELISM=false
DATA_DIR="data/unit_test_medium/"
GRAPH_NAME="influence_graphs_unit_test.jsonl"
MODEL_TYPE="$1"

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

which python
python src/gnn_qa/run.py "${ARGS[@]}"
