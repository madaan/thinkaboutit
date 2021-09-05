#!/bin/bash
export PYTHONPATH=".:../:.:src:../:../../"
export TOKENIZERS_PARALLELISM=false
python src/gnn_qa/run.py --dataset_basedir data/unit_test_medium/ \
    --graphs_file_name influence_graphs_unit_test.jsonl \
    --lr 2e-5 --gpus 1 --max_epochs 2 --accelerator ddp \
    --merge_nodes_op join --format phyu
