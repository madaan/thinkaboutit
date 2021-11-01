#!/bin/bash
GPU="${1}"
MODEL_TYPE=${2}
CKPT="${3}"
OUTPUT_LOC="${4}"
GRAPH_FILE_NAME="${5}"
export CUDA_VISIBLE_DEVICES=${GPU} && python src/model/${MODEL_TYPE}/infer.py --ckpt ${CKPT} \
    --paths_output_loc ${OUTPUT_LOC}\
     --graphs_file_name ${GRAPH_FILE_NAME}
