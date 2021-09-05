#!/bin/bash
python src/gnn_qa/run.py --dataset_basedir data/qa/unit_test/ --lr 1e-3 --gpus 1 --max_epochs 20 --accelerator ddp --sentence_encoder bert-base-uncased
