#!/bin/bash
# runs all the inferences to generate results in table 2

mkdir -p results/table2/

# String based models
mkdir -p results/table2/string/
bash scripts/infer.sh 0 str pre_trained_models/corrected_graphs/str/atomic/epoch\=13-step\=14768-val_acc_epoch\=0.7971.ckpt results/table2/string/atomic.csv influence_graphs_corr_v3.jsonl
bash scripts/infer.sh 0 str pre_trained_models/corrected_graphs/str/snli/epoch\=14-step\=40193-val_acc_epoch\=0.8471.ckpt results/table2/string/snli.csv influence_graphs_corr_v3.jsonl
bash scripts/infer.sh 1 str pre_trained_models/corrected_graphs/str/social/epoch\=4-step\=10830-val_acc_epoch\=0.8635.ckpt results/table2/string/social.csv influence_graphs_corr_v3.jsonl

# MOE based models
mkdir -p results/table2/moe/
bash scripts/infer.sh 0 moe pre_trained_models/corrected_graphs/moe/atomic/epoch\=28-step\=31178-val_acc_epoch\=0.7995.ckpt results/table2/moe/atomic.csv influence_graphs_corr_v3.jsonl
bash scripts/infer.sh 0 moe pre_trained_models/corrected_graphs/moe/snli/epoch\=27-step\=76229-val_acc_epoch\=0.8448.ckpt results/table2/moe/snli.csv influence_graphs_corr_v3.jsonlbash scripts/infer.sh 0 moe pre_trained_models/corrected_graphs/moe/social/epoch\=29-step\=72209-val_acc_epoch\=0.8824.ckpt results/table2/moe/social.csv influence_graphs_corr_v3.jsonl
bash scripts/infer.sh 0 moe pre_trained_models/corrected_graphs/moe/social/epoch\=29-step\=72209-val_acc_epoch\=0.8824.ckpt results/table2/moe/social.csv influence_graphs_corr_v3.jsonl
