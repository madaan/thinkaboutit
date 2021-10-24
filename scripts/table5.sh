#!/bin/bash
# runs all the inferences to generate results in table 5

GPU=0
mkdir -p results/table5/

# String based models
mkdir -p results/table5/string/
bash scripts/infer.sh ${GPU} str pre_trained_models/corrected_graphs/str/atomic/epoch\=13-step\=14768-val_acc_epoch\=0.7971.ckpt results/table5/string/atomic.csv influence_graphs_cleaned.jsonl
bash scripts/infer.sh ${GPU} str pre_trained_models/corrected_graphs/str/snli/epoch\=14-step\=40193-val_acc_epoch\=0.8471.ckpt results/table5/string/snli.csv influence_graphs_cleaned.jsonl
bash scripts/infer.sh ${GPU} str pre_trained_models/corrected_graphs/str/social/epoch\=4-step\=10830-val_acc_epoch\=0.8635.ckpt results/table5/string/social.csv influence_graphs_cleaned.jsonl

# MOE based models
mkdir -p results/table5/moe/
bash scripts/infer.sh ${GPU} moe pre_trained_models/corrected_graphs/moe/atomic/epoch\=28-step\=31178-val_acc_epoch\=0.7995.ckpt results/table5/moe/atomic.csv influence_graphs_cleaned.jsonl
bash scripts/infer.sh ${GPU} moe pre_trained_models/corrected_graphs/moe/snli/epoch\=27-step\=76229-val_acc_epoch\=0.8448.ckpt results/table5/moe/snli.csv influence_graphs_cleaned.jsonlbash scripts/infer.sh 0 moe pre_trained_models/corrected_graphs/moe/social/epoch\=29-step\=72209-val_acc_epoch\=0.8824.ckpt results/table5/moe/social.csv influence_graphs_cleaned.jsonl
bash scripts/infer.sh ${GPU} moe pre_trained_models/corrected_graphs/moe/social/epoch\=29-step\=72209-val_acc_epoch\=0.8824.ckpt results/table5/moe/social.csv influence_graphs_cleaned.jsonl

# GCN based models
mkdir -p results/table5/gcn/
bash scripts/infer.sh ${GPU} gcn pre_trained_models/corrected_graphs/gcn/atomic/epoch=28-step=31178-val_acc_epoch=0.7857.ckpt results/table5/moe/atomic.csv influence_graphs_cleaned.jsonl
bash scripts/infer.sh ${GPU} gcn pre_trained_models/corrected_graphs/gcn/social/epoch=28-step=68599-val_acc_epoch=0.8795.ckpt results/table5/moe/social.csv influence_graphs_cleaned.jsonl
bash scripts/infer.sh ${GPU} gcn pre_trained_models/corrected_graphs//gcn/snli/epoch=24-step=69298-val_acc_epoch=0.8398.ckpt results/table5/moe/snli.csv influence_graphs_cleaned.jsonl

# GCN-MOE based models
mkdir -p results/table5/gcn_moe
bash scripts/infer.sh ${GPU} gcn_moe pre_trained_models/corrected_graphs/gcn_moe/atomic/epoch=27-step=30631-val_acc_epoch=0.7914.ckpt results/table5/gcn_moe/atomic.csv influence_graphs_cleaned.jsonl
bash scripts/infer.sh ${GPU} gcn_moe pre_trained_models/corrected_graphs/gcn_moe/social/epoch=28-step=69802-val_acc_epoch=0.8771.ckpt results/table5/gcn_moe/social.csv influence_graphs_cleaned.jsonl
bash scripts/infer.sh ${GPU} gcn_moe pre_trained_models/corrected_graphs//gcn_moe/snli/epoch=21-step=59597-val_acc_epoch=0.8364.ckpt results/table5/gcn_moe/snli.csv influence_graphs_cleaned.jsonl
