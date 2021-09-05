export SEED=43 && export DSET="atomic" && export CUDA_VISIBLE_DEVICES=2 && export OP="concat" &&
    nohup python -u src/gnn_qa/run.py --dataset_basedir data/defeasible_graph_augmented_qa//t5/"${DSET}"/ \
        --graphs_file_name influence_graphs_corr_v3.jsonl --n_heads 1 --accumulate_grad_batches 2 --max_epochs 30 \
        --lr 2e-5 --default_root_dir "lightning_logs/defeasible_graph_augmented_qa/${DSET}/t5/hier_moe/corr_v3_${SEED}" \
        --model_name roberta-base --batch_size 16 \
        --n_class 2 --warmup_prop 0.10 --seed ${SEED} --embedding_op "${OP}" >logs/"${DSET}"_corr_v3_hier_moe_${SEED}.log &
