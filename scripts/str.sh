export CUDA_VISIBLE_DEVICES=4 && export OP="concat" && nohup python -u src/gnn_qa/run.py \
    --dataset_basedir data/defeasible_graph_augmented_qa//t5/social/ \
    --graphs_file_name influence_graphs_ggen2.jsonl \
    --accumulate_grad_batches 2 \
    --max_epochs 30 --lr 2e-5 \
    --default_root_dir "lightning_logs/defeasible_graph_augmented_qa/social/t5/interactive_curie/ggen2/" \
    --model_name roberta-base --batch_size 16 --n_class 2 --warmup_prop 0.10 \
    --seed 10725 --gpus 1 \
    --merge_nodes_op join --format phyu \
    --use_graphs
