## GraphAugmentedQA

The goal of this repo is to explore methods that can leverage auxiliary graphs for generation and QA. We want to develop techniques that not only help with the performance, but also help in obtaining useful explanations from the graphs.


### TL; DR

From the `gcn` branch, run `bash src/gnn_qa/ut_medium.sh`.


### Running unit test

From the root directory of this project, run:
```sh
bash src/gnn_qa/ut_medium.sh 
```
- The unit data has 100 examples in each of the splits, and the same data is used for all three splits. This means that the accuracy should be 100% in ~15 epochs.

- The lightning logs are stored in `lightning_logs/version_x`. 
The logs can be checked using:
```sh
tensorboard --logdir lightning_logs/version_2 --port 9803 --bind_all
```
Depending on the machine this is executed from, this would expose the tensorboard on port 9803 (e.g., sa.lti.cs.cmu.edu:9803, if the port is open).

- For inference, use:
```sh
python src/gnn_qa/infer.py [PATH_TO_CHECKPOINT]
```

E.g.,:
```sh
lightning_logs/rgcn_pos_enc_after_gcn_att/lightning_logs/version_0/checkpoints/epoch\=21-step\=13350-val_acc_epoch\=0.73.ckpt
```

### Code overview

1. `run.py` is the run script.

### TODO:
- [x] R-GCN (note: no obvious improvements in the performance)
- [x] Positional embeddings
- [ ] LR scheduler
- [ ] Concat entire graph as a string OR using custom attention.

### Ideas: 
- [ ] Localized graph attention: pay attention to only certain parts of the network as you generate.
- [ ] Graph positional embeddings

### Ablations:
- [ ] positional embeddings before or after GCN? Maybe it doesn't matter, but perhaps after GCN is better as GCN cannot possibly utilize positional information.


