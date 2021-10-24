# Code and data for *Think about it! Using inference graphs for defeasible reasoning!*


## Setting up:
- Download the pre-trained models from [here](https://drive.google.com/file/d/1QKSnMLpt0TfM-Jxu-eI-c92qHSjcIAov/view?usp=sharing) (or if you use gdown, you can run `gdown --id 1QKSnMLpt0TfM-Jxu-eI-c92qHSjcIAov` to directly download the models zip (23GB)).
-   Download the data directory from [here](https://drive.google.com/drive/folders/1iexS3RrtSl3T2B2fGDCz9m0nVotula8x?usp=sharing) and unzip in the the root folder (or run `gdown --id 1iexS3RrtSl3T2B2fGDCz9m0nVotula8x`).

## Use cases

1. **Inference**
 - Use `scripts/table5.sh` to run inference for all models/dataset. This will recreate the numbers presented in Table 5 in the paper.
- Each output file contains per-sample inferred and true labels, as well as the MOE gate values if applicable.

2. **Training**

- You can run training using the `scripts/train.sh` script.
  
Usage:

```sh
scripts/train.sh MODEL_TYPE DATA_DIR GRAPH_NAME GPU
```

    where:
    - MODEL_TYPE: one of `str`, `moe`, `gcn`, `gcn_moe`
    - DATA_DIR: path to the directory containing the dataset.
    - GRAPH_NAME: name of the graph to be used for training.
    - GPU: GPU to use. If not specified, will use the first available GPU.

- Sample unit data is located in `data/unit_test`. The following command runs a unit test:
```sh
bash scripts/train.sh moe data/unit_test/ influence_graphs.jsonl 0
```

## Data and pre-trained models listing

### Data




The structure of `data` after unzipping will be as follows:
```
data
├── defeasible_graph_augmented_qa
│   └── t5
│       ├── atomic
│       │   ├── influence_graphs_cleaned.jsonl
│       │   ├── influence_graphs_noisy.jsonl
│       │   ├── qa-dev.jsonl
│       │   ├── qa-test.jsonl
│       │   └── qa-train.jsonl
│       ├── snli
│       │   ├── influence_graphs_cleaned.jsonl
│       │   ├── influence_graphs_noisy.jsonl
│       │   ├── qa-dev.jsonl
│       │   ├── qa-test.jsonl
│       │   └── qa-train.jsonl
│       └── social
│           ├── influence_graphs_cleaned.jsonl
│           ├── influence_graphs_noisy.jsonl
│           ├── qa-dev.jsonl
│           ├── qa-test.jsonl
│           └── qa-train.jsonl
└── unit_test
    ├── influence_graphs.jsonl
    ├── qa-dev.jsonl
    ├── qa-test.jsonl
    └── qa-train.jsonl
```

Here, each domain (`snli`, `atomic`, `social`) has its own folder, which contains the  `qa` files and the cleaned and noisy influence graphs (`influence_graphs_cleaned.jsonl` and `influence_graphs_noisy.jsonl`, respectively).
The details on graph generation are described [here](https://aclanthology.org/2021.findings-acl.456.pdf).


### Pre-trained models

- The pre-trained models for the three domains with clean and noisy graphs are located [here](https://drive.google.com/file/d/1QKSnMLpt0TfM-Jxu-eI-c92qHSjcIAov/view?usp=sharing)
```
pre_trained_models
├── corrected_graphs
│   ├── gcn
│   │   ├── atomic
│   │   │   └── epoch=28-step=31178-val_acc_epoch=0.7857.ckpt
│   │   ├── snli
│   │   │   └── epoch=24-step=69298-val_acc_epoch=0.8398.ckpt
│   │   └── social
│   │       └── epoch=28-step=68599-val_acc_epoch=0.8795.ckpt
│   ├── gcn_moe
│   │   ├── atomic
│   │   │   └── epoch=27-step=30631-val_acc_epoch=0.7914.ckpt
│   │   ├── snli
│   │   │   └── epoch=21-step=59597-val_acc_epoch=0.8364.ckpt
│   │   └── social
│   │       └── epoch=28-step=69802-val_acc_epoch=0.8771.ckpt
│   ├── moe
│   │   ├── atomic
│   │   │   └── epoch=28-step=31178-val_acc_epoch=0.7995.ckpt
│   │   ├── snli
│   │   │   └── epoch=27-step=76229-val_acc_epoch=0.8448.ckpt
│   │   └── social
│   │       └── epoch=29-step=72209-val_acc_epoch=0.8824.ckpt
│   └── str
│       ├── atomic
│       │   └── epoch=13-step=14768-val_acc_epoch=0.7971.ckpt
│       ├── snli
│       │   └── epoch=14-step=40193-val_acc_epoch=0.8471.ckpt
│       └── social
│           └── epoch=4-step=10830-val_acc_epoch=0.8635.ckpt
└── original_graphs
    ├── gcn
    │   ├── atomic
    │   │   └── epoch=19-step=21879-val_acc_epoch=0.7833.ckpt
    │   ├── snli
    │   │   └── epoch=23-step=65141-val_acc_epoch=0.8398.ckpt
    │   └── social
    │       └── epoch=25-step=61378-val_acc_epoch=0.8785.ckpt
    ├── moe
    │   ├── atomic
    │   │   └── epoch=4-step=5469-val_acc_epoch=0.7956.ckpt
    │   ├── snli
    │   │   └── epoch=18-step=52666-val_acc_epoch=0.8426.ckpt
    │   └── social
    │       └── epoch=28-step=69802-val_acc_epoch=0.8819.ckpt
    └── str
        ├── atomic
        │   └── epoch=13-step=14768-val_acc_epoch=0.7971.ckpt
        ├── snli
        │   └── epoch=7-step=22174-val_acc_epoch=0.8415.ckpt
        └── social
            └── epoch=5-step=13237-val_acc_epoch=0.8712.ckpt
```