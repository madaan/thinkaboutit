# Dataset

## Getting the dataset

- Download the data directory from [here](https://drive.google.com/drive/folders/1iexS3RrtSl3T2B2fGDCz9m0nVotula8x?usp=sharing) and unzip in the the root folder (or run `gdown --id 1jz706EyCT__RHjSeUIQKQBI-XC4fswTt`).

- The structure of `data` after unzipping will be as follows:

```js
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

## Defeasible QA sample point

- Defeasible inference is a mode of reasoning given a premise, a hypothesis might be strengthened or weakened in the light of new evidence. A large dataset of such defeasible queries was introduced in this [work](https://aclanthology.org/2020.findings-emnlp.418.pdf).

- We supply their dataset for each domain (`snli`, `atomic`, `social`) in jsonl files.

- Each line in qa-{train/test/dev}.jsonl has the following format:

```js
{
    "question": n/a,
    "answer": "attenuator",
    "eg_type": "snli",
    "id": "train-snli-52235",
    "graph": n/a,
    "hypo": "people are getting ready to go swimming",
    "premise": "people are setting up chairs on a beach",
    "update": "the people are holding volleyballs",
    "metadata": {
        "graph_id": "train-snli-52235"
    }
}
```

Here the `hypo`, `premise`, `update`, and `answer` fields form the defeasible query. The `eg_type` field indicates the domain of the example.
`graph_id` is the id of the graph in the corresponding `influence_graphs_cleaned.jsonl` or `influence_graphs_noisy.jsonl` file.

The graph corresponding to the above query is:


```js
{
    "Y_affects_outcome": "more",
    "Z": [
        "the people are getting ready to play volleyball [OR] the volleyball court is in good condition"
    ],
    "V": [
        "the people are not allowed to play volleyball [OR] the volleyball court is under repair"
    ],
    "X": [
        "the people are holding volleyballs"
    ],
    "U": [
        "the people are not setting up chairs [OR] the people are not holding volleyballs"
    ],
    "W": [
        "there is no volleyball net [OR] the volleyballs are broken"
    ],
    "Y": [
        "the people are setting up a net"
    ],
    "para_outcome_accelerate": [
        "MORE people are getting ready to go swimming?"
    ],
    "para_outcome_decelerate": [
        "LESS people are getting ready to go swimming"
    ],
    "graph_id": "train-snli-52235",
    "para_id": "train-snli-52235",
    "paragraph": "train-snli-52235",
    "prompt": "train-snli-52235"
}
```
