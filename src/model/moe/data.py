"""Wrapper for a conditional generation dataset present in 2 tab-separated columns:
source[TAB]target
"""
import logging
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
from src.model.moe.influence_graph import InfluenceGraph

label_dict = {"less": 0, "attenuator": 0,
              "more": 1, "intensifier": 1, "no_effect": 2}
rev_label_dict = defaultdict(list)

for k, v in label_dict.items():
    rev_label_dict[v].append(k)

rev_label_dict = {k: "/".join(v) for k, v in rev_label_dict.items()}


class GraphQaDataModule(pl.LightningDataModule):
    def __init__(self, basedir: str, tokenizer_name: str, batch_size: int,
                 graphs_file_name: str, num_workers: int = 16, **kwargs):
        super().__init__()
        self.basedir = basedir
        self.graphs_file_name = graphs_file_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=True)

    def train_dataloader(self):
        dataset = GraphQADataset(tokenizer=self.tokenizer,
                                 qa_pth=f"{self.basedir}/qa-train.jsonl", graph_pth=f"{self.basedir}/{self.graphs_file_name}")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, collate_fn=GraphQADataset.collate_pad)

    def val_dataloader(self):
        dataset = GraphQADataset(tokenizer=self.tokenizer,
                                 qa_pth=f"{self.basedir}/qa-dev.jsonl", graph_pth=f"{self.basedir}/{self.graphs_file_name}")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=GraphQADataset.collate_pad)

    def test_dataloader(self):
        dataset = GraphQADataset(tokenizer=self.tokenizer,
                                 qa_pth=f"{self.basedir}/qa-test.jsonl", graph_pth=f"{self.basedir}/{self.graphs_file_name}")
        return DataLoader(dataset=dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=GraphQADataset.collate_pad)


class GraphQADataset(Dataset):
    def __init__(self, tokenizer, qa_pth: str, graph_pth: str) -> None:
        super().__init__()
        self.qa_pth = qa_pth
        self.graph_pth = graph_pth
        self.tokenizer = tokenizer
        self.read_graphs()
        self.read_qa()

    def read_graphs(self):
        influence_graphs = pd.read_json(
            self.graph_pth, orient='records', lines=True).to_dict(orient='records')
        self.graphs = {}
        for graph_dict in tqdm(influence_graphs, desc="Reading graphs", total=len(influence_graphs)):
            self.graphs[str(graph_dict["graph_id"])] = InfluenceGraphNNData.make_data_from_dict(
                graph_dict, self.tokenizer)

    def read_qa(self):
        logging.info("Reading data from {}".format(self.qa_pth))
        data = pd.read_json(self.qa_pth, orient="records", lines=True)
        self.questions, self.answer_labels, self.graph_ids, self.paragraphs = [], [], [], []
        logging.info(f"Reading QA file from {self.qa_pth}")
        self.PHUs = []
        for i, row in tqdm(data.iterrows(), total=len(data), desc="Reading QA examples"):
            self.answer_labels.append(row["question"]["answer_label"].strip())
            para = " ".join([p.strip() for p in row["question"]
                             ["para_steps"] if len(p) > 0])
            question = row["question"]["stem"].strip()
            self.questions.append(question)
            self.paragraphs.append(para)
            self.graph_ids.append(row["metadata"]["graph_id"])
            if 'premise' in row:
                self.PHUs.append(f"{row['premise']}</s></s>{row['hypo']}</s></s>{row['update']}")
            else:
                self.PHUs.append(para + "</s></s>" + question)

        # encoded_input = self.tokenizer(self.paragraphs, self.questions)
        encoded_input = self.tokenizer(self.PHUs)
        self.input_ids = encoded_input["input_ids"]
        if "token_type_ids" in encoded_input:
            self.token_type_ids = encoded_input["token_type_ids"]
        else:
            # only BERT uses it anyways, so just set it to 0
            self.token_type_ids = [[0] * len(s)
                                   for s in encoded_input["input_ids"]]

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, i):
        # We???ll pad at the batch level.
        return (self.input_ids[i], self.token_type_ids[i], self.graphs[self.graph_ids[i]], self.answer_labels[i])

    @staticmethod
    def collate_pad(batch):
        max_ques_token_len = 0
        max_node_token_len = 0
        num_elems = len(batch)
        for i in range(num_elems):
            tokens, type_ids, graph, label = batch[i]
            max_ques_token_len = max(max_ques_token_len, len(tokens))

            for j in range(InfluenceGraphNNData.num_nodes_per_graph):
                max_node_token_len = max(max_node_token_len, len(graph[j]))

        tokens = torch.zeros(num_elems, max_ques_token_len).long()
        tokens_mask = torch.zeros(num_elems, max_ques_token_len).long()
        token_type_ids = torch.zeros(num_elems, max_ques_token_len).long()
        labels = torch.zeros(num_elems).long()
        graphs = torch.zeros(num_elems * InfluenceGraphNNData.num_nodes_per_graph, max_node_token_len).long()
        graph_masks = torch.zeros(num_elems * InfluenceGraphNNData.num_nodes_per_graph, max_node_token_len).long()
        for i in range(num_elems):
            toks, type_ids, graph, label = batch[i]
            length = len(toks)
            tokens[i, :length] = torch.LongTensor(toks)
            token_type_ids[i, :length] = torch.LongTensor(type_ids)
            tokens_mask[i, :length] = 1
            # graphs.append(graph)
            labels[i] = label_dict[label]
            for j in range(InfluenceGraphNNData.num_nodes_per_graph):
                graphs[i * InfluenceGraphNNData.num_nodes_per_graph + j, :len(graph[j])] = torch.LongTensor(graph[j])
                graph_masks[i * InfluenceGraphNNData.num_nodes_per_graph + j, :len(graph[j])] = 1
        return [tokens, token_type_ids, tokens_mask, graphs, graph_masks, labels]


class InfluenceGraphNNData:
    """
    V       Z
    |     /
    -   +
    | /
    X       U
    | \     |
    -   +   -
    |     \ |
    W       Y
    | \   / |
    -   +   -
    | /   \ |
    L       M
    """
    # node_index = {
    #     "V": 0, "Z": 1, "X": 2, "U": 3, "W": 4, "Y": 5, "dec": 6, "acc": 7}
    node_index = [("V",  0), ("Z", 1), ("U", 2), ("W", 3), ("Y", 4)]
    num_nodes_per_graph = len(node_index)
    node_to_desc = {
        "V": "external negative = ",
        "Z": "external positive = ",
        "U": "external to Y = ",
        "Y": "mediator positive = ",
        "W": "mediator negative = ",
    }
    # index_node = {v: k for k, v in node_index.items()}
    edge_index = [[0, 1, 2, 2, 3, 4, 4, 5, 5],
                  [2, 2, 4, 5, 5, 6, 7, 6, 7]]
    EDGE_TYPE_HELPS, EDGE_TYPE_HURTS = 0, 1
    
    max_length = 35
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    @staticmethod
    def make_data_from_dict(graph_dict: dict, tokenizer):
        igraph = InfluenceGraph(graph_dict)
        if igraph.graph["Y_affects_outcome"] == "more":
            # the final edges depend on para outcome
            edge_features = [InfluenceGraphNNData.EDGE_TYPE_HURTS, InfluenceGraphNNData.EDGE_TYPE_HELPS, InfluenceGraphNNData.EDGE_TYPE_HURTS, InfluenceGraphNNData.EDGE_TYPE_HELPS,
                             InfluenceGraphNNData.EDGE_TYPE_HURTS, InfluenceGraphNNData.EDGE_TYPE_HELPS, InfluenceGraphNNData.EDGE_TYPE_HURTS, InfluenceGraphNNData.EDGE_TYPE_HURTS, InfluenceGraphNNData.EDGE_TYPE_HELPS]
        else:
            edge_features = [InfluenceGraphNNData.EDGE_TYPE_HURTS, InfluenceGraphNNData.EDGE_TYPE_HELPS, InfluenceGraphNNData.EDGE_TYPE_HURTS, InfluenceGraphNNData.EDGE_TYPE_HELPS,
                             InfluenceGraphNNData.EDGE_TYPE_HURTS, InfluenceGraphNNData.EDGE_TYPE_HURTS, InfluenceGraphNNData.EDGE_TYPE_HELPS, InfluenceGraphNNData.EDGE_TYPE_HELPS, InfluenceGraphNNData.EDGE_TYPE_HURTS]

        node_sentences = []
        for node, node_idx in InfluenceGraphNNData.node_index:
            if node in igraph.nodes_dict and len(igraph.nodes_dict[node]) > 0:
                # node_sentences.append(InfluenceGraphNNData.node_to_desc[node] + " [OR] ".join(igraph.nodes_dict[node]))
                node_sentences.append(" [OR] ".join(igraph.nodes_dict[node]))
            else:
                node_sentences.append(tokenizer.pad_token)
        encoding_dict = tokenizer(
            node_sentences, max_length=InfluenceGraphNNData.max_length, truncation=True)
        return encoding_dict["input_ids"]
        # return Data(graph_id=str(igraph.graph_id), num_nodes=len(InfluenceGraphNNData.node_index), tokens=encoding_dict["input_ids"],
        #             edge_index=torch.tensor(InfluenceGraphNNData.edge_index).long(), edge_attr=torch.tensor(edge_features).long())


if __name__ == "__main__":
    import sys
    dm = GraphQaDataModule(
        basedir=sys.argv[1], model_name=sys.argv[2], batch_size=32)
    for (tokens, tokens_mask, graphs, labels) in dm.train_dataloader():
        print(torch.tensor(graphs[0].tokens).shape)
