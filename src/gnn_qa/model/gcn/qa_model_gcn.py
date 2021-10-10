import argparse
from transformers import BertConfig
import torch
from torch_geometric.data import Data
from typing import List, Any
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel
import torch.nn as nn
from torch_geometric.data import Batch
from argparse import ArgumentParser
from transformers.models.bert.modeling_bert import BertPooler
from src.gnn_qa.utils import MultiheadedAttention, PositionalEncoding
from src.gnn_qa.model.gcn.ignn import InfluenceGraphGNN


class GraphQaModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams) if isinstance(hparams, argparse.Namespace) else hparams)
        self.save_hyperparameters(hparams)
        config = BertConfig()
        # self.model = BertForSequenceClassification.from_pretrained(self.hparams.model_name, num_labels=self.hparams.n_class)
        self.roberta = AutoModel.from_pretrained(self.hparams.model_name)
        self.pooler = BertPooler(config)
        self.gcn = InfluenceGraphGNN(
            num_in_features=self.hparams.h_dim,
            num_out_features=self.hparams.h_dim,
            num_relations=self.hparams.num_relations,
            rgcn=self.hparams.use_rgcn,
        )
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.h_dim)
        self.attention = MultiheadedAttention(
            h_dim=self.hparams.h_dim, kqv_dim=self.hparams.kqv_dim, n_heads=self.hparams.n_heads
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.hparams.n_class)
        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--min_lr", default=0, type=float, help="Minimum learning rate.")
        parser.add_argument("--use_rgcn", action="store_true")
        parser.add_argument("--h_dim", type=int, help="Size of the hidden dimension.", default=768)
        parser.add_argument("--n_heads", type=int, help="Number of attention heads.", default=3)
        parser.add_argument("--kqv_dim", type=int, help="Dimensionality of the each attention head.", default=256)
        parser.add_argument("--n_class", type=int, help="Number of classes.", default=3)
        parser.add_argument("--lr", default=5e-4, type=float, help="Initial learning rate.")
        parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay rate.")
        parser.add_argument("--warmup_prop", default=0.0, type=float, help="Warmup proportion.")
        parser.add_argument(
            "--num_relations", default=2, type=int, help="The number of relations (edge types) in your graph."
        )
        parser.add_argument("--model_name", default="bert-base-uncased", help="Model to use.")
        return parser

    @staticmethod
    def set_train_steps(hparams, train_dataloader):
        total_devices = hparams.num_gpus * hparams.num_nodes
        train_batches = len(train_dataloader) // total_devices
        train_steps = (hparams.max_epochs * train_batches) // hparams.accumulate_grad_batches
        hparams.train_steps = train_steps
        hparams.warmup_steps = int(hparams.train_steps * hparams.warmup_prop)

    def configure_optimizers(self):
        def _is_roberta(param_name: str):
            return "roberta" in param_name

        params = list(self.named_parameters())
        grouped_parameters = [
            {"params": [p for n, p in params if _is_roberta(n)], "lr": self.hparams.lr},
            {"params": [p for n, p in params if not _is_roberta(n)], "lr": self.hparams.lr * 10},
        ]
        optimizer = AdamW(grouped_parameters, lr=self.hparams.lr, eps=1e-8)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.train_steps
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def forward(self, batch):
        question_tokens, question_type_ids, question_masks, graphs, labels = batch

        # step 1: encode the question/paragraph
        question_cls_embeddeding = self.forward_bert(
            input_ids=question_tokens, token_type_ids=question_type_ids, attention_mask=question_masks
        )

        # step 2: encode the nodes
        nodes_gcn_embedding = self.forward_gcn(graphs, labels)  # (B, num_nodes, h_dim)

        # nodes_gcn_embedding = self.positional_encoding(nodes_gcn_embedding)

        # step 3: get attention vector, graph representation
        attention_vector, attention_weighted_gcn = self.attention(KV=nodes_gcn_embedding, Q=question_cls_embeddeding)

        # step 4: classify
        logits = self.classifier(torch.cat([question_cls_embeddeding, attention_weighted_gcn], dim=-1))

        predicted_labels = torch.argmax(logits, -1)
        acc = torch.true_divide((predicted_labels == labels).sum(), labels.shape[0])
        return attention_vector, logits, acc

    def forward_bert(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None):
        """Returns the pooled token from BERT"""
        outputs = self.roberta(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        hidden_states = outputs["hidden_states"]
        cls_embeddeding = self.dropout(self.pooler(hidden_states[-1]))
        return cls_embeddeding

    def forward_gcn(self, graphs: List[Data], labels):
        """Runs GCN on a given list of graphs. This function is responsible for two
        steps:
            i) Encodes all the nodes in all the graphs using the BERT encoder.
            ii) Runs a GCN
        graph ([type]): [description]
        """
        bsz = len(graphs)
        graphs = self.encode_nodes(graphs, labels)
        graphs_batched = Batch.from_data_list(graphs)  #  pytorch geometric batch
        x = self.gcn(graphs_batched)  #  (bsz x num_nodes, 768)
        x = x.view(bsz, -1, self.hparams.h_dim)  #  (bsz, num_nodes, 768)
        return x

    def encode_nodes(self, graphs: List[Data], labels):
        """Encodes all the nodes in the given graphs"""

        #  collate the input tensors
        # label_to_tokens = {0:  [101, 2625, 102], 1: [101, 2062, 102], 2: [101, 3904, 102]}
        batch_sz = len(graphs)
        nodes_per_graph = graphs[0].num_nodes
        total_nodes = sum([graph.num_nodes for graph in graphs])
        max_token_len = max([len(token) for graph in graphs for token in graph.tokens])
        tokens = torch.zeros(total_nodes, max_token_len).long()
        tokens_mask = torch.zeros(total_nodes, max_token_len).long()
        num_nodes_processed = 0
        for i, graph in enumerate(graphs):
            for j, node_tokens in enumerate(graph.tokens):
                # if j == 0: # HACK that we use for debugging attention
                #     node_tokens = label_to_tokens[labels[i].item()] + node_tokens
                length = len(node_tokens)
                tokens[num_nodes_processed + j, :length] = torch.LongTensor(node_tokens)
                tokens_mask[num_nodes_processed + j, :length] = 1
            num_nodes_processed += graph.num_nodes
        #  get CLS token for the collated nodes
        cls_embeddeding = self.forward_bert(
            input_ids=tokens.to(self.device), attention_mask=tokens_mask.to(self.device)
        )  #  (num_graph x num_nodes, 768)
        cls_embeddeding = cls_embeddeding.view(batch_sz, nodes_per_graph, -1)  #  (num_graph, num_nodes, 768)
        # return cls_embeddeding
        # cls_embeddeding = self.positional_encoding(cls_embeddeding)

        # assign the graph.x to the correct graphs
        for i, graph in enumerate(graphs):
            # graph.x = cls_embeddeding[num_nodes_processed:num_nodes_processed + graph.num_nodes]
            graph.x = cls_embeddeding[i]
            graph.x.to(self.device)
            del graph.tokens  # we don't need the tokens anymore

        return graphs

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        attention, logits, acc = self(batch)
        loss = self.loss(logits, batch[-1])
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "attention": attention}

    def validation_step(self, batch, batch_idx):
        # Load the data into variables
        attention, logits, acc = self(batch)

        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(logits, batch[-1])

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "attention": attention}

    def test_step(self, batch, batch_idx):
        # Load the data into variables
        attention, logits, acc = self(batch)

        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(logits, batch[-1])
        return {"loss": loss, "attention": attention}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        total_entropy, num_elements = 0.0, 0
        tmp = []
        with torch.no_grad():
            for output in outputs:
                x = output["attention"].squeeze(1)
                # logging.info(x[0], x.sum(axis=-1))
                tmp.append(x.argmax(axis=1))
                total_entropy += torch.sum(-x * torch.log(x), axis=1).sum()
                num_elements += x.shape[0]
            self.logger.experiment.add_histogram(
                "val_nodes_attended_to", torch.cat(tmp, 0), self.current_epoch
            )  #  histogram over the nodes attended to
            self.log(
                "val_attention_entropy", total_entropy / num_elements, on_epoch=True, prog_bar=True
            )  #  hopefully goes down as attention becomes more peaky
        return super().validation_epoch_end(outputs)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        tqdm_dict.pop("val_loss_step", None)
        tqdm_dict.pop("val_acc_step", None)
        return tqdm_dict


if __name__ == "__main__":
    sentences = [
        "This framework generates embeddings for each input sentence",
        "Sentences are passed as a list of string.",
        "The quick brown fox jumps over the lazy dog.",
    ]
