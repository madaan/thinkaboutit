import argparse
import torch
from typing import List, Any
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import AdamW
import logging
from transformers import AutoModel
import torch.nn as nn
from argparse import ArgumentParser
from transformers import get_linear_schedule_with_warmup

from src.gnn_qa.model.moe.data import InfluenceGraphNNData
from src.gnn_qa.utils import CustomRobertaClassificationHead
from src.gnn_qa.moe import GatingNetwork, ExpertModel
from src.gnn_qa.model.moe.graph_experts import GraphExpert


class GraphQaModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams) if isinstance(hparams, argparse.Namespace) else hparams)
        self.save_hyperparameters(hparams)
        self.roberta = AutoModel.from_pretrained(self.hparams.model_name)
        self.roberta.config.num_labels = self.hparams.n_class
        # self.attention = MultiheadedAttention(h_dim=self.hparams.h_dim, kqv_dim=self.hparams.kqv_dim, n_heads=self.hparams.n_heads)
        # self.multihead_attn = nn.MultiheadAttention(
        #     embed_dim=self.hparams.h_dim, num_heads=self.hparams.n_heads)
        # self.classifier = nn.Linear(config.hidden_size * 2, self.hparams.n_class)
        # self.mapping = nn.Linear(config.hidden_size * 2, config.hidden_size)
        if self.hparams.embedding_op == "concat":
            # self.moe = MixtureOfExperts(num_experts=1, num_layers=1, input_size=768 * 2,
            # hidden_size=768, output_size=768, gating_input_detach=False)
            self.question_expert = ExpertModel(
                num_layers=1,
                input_size=self.roberta.config.hidden_size,
                output_size=self.roberta.config.hidden_size,
                hidden_size=self.roberta.config.hidden_size,
            )
            self.graph_expert = GraphExpert(
                num_layers=1,
                input_size=self.roberta.config.hidden_size,
                output_size=self.roberta.config.hidden_size,
                hidden_size=self.roberta.config.hidden_size,
            )
            self.graph_expert_2 = ExpertModel(
                num_layers=1,
                input_size=self.roberta.config.hidden_size,
                output_size=self.roberta.config.hidden_size,
                hidden_size=self.roberta.config.hidden_size,
            )
            self.gating = GatingNetwork(self.roberta.config.hidden_size * 2, 2)

            self.classifier = CustomRobertaClassificationHead(config=self.roberta.config)
        elif self.hparams.embedding_op == "add" or self.hparams.embedding_op == "only_phu":
            self.classifier = CustomRobertaClassificationHead(config=self.roberta.config)
        elif self.hparams.embedding_op == "both":
            self.ques_classifier = CustomRobertaClassificationHead(config=self.roberta.config)
            self.graph_classifier = CustomRobertaClassificationHead(config=self.roberta.config)
            # self.classifier = nn.Sequential(
        #     nn.Linear(self.hparams.h_dim * 2, self.hparams.h_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hparams.h_dim, self.hparams.n_class))
        self.loss = nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=-1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--min_lr", default=0, type=float, help="Minimum learning rate.")
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
        parser.add_argument("--embedding_op", type=str, help="The operation to be used to concatenate the embeddings")
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
        question_tokens, question_type_ids, question_masks, graphs, graph_masks, labels = batch

        # step 1: encode the question/paragraph
        question_cls_embeddeding = self.forward_roberta(
            input_ids=question_tokens, token_type_ids=question_type_ids, attention_mask=question_masks
        )

        # step 2: encode the nodes
        nodes_cls_embedding = self.encode_nodes(graphs, graph_masks)  # (B, num_nodes, h_dim)
        # nodes_gcn_embedding = self.positional_encoding(nodes_gcn_embedding)

        # step 3: get attention vector, graph representation
        # attention_vector, attention_weighted_gcn = self.attention(KV=nodes_gcn_embedding, Q=question_cls_embeddeding)
        # attention_output, attention_weights = self.multihead_attn(key=nodes_cls_embedding, value=nodes_cls_embedding,
        #                                                           query=question_cls_embeddeding.unsqueeze(0))
        # attention_output = attention_output.squeeze(0)
        # step 4: classify
        if self.hparams.embedding_op == "concat":
            bsz, hsz = question_cls_embeddeding.shape
            # Pool the graphs using an MOE
            graph_expert_output, graph_expert_probs = self.graph_expert(nodes_cls_embedding)

            #  the gating is determined by the original input
            input = torch.cat([question_cls_embeddeding, graph_expert_output], dim=-1)  # B x 1 x 1536
            # input = torch.cat([question_cls_embeddeding, attention_output], dim=-1).view(bsz, hsz, 2).unsqueeze(1)  # B x 1 x 768 x 2
            experts_logits = self.gating(input)
            expert_probs = self.softmax(experts_logits).unsqueeze(1)  # B x 1 x 2

            # Run the second level of experts
            question_expert_output = self.question_expert(question_cls_embeddeding)
            graph_expert_output_l2 = self.graph_expert_2(graph_expert_output)
            output_level_2 = torch.cat([question_expert_output, graph_expert_output_l2], dim=-1)  # B x 1 x 1536

            # Combine them using the gates
            combined_output = expert_probs.mul(output_level_2.view(bsz, hsz, 2))
            combined_output = combined_output.sum(dim=-1).squeeze(1)

            logits = self.classifier(combined_output)
        elif self.hparams.embedding_op == "only_phu":
            logits = self.classifier(question_cls_embeddeding)

        predicted_labels = torch.argmax(logits, -1)
        acc = torch.true_divide((predicted_labels == labels).sum(), labels.shape[0])
        return graph_expert_probs.detach(), logits, acc, expert_probs

    def setup(self, stage):
        if stage == "fit":
            total_devices = self.hparams.num_gpus * self.hparams.num_nodes
            train_batches = len(self.train_dataloader()) // total_devices
            self.train_steps = (self.hparams.max_epochs * train_batches) // self.hparams.accumulate_grad_batches

    def forward_roberta(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None
    ):
        """Returns the pooled token from BERT"""
        last_hidden_state = self.roberta(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        ).last_hidden_state
        return last_hidden_state[:, 0, :]

    def encode_nodes(self, graphs, graph_masks):
        """Encodes all the nodes in the given graphs"""
        batch_sz = len(graphs) // InfluenceGraphNNData.num_nodes_per_graph
        cls_embedding = self.forward_roberta(input_ids=graphs, attention_mask=graph_masks)
        cls_embedding = cls_embedding.view(
            batch_sz, InfluenceGraphNNData.num_nodes_per_graph, -1
        )  # (num_graph, num_nodes, 768)
        return cls_embedding

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        attention, logits, acc, expert_probs = self(batch)
        loss = self.loss(logits, batch[-1])
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "train_gates",
            expert_probs.squeeze(1)[:, 1].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": loss, "attention": attention}

    def validation_step(self, batch, batch_idx):
        # Load the data into variables
        attention, logits, acc, expert_probs = self(batch)

        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(logits, batch[-1])

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "val_gates",
            expert_probs.squeeze(1)[:, 1].mean(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
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
        # attention bz x num_nodes
        top2_attn = []
        with torch.no_grad():
            for output in outputs:
                x = output["attention"].squeeze(1)
                # logging.info(x[0], x.sum(axis=-1))
                tmp.append(x.argmax(axis=1))
                top2_attn.extend(x.topk(2).indices.tolist())
                total_entropy += torch.sum(-x * torch.log(x), axis=1).sum()
                num_elements += x.shape[0]
            self.logger.experiment.add_histogram(
                "val_nodes_attended_to", torch.cat(tmp, 0), self.current_epoch
            )  # histogram over the nodes attended to
            self.logger.experiment.add_histogram(
                "val_nodes_attended_to_top2", torch.tensor(top2_attn), self.current_epoch
            )  # histogram over the nodes attended to

            self.log(
                "val_attention_entropy", total_entropy / num_elements, on_epoch=True, prog_bar=True
            )  # hopefully goes down as attention becomes more peaky
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
