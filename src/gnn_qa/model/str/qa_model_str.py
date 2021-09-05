import argparse
import torch
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
from argparse import ArgumentParser
from transformers import RobertaForSequenceClassification
from pytorch_lightning.core.lightning import LightningModule


class GraphQaModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams) if isinstance(hparams, argparse.Namespace) else hparams)
        self.save_hyperparameters(hparams)
        self.save_hyperparameters()
        self.model = RobertaForSequenceClassification.from_pretrained(self.hparams.model_name)
        self.loss = nn.CrossEntropyLoss()

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
        parser.add_argument("--model_name", default="bert-base-uncased", help="Model to use.")
        return parser

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, eps=1e-8)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def forward(self, batch):
        question_tokens, question_type_ids, question_masks, graphs, labels = batch

        logits = self.forward_roberta(input_ids=question_tokens, attention_mask=question_masks)

        predicted_labels = torch.argmax(logits, -1)
        acc = torch.true_divide((predicted_labels == labels).sum(), labels.shape[0])
        return torch.rand(len(graphs), 8), logits, acc

    def forward_roberta(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Returns the pooled token from RoBERTa."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        # Load the data into variables
        attention, logits, acc = self(batch)
        loss = self.loss(logits, batch[-1])
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

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

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        tqdm_dict.pop("val_loss_step", None)
        tqdm_dict.pop("val_acc_step", None)
        return tqdm_dict

    def setup(self, stage):
        if stage == "fit":
            total_devices = self.hparams.num_gpus * self.hparams.num_nodes
            train_batches = len(self.train_dataloader()) // total_devices
            self.train_steps = (self.hparams.max_epochs * train_batches) // self.hparams.accumulate_grad_batches
            self.total_steps = self.train_steps

    @staticmethod
    def set_train_steps(hparams, train_dataloader):
        total_devices = hparams.num_gpus * hparams.num_nodes
        train_batches = len(train_dataloader) // total_devices
        train_steps = (hparams.max_epochs * train_batches) // hparams.accumulate_grad_batches
        hparams.train_steps = train_steps
        hparams.warmup_steps = int(hparams.train_steps * hparams.warmup_prop)

if __name__ == "__main__":
    sentences = [
        "This framework generates embeddings for each input sentence",
        "Sentences are passed as a list of string.",
        "The quick brown fox jumps over the lazy dog.",
    ]
