import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import random
import numpy as np
import pytorch_lightning as pl
import logging
from argparse import ArgumentParser
import resource
from pytorch_lightning.plugins import DDPPlugin

logging.basicConfig(level=logging.INFO)
MODEL = "str"


class GraphQaTrainer(object):
    def __init__(self) -> None:
        super().__init__()
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

        self.init_args()
        self.seed_everything()
        self.init_dm()
        self.init_model()
        self.init_trainer()

    def init_args(self):
        parser = ArgumentParser()
        parser.add_argument("--num_gpus", type=int)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--seed", help="the most important input", type=int)
        parser.add_argument("--clip_grad", type=float, default=1.0)
        parser.add_argument("--dataset_basedir", help="Base directory where the dataset is located.", type=str)
        parser.add_argument("--graphs_file_name", help="File that contains the generated influence graphs.", type=str)
        parser.add_argument("--format", help="Training data format", type=str)
        parser.add_argument(
            "--merge_nodes_op", help="Operation to merge graphs with [join, random, first, last]", type=str
        )
        parser.add_argument("--use_graphs", action="store_true", default=True)  # TODO remove this option
        parser.add_argument("--model_type", help="Name of the model", choices=["str", "gcn", "moe", "gcn_moe"])
        parser.add_argument("--node_op", help="Operation to merge nodes with [all, first, last]", type=str)
        args, _ = parser.parse_known_args()
        print(args.model_type)
        self.data_module, self.qa_model = self.get_model_data_class(args.model_type)
        parser = pl.Trainer.add_argparse_args(parser)
        parser = self.qa_model.add_model_specific_args(parser)
        self.args = parser.parse_args()
        self.args.num_gpus = len(str(self.args.gpus).split(","))
        self.args.basedir = self.args.dataset_basedir
        self.args.tokenizer_name = self.args.model_name

    def seed_everything(self):
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        pl.utilities.seed.seed_everything(self.args.seed)
        pytorch_lightning.seed_everything(self.args.seed)

    def get_train_steps(self, dm):
        total_devices = self.args.num_gpus * self.args.num_nodes
        train_batches = len(dm.train_dataloader()) // total_devices
        return (self.args.max_epochs * train_batches) // self.args.accumulate_grad_batches

    def get_model_data_class(self, model_type: str):
        if model_type == "moe":
            from src.gnn_qa.model.moe.qa_model_moe import GraphQaModel
            from src.gnn_qa.model.moe.data import GraphQaDataModule
        elif model_type == "gcn":
            from src.gnn_qa.model.gcn.qa_model_gcn import GraphQaModel
            from src.gnn_qa.model.gcn.data import GraphQaDataModule
        elif model_type == "gcn_moe":
            from src.gnn_qa.model.gcn_moe.qa_model_gcn_moe import GraphQaModel
            from src.gnn_qa.model.gcn_moe.data import GraphQaDataModule
        elif model_type == "str":
            from src.gnn_qa.model.str.qa_model_str import GraphQaModel
            from src.gnn_qa.model.str.data import GraphQaDataModule
        return GraphQaDataModule, GraphQaModel

    def init_dm(self):
        # Step 1: Init Data
        logging.info("Loading the data module")
        self.dm = self.data_module(
            # basedir=self.args.dataset_basedir,
            # graphs_file_name=self.args.graphs_file_name,
            # tokenizer_name=self.args.model_name,
            # batch_size=self.args.batch_size,
            **vars(self.args)
        )
        # Add information about training steps using the dataloader
        self.qa_model.set_train_steps(hparams=self.args, train_dataloader=self.dm.train_dataloader())

    def init_model(self):
        # Step 2: Init Model
        logging.info("Initializing the model")
        self.model = self.qa_model(hparams=self.args)
        self.lr_monitor = LearningRateMonitor(logging_interval="step")
        # Step 3: Start
        logging.info("Starting the training")
        self.checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{step}-{val_acc_epoch:.4f}",
            save_top_k=5,
            verbose=True,
            monitor="val_acc_epoch",
            mode="max",
        )

    def init_trainer(self):
        self.trainer = pl.Trainer.from_argparse_args(
            self.args,
            callbacks=[self.checkpoint_callback, self.lr_monitor],
            val_check_interval=0.5,
            gradient_clip_val=self.args.clip_grad,
            track_grad_norm=2,
            accumulate_grad_batches=self.args.accumulate_grad_batches,
            plugins=[DDPPlugin(find_unused_parameters=True)],
            gpus=1,
        )

    def fit(self):
        self.trainer.fit(self.model, self.dm)


trainer = GraphQaTrainer()
trainer.fit()
