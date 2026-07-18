import torch
import argparse

from clearml import Task
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger


def trainer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=300000, help='Number of training epochs')
    args, _ = parser.parse_known_args()
    return args


def train_with_clearml(task_name, model, dm):
    task = Task.init(task_name=task_name, project_name="Simple Nerf")
    model = train(model, dm)
    return model


def train(model, dm, use_early_stopping=True):
    torch.set_float32_matmul_precision('medium')

    logger = TensorBoardLogger("tb_logs", name="my_model")

    trainer = Trainer(
        max_epochs=300000,
        accelerator="gpu",
        devices=[0],
        strategy="ddp_find_unused_parameters_true",
        logger=logger
    )
        
    trainer.fit(model, dm)
    return model