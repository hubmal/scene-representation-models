import torch
import argparse

from clearml import Task
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def trainer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=50, help='Number of training epochs')
    args, _ = parser.parse_known_args()
    return args


def train_with_clearml(task_name, model, dm):
    task = Task.init(task_name=task_name, project_name="Simple Nerf")
    model = train(model, dm)
    return model


def train(model, dm, use_early_stopping=True):
    torch.set_float32_matmul_precision('medium')

    trainer = Trainer(
        max_epochs=100,
        accelerator="cpu",
        # devices=[0],
        strategy="ddp_find_unused_parameters_true"
    )
        
    trainer.fit(model, dm)
    return model