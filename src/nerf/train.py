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


def train_with_clearml(task_name, *args, tags=None, return_best_model_path=False):
    task = Task.init(task_name=task_name, project_name="segment cardiag", tags=tags)
    model = train(task_name, *args)
    return model


def train(task_name, model, dm, args, use_early_stopping=True):
    torch.set_float32_matmul_precision('medium')

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="cpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true"
    )
        
    trainer.fit(model, dm)
    return model