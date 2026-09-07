import torch
import argparse

from clearml import Task
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def train_with_clearml(task_name, model, dm, max_epochs):
    Task.init(task_name=task_name, project_name=task_name)
    model = train(model, dm, max_epochs, task_name)
    return model

def train(model, dm, max_epochs, task_name):
    torch.set_float32_matmul_precision('medium')

    logger = TensorBoardLogger("tb_logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=task_name+'-epoch-{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=50,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=[0],
        logger=logger
    )
        
    trainer.fit(model, dm)
    return model