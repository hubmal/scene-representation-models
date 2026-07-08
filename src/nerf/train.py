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
    model, best_model_path = train(task_name, *args)
    task.close()
    if return_best_model_path:
        return model, best_model_path
    return model


def train(task_name, model, dm, args, use_early_stopping=True):
    torch.set_float32_matmul_precision('medium')

    early_stop_callback = EarlyStopping(
            monitor="val/f1_fg_mean",
            patience=100,
            verbose=True,
            mode="max",
            strict=True
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=task_name + '-best',
        monitor="val/f1_fg_mean",
        mode="max",
        save_top_k=1,
        enable_version_counter=False,
    )
    callbacks = [checkpoint_callback, early_stop_callback] if use_early_stopping else [checkpoint_callback]

    if hasattr(args, 'max_epochs'):
        trainer = Trainer(
            max_epochs=args.max_epochs,
            callbacks=callbacks,
            accelerator="gpu",
            devices=-1,
            strategy="ddp_find_unused_parameters_true"
        )
    else:
        trainer = Trainer(
        max_steps=args.max_steps,
        val_check_interval=1.0, 
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true"
    )
        
    trainer.fit(model, dm)
    return model, trainer.checkpoint_callback.best_model_path
