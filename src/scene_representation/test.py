import torch
from pytorch_lightning import Trainer


def test(model, dm, ckpt_path):
    torch.set_float32_matmul_precision('medium')
    model = model.load_from_checkpoint(ckpt_path)
    model.eval()
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True
    )
    trainer.test(model, dm)