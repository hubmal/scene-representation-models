from nerf.train import train_with_clearml
from nerf.model.nerf.lightning import NerfTrainer
from nerf.data.lightning import get_datamodule


train_with_clearml(
    "simple_nerf",
    NerfTrainer(),
    get_datamodule()
)