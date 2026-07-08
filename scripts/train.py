from nerf.train import train_with_clearml
from nerf.model.lightning import SpecificTrainer
from nerf.data.lightning import get_datamodule


train_with_clearml(
    "simple_nerf",
    SpecificTrainer(),
    get_datamodule()
)