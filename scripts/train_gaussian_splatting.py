from nerf.train import train_with_clearml
from nerf.model.gaussian_splatting.lightning import GaussianSplattingTrainer
from nerf.data.lightning import get_datamodule


train_with_clearml(
    "simple_3dgs",
    GaussianSplattingTrainer(),
    get_datamodule()
)