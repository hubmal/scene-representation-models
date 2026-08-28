from scene_representation.train import train_with_clearml
from scene_representation.model.gaussian_splatting.lightning import GaussianSplattingTrainer
from scene_representation.data.lightning import get_datamodule


train_with_clearml(
    "3DGS",
    GaussianSplattingTrainer(),
    get_datamodule(downsample_factor=8),
    max_epochs=50
)