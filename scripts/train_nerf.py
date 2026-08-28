from scene_representation.train import train_with_clearml
from scene_representation.model.nerf.lightning import NerfTrainer
from scene_representation.data.lightning import get_datamodule


train_with_clearml(
    "NeRF",
    NerfTrainer(),
    get_datamodule(),
    max_epochs=3000
)