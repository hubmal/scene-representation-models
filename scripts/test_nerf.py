import argparse
from scene_representation.test import test
from scene_representation.model.nerf.lightning import NerfTrainer
from scene_representation.data.lightning import get_datamodule


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, help="Path to model checkpoint")
args = parser.parse_args()

test(
    NerfTrainer,
    get_datamodule(downsample_factor=1),
    args.ckpt_path
)