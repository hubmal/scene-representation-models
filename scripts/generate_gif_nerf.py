import argparse
import numpy as np
from PIL import Image
from scene_representation.infer import infer
from scene_representation.model.nerf.lightning import NerfTrainer
from scene_representation.data.lightning import get_datamodule


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, help="Path to model checkpoint")
args = parser.parse_args()

preds = infer(
    NerfTrainer,
    get_datamodule(downsample_factor=1),
    args.ckpt_path
)

frames = [Image.fromarray((np.asarray(pred, dtype=np.float32) * 255).astype(np.uint8)) for pred in preds]
frames[0].save(
    "assets/nerf_result.gif",
    save_all=True,
    append_images=frames[1:],
    duration=150,
    loop=0
)