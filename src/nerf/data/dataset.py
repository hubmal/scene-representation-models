import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
from PIL import Image
from torchvision.datasets import VisionDataset


class SpecificDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path] = None,
        _set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.ratio = 0.9
        images, poses, focal = self._load_npz("data/tiny_nerf_data.npz")
        self.images = images[:int(self.ratio * len(images))] if _set == "train" else images[int(self.ratio * len(images)):]
        self.poses = poses[:int(self.ratio * len(poses))] if _set == "train" else poses[int(self.ratio * len(poses)):]
        self.focal = focal
    
    def _load_npz(self, path):
        with np.load(path) as data:
            images = data['images']
            poses = data['poses']
            focal = data['focal']

        return images, poses, focal

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.images[index], self.poses[index], self.focal