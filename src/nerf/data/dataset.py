import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
from PIL import Image
from torchvision.datasets import VisionDataset


class SpecificDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        images, poses, focal = self._load_npz("data/tiny_nerf_data.npz")
        self.images = images
        self.poses = poses
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