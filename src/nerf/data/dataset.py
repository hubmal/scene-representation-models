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

        self.trans
        self.samples = []
        
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input_path, label_path = self.samples[index]

        input_image = Image.open(input_path).convert("L")
        label_mask = Image.open(label_path).convert("L")

        if self.transforms is not None:
            input_image, label_mask = self.transforms(input_image, label_mask)

        return input_image, label_mask