import pytorch_lightning as pl
from torch.utils.data import DataLoader

from nerf.data.dataset import SpecificDataset


class SpecificDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = None,
        batch_size: int = 1,
        num_workers: int = 4,
        *args, **kwargs
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args
        self.kwargs = kwargs
  
    def setup(self, stage):
        self.train_dataset = SpecificDataset(
            self.root,  
            *self.args, **self.kwargs
        )
        self.val_dataset = SpecificDataset(
            self.root, 
            *self.args, **self.kwargs
        )
        self.test_dataset = SpecificDataset(
            self.root, 
            *self.args, **self.kwargs
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

def get_datamodule(*args, **kwargs):
    return SpecificDataModule(*args, **kwargs)

if __name__ == "__main__":
    datamodule = get_datamodule()
    datamodule.setup("fit")
    print(next(iter(datamodule.train_dataloader())))