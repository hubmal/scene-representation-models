import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import JaccardIndex, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms.functional as F


class SpecificTrainer(pl.LightningModule):
    def __init__(self):
        super(SpecificTrainer, self).__init__()
        
        # self.model =
        # self.criterion =
        # self.lr
        # self.f1 = F1Score(task="binary")

    def forward(self, x):
        pass
        # return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            # lr=self.lr,
            # weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimizer,
            "monitor": "val_loss"
        }

    def any_step(self, batch, batch_idx, mode):
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = (torch.sigmoid(logits) > 0.5).long()

        if batch_idx == 0:
            self.log_debug_samples(images, preds, masks, mode)

        return loss, logits, preds

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.any_step(batch, batch_idx, "train")
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, preds = self.any_step(batch, batch_idx, "val")
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        images, masks = batch

        if batch_idx == 0:
            self.log_debug_samples(images, preds, masks, "val")
        
        device = logits.device
        self.jaccard.to(device)
        self.f1.to(device)
    
        self.jaccard.update(preds, masks.long())
        self.f1.update(preds, masks.long())        
        
        self.log("val_jaccard", self.jaccard.compute(), on_epoch=True, prog_bar=True)
        self.log("val_f1", self.f1.compute(), on_epoch=True, prog_bar=True)

        self.jaccard.reset()
        self.f1.reset()
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        preds = (torch.sigmoid(logits) > 0.5).long()

        device = logits.device
        self.jaccard.to(device)
        self.f1.to(device)
    
        self.jaccard.update(preds, masks.long())
        self.f1.update(preds, masks.long())  
        
        self.log("test_jaccard", self.jaccard.compute(), on_epoch=True, prog_bar=True)
        self.log("test_f1", self.f1.compute(), on_epoch=True, prog_bar=True)
        
        self.jaccard.reset()
        self.f1.reset()
        
        return 0
    
    def log_debug_samples(self, imgs, preds, labels, mode):
        imgs = imgs.detach().cpu()
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        imgs = F.rgb_to_grayscale(torch.clamp(imgs / 2 + 0.5, 0.0, 1.0))

        grid = torch.cat([imgs, preds, labels], dim=3)
        grid = torchvision.utils.make_grid(grid, nrow=2)

        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_image(f"{mode}_debug_samples", grid, self.current_epoch)
    