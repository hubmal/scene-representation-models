import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import JaccardIndex, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms.functional as F

from nerf.model.mlp import MLP


class NerfTrainer(pl.LightningModule):
    def __init__(self):
        super(NerfTrainer, self).__init__()
        
        self.model = MLP()
        self.criterion = nn.MSELoss()
        self.lr = 1e-4
        self.t_n = 2.0
        self.t_f = 6.0
        self.n = 60

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
    
    def _get_camera_direction_vectors(self, image, focal_length):
        H, W = image.shape[:2]
        xs = (torch.range(0, H - 1).float() + 0.5 - H // 2) / focal_length
        ys = (torch.range(0, W - 1).float() + 0.5 - W // 2) / focal_length
        X_camera_system_coords, Y_camera_system_coords = torch.meshgrid(xs, ys)
        X_camera_system_coords = X_camera_system_coords.reshape(-1)
        Y_camera_system_coords = Y_camera_system_coords.reshape(-1)
        Z = -torch.ones_like(X_camera_system_coords, dtype=torch.float32)
        return torch.stack((X_camera_system_coords, Y_camera_system_coords, Z), axis=-1)

    def any_step(self, batch, batch_idx, mode):
        images, poses, focal_lengths = batch
        image = torch.tensor(images[0, :]) # H x W x 3
        pose = torch.tensor(poses[0, :]) # 4 x 4
        focal_length = torch.tensor(focal_lengths[0]) # 1
        
        # Step 2-4: Ray casting
        camera_direction_vectors_camera_coords = self._get_camera_direction_vectors(image, focal_length) # HW x 3
        camera_direction_vectors_camera_coords = torch.permute(camera_direction_vectors_camera_coords, (1, 0)) # 3 x HW
        camera_direction_vectors_world_coords = torch.matmul(pose[:3, :3], camera_direction_vectors_camera_coords) # 3 x HW
        camera_direction_vectors_world_coords = torch.permute(camera_direction_vectors_world_coords, (1, 0)) # HW x 3
        
        # Step 5: Ray Marching
        camera_center = pose[:3, 3] # (3,)
        camera_direction = pose[:3, 2] # (3,)
        ray_points = torch.linspace(self.t_n, self.t_f, self.n) #TODO: Exchange to random sampling, # (n,)

        # Step 6: Prepare input for MLP
        t =  torch.broadcast_to(ray_points, (*camera_direction_vectors_world_coords.shape, len(ray_points))) # HW x 3 x n
        all_points = camera_center[None, :, None] + t * camera_direction_vectors_world_coords[:, :, None] # HW x 3 x n
        all_points = torch.permute(all_points, (0, 2, 1)) # HW x n x 3
        all_points = torch.reshape(all_points, (-1, 3)) # HWn x 3
        extended_camera_direction = torch.broadcast_to(camera_direction, (all_points.shape[0], 3)) #TODO: Exchange to angles, HWn x 3
        input_tensor = torch.concat((all_points, extended_camera_direction), axis=-1) # HWn x 6
        
        # Step 7: Pass input through a model
        output_tensor = self.model(input_tensor) # HWn x 4

        # Step 8: Pixel reconstruction (Classic Volume Rendering)
        output_tensor = torch.reshape(output_tensor, (image.shape[0] * image.shape[1], 4, len(ray_points))) # HW x 4 x n
        T = torch.exp(torch.cumsum(output_tensor[:, 3:4, :], dim=-1)) # HW x 1 x n
        sigma = output_tensor[:, 3:4, :] # HW x 1 x n
        c = output_tensor[:, :3, :] # HW x 3 x n
        color_map = torch.sum(T * (1 - torch.exp(-sigma)) * c, dim=-1) #TODO  HW x 3
        color_map = torch.reshape(color_map, (image.shape[0], image.shape[1], 3))

        # Step 9-10: Loss calculation
        loss = self.criterion(color_map, image)

        # if batch_idx == 0:
        #     self.log_debug_samples(images, preds, masks, mode)

        return loss

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
    