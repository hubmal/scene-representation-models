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
        xs = (torch.range(0, H - 1, device=self.device).float() + 0.5 - H // 2) / focal_length
        ys = (torch.range(0, W - 1, device=self.device).float() + 0.5 - W // 2) / focal_length
        X_camera_system_coords, Y_camera_system_coords = torch.meshgrid(xs, ys)
        X_camera_system_coords = X_camera_system_coords.reshape(-1)
        Y_camera_system_coords = Y_camera_system_coords.reshape(-1)
        Z = -torch.ones_like(X_camera_system_coords, dtype=torch.float32)
        return torch.stack((X_camera_system_coords, Y_camera_system_coords, Z), axis=-1)
    
    def _sample_ray_tracing_points(self):
        array = np.linspace(self.t_n, self.t_f, self.n + 1)
        assert len(array) > 1
        low = array[:-1]
        high = array[1:]
        sampled = np.random.uniform(low, high)
        return torch.tensor(sampled, device=self.device, dtype=torch.float32)

    def any_step(self, batch, batch_idx, mode):
        images, poses, focal_lengths = batch
        image = torch.tensor(images[0, :], device=self.device) # H x W x 3
        pose = torch.tensor(poses[0, :], device=self.device) # 4 x 4
        focal_length = torch.tensor(focal_lengths[0], device=self.device) # 1
        
        # Step 2-4: Ray casting
        camera_direction_vectors_camera_coords = self._get_camera_direction_vectors(image, focal_length) # HW x 3
        camera_direction_vectors_camera_coords = torch.permute(camera_direction_vectors_camera_coords, (1, 0)) # 3 x HW
        camera_direction_vectors_world_coords = torch.matmul(pose[:3, :3], camera_direction_vectors_camera_coords) # 3 x HW
        camera_direction_vectors_world_coords = torch.permute(camera_direction_vectors_world_coords, (1, 0)) # HW x 3
        
        # Step 5: Ray Marching
        camera_center = pose[:3, 3] # (3,)
        camera_direction = pose[:3, 2] # (3,)
        ray_points = self._sample_ray_tracing_points() # (n,)

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
        delta = (ray_points - torch.cat([torch.tensor([self.t_n], device=self.device), ray_points[:-1]]))[None, None, :] # 1 x 1 x n
        c = output_tensor[:, :3, :] # HW x 3 x n
        color_map = torch.sum(T * (1 - torch.exp(-sigma * delta)) * c, dim=-1) # HW x 3
        color_map = torch.reshape(color_map, (image.shape[0], image.shape[1], 3))

        # Step 9-10: Loss calculation
        loss = self.criterion(color_map, image)

        if batch_idx == 0:
            self.log_debug_samples(color_map, image, mode)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.any_step(batch, batch_idx, "train")
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.any_step(batch, batch_idx, "val")
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)        
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
    
    def log_debug_samples(self, pred, img, mode):
        img = img.detach().cpu().permute(2, 0, 1)
        pred = pred.detach().cpu().permute(2, 0, 1)

        pred = torch.clamp(pred, 0.0, 1.0)

        # grid = torch.cat([img, pred], dim=0)
        grid = torchvision.utils.make_grid([img, pred])

        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_image(f"{mode}_debug_samples", grid, self.current_epoch)
    