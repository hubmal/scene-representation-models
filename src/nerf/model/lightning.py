import numpy as np
import torch
import torch.nn as nn
from torch import math
import pytorch_lightning as pl
from torchmetrics.classification import JaccardIndex, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms.functional as F

from nerf.model.mlp import MLP


class NerfTrainer(pl.LightningModule):
    def __init__(self):
        super(NerfTrainer, self).__init__()
        
        self.coarse_model = MLP()
        self.fine_model = MLP()
        self.criterion = nn.MSELoss()
        self.lr = 1e-4
        self.t_n = 2.0
        self.t_f = 6.0
        self.n_c = 64
        self.n_f = 124
        self.lr = 5e-4
        self.weight_decay = 5e-5

    def forward(self, x):
        pass
        # return self.model(x)
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            list(self.coarse_model.parameters()) + list(self.fine_model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        return opt
    
    def _get_camera_direction_vectors(self, image, focal_length):
        H, W = image.shape[:2]
        xs = (torch.range(0, H - 1, device=self.device).float() + 0.5 - H // 2) / focal_length
        ys = (torch.range(0, W - 1, device=self.device).float() + 0.5 - W // 2) / focal_length
        X_camera_system_coords, Y_camera_system_coords = torch.meshgrid(xs, ys)
        X_camera_system_coords = X_camera_system_coords.reshape(-1)
        Y_camera_system_coords = Y_camera_system_coords.reshape(-1)
        Z = -torch.ones_like(X_camera_system_coords, dtype=torch.float32)
        return torch.stack((X_camera_system_coords, Y_camera_system_coords, Z), axis=-1)
    
    def _sample_points_for_coarse_network(self, pixels_num):
        array = torch.linspace(self.t_n, self.t_f, self.n_c + 1, device=self.device)
        array = torch.broadcast_to(array, (pixels_num, self.n_c + 1))
        low = array[:, :-1]
        high = array[:, 1:]
        u = torch.rand_like(low)
        sampled = low + (high - low) * u
        return sampled
    
    def _sample_points_for_fine_network(self, probs):
        eps = 1e-5
        probs = torch.clamp(probs, min=eps)
        array = torch.linspace(self.t_n, self.t_f, self.n_c + 1, device=self.device)
        assert len(array) > 1
        low_bounds = array[:-1]
        high_bounds = array[1:]
        distribution = torch.distributions.Categorical(probs=probs)
        sampled_bins_indices = distribution.sample((self.n_f,)).T.to(self.device)
        low = low_bounds[sampled_bins_indices]
        high = high_bounds[sampled_bins_indices]
        u = torch.rand_like(low)
        sampled = low + (high - low) * u
        return sampled
    
    def _apply_positional_encoding(self, p, L):
        aranged_L = torch.arange(L, device=self.device)[None, None, :]
        x =  torch.pow(2 * torch.ones_like(aranged_L), aranged_L) * math.pi * p[:, :, None]
        sin_basis = torch.sin(x)
        cos_basis = torch.cos(x)
        out = torch.stack((sin_basis, cos_basis), dim=3).flatten(start_dim=2).flatten(start_dim=1)
        return out
    
    def _render_volume(self, ray_points, camera_center, camera_direction, camera_direction_vectors_world_coords, image_shape, coarse_network=True):
        # Step 6: Prepare input for MLP
        t =  torch.broadcast_to(ray_points[:, None, :], (*camera_direction_vectors_world_coords.shape, ray_points.shape[-1])) # HW x 3 x n
        all_points = camera_center[None, :, None] + t * camera_direction_vectors_world_coords[:, :, None] # HW x 3 x n
        all_points = torch.permute(all_points, (0, 2, 1)) # HW x n x 3
        all_points = torch.reshape(all_points, (-1, 3)) # HWn x 3
        extended_camera_direction = torch.broadcast_to(camera_direction, (all_points.shape[0], 3)) #TODO: Exchange to angles, HWn x 3
        
        # Positional encoding
        all_points = self._apply_positional_encoding(all_points, L=10)
        extended_camera_direction = self._apply_positional_encoding(extended_camera_direction, L=4)
        input_tensor = torch.concat((all_points, extended_camera_direction), axis=-1) # HWn x 6
        
        # Step 7: Pass input through a model
        if coarse_network:
            output_tensor = self.coarse_model(input_tensor) # HWn x 4
        else:
            output_tensor = self.fine_model(input_tensor) # HWn x 4

        # Step 8: Pixel reconstruction (Classic Volume Rendering)
        output_tensor = torch.reshape(output_tensor, (image_shape[0] * image_shape[1], ray_points.shape[-1], 4)) # HW x n x 4
        output_tensor = torch.permute(output_tensor, (0, 2, 1)) # HW x 4 x n
        sigma = output_tensor[:, 3:4, :] # HW x 1 x n
        delta = (ray_points - torch.cat([torch.ones_like(ray_points[:, 0:1]) * self.t_n, ray_points[:, :-1]], dim=1))[:, None, :] # HW x 1 x n
        T = torch.exp(torch.cumsum(sigma * delta, dim=-1)) # HW x 1 x n
        c = output_tensor[:, :3, :] # HW x 3 x n
        color_weights = T * (1 - torch.exp(-sigma * delta)) # 10000, 1, 60
        color_map = torch.sum(color_weights * c, dim=-1) # HW x 3

        if coarse_network:
            return color_map, color_weights.squeeze(1)
        return color_map

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
        camera_direction = nn.functional.normalize(pose[:3, 2], dim=0) # (3,)
        coarse_ray_points = self._sample_points_for_coarse_network(image.shape[0] * image.shape[1]) # HW x n

        coarse_color_map, color_weights = self._render_volume(coarse_ray_points, camera_center, camera_direction, camera_direction_vectors_world_coords, image.shape[:2], coarse_network=True)
        coarse_color_map = torch.reshape(coarse_color_map, (image.shape[0], image.shape[1], 3)) # H x W x 3

        fine_ray_points = self._sample_points_for_fine_network(color_weights)
        fine_ray_points = torch.cat([coarse_ray_points, fine_ray_points], dim=1)
        fine_ray_points, _ = torch.sort(fine_ray_points, dim=1)
        fine_color_map = self._render_volume(fine_ray_points, camera_center, camera_direction, camera_direction_vectors_world_coords, image.shape[:2], coarse_network=False)
        fine_color_map = torch.reshape(fine_color_map, (image.shape[0], image.shape[1], 3)) # H x W x 3

        # Step 9-10: Loss calculation
        loss = self.criterion(coarse_color_map, image) + self.criterion(fine_color_map, image)
   
        if batch_idx == 0:
            self.log_debug_samples(coarse_color_map, fine_color_map, image, mode)

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
    
    def log_debug_samples(self, pred1, pred2, img, mode):
        img = img.detach().cpu().permute(2, 0, 1)
        pred1 = pred1.detach().cpu().permute(2, 0, 1)
        pred2 = pred2.detach().cpu().permute(2, 0, 1)

        pred1 = torch.clamp(pred1, 0.0, 1.0)
        pred2 = torch.clamp(pred2, 0.0, 1.0)

        # grid = torch.cat([img, pred], dim=0)
        grid = torchvision.utils.make_grid([img, pred1, pred2])

        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_image(f"{mode}_debug_samples", grid, self.current_epoch)
    