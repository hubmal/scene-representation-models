import torch
import torch.nn as nn
from torch import math
import pytorch_lightning as pl
import torchvision
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from scene_representation.model.nerf.mlp import MLP


class NerfTrainer(pl.LightningModule):
    def __init__(self):
        super(NerfTrainer, self).__init__()
        
        self.coarse_model = MLP(in_location_channels=60+3, in_direction_channels=24+3)
        self.fine_model = MLP(in_location_channels=60+3, in_direction_channels=24+3)
        
        self.rays_batch_size = 4096
        self.t_n = 2.0
        self.t_f = 6.0
        self.n_c = 64
        self.n_f = 124
        
        self.criterion = nn.MSELoss()
        self.lr = 5e-4
        self.weight_decay = 5e-5

        self.psnr_coarse = PSNR(data_range=1.0)
        self.psnr_fine = PSNR(data_range=1.0)

        self.test_psnr = PSNR(data_range=1.0)
        self.test_ssim = SSIM(data_range=1.0)
        self.test_lpips = LPIPS(normalize=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            list(self.coarse_model.parameters()) + list(self.fine_model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return opt

    def _prepare_for_metrics_calculation(self, x):
        x = torch.clamp(x, 0.0, 1.0)
        if x.shape[2] == 1 or x.shape[2] == 3:
            x = torch.permute(x, (2, 0, 1))
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return x
    
    def _sample_points_for_coarse_network(self, pixels_num):
        array = torch.linspace(self.t_n, self.t_f, self.n_c + 1, device=self.device)
        array = torch.broadcast_to(array, (pixels_num, self.n_c + 1))
        low = array[:, :-1]
        high = array[:, 1:]
        u = torch.rand_like(low)
        sampled = low + (high - low) * u
        return sampled
    
    @torch.no_grad
    def _sample_points_for_fine_network(self, coarse_ray_points, probs):
        probs = torch.clamp(probs, min=1e-5)
        bins = 0.5 * (coarse_ray_points[:, 1:] + coarse_ray_points[:, :-1])
        bins = torch.cat([torch.ones_like(bins[:, 0:1]) * self.t_n, bins, torch.ones_like(bins[:, 0:1]) * self.t_f], dim=-1)
        distribution = torch.distributions.Categorical(probs=probs)
        sampled_bins_indices = distribution.sample((self.n_f,)).T.to(self.device)
        low = torch.gather(bins, dim=1, index=sampled_bins_indices)
        high = torch.gather(bins, dim=1, index=sampled_bins_indices+1)
        u = torch.rand_like(low)
        sampled = low + (high - low) * u
        return sampled
    
    def _apply_positional_encoding(self, p, L):
        aranged_L = torch.arange(L, device=self.device)[None, None, :]
        x =  torch.pow(2 * torch.ones_like(aranged_L), aranged_L) * math.pi * p[:, :, None]
        sin_basis = torch.sin(x)
        cos_basis = torch.cos(x)
        encoded = torch.stack((sin_basis, cos_basis), dim=3).flatten(start_dim=2).flatten(start_dim=1)
        out = torch.cat((p, encoded), dim=1)
        return out
    
    def _synthesize(self, ray_points, camera_center, camera_direction, ray_points_num, coarse_network=True):
        # Input preparing
        t = torch.broadcast_to(ray_points[:, None, :], (*camera_direction.shape, ray_points.shape[-1]))
        camera_direction = torch.broadcast_to(camera_direction[:, :, None], (*camera_direction.shape, ray_points.shape[-1]))
        all_points = camera_center[None, :, None] + t * camera_direction
        all_points = torch.permute(all_points, (0, 2, 1))
        all_points = torch.reshape(all_points, (-1, 3))
        camera_direction = torch.permute(camera_direction, (0, 2, 1)).reshape(-1, 3)
        
        # Positional encoding
        camera_direction = nn.functional.normalize(camera_direction, dim=1)
        all_points = self._apply_positional_encoding(all_points, L=10)
        camera_direction = self._apply_positional_encoding(camera_direction, L=4)
        input_tensor = torch.concat((all_points, camera_direction), axis=-1)
        
        # Passing input through a model
        if coarse_network:
            output_tensor = self.coarse_model(input_tensor)
        else:
            output_tensor = self.fine_model(input_tensor)

        # Pixel reconstruction (Classic Volume Rendering)
        output_tensor = torch.reshape(output_tensor, (ray_points_num, ray_points.shape[-1], 4))
        output_tensor = torch.permute(output_tensor, (0, 2, 1))
        sigma = output_tensor[:, 3:4, :]
        delta = (torch.cat([ray_points[:, 1:], torch.ones_like(ray_points[:, 0:1]) * self.t_f], dim=1) - ray_points)[:, None, :]
        T = torch.exp(torch.cumsum(torch.cat([torch.zeros_like(sigma[:, :, 0:1], device=self.device), (-sigma * delta)[:, :, :-1]], dim=-1), dim=-1))
        c = output_tensor[:, :3, :]
        color_weights = T * (1 - torch.exp(-sigma * delta))
        output = torch.sum(color_weights * c, dim=-1)
        # white background
        acc_map = torch.sum(color_weights, dim=-1) 
        output = output + (1.0 - acc_map)

        if coarse_network:
            return output, color_weights.squeeze(1)
        return output

    def _render_view(self, rays_d_batch, ray_points_num, pose):
        coarse_ray_points = self._sample_points_for_coarse_network(ray_points_num)
        coarse_output, color_weights = self._synthesize(coarse_ray_points, pose[:3, 3], rays_d_batch, ray_points_num, coarse_network=True)

        probs = nn.functional.normalize(color_weights, dim=1)
        fine_ray_points = self._sample_points_for_fine_network(coarse_ray_points, probs)
        fine_ray_points = torch.cat([coarse_ray_points, fine_ray_points], dim=1)
        fine_ray_points, _ = torch.sort(fine_ray_points, dim=1)
        fine_output = self._synthesize(fine_ray_points, pose[:3, 3], rays_d_batch, ray_points_num, coarse_network=False)

        return coarse_output, fine_output

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)
    
    def any_step(self, batch, batch_idx, mode):
        images, poses, _, camera_direction_vectors_world_coords = batch

        image = images[0, ...]
        pose = poses[0, ...]
        camera_direction_vectors_world_coords = camera_direction_vectors_world_coords[0, ...]
        image_flattened = image.reshape(-1, 3) 
    
        if mode == "train":
            sampled_indices = torch.randint(0, image.shape[0] * image.shape[1], (self.rays_batch_size,), device=self.device)
            rays_d_batch = camera_direction_vectors_world_coords[sampled_indices]
            image_batch = image_flattened[sampled_indices]
            ray_points_num = self.rays_batch_size
            coarse_output, fine_output = self._render_view(rays_d_batch, ray_points_num, pose)
        else:
            chunk_size = 100 * 100
            coarse_output = torch.empty_like(camera_direction_vectors_world_coords, device=self.device)
            fine_output = torch.empty_like(camera_direction_vectors_world_coords, device=self.device)
            for i in range(0, len(camera_direction_vectors_world_coords), chunk_size):
                rays_d_batch = camera_direction_vectors_world_coords[i:i+chunk_size, ...]
                image_batch = image_flattened
                ray_points_num = chunk_size
                coarse_output[i:i+chunk_size, ...], fine_output[i:i+chunk_size, ...] = self._render_view(rays_d_batch, ray_points_num, pose)

        loss = self.criterion(coarse_output, image_batch) + self.criterion(fine_output, image_batch)
        self.log(f"{mode}_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        return loss, coarse_output, fine_output, image

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.any_step(batch, batch_idx, "train")
        
        return loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % 300 == 0:
            loss, coarse_output, fine_output, image = self.any_step(batch, batch_idx, "val")
            
            coarse_output = coarse_output.reshape(image.shape[0], image.shape[1], -1)
            fine_output = fine_output.reshape(image.shape[0], image.shape[1], -1)
            self.psnr_coarse.update(coarse_output, image)
            self.psnr_fine.update(fine_output, image)
            
            if self.current_epoch % 5 == 0 and batch_idx == 0:
                self.log_debug_samples(image, coarse_output, fine_output, "val")

            return loss

    def test_step(self, batch, batch_idx):
        loss, _, output, image = self.any_step(batch, batch_idx, "test")
        output = output.reshape(image.shape[0], image.shape[1], -1)
        self.test_psnr.update(output, image)
        self.test_ssim.update(self._prepare_for_metrics_calculation(output), self._prepare_for_metrics_calculation(image))
        self.test_lpips.update(self._prepare_for_metrics_calculation(output), self._prepare_for_metrics_calculation(image))
        return loss
    
    def on_validation_epoch_end(self):
        psnr_coarse_value = self.psnr_coarse.compute()
        psnr_fine_value = self.psnr_fine.compute()

        self.log("val_psnr_coarse", psnr_coarse_value, on_epoch=True, prog_bar=True)
        self.log("val_psnr_fine", psnr_fine_value, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        test_psnr_value = self.test_psnr.compute()
        test_ssim_value = self.test_ssim.compute()
        test_lpips_value = self.test_lpips.compute()

        self.log("test_psnr", test_psnr_value, on_epoch=True, prog_bar=True)
        self.log("test_ssim", test_ssim_value, on_epoch=True, prog_bar=True)
        self.log("test_lpips", test_lpips_value, on_epoch=True, prog_bar=True)
        
    def log_debug_samples(self, img, pred1, pred2, mode):
        if self.logger is not None and hasattr(self.logger, "experiment"):
            img = img.detach().cpu().permute(2, 0, 1)
            pred1 = pred1.detach().cpu().permute(2, 0, 1)
            pred2 = pred2.detach().cpu().permute(2, 0, 1)
    
            pred1 = torch.clamp(pred1, 0.0, 1.0)
            pred2 = torch.clamp(pred2, 0.0, 1.0)
    
            grid = torchvision.utils.make_grid([img, pred1, pred2])
            self.logger.experiment.add_image(f"{mode}_debug_samples", grid, self.current_epoch)
    