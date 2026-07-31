import torch
import torch.nn as nn
from torch import math
import pytorch_lightning as pl
import torchvision
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

from nerf.model.gaussian_splatting.pcd import PCD


class GaussianSplattingTrainer(pl.LightningModule):
    def __init__(self):
        super(GaussianSplattingTrainer, self).__init__()

        self.num_points = 5000
        self.positions = torch.rand(self.num_points, 3) * 2 - 1
        self.scaling_vectors = torch.rand(self.num_points, 3) * 2 - 1
        self.quaternions = torch.rand(self.num_points, 4) * 2 - 1
        self.colors = torch.rand(self.num_points, 3)
        self.opacities = torch.rand(self.num_points, 1)
        
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

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            list(self.coarse_model.parameters()) + list(self.fine_model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return opt
    
    def _get_camera_direction_vectors(self, image, focal_length):
        H, W = image.shape[:2]
        y, x = torch.meshgrid(
            torch.arange(H, device=self.device).float(),
            torch.arange(W, device=self.device).float(),
            indexing="ij"
        )
        dirs_x = (x + 0.5 - W // 2) / focal_length
        dirs_y = -(y + 0.5 - H // 2) / focal_length
        dirs_z = -torch.ones_like(dirs_x, dtype=torch.float32)
        ys = (torch.arange(H, device=self.device).float() + 0.5 - H // 2) / focal_length
        xs = (torch.arange(W, device=self.device).float() + 0.5 - W // 2) / focal_length
        X_camera_system_coords, Y_camera_system_coords = torch.meshgrid(xs, ys, indexing='ij')
        X_camera_system_coords = X_camera_system_coords.reshape(-1)
        Y_camera_system_coords = Y_camera_system_coords.reshape(-1)
        Z = -torch.ones_like(X_camera_system_coords, dtype=torch.float32)
        return torch.stack((dirs_x, dirs_y, dirs_z), axis=-1).reshape(-1, 3)
    
    def _sample_points_for_coarse_network(self, pixels_num):
        array = torch.linspace(self.t_n, self.t_f, self.n_c + 1, device=self.device)
        array = torch.broadcast_to(array, (pixels_num, self.n_c + 1))
        low = array[:, :-1]
        high = array[:, 1:]
        u = torch.rand_like(low)
        sampled = low + (high - low) * u
        return sampled
    
    @torch.no_grad
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
    
    def _render_volume(self, ray_points, camera_center, camera_direction, ray_points_num, coarse_network=True):
        # Step 6: Prepare input for MLP
        t =  torch.broadcast_to(ray_points[:, None, :], (*camera_direction.shape, ray_points.shape[-1])) # batch_size x 3 x n
        camera_direction = torch.broadcast_to(camera_direction[:, :, None], (*camera_direction.shape, ray_points.shape[-1])) # batch_size x 3 x n
        all_points = camera_center[None, :, None] + t * camera_direction # batch_size x 3 x n
        all_points = torch.permute(all_points, (0, 2, 1)) # batch_size x n x 3
        all_points = torch.reshape(all_points, (-1, 3)) # batch_size*n x 3
        camera_direction = torch.permute(camera_direction, (0, 2, 1)).reshape(-1, 3) # batch_size*n x 3
        del t
        
        # Positional encoding
        all_points = self._apply_positional_encoding(all_points, L=10)
        camera_direction = self._apply_positional_encoding(camera_direction, L=4)
        input_tensor = torch.concat((all_points, camera_direction), axis=-1) # batch_size*n x 6
        del all_points
        del camera_direction
        
        # Step 7: Pass input through a model
        if coarse_network:
            output_tensor = self.coarse_model(input_tensor) # batch_size*n x 4
        else:
            output_tensor = self.fine_model(input_tensor) # batch_size*n x 4

        # Step 8: Pixel reconstruction (Classic Volume Rendering)
        output_tensor = torch.reshape(output_tensor, (ray_points_num, ray_points.shape[-1], 4)) # batch_size x n x 4
        output_tensor = torch.permute(output_tensor, (0, 2, 1)) # batch_size x 4 x n
        sigma = output_tensor[:, 3:4, :] # batch_size x 1 x n
        delta = (torch.cat([ray_points[:, 1:], torch.ones_like(ray_points[:, 0:1]) * self.t_f], dim=1) - ray_points)[:, None, :] # batch_size x 1 x n
        T = torch.exp(torch.cumsum(torch.cat([torch.zeros_like(sigma[:, :, 0:1], device=self.device), (-sigma * delta)[:, :, :-1]], dim=-1), dim=-1)) # batch_size x 1 x n
        c = output_tensor[:, :3, :] # batch_size x 3 x n
        color_weights = T * (1 - torch.exp(-sigma * delta)) # batch_size x 1 x 60
        output = torch.sum(color_weights * c, dim=-1) # batch_size x 3
        # white background
        acc_map = torch.sum(color_weights, dim=-1) 
        output = output + (1.0 - acc_map)

        if coarse_network:
            return output, color_weights.squeeze(1)
        return output

    def _cull_gaussians(self, camera_params):
        pass

    def _screenspace_gaussians(self, camera_params):
        pass

    def _create_tiles(self, w, h):
        return torch.zeros(w, h)

    def _duplicate_with_keys(self):
        pass

    def _sort_by_keys(self, keys, indices):
        pass

    def _identify_tile_ranges(tiles, keys):
        pass

    def _get_tile_range(self, ranges, tiles):
        pass

    def _blend_in_order(self, i, indices, ranges, keys):
        pass

    def _project_means(self, w, h, extrinsic_matrix, focal_length):
        intrinsic_matrix = torch.tensor([   
            [focal_length, 0, w // 2],
            [0, focal_length, h // 2]
        ], dtype=torch.float32, device=self.device)
        means_3d = torch.cat([self.positions, torch.ones((self.positions.shape[0], 1))], axis=-1).permute(1, 0)
        W_mi = torch.matmul(extrinsic_matrix[:3, :], means_3d)
        means_2d = torch.matmul(intrinsic_matrix, W_mi / W_mi[2]).permute(1, 0)
        return means_2d, W_mi

    def _project_cov_matrices(self, means_camera, extrinsic_matrix, focal_length):
        x_c, y_c, z_c = means_camera
        cov_matrices_3d = self._create_covariance_matrices()
        jacobian = torch.stack([
            focal_length / z_c,
            torch.zeros_like(x_c),
            -focal_length * x_c / (z_c * z_c),
            torch.zeros_like(x_c),
            focal_length / z_c,
            -focal_length * y_c / (z_c * z_c),
        ]).reshape(-1, 2, 3) @ extrinsic_matrix[:3, :3]
        return jacobian @ extrinsic_matrix[:3, :3] @ cov_matrices_3d @ extrinsic_matrix[:3, :3].transpose(-2, -1) @ jacobian.transpose(-2, -1)

    def _create_covariance_matrices(self):
        r, i, j, k = self.quaternions.permute(1, 0)
        rotation_matrices = 2 * torch.stack([
            1/2 - (j * j + k * k),
            i * j - r * k,
            i * k + r * j,
            i * j + r * k,
            1/2 - (i * i + k * k),
            j * k - r * i,
            i * k - r * j,
            j * k + r * i,
            1/2 - (i * i + j * j)
        ]).reshape(-1, 3, 3)
        scaling_matrices = torch.diag_embed(self.scaling_vectors)
        return rotation_matrices @ scaling_matrices @ scaling_matrices.transpose(-2, -1) @ rotation_matrices.transpose(-2, -1)

    def _rasterize(self, w, h, extrinsic_matrix, focal_length):
        means_2d, means_camera = self._project_means(extrinsic_matrix, focal_length)
        cov_matrices = self._project_cov_matrices(means_camera, extrinsic_matrix, focal_length)
        self._cull_gaussians(extrinsic_matrix, focal_length)
        self._screenspace_gaussians(extrinsic_matrix, focal_length)
        tiles = self._create_tiles(w, h)
        indices, keys = self._duplicate_with_keys()
        self._sort_by_keys(keys, indices)
        ranges = self._identify_tile_ranges(tiles, keys)
        for tile in tiles:
            for pixel in tiles:
                range = self._get_tile_range(ranges, tile)
                self._blend_in_order(pixel, indices, range, keys)

    def any_step(self, batch, batch_idx, mode):
        images, poses, focal_lengths = batch
        image_rgba = torch.tensor(images[0, :], device=self.device) # H x W x 4
        pose = torch.tensor(poses[0, :], device=self.device) # 4 x 4
        focal_length = torch.tensor(focal_lengths[0], device=self.device) # 1

        rgb = image_rgba[..., :3]
        alpha = image_rgba[..., 3:4]
        image = rgb * alpha + 1.0 * (1.0 - alpha)

        output = self._rasterize(image.shape[0], image.shape[1], (pose, focal_length))
        
        # Step 2-4: Ray casting
        camera_direction_vectors_camera_coords = self._get_camera_direction_vectors(image, focal_length) # HW x 3
        camera_direction_vectors_camera_coords = torch.permute(camera_direction_vectors_camera_coords, (1, 0)) # 3 x HW
        camera_direction_vectors_world_coords = torch.matmul(pose[:3, :3], camera_direction_vectors_camera_coords) # 3 x HW
        camera_direction_vectors_world_coords = torch.permute(camera_direction_vectors_world_coords, (1, 0)) # HW x 3
        camera_direction_vectors_world_coords = nn.functional.normalize(camera_direction_vectors_world_coords, dim=1) # HW x 3

        image_flattened = image.reshape(-1, 3) # HW x 3
    
        if mode == "train":
            sampled_indices = torch.randint(0, image.shape[0] * image.shape[1], (self.rays_batch_size,), device=self.device)
            
            rays_d_batch = camera_direction_vectors_world_coords[sampled_indices]
            image_batch = image_flattened[sampled_indices]
            ray_points_num = self.rays_batch_size
        else:
            rays_d_batch = camera_direction_vectors_world_coords
            image_batch = image_flattened
            ray_points_num = image.shape[0] * image.shape[1]
            
        # Step 5: Ray Marching
        camera_center = pose[:3, 3] # (3,)
        coarse_ray_points = self._sample_points_for_coarse_network(ray_points_num) # batch_size x n

        coarse_output, color_weights = self._render_volume(coarse_ray_points, camera_center, rays_d_batch, ray_points_num, coarse_network=True)

        fine_ray_points = self._sample_points_for_fine_network(color_weights)
        fine_ray_points = torch.cat([coarse_ray_points, fine_ray_points], dim=1)
        fine_ray_points, _ = torch.sort(fine_ray_points, dim=1)
        fine_output = self._render_volume(fine_ray_points, camera_center, rays_d_batch, ray_points_num, coarse_network=False)

        # Step 9-10: Loss calculation
        loss = self.criterion(coarse_output, image_batch) + self.criterion(fine_output, image_batch)
        self.log(f"{mode}_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        return loss, coarse_output, fine_output, image

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.any_step(batch, batch_idx, "train")
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, coarse_output, fine_output, image = self.any_step(batch, batch_idx, "val")
        
        coarse_output = coarse_output.reshape(image.shape[0], image.shape[1], -1)
        fine_output = fine_output.reshape(image.shape[0], image.shape[1], -1)

        self.psnr_coarse.update(coarse_output, image)
        self.psnr_fine.update(fine_output, image)
        
        if self.current_epoch % 100 == 0 and batch_idx == 0:
            self.log_debug_samples(image, coarse_output, fine_output, "val")

        return loss
    
    def on_validation_epoch_end(self):
        psnr_coarse_value = self.psnr_coarse.compute()
        psnr_fine_value = self.psnr_fine.compute()

        self.log("val_psnr_coarse", psnr_coarse_value, on_epoch=True, prog_bar=True)
        self.log("val_psnr_fine", psnr_fine_value, on_epoch=True, prog_bar=True)
    
    def log_debug_samples(self, img, pred1, pred2, mode):
        img = img.detach().cpu().permute(2, 0, 1)
        pred1 = pred1.detach().cpu().permute(2, 0, 1)
        pred2 = pred2.detach().cpu().permute(2, 0, 1)

        pred1 = torch.clamp(pred1, 0.0, 1.0)
        pred2 = torch.clamp(pred2, 0.0, 1.0)

        # grid = torch.cat([img, pred], dim=0)
        grid = torchvision.utils.make_grid([img, pred1, pred2])

        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_image(f"{mode}_debug_samples", grid, self.current_epoch)
    