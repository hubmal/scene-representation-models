# CO ZOSTALO:
# NAPISANIE RASTERIZERA W TRITONIE
# CULL_GAUSSIANS
# LEPSZA INICJALIZACJA (NA INNYM DATASECIE)

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from shapely import STRtree
from shapely.geometry import box, Point
from sklearn.neighbors import NearestNeighbors
import torchvision
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from piqa.ssim import SSIM
import time

from scene_representation.model.gaussian_splatting.utils import inverse_sigmoid
from scene_representation.model.gaussian_splatting.spherical_harmonics import sh_constants


class GaussianSplattingTrainer(pl.LightningModule):
    def __init__(self):
        super(GaussianSplattingTrainer, self).__init__()

        self.num_points = 100000
        self.positions = nn.Parameter(torch.rand(self.num_points, 3) * 2 - 1)
        self.scaling_vectors = self._initialize_scaling_vectors(self.positions.cpu().detach().numpy())
        self.quaternions = nn.Parameter(torch.cat([torch.ones(self.num_points, 1), torch.zeros(self.num_points, 3)], dim=-1))
        # self.colors = nn.Parameter(torch.rand(self.num_points, 3))
        self.sh_weights = nn.Parameter(torch.rand(self.num_points, 3, 16))
        self.opacities = nn.Parameter(inverse_sigmoid(torch.ones(self.num_points, 1) * 0.1))
        self.tiles_size = 50
        self.tiles_num_h = 16
        self.tiles_num_w = 16

        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densification_interval = 100
        self.densify_grad_threshold = 0.0002
        self.max_scale_threshold = 0.04
        self.pruning_threshold = 0.005
        self.scale_divisor = 1.6
        self.opacity_reset_interval = 3000
        self.opacity_reset_value = 0.01
        
        self.l1_loss = nn.L1Loss()
        self.ssim = SSIM()
        self.ssim_lambda = 0.2

        self.psnr = PSNR(data_range=1.0)
        self.test_psnr = PSNR(data_range=1.0)

    def configure_optimizers(self):
        return torch.optim.AdamW([
        {"params": [self.positions], "lr": 1.6e-4, "weight_decay": 1.6e-6},
        {"params": [self.scaling_vectors], "lr": 5e-3, "weight_decay": 0},
        {"params": [self.quaternions], "lr": 1e-3, "weight_decay": 0},
        # {"params": [self.colors], "lr": 2.5e-3, "weight_decay": 0},
        {"params": [self.sh_weights], "lr": 2.5e-3, "weight_decay": 0},
        {"params": [self.opacities], "lr": 5e-2, "weight_decay": 0},
    ])

    def _initialize_scaling_vectors(self, positions):
        nbrs = NearestNeighbors(n_neighbors=4, algorithm="ball_tree").fit(positions)
        distances, _ = nbrs.kneighbors(positions)
        avg_distances = torch.mean(torch.tensor(distances[:, 1:] ** 2, dtype=torch.float32), dim=1)
        avg_distances = torch.clip(avg_distances, 1e-7, None)
        return nn.Parameter(torch.log(torch.sqrt(avg_distances[:, None]).repeat(1, 3)))

    def _if_densification(self):
        return (
            self.current_epoch > 0 and
            self.global_step >= self.densify_from_iter and
            self.global_step <= self.densify_until_iter and
            self.global_step % self.densification_interval == 0 
        )

    def _if_opacity_reset(self):
        return self._if_densification() and self.global_step % self.opacity_reset_interval == 0

    def _cull_gaussians(self, extrinsic_matrix):
        def cull(params, index, optimizer, indices):
            stored_grad = params.grad
            stored_state = optimizer.state.get(params, None)  
            if stored_state:
                stored_state["exp_avg"] = stored_state["exp_avg"][indices]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][indices]
                del optimizer.state[params]
                params = nn.Parameter(
                    (params[indices].detach().requires_grad_(True))
                )
                optimizer.state[params] = stored_state
            else:
                params = nn.Parameter(
                    (params[indices].detach().requires_grad_(True))
                )
            params.grad = stored_grad[indices] if stored_grad is not None else None
            optimizer.param_groups[index]["params"][0] = params
            return params
         
        means_3d = torch.cat([self.positions, torch.ones((self.positions.shape[0], 1), device=self.device)], axis=-1).permute(1, 0)
        means_3d_camera = torch.matmul(extrinsic_matrix[:3, :], means_3d)
        indices = torch.argwhere(means_3d_camera[2] >= 0.01)
        optimizer = self.optimizers().optimizer
        
        self.positions = cull(self.positions, 0, optimizer, indices)
        self.scaling_vectors = cull(self.scaling_vectors, 1, optimizer, indices)
        self.quaternions = cull(self.quaternions, 2, optimizer, indices)
        # self.colors = cull(self.colors, 3, optimizer, indices)
        self.sh_weights = cull(self.sh_weights, 3, optimizer, indices)
        self.opacities = cull(self.opacities, 4, optimizer, indices)
        

    def _prune_gaussians(self):
        if self.current_epoch < 1:
            return
        def prune(params, index, optimizer, indices):
            stored_grad = params.grad
            stored_state = optimizer.state.get(params, None)  
            if stored_state:
                stored_state["exp_avg"] = stored_state["exp_avg"][indices]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][indices]
                del optimizer.state[params]
                params = nn.Parameter(
                    (params[indices].detach().requires_grad_(True))
                )
                optimizer.state[params] = stored_state
            else:
                params = nn.Parameter(
                    (params[indices].detach().requires_grad_(True))
                )
            params.grad = stored_grad[indices] if stored_grad is not None else None
            optimizer.param_groups[index]["params"][0] = params
            return params
        
        indices = torch.argwhere(torch.sigmoid(self.opacities) >= self.pruning_threshold)[:, 0]
        optimizer = self.optimizers().optimizer

        self.positions = prune(self.positions, 0, optimizer, indices)
        self.scaling_vectors = prune(self.scaling_vectors, 1, optimizer, indices)
        self.quaternions = prune(self.quaternions, 2, optimizer, indices)
        # self.colors = prune(self.colors, 3, optimizer, indices)
        self.sh_weights = prune(self.sh_weights, 3, optimizer, indices)
        self.opacities = prune(self.opacities, 4, optimizer, indices)


    def _split_gaussians(self, indices):
        def split(params, index, optimizer, indices, sampling=False, scaling=False):
            cloned_params = params.clone().detach()
            new_params = params[indices].clone().detach()
            if sampling:
                cov_matrices = self._create_covariance_matrices()[indices].float()
                multivariate_normal = torch.distributions.MultivariateNormal(new_params, cov_matrices)
                new_params = multivariate_normal.sample((1,)).squeeze(0) # TODO: Czy potrzebny clip?
            if scaling:
                new_params -= math.log(self.scale_divisor)
                cloned_params[indices] = new_params
            cloned_params = torch.cat([new_params.clone().detach(), cloned_params], dim=0)
            stored_state = optimizer.state.get(params, None)  
            if stored_state:
                stored_state["exp_avg"] = torch.cat([stored_state["exp_avg"], stored_state["exp_avg"][indices]], dim=0)
                stored_state["exp_avg_sq"] = torch.cat([stored_state["exp_avg_sq"], stored_state["exp_avg_sq"][indices]], dim=0)
                del optimizer.state[params]
                params = nn.Parameter(
                    (cloned_params.detach().requires_grad_(True))
                )
                optimizer.state[params] = stored_state
            else:
                params = nn.Parameter(
                    (cloned_params.detach().requires_grad_(True))
                )
            optimizer.param_groups[index]["params"][0] = params
            return params
        
        optimizer = self.optimizers().optimizer
        self.positions = split(self.positions, 0, optimizer, indices, sampling=True)
        self.scaling_vectors = split(self.scaling_vectors, 1, optimizer, indices, scaling=True)
        self.quaternions = split(self.quaternions, 2, optimizer, indices)
        self.sh_weights = split(self.sh_weights, 3, optimizer, indices)
        # self.colors = split(self.colors, 3, optimizer, indices)
        self.opacities = split(self.opacities, 4, optimizer, indices)

    def _clone_gaussians(self, indices, grad):
        def cloned(params, index, optimizer, indices, grad=None):
            if grad is not None:
                cloned_params = torch.cat([params.clone().detach(), params[indices].clone().detach() + 1.6e-4 * grad[indices]], dim=0)
            else:
                cloned_params = torch.cat([params.clone().detach(), params[indices].clone().detach()], dim=0)
            stored_state = optimizer.state.get(params, None)  
            if stored_state:
                stored_state["exp_avg"] = torch.cat([stored_state["exp_avg"], stored_state["exp_avg"][indices]], dim=0)
                stored_state["exp_avg_sq"] = torch.cat([stored_state["exp_avg_sq"], stored_state["exp_avg_sq"][indices]], dim=0)
                del optimizer.state[params]
                params = nn.Parameter(
                    (cloned_params.detach().requires_grad_(True))
                )
                optimizer.state[params] = stored_state
            else:
                params = nn.Parameter(
                    (cloned_params.detach().requires_grad_(True))
                )
            optimizer.param_groups[index]["params"][0] = params
            return params
        
        optimizer = self.optimizers().optimizer
        self.positions = cloned(self.positions, 0, optimizer, indices, grad)
        self.scaling_vectors = cloned(self.scaling_vectors, 1, optimizer, indices)
        self.quaternions = cloned(self.quaternions, 2, optimizer, indices)
        # self.colors = cloned(self.colors, 3, optimizer, indices)
        self.sh_weights = cloned(self.sh_weights, 3, optimizer, indices)
        self.opacities = cloned(self.opacities, 4, optimizer, indices)

    def _densify_gaussians(self):
        if self.current_epoch < 1:
            return
        optimizer = self.optimizers().optimizer
        grad = optimizer.param_groups[0]["params"][0].grad
        over_recon_indices = torch.argwhere(
            torch.logical_and(
                torch.linalg.vector_norm(grad, dim=1) > self.densify_grad_threshold,
                torch.abs(torch.max(torch.exp(self.scaling_vectors), dim=1)[0]) > self.max_scale_threshold
            )
        ).flatten()        
        under_recon_indices = torch.argwhere(
            torch.logical_and(
                torch.linalg.vector_norm(grad, dim=1) > self.densify_grad_threshold,
                torch.abs(torch.max(torch.exp(self.scaling_vectors), dim=1)[0]) <= self.max_scale_threshold
            )
        ).flatten()
        if len(under_recon_indices) != 0:
            self._clone_gaussians(under_recon_indices, grad)
        if len(over_recon_indices) != 0:
            self._split_gaussians(over_recon_indices)

    def _safe_eigvalsh_chunked(self, cov_matrices, chunk_size=50000):
        results = []
        for i in range(0, cov_matrices.shape[0], chunk_size):
            results.append(torch.linalg.eigvalsh(cov_matrices[i:i+chunk_size]))
        return torch.cat(results, dim=0)

    def _duplicate_with_keys(self, w, h, means_2d, cov_matrices, z_coords):
        def in_tile(tiles_indices, centers, radiuses):
            h1 = (tiles_indices // self.tiles_num_h) * self.tiles_size
            h2 = ((tiles_indices // self.tiles_num_h) + 1) * self.tiles_size
            w1 = (tiles_indices % self.tiles_num_w) * self.tiles_size
            w2 = ((tiles_indices % self.tiles_num_w) + 1) * self.tiles_size
            h2, w2 = h2.clamp(max=h), w2.clamp(max=w)
            cx, cy = centers[:, 0], centers[:, 1]
            closest_x = cx[None, :].clamp(min=w1[:, None], max=w2[:, None])
            closest_y = cy[None, :].clamp(min=h1[:, None], max=h2[:, None])
            mask = (closest_x - cx[None, :]) ** 2 + (closest_y - cy[None, :]) ** 2 <= radiuses[None, :] ** 2
            return mask
        eigenvalues = self._safe_eigvalsh_chunked(cov_matrices)
        eigenvalues = eigenvalues.real
        max_eigenvalues, _ = eigenvalues.max(dim=1, keepdim=False)
        radiuses = torch.ceil(3 * torch.sqrt(max_eigenvalues))
        gaussians_for_tiles = {}
        masks = in_tile(torch.arange(self.tiles_num_w * self.tiles_num_h, device=self.device), means_2d, radiuses)
        for tile_idx, mask in enumerate(masks):
            indices = mask.nonzero(as_tuple=True)[0]
            if indices.shape[0] == 0:
                gaussians_for_tiles[tile_idx] = indices
            else:
                gaussians_for_tiles[tile_idx] = indices[torch.argsort(z_coords[indices])]
        return gaussians_for_tiles

    def _gaussian_weight(self, tile_pixels, means_2d_sorted, cov_matrices_sorted):
        dx = tile_pixels[..., 0] - means_2d_sorted[:, 0]
        dy = tile_pixels[..., 1] - means_2d_sorted[:, 1]

        a = cov_matrices_sorted[:, 0, 0]
        b = cov_matrices_sorted[:, 0, 1]
        c = cov_matrices_sorted[:, 1, 1]

        det = a * c - b * b
        det = det.clamp(min=1e-8)
        mdist = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det
        return torch.exp(-0.5 * mdist)

    def _calculate_colors(self, sh_weights, positions, camera_pos):
        direction_vectors = torch.nn.functional.normalize(positions - camera_pos[None, :], dim=-1).permute(1, 0)
        x, y, z = direction_vectors
        coeffs = torch.stack([
            sh_constants[0] * torch.ones_like(x), -sh_constants[1] * y, sh_constants[2] * z, -sh_constants[3] * x,
            sh_constants[4] * x * y, sh_constants[5] * y * z, sh_constants[6] * (2 * z * z - x * x - y * y),
            sh_constants[7] * x * z, sh_constants[8] * (x * x - y * y), sh_constants[9] * y * (3 * x * x - y * y),
            sh_constants[10] * x * y * z, sh_constants[11] * y * (4 * z * z - x * x - y * y),
            sh_constants[12] * z * (2 * z * z - 3 * x * x - 3 * y * y),
            sh_constants[13] * x * (4 * z * z - x * x - y * y),
            sh_constants[14] * z * (x * x - y * y), sh_constants[15] * x * (x * x - 3 * y * y)
        ], dim=-1)
        return torch.sigmoid(torch.sum(coeffs[:, None, :] * sh_weights, dim=-1))
    
    def _blend_in_order(self, w, h, gaussians_for_tiles, means_2d, cov_matrices, camera_pos):
        image = torch.ones((h, w, 3), device=self.device)
        for tile_idx, indices in gaussians_for_tiles.items():
            if indices.shape[0] == 0:
                continue
            positions = self.positions[indices.cpu()].cuda()
            means_2d_sorted = means_2d[indices]
            cov_matrices_sorted = cov_matrices[indices.cpu()].cuda()
            # colors = self.colors[indices.cpu()].cuda()
            sh_weights = self.sh_weights[indices.cpu()].cuda()
            colors = self._calculate_colors(sh_weights, positions, camera_pos)
            opacities = torch.sigmoid(self.opacities[indices.cpu()].cuda())
            # multivariate_normal = torch.distributions.MultivariateNormal(means_2d_sorted, cov_matrices_sorted)
            h1, h2, w1, w2 = (tile_idx // self.tiles_num_h) * self.tiles_size, ((tile_idx // self.tiles_num_h) + 1) * self.tiles_size, (tile_idx % self.tiles_num_w) * self.tiles_size, ((tile_idx % self.tiles_num_w) + 1) * self.tiles_size
            h2, w2 = min(h, h2), min(w, w2)
            y, x = torch.meshgrid(
                torch.arange(h1, h2, device=self.device).float(),
                torch.arange(w1, w2, device=self.device).float(),
                indexing="ij"
            )
            tile_pixels = torch.stack((x, y), axis=-1)
            tile_pixels = torch.broadcast_to(tile_pixels, (opacities.shape[0], *tile_pixels.shape))
            colors = colors.transpose(-2, -1)[None, None, :, :]
            gauss_weight = self._gaussian_weight(tile_pixels.permute(1, 2, 0, 3), means_2d_sorted, cov_matrices_sorted)                  # BEZ normalizacji!

            alfas = opacities.transpose(-2, -1) * gauss_weight
            alfas = torch.clamp(alfas, max=0.99)
            alfas = alfas[:, :, None, :]
            transmittance = torch.cumprod(torch.cat([torch.ones_like(alfas[:, :, :, 0:1], device=self.device), (1 - alfas)[:, :, :, :-1]], dim=-1), dim=-1)
            transmittance_final = transmittance[:, :, :, -1] * (1 - alfas[:, :, :, -1])
            image[h1:h2, w1:w2, :] *= transmittance_final
            image[h1:h2, w1:w2, :] += torch.sum(colors * alfas * transmittance, dim=-1)
        image = torch.clamp(image, 0.0, 1.0)
        return image

    def _project_means(self, w, h, extrinsic_matrix, focal_length):
        intrinsic_matrix = torch.tensor([   
            [focal_length, 0, w // 2],
            [0, focal_length, h // 2]
        ], dtype=torch.float32, device=self.device)
        means_3d = torch.cat([self.positions, torch.ones((self.positions.shape[0], 1), device=self.device)], axis=-1).permute(1, 0)
        W_mi = torch.matmul(extrinsic_matrix[:3, :], means_3d)
        means_2d = torch.matmul(intrinsic_matrix, W_mi / W_mi[2]).permute(1, 0)
        return means_2d, W_mi

    def _project_cov_matrices(self, means_camera, extrinsic_matrix, focal_length):
        extrinsic_matrix = extrinsic_matrix.double()
        focal_length = focal_length.double()
        x_c, y_c, z_c = means_camera.double()
        z_c = z_c.clamp(min=0.2)
        cov_matrices_3d = self._create_covariance_matrices()
        jacobian = torch.stack([
            focal_length / z_c,
            torch.zeros_like(x_c),
            -focal_length * x_c / (z_c * z_c),
            torch.zeros_like(x_c),
            focal_length / z_c,
            -focal_length * y_c / (z_c * z_c),
        ], dim=-1).reshape(-1, 2, 3)
        cov_matrices_2d = jacobian @ extrinsic_matrix[:3, :3] @ cov_matrices_3d @ extrinsic_matrix[:3, :3].transpose(-2, -1) @ jacobian.transpose(-2, -1)
        cov_matrices_2d = (cov_matrices_2d + cov_matrices_2d.transpose(-2, -1)) / 2 # avoid numerical errors - matrix should be symmetric
        cov_matrices_2d[:, 1, 1] += 0.3
        return cov_matrices_2d.float()

    def _create_covariance_matrices(self):
        quaternions = torch.nn.functional.normalize(self.quaternions, dim=-1).double()
        r, i, j, k = quaternions.permute(1, 0)
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
        ], dim=-1).reshape(-1, 3, 3)
        scales = torch.exp(self.scaling_vectors.clamp(min=-4, max=1.0)).double()
        scaling_matrices = torch.diag_embed(scales)
        return rotation_matrices @ scaling_matrices @ scaling_matrices.transpose(-2, -1) @ rotation_matrices.transpose(-2, -1)

    def _rasterize(self, h, w, extrinsic_matrix, focal_length, batch_idx):
        R = extrinsic_matrix[:3, :3]
        t = extrinsic_matrix[:3, 3]
        camera_pos = -R.T @ t
        means_2d, means_camera = self._project_means(w, h, extrinsic_matrix, focal_length)
        cov_matrices = self._project_cov_matrices(means_camera, extrinsic_matrix, focal_length)
        gaussians_for_tiles = self._duplicate_with_keys(w, h, means_2d, cov_matrices, means_camera[2, :])
        image = self._blend_in_order(w, h, gaussians_for_tiles, means_2d, cov_matrices, camera_pos)
        return image

    def any_step(self, batch, batch_idx, mode):
        images, poses, focal_lengths, _ = batch

        image = images[0, ...]
        pose = poses[0, ...]
        focal_length = focal_lengths[0, ...]

        extrinsic_matrix = torch.linalg.inv(pose)
        extrinsic_matrix[1, :] *= -1 
        extrinsic_matrix[2, :] *= -1

        output = self._rasterize(image.shape[0], image.shape[1], extrinsic_matrix, focal_length, batch_idx)
        output = output.permute(2, 0, 1)
        image = image.permute(2, 0, 1)
        loss = (1 - self.ssim_lambda) * self.l1_loss(output, image) + self.ssim_lambda * (1 - self.ssim(output.unsqueeze(0), image.unsqueeze(0))) / 2

        if mode != "predict":
            self.log(f"{mode}_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        return loss, output, image

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.any_step(batch, batch_idx, "train")
        
        return loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % 10 == 0:
            loss, output, image = self.any_step(batch, batch_idx, "val")

            self.psnr.update(output, image)
            
            if batch_idx % 10 == 0:
                self.log_debug_samples(image, output, "val", batch_idx)

            return loss

    def test_step(self, batch, batch_idx):
        loss, output, image = self.any_step(batch, batch_idx, "test")
        self.test_psnr.update(output, image)
        return loss

    def predict_step(self, batch, batch_idx):
        _, output, _ = self.any_step(batch, batch_idx, "predict")
        return output

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._if_densification():
            self._prune_gaussians()
            self._densify_gaussians()
        if self._if_opacity_reset() or self.global_step == self.densify_from_iter:
            self.opacities.data.clamp_(max=inverse_sigmoid(self.opacity_reset_value))
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_end(self):
        print(self.positions.shape[0])
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self):
        psnr_value = self.psnr.compute()
        self.log("val_psnr", psnr_value, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        test_psnr_value = self.test_psnr.compute()
        self.log("test_psnr", test_psnr_value, on_epoch=True, prog_bar=True)
        
    def log_debug_samples(self, img, pred, mode, batch_idx=""):
        img = img.detach().cpu()
        pred = pred.detach().cpu()

        pred = torch.clamp(pred, 0.0, 1.0)
        grid = torchvision.utils.make_grid([img, pred])

        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_image(f"{mode}_debug_samples{batch_idx}", grid, self.current_epoch)
