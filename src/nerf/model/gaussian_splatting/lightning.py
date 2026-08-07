import torch
import torch.nn as nn
from torch import math
import pytorch_lightning as pl
from shapely.geometry import box, Point, Polygon
import torchvision
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from piqa.ssim import SSIM

from nerf.model.gaussian_splatting.pcd import PCD


class GaussianSplattingTrainer(pl.LightningModule):
    def __init__(self):
        super(GaussianSplattingTrainer, self).__init__()

        self.num_points = 5000
        self.positions = nn.Parameter(torch.rand(self.num_points, 3) * 2 - 1)
        self.scaling_vectors = nn.Parameter(torch.rand(self.num_points, 3) * 2 - 1)
        self.quaternions = nn.Parameter(torch.rand(self.num_points, 4) * 2 - 1)
        self.colors = nn.Parameter(torch.rand(self.num_points, 3))
        self.opacities = nn.Parameter(torch.rand(self.num_points, 1))
        self.tiles_size = 50
        self.tiles_num_h = 4
        self.tiles_num_w = 4
        
        self.rays_batch_size = 4096
        self.t_n = 2.0
        self.t_f = 6.0
        self.n_c = 64
        self.n_f = 124
        
        self.l1_loss = nn.L1Loss()
        self.ssim = SSIM()
        self._lambda = 0.2
        self.lr = 5e-4
        self.weight_decay = 5e-5

        self.psnr = PSNR(data_range=1.0)

    def configure_optimizers(self):
        # opt = torch.optim.AdamW(
        #     list(self.coarse_model.parameters()) + list(self.fine_model.parameters()),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay
        # )
        # return opt
        return []

    def _cull_gaussians(self, camera_params):
        pass

    def _duplicate_with_keys(self, w, h, means_2d, cov_matrices, z_coords):
        def in_tile(tile_idx, center, radius):
            h1, h2, w1, w2 = (tile_idx // self.tiles_num_h) * self.tiles_size, ((tile_idx // self.tiles_num_h) + 1) * self.tiles_size, (tile_idx % self.tiles_num_w) * self.tiles_size, ((tile_idx % self.tiles_num_w) + 1) * self.tiles_size
            h2, w2 = max(h, h2), max(w, w2)
            tile = box(w1, h1, w2, h2)
            gaussian = Point(center.cpu().detach().numpy()).buffer(radius.cpu().detach().numpy())
            return (tile.intersects(gaussian))
        eigenvalues, _ = torch.linalg.eig(cov_matrices)
        eigenvalues = eigenvalues.real
        max_eigenvalues, _ = eigenvalues.max(dim=1, keepdim=False)
        radiuses = torch.ceil(3 * torch.sqrt(max_eigenvalues))
        gaussians_for_tiles = {}
        for tile_idx in range(self.tiles_num_w * self.tiles_num_h):
            unsorted_gaussians = []
            for idx, (mean, radius) in enumerate(zip(means_2d, radiuses)):
                if in_tile(tile_idx, mean, radius):
                    unsorted_gaussians.append((idx, z_coords[idx]))
            gaussians_for_tiles[tile_idx] = torch.tensor(sorted(unsorted_gaussians, key=lambda elem : elem[1]), device=self.device)[:, 0].int()
        return gaussians_for_tiles

    def _blend_in_order(self, w, h, gaussians_for_tiles, means_2d, cov_matrices):
        image = torch.zeros((w, h, 3), device=self.device)
        for tile_idx, indices in gaussians_for_tiles.items():
            means_2d_sorted = means_2d[indices]
            # # for testing
            A = torch.randn(5000, 2, 2)
            cov_matrices = A @ A.transpose(-1, -2) + torch.eye(2) * 1e-3 
            cov_matrices_sorted = cov_matrices[indices.cpu()].cuda()
            colors = self.colors[indices.cpu()].cuda()
            opacities = self.opacities[indices.cpu()].cuda()
            multivariate_normal = torch.distributions.MultivariateNormal(means_2d_sorted, cov_matrices_sorted)
            h1, h2, w1, w2 = (tile_idx // self.tiles_num_h) * self.tiles_size, ((tile_idx // self.tiles_num_h) + 1) * self.tiles_size, (tile_idx % self.tiles_num_w) * self.tiles_size, ((tile_idx % self.tiles_num_w) + 1) * self.tiles_size
            h2, w2 = min(h, h2), min(w, w2)
            y, x = torch.meshgrid(
                torch.arange(h1, h2, device=self.device).float(),
                torch.arange(w1, w2, device=self.device).float(),
                indexing="ij"
            )
            tile_pixels = torch.stack((y, x), axis=-1)
            tile_pixels = torch.broadcast_to(tile_pixels, (opacities.shape[0], *tile_pixels.shape))
            alfas = opacities.transpose(-2, -1) * torch.exp(multivariate_normal.log_prob(tile_pixels.permute(1, 2, 0, 3)))
            colors = colors.transpose(-2, -1)[None, None, :, :]
            alfas = alfas[:, :, None, :]
            transmittance = torch.cumprod(1 - alfas, dim=-1)
            image[h1:h2, w1:w2, :] = torch.sum(colors * alfas * transmittance, dim=-1)
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
        x_c, y_c, z_c = means_camera
        cov_matrices_3d = self._create_covariance_matrices()
        jacobian = torch.stack([
            focal_length / z_c,
            torch.zeros_like(x_c),
            -focal_length * x_c / (z_c * z_c),
            torch.zeros_like(x_c),
            focal_length / z_c,
            -focal_length * y_c / (z_c * z_c),
        ]).reshape(-1, 2, 3)
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
        # self._cull_gaussians(extrinsic_matrix, focal_length)
        means_2d, means_camera = self._project_means(w, h, extrinsic_matrix, focal_length)
        cov_matrices = self._project_cov_matrices(means_camera, extrinsic_matrix, focal_length)
        gaussians_for_tiles = self._duplicate_with_keys(w, h, means_2d, cov_matrices, means_camera[2, :])
        image = self._blend_in_order(w, h, gaussians_for_tiles, means_2d, cov_matrices)
        return image

    def any_step(self, batch, batch_idx, mode):
        images, poses, focal_lengths = batch
        image_rgba = torch.tensor(images[0, :], device=self.device) # H x W x 4
        pose = torch.tensor(poses[0, :], device=self.device) # 4 x 4
        focal_length = torch.tensor(focal_lengths[0], device=self.device) # 1

        rgb = image_rgba[..., :3]
        alpha = image_rgba[..., 3:4]
        image = rgb * alpha + 1.0 * (1.0 - alpha)

        output = self._rasterize(image.shape[0], image.shape[1], pose, focal_length)
        loss = (1 - self._lambda) * self.l1_loss(output, image) + self._lambda * (1 - self.ssim(output, image)) / 2
        self.log(f"{mode}_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        return loss, output, image

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.any_step(batch, batch_idx, "train")
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output, image = self.any_step(batch, batch_idx, "val")

        self.psnr.update(output, image)
        
        if self.current_epoch % 100 == 0 and batch_idx == 0:
            self.log_debug_samples(image, output, "val")

        return loss
    
    def on_validation_epoch_end(self):
        psnr_coarse_value = self.psnr.compute()
        psnr_fine_value = self.psnr_fine.compute()

        self.log("val_psnr_coarse", psnr_coarse_value, on_epoch=True, prog_bar=True)
        self.log("val_psnr_fine", psnr_fine_value, on_epoch=True, prog_bar=True)
    
    def log_debug_samples(self, img, pred, mode):
        img = img.detach().cpu().permute(2, 0, 1)
        pred = pred.detach().cpu().permute(2, 0, 1)

        pred = torch.clamp(pred, 0.0, 1.0)
        grid = torchvision.utils.make_grid([img, pred])

        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_image(f"{mode}_debug_samples", grid, self.current_epoch)
    