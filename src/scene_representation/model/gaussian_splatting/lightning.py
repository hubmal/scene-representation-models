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

from scene_representation.model.gaussian_splatting.utils import inverse_sigmoid


class GaussianSplattingTrainer(pl.LightningModule):
    def __init__(self):
        super(GaussianSplattingTrainer, self).__init__()

        # self.num_points = 5000
        self.num_points = 500
        self.positions = nn.Parameter(torch.rand(self.num_points, 3) * 2 - 1)
        self.scaling_vectors = self._initialize_scaling_vectors(self.positions.cpu().detach().numpy())
        self.quaternions = nn.Parameter(torch.cat([torch.ones(self.num_points, 1), torch.zeros(self.num_points, 3)], dim=-1))
        self.colors = nn.Parameter(torch.rand(self.num_points, 3))
        self.opacities = nn.Parameter(inverse_sigmoid(torch.ones(self.num_points, 1) * 0.1))
        self.tiles_size = 25
        self.tiles_num_h = 4
        self.tiles_num_w = 4
        self.pruning_threshold = 0.005
        self.densification_interval = 100
        self.decreasing_alpha_interval = 300
        self.pos_grad_threshold = 2e-4
        self.max_scale_threshold = 0.01
        self.scale_divisor = 1.6
        
        self.l1_loss = nn.L1Loss()
        self.ssim = SSIM()
        self._lambda = 0.2
        self.lr = 5e-4
        self.weight_decay = 5e-5

        self.psnr = PSNR(data_range=1.0)

    def configure_optimizers(self):
        return torch.optim.AdamW([
        {"params": [self.positions], "lr": 1.6e-4, "weight_decay": 1.6e-6},
        {"params": [self.scaling_vectors], "lr": 5e-3, "weight_decay": 0},
        {"params": [self.quaternions], "lr": 1e-3, "weight_decay": 0},
        {"params": [self.colors], "lr": 2.5e-3, "weight_decay": 0},
        {"params": [self.opacities], "lr": 5e-2, "weight_decay": 0},
    ])

    def _initialize_scaling_vectors(self, positions):
        nbrs = NearestNeighbors(n_neighbors=4, algorithm="ball_tree").fit(positions)
        distances, _ = nbrs.kneighbors(positions)
        avg_distances = torch.mean(torch.tensor(distances[:, 1:] ** 2, dtype=torch.float32), dim=1)
        avg_distances = torch.clip(avg_distances, 1e-7, None)
        return nn.Parameter(torch.log(torch.sqrt(avg_distances[:, None]).repeat(1, 3)))

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
        self.colors = cull(self.colors, 3, optimizer, indices)
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
            params.grad = stored_grad[indices]
            optimizer.param_groups[index]["params"][0] = params
            return params
        
        indices = torch.argwhere(torch.sigmoid(self.opacities) >= self.pruning_threshold)[:, 0]
        optimizer = self.optimizers().optimizer

        self.positions = prune(self.positions, 0, optimizer, indices)
        self.scaling_vectors = prune(self.scaling_vectors, 1, optimizer, indices)
        self.quaternions = prune(self.quaternions, 2, optimizer, indices)
        self.colors = prune(self.colors, 3, optimizer, indices)
        self.opacities = prune(self.opacities, 4, optimizer, indices)


    def _split_gaussians(self, indices):
        def split(params, index, optimizer, indices, sampling=False, scaling=False):
            cloned_params = params.clone().detach()
            new_params = params[indices].clone().detach()
            if sampling:
                cov_matrices = self._create_covariance_matrices()[indices]
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
        self.colors = split(self.colors, 3, optimizer, indices)
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
        self.colors = cloned(self.colors, 3, optimizer, indices)
        self.opacities = cloned(self.opacities, 4, optimizer, indices)

    def _densify_gaussians(self):
        if self.current_epoch < 1:
            return
        optimizer = self.optimizers().optimizer
        grad = optimizer.param_groups[0]["params"][0].grad
        over_recon_indices = torch.argwhere(
            torch.logical_and(
                torch.linalg.vector_norm(grad, dim=1) > self.pos_grad_threshold,
                torch.abs(torch.max(torch.exp(self.scaling_vectors), dim=1)[0]) > self.max_scale_threshold
            )
        ).flatten()        
        under_recon_indices = torch.argwhere(
            torch.logical_and(
                torch.linalg.vector_norm(grad, dim=1) > self.pos_grad_threshold,
                torch.abs(torch.max(torch.exp(self.scaling_vectors), dim=1)[0]) <= self.max_scale_threshold
            )
        ).flatten()
        if len(under_recon_indices) != 0:
            self._clone_gaussians(under_recon_indices, grad)
        if len(over_recon_indices) != 0:
            self._split_gaussians(over_recon_indices)

    def _duplicate_with_keys(self, w, h, means_2d, cov_matrices, z_coords):
        def in_tile(tile_idx, centers, radiuses):
            h1, h2, w1, w2 = (tile_idx // self.tiles_num_h) * self.tiles_size, ((tile_idx // self.tiles_num_h) + 1) * self.tiles_size, (tile_idx % self.tiles_num_w) * self.tiles_size, ((tile_idx % self.tiles_num_w) + 1) * self.tiles_size
            h2, w2 = min(h, h2), min(w, w2)
            tile = [box(w1, h1, w2, h2)]
            tree_gaussians = [Point(center).buffer(radius) for center, radius in zip(centers.cpu().detach().numpy(), radiuses.cpu().detach().numpy())]
            tree = STRtree(tree_gaussians)
            pairs = tree.query(tile, predicate="intersects")
            return pairs[1, :]
        eigenvalues, _ = torch.linalg.eig(cov_matrices)
        eigenvalues = eigenvalues.real
        max_eigenvalues, _ = eigenvalues.max(dim=1, keepdim=False)
        radiuses = torch.ceil(2 * torch.sqrt(max_eigenvalues)) # moze do zmiany na 3
        gaussians_for_tiles = {}
        for tile_idx in range(self.tiles_num_w * self.tiles_num_h):
            indices_list = in_tile(tile_idx, means_2d, radiuses)
            if len(indices_list) == 0:
                gaussians_for_tiles[tile_idx] = torch.tensor([], device=self.device)
            else:
                indices = torch.tensor(indices_list, device=self.device)
                gaussians_for_tiles[tile_idx] = indices[torch.argsort(z_coords[indices])]
        return gaussians_for_tiles

    def _blend_in_order(self, w, h, gaussians_for_tiles, means_2d, cov_matrices):
        image = torch.zeros((h, w, 3), device=self.device)
        for tile_idx, indices in gaussians_for_tiles.items():
            if indices.shape[0] == 0:
                continue
            means_2d_sorted = means_2d[indices]
            cov_matrices_sorted = cov_matrices[indices.cpu()].cuda()
            colors = self.colors[indices.cpu()].cuda()
            opacities = torch.sigmoid(self.opacities[indices.cpu()].cuda())
            multivariate_normal = torch.distributions.MultivariateNormal(means_2d_sorted, cov_matrices_sorted)
            h1, h2, w1, w2 = (tile_idx // self.tiles_num_h) * self.tiles_size, ((tile_idx // self.tiles_num_h) + 1) * self.tiles_size, (tile_idx % self.tiles_num_w) * self.tiles_size, ((tile_idx % self.tiles_num_w) + 1) * self.tiles_size
            h2, w2 = min(h, h2), min(w, w2)
            y, x = torch.meshgrid(
                torch.arange(h1, h2, device=self.device).float(),
                torch.arange(w1, w2, device=self.device).float(),
                indexing="ij"
            )
            tile_pixels = torch.stack((x, y), axis=-1)
            tile_pixels = torch.broadcast_to(tile_pixels, (opacities.shape[0], *tile_pixels.shape))
            alfas = opacities.transpose(-2, -1) * torch.exp(multivariate_normal.log_prob(tile_pixels.permute(1, 2, 0, 3)))
            colors = colors.transpose(-2, -1)[None, None, :, :]
            alfas = alfas[:, :, None, :]
            transmittance = torch.cumprod(torch.cat([torch.ones_like(alfas[:, :, :, 0:1], device=self.device), (1 - alfas)[:, :, :, :-1]], dim=-1), dim=-1)
            image[h1:h2, w1:w2, :] = torch.sum(colors * alfas * transmittance, dim=-1)
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
        x_c, y_c, z_c = means_camera
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
        cov_matrices_2d = cov_matrices_2d + 1e-3 * (cov_matrices_2d[:, 0, 0] + cov_matrices_2d[:, 1, 1])[:, None, None] * torch.eye(cov_matrices_2d.shape[-1], device=cov_matrices_2d.device) # eigenvalues should not be too small
        return cov_matrices_2d, cov_matrices_3d 

    def _create_covariance_matrices(self):
        quaternions = torch.nn.functional.normalize(self.quaternions, dim=-1)
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
        scaling_matrices = torch.diag_embed(torch.exp(self.scaling_vectors))
        return rotation_matrices @ scaling_matrices @ scaling_matrices.transpose(-2, -1) @ rotation_matrices.transpose(-2, -1)

    def _rasterize(self, h, w, extrinsic_matrix, focal_length, batch_idx):
        # self._cull_gaussians(extrinsic_matrix)
        means_2d, means_camera = self._project_means(w, h, extrinsic_matrix, focal_length)
        # self._if_uniform_3d_gaussians(self.positions)
        # self.debug_means(self.positions, means_2d, means_camera, extrinsic_matrix)
        cov_matrices, cov_matrices_3d = self._project_cov_matrices(means_camera, extrinsic_matrix, focal_length)
        # self.debug_covariances(self.positions, means_2d, means_camera, extrinsic_matrix, cov_matrices, cov_matrices_3d)
        gaussians_for_tiles = self._duplicate_with_keys(w, h, means_2d, cov_matrices, means_camera[2, :])
        # if batch_idx % 10 == 0:
        #     for i, (_, gaussian) in enumerate(gaussians_for_tiles.items()):
        #         self.debug_covariances_in_tiles(self.positions, means_2d, means_camera, extrinsic_matrix, cov_matrices, cov_matrices_3d, gaussian, i, batch_idx)
        image = self._blend_in_order(w, h, gaussians_for_tiles, means_2d, cov_matrices)
        return image

    def any_step(self, batch, batch_idx, mode):
        images, poses, focal_lengths, _ = batch

        image = images[0, ...]
        pose = poses[0, ...]
        focal_length = focal_lengths[0, ...]

        extrinsic_matrix = torch.linalg.inv(pose)

        # print(self.opacities)

        # if self.global_step % self.densification_interval == 0:
        #     self._densify_gaussians()
        output = self._rasterize(image.shape[0], image.shape[1], extrinsic_matrix, focal_length, batch_idx)
        output = output.permute(2, 0, 1)
        image = image.permute(2, 0, 1)
        loss = (1 - self._lambda) * self.l1_loss(output, image) + self._lambda * (1 - self.ssim(output.unsqueeze(0), image.unsqueeze(0))) / 2
        self.log(f"{mode}_loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        return loss, output, image

    def training_step(self, batch, batch_idx):
        # if batch_idx > 5:
        #     return
        # print(self.positions)
        loss, _, _ = self.any_step(batch, batch_idx, "train")
        
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx > 5:
            return
        loss, output, image = self.any_step(batch, batch_idx, "val")

        self.psnr.update(output, image)
        
        if batch_idx == 0:
            self.log_debug_samples(image, output, "val")

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch > 0 and self.global_step % self.densification_interval == 0:
            self._prune_gaussians()
            self._densify_gaussians()
        # if self.current_epoch > 0 and self.global_step % self.decreasing_alpha_interval == 0:
        #     self.opacities.data.fill_(inverse_sigmoid(torch.tensor(1e-6))) # TODO: wybrać
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def on_validation_epoch_end(self):
        psnr_value = self.psnr.compute()
        self.log("val_psnr", psnr_value, on_epoch=True, prog_bar=True)
    
    def log_debug_samples(self, img, pred, mode):
        img = img.detach().cpu()
        pred = pred.detach().cpu()

        pred = torch.clamp(pred, 0.0, 1.0)
        grid = torchvision.utils.make_grid([img, pred])

        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_image(f"{mode}_debug_samples", grid, self.current_epoch)

    def _if_uniform_3d_gaussians(self, positions):
        import matplotlib.pyplot as plt
        import os

        w, h = 400, 400
        focal_length = 300.0

        # --- plot ---------------------------------------------------------------
        fig = plt.figure(figsize=(11, 5))

        import numpy as np
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax3d.scatter(*zip(*positions.tolist()), s=60)
        for pos in positions.tolist():
            ax3d.text(*pos, "")
        # ax3d.plot(*zip(camera_pos, target), c="gray", linestyle="--", label="optical axis")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.set_title("3D scene")
        ax3d.legend()

        ax2d = fig.add_subplot(1, 2, 2)
        ax2d.add_patch(plt.Rectangle((0, 0), w, h, fill=False, edgecolor="black"))
        ax2d.set_xlim(-0.2 * w, 1.2 * w)
        ax2d.set_ylim(1.2 * h, -0.2 * h)  # image convention: y grows downward
        ax2d.set_xlabel("u (pixels)")
        ax2d.set_ylabel("v (pixels)")
        ax2d.set_title("2D projection (image plane)")
        ax2d.set_aspect("equal")

        fig.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", "means_3d.png")
        fig.savefig(out_path, dpi=150)
        print(f"saved figure to {out_path}")
        plt.show()

        print(positions)
            
    def debug_means(self, positions, means_2d, means_camera, extrinsic_matrix):

        import matplotlib.pyplot as plt
        import os
        colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]

        w, h = 100, 100
        focal_length = 300.0

        # --- run the function under test --------------------------------------
        means_2d = means_2d.cpu().detach().numpy()[:5]
        depths = means_camera[2].cpu().detach().numpy()[:5]  # camera-space z (depth) per point
        means_camera = means_camera.permute(1, 0).cpu().detach().numpy()[:5]

        # --- plot ---------------------------------------------------------------
        fig = plt.figure(figsize=(11, 5))

        R = extrinsic_matrix[:3, :3]
        T = extrinsic_matrix[:3, 3]
        camera_pos = -torch.matmul(torch.linalg.inv(R), T)
        camera_pos = camera_pos.cpu().detach().numpy()
        import numpy as np
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        positions = positions
        ax3d.scatter(*zip(*means_camera.tolist()), c=colors, s=60)
        for pos in means_camera.tolist():
            ax3d.text(*pos, "")
        ax3d.scatter(0, 0, 0, c="black", marker="^", s=100, label="camera")
        # ax3d.plot(*zip(camera_pos, target), c="gray", linestyle="--", label="optical axis")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.set_title("3D scene")
        ax3d.legend()

        ax2d = fig.add_subplot(1, 2, 2)
        ax2d.scatter(means_2d[:, 0], means_2d[:, 1], c=colors, s=60)
        ax2d.add_patch(plt.Rectangle((0, 0), w, h, fill=False, edgecolor="black"))
        ax2d.set_xlim(-0.2 * w, 1.2 * w)
        ax2d.set_ylim(1.2 * h, -0.2 * h)  # image convention: y grows downward
        ax2d.set_xlabel("u (pixels)")
        ax2d.set_ylabel("v (pixels)")
        ax2d.set_title("2D projection (image plane)")
        ax2d.set_aspect("equal")

        fig.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", "project_means.png")
        fig.savefig(out_path, dpi=150)
        print(f"saved figure to {out_path}")
        plt.show()

        print("camera-space depth (z_c) per gaussian:", depths)
        # print(F"COLORS: {colors}")
        print(f"Means positions in camera system: {means_camera}")
        print(f"Camera position in world system: {camera_pos}")
        print(f"Means positions in 2D: {means_2d}")


    def debug_covariances(self, positions, means_2d, means_camera, extrinsic_matrix, cov_matrices_2d, cov_matrices_3d):
            import matplotlib.pyplot as plt
            import os
            from scene_representation.model.gaussian_splatting.utils import ellipsoid_surface, ellipse_points
            colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    
            w, h = 100, 100
            focal_length = 300.0
    
            # --- run the function under test --------------------------------------
            means_2d = means_2d[:5]
            depths = means_camera[2].cpu().detach().numpy()[:5]  # camera-space z (depth) per point
            means_camera = means_camera.permute(1, 0).cpu().detach().numpy()[:5]
            cov_matrices_2d = cov_matrices_2d[:5]
            cov_matrices_3d = cov_matrices_3d[:5]

            R = extrinsic_matrix[:3, :3]
            T = extrinsic_matrix[:3, 3]
            camera_pos = -torch.matmul(torch.linalg.inv(R), T)
            camera_pos = camera_pos.cpu().detach().numpy()
            target = (0.0, 0.0, 0.0)
            # --- plot ---------------------------------------------------------------
            fig = plt.figure(figsize=(11, 5))

            ax3d = fig.add_subplot(1, 2, 1, projection="3d")
            for mean, cov, c in zip(positions, cov_matrices_3d, colors):
                X, Y, Z = ellipsoid_surface(mean, cov, n_std=2.0)
                ax3d.plot_wireframe(X, Y, Z, color=c, alpha=0.5, linewidth=0.5, rstride=2, cstride=2)
                ax3d.scatter(*mean.tolist(), c=c, s=20)
            ax3d.scatter(*camera_pos, c="black", marker="^", s=100, label="camera")
            ax3d.plot(*zip(camera_pos, target), c="gray", linestyle="--")
            ax3d.set_xlabel("x")
            ax3d.set_ylabel("y")
            ax3d.set_zlabel("z")
            ax3d.set_title("3D gaussians (2-std ellipsoids)")
    
    
            ax2d = fig.add_subplot(1, 2, 2)
            ax2d.add_patch(plt.Rectangle((0, 0), w, h, fill=False, edgecolor="black"))
            for mean2d, cov2d, c in zip(means_2d, cov_matrices_2d, colors):
                ex, ey = ellipse_points(mean2d, cov2d, n_std=2.0)
                ax2d.plot(ex, ey, color=c)
                ax2d.scatter(*mean2d.tolist(), c=c, s=20)
            ax2d.set_xlim(-0.2 * w, 1.2 * w)
            ax2d.set_ylim(1.2 * h, -0.2 * h)  # image convention: y grows downward
            ax2d.set_xlabel("u (pixels)")
            ax2d.set_ylabel("v (pixels)")
            ax2d.set_title("projected 2-std ellipses")
            ax2d.set_aspect("equal")
            ax2d.legend(fontsize=8, loc="upper right")
    
            fig.tight_layout()
            os.makedirs("outputs", exist_ok=True)
            out_path = os.path.join("outputs", "project_covariances.png")
            fig.savefig(out_path, dpi=150)
            print(f"saved figure to {out_path}")
            plt.show()
    
            # print("camera-space depth (z_c) per gaussian:", depths)
            # print(F"COLORS: {colors}")
            # print(f"Means positions in camera system: {means_camera}")
            # print(f"Camera position in world system: {camera_pos}")
            # print(f"Means positions in 2D: {means_2d}")

            print(cov_matrices_3d)
            print(cov_matrices_2d)
            print(self.scaling_vectors[:5])
    def debug_covariances_in_tiles(self, positions, means_2d, means_camera, extrinsic_matrix, cov_matrices_2d, cov_matrices_3d, indices, i, batch_idx):
        import matplotlib.pyplot as plt
        import os
        from scene_representation.model.gaussian_splatting.utils import ellipsoid_surface, ellipse_points
        # colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]

        w, h = 100, 100
        focal_length = 300.0

        # --- run the function under test --------------------------------------
        means_2d = means_2d[indices]
        # depths = means_camera[2].cpu().detach().numpy()[:5]  # camera-space z (depth) per point
        # means_camera = means_camera.permute(1, 0).cpu().detach().numpy()[:5]
        cov_matrices_2d = cov_matrices_2d[indices]
        cov_matrices_3d = cov_matrices_3d[indices]

        R = extrinsic_matrix[:3, :3]
        T = extrinsic_matrix[:3, 3]
        camera_pos = -torch.matmul(torch.linalg.inv(R), T)
        camera_pos = camera_pos.cpu().detach().numpy()
        target = (0.0, 0.0, 0.0)
        # --- plot ---------------------------------------------------------------
        fig = plt.figure(figsize=(11, 5))

        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        for mean, cov in zip(positions, cov_matrices_3d):
            X, Y, Z = ellipsoid_surface(mean, cov, n_std=2.0)
            ax3d.plot_wireframe(X, Y, Z, alpha=0.5, linewidth=0.5, rstride=2, cstride=2)
            ax3d.scatter(*mean.tolist(), s=20)
        ax3d.scatter(*camera_pos, c="black", marker="^", s=100, label="camera")
        ax3d.plot(*zip(camera_pos, target), c="gray", linestyle="--")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        ax3d.set_title("3D gaussians (2-std ellipsoids)")


        ax2d = fig.add_subplot(1, 2, 2)
        ax2d.add_patch(plt.Rectangle((0, 0), w, h, fill=False, edgecolor="black"))
        for mean2d, cov2d in zip(means_2d, cov_matrices_2d):
            ex, ey = ellipse_points(mean2d, cov2d, n_std=2.0)
            ax2d.plot(ex, ey)
            ax2d.scatter(*mean2d.tolist(), s=20)
        ax2d.set_xlim(-0.2 * w, 1.2 * w)
        ax2d.set_ylim(1.2 * h, -0.2 * h)  # image convention: y grows downward
        ax2d.set_xlabel("u (pixels)")
        ax2d.set_ylabel("v (pixels)")
        ax2d.set_title("projected 2-std ellipses")
        ax2d.set_aspect("equal")
        ax2d.legend(fontsize=8, loc="upper right")

        fig.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"project_covariances_tiles_{batch_idx}_{i}.png")
        fig.savefig(out_path, dpi=150)
        print(f"saved figure to {out_path}")
        plt.show()

        # print("camera-space depth (z_c) per gaussian:", depths)
        # print(F"COLORS: {colors}")
        # print(f"Means positions in camera system: {means_camera}")
        # print(f"Camera position in world system: {camera_pos}")
        # print(f"Means positions in 2D: {means_2d}")

        # print(cov_matrices_3d)
        # print(cov_matrices_2d)
        # print(self.scaling_vectors[:5])
