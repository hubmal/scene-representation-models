import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image


class LegoDataset(Dataset):
    def __init__(self, root_dir, split="train", downsample_factor=4):
        self.root_dir = root_dir
        self.split = split
        self.cached_dir = os.path.join(self.root_dir, "cached", self.split)
        
        json_path = os.path.join(root_dir, f"transforms_{split}.json")
        with open(json_path, 'r') as f:
            self.meta = json.load(f)
            
        self.camera_angle_x = self.meta['camera_angle_x']
        self.frames = self.meta['frames']
        self.downsample_factor = downsample_factor

        file_count = len([
            f for f in os.listdir(self.cached_dir) if os.path.isfile(os.path.join(self.cached_dir, f))
        ]) if os.path.exists(self.cached_dir) else 0
        if not file_count == len(self.frames):
            self._cache_data()

    def _cache_data(self):
        os.makedirs(self.cached_dir, exist_ok=True)
        for frame in tqdm(self.frames, desc=f"Caching {self.split} data"):
            img_name = frame['file_path'].lstrip('./') + ".png"
            img_path = os.path.join(self.root_dir, img_name)
            
            image_rgba = Image.open(img_path).convert("RGBA")
            original_W, original_H = image_rgba.size
            
            if self.downsample_factor > 1:
                new_W = original_W // self.downsample_factor
                new_H = original_H // self.downsample_factor
                image_rgba = image_rgba.resize((new_W, new_H), Image.Resampling.LANCZOS)

            image_rgba = np.array(image_rgba) / 255.0
            pose = np.array(frame['transform_matrix'])
            
            H, W = image_rgba.shape[:2]
            focal_length = 0.5 * W / np.tan(0.5 * self.camera_angle_x)

            rgb = image_rgba[..., :3]
            alpha = image_rgba[..., 3:4]
            image = rgb * alpha + 1.0 * (1.0 - alpha)

            camera_direction_vectors_camera_coords = self._get_camera_direction_vectors(image, focal_length)
            camera_direction_vectors_camera_coords = np.transpose(camera_direction_vectors_camera_coords, (1, 0))
            camera_direction_vectors_world_coords = np.matmul(pose[:3, :3], camera_direction_vectors_camera_coords)
            camera_direction_vectors_world_coords = np.transpose(camera_direction_vectors_world_coords, (1, 0))

            np.savez(
                os.path.join(self.cached_dir, os.path.splitext(os.path.basename(frame['file_path']))[0]),
                image=image,
                pose=pose,
                focal_length=focal_length,
                camera_direction_vectors_world_coords=camera_direction_vectors_world_coords
            )

    def _get_camera_direction_vectors(self, image, focal_length):
        H, W = image.shape[:2]
        y, x = np.meshgrid(
            np.arange(H, dtype=np.float32),
            np.arange(W, dtype=np.float32),
            indexing="ij"
        )
        dirs_x = (x + 0.5 - W // 2) / focal_length
        dirs_y = -(y + 0.5 - H // 2) / focal_length
        dirs_z = -np.ones_like(dirs_x, dtype=np.float32)
        return np.stack((dirs_x, dirs_y, dirs_z), axis=-1).reshape(-1, 3)
    
    def __len__(self):
        return len(self.frames)

    # def __getitem__(self, idx):
    #     frame = self.frames[idx]

    #     img_name = frame['file_path'].lstrip('./') + ".png"
    #     img_path = os.path.join(self.root_dir, img_name)
        
    #     image = Image.open(img_path).convert("RGBA")
    #     original_W, original_H = image.size
        
    #     if self.downsample_factor > 1:
    #         new_W = original_W // self.downsample_factor
    #         new_H = original_H // self.downsample_factor
    #         image = image.resize((new_W, new_H), Image.Resampling.LANCZOS)

    #     image = self.transform(image)
    #     image = image.permute(1, 2, 0)
        
    #     pose = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        
    #     H, W = image.shape[:2]
    #     focal_length = 0.5 * W / torch.tan(torch.tensor(0.5 * self.camera_angle_x))
        
    #     return image, pose, focal_length

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.cached_dir, f"r_{idx}.npz"))
        image = torch.tensor(data["image"], dtype=torch.float32)
        pose = torch.tensor(data["pose"], dtype=torch.float32)
        focal_length = torch.tensor(data["focal_length"], dtype=torch.float32)
        camera_direction_vectors_world_coords = torch.tensor(data["camera_direction_vectors_world_coords"], dtype=torch.float32)
        return image, pose, focal_length, camera_direction_vectors_world_coords