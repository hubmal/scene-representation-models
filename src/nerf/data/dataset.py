import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class LegoDataset(Dataset):
    def __init__(self, root_dir, split="train", downsample_factor=4):
        self.root_dir = root_dir
        self.split = split
        
        json_path = os.path.join(root_dir, f"transforms_{split}.json")
        with open(json_path, 'r') as f:
            self.meta = json.load(f)
            
        self.camera_angle_x = self.meta['camera_angle_x']
        self.frames = self.meta['frames']
        self.downsample_factor = downsample_factor
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]

        img_name = frame['file_path'].lstrip('./') + ".png"
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert("RGBA")
        original_W, original_H = image.size
        
        if self.downsample_factor > 1:
            new_W = original_W // self.downsample_factor
            new_H = original_H // self.downsample_factor
            image = image.resize((new_W, new_H), Image.Resampling.LANCZOS)

        image = self.transform(image)
        image = image.permute(1, 2, 0)
        
        pose = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        
        H, W = image.shape[:2]
        focal_length = 0.5 * W / torch.tan(torch.tensor(0.5 * self.camera_angle_x))
        
        return image, pose, focal_length