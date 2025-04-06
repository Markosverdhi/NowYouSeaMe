import os
import argparse

import torch
from time import time
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.models import models
import kornia.morphology as morph
from PIL import Image
try:
    from tqdm import trange
except:
    trange = range

class paired_rgb_depth_dataset(Dataset):
    def __init__(self, image_path, depth_path, openni_depth, mask_max_depth, image_height, image_width, device):
        self.image_dir = image_path
        self.depth_dir = depth_path
        self.image_files = sorted(os.listdir(image_path))
        self.depth_files = sorted(os.listdir(depth_path))
        self.device = device
        self.openni_depth = openni_depth
        self.mask_max_depth = mask_max_depth
        self.crop = (0, 0, image_height, image_width)
        self.depth_perc = 0.0001
        self.kernel = torch.ones(3, 3).to(device=device)
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.crop[2], self.crop[3]), transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.PILToTensor(),
        ])
        assert len(self.image_files) == len(self.depth_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        fname = self.image_files[index]
        image = Image.open(os.path.join(self.image_dir, fname))
        depth_fname = self.depth_files[index]
        depth = Image.open(os.path.join(self.depth_dir, depth_fname))
        depth_transformed: torch.Tensor = self.image_transforms(depth).float().to(device=self.device)
        if self.openni_depth:
            depth_transformed = depth_transformed / 1000.
        if self.mask_max_depth:
            depth_transformed[depth_transformed == 0.] = depth_transformed.max()
        low, high = torch.nanquantile(depth_transformed, self.depth_perc), torch.nanquantile(depth_transformed,
                                                                                             1. - self.depth_perc)
        depth_transformed[(depth_transformed < low) | (depth_transformed > high)] = 0.
        depth_transformed = torch.squeeze(morph.closing(torch.unsqueeze(depth_transformed, dim=0), self.kernel), dim=0)
        left_transformed: torch.Tensor = self.image_transforms(image).to(device=self.device) / 255.
        return left_transformed, depth_transformed, [fname]


class BackscatterNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=False).features
        self.features = mobilenet
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # MobileNetV2 outputs 1280 channels by default
        self.fc = nn.Linear(1280, 3)  # Predict 3 correction parameters
        self.sigmoid = nn.Sigmoid()

    def forward(self, depth):
        # Use depth as input (ensure it has the proper shape)
        x = self.features(depth)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return self.sigmoid(out)

class DeattenuateNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=False).features
        self.features = mobilenet
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # This example assumes you combine information from the direct image and depth
        self.fc = nn.Linear(1280, 1)  # Predict a single attenuation factor
        self.output_act = nn.Sigmoid()

    def forward(self, direct, depth):
        # Example: concatenate along the channel dimension (you might need a 1x1 conv to adjust channels)
        x = torch.cat([direct, depth], dim=1)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return self.output_act(out)


class BackscatterLoss(nn.Module):
    def __init__(self, table_size=256, cost_ratio=1000.0):
        super().__init__()
        # A learnable 1D lookup table
        self.table = nn.Parameter(torch.rand(table_size))
        self.cost_ratio = cost_ratio
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)

    def forward(self, direct):
        # Assume direct is normalized in [0, 1]. Quantize to get table indices.
        indices = (direct * (self.table.numel() - 1)).long().clamp(0, self.table.numel() - 1)
        # For a differentiable lookup, you might implement linear interpolation.
        # Here we simply index the table for illustration.
        table_values = self.table[indices]
        # Compute a lookup-based loss term.
        lookup_loss = ((direct - table_values)**2).mean()
        pos = self.l1(torch.relu(direct), torch.zeros_like(direct))
        neg = self.smooth_l1(torch.relu(-direct), torch.zeros_like(direct))
        return self.cost_ratio * neg + pos + lookup_loss

class DeattenuationLoss(nn.Module):
    def __init__(self, table_size=256, target_intensity=0.5):
        super().__init__()
        self.table = nn.Parameter(torch.rand(table_size))
        self.target_intensity = target_intensity
        self.mse = nn.MSELoss()

    def forward(self, direct, J):
        # Quantize J (assumed normalized) to index into the lookup table.
        indices = (J * (self.table.numel() - 1)).long().clamp(0, self.table.numel() - 1)
        table_values = self.table[indices]
        lookup_loss = ((J - table_values)**2).mean()
        saturation_loss = ((torch.relu(-J) + torch.relu(J - 1))**2).mean()
        channel_intensities = J.mean(dim=[2, 3], keepdim=True)
        intensity_loss = ((channel_intensities - self.target_intensity)**2).mean()
        spatial_variation_loss = self.mse(J.std(dim=[2, 3]), direct.std(dim=[2, 3]))
        return saturation_loss + intensity_loss + spatial_variation_loss + lookup_loss
