import os
import argparse

import torch
from time import time
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.models as models
import kornia.morphology as morph
from PIL import Image
try:
    from tqdm import trange
except:
    trange = range

class paired_rgb_depth_dataset(Dataset):
    def __init__(self, image_path, depth_path, openni_depth, mask_max_depth, device,
                 patch_height=400, overlap=40):
        self.image_dir = image_path
        self.depth_dir = depth_path
        self.image_files = sorted(os.listdir(image_path))
        self.depth_files = sorted(os.listdir(depth_path))
        self.device = device
        self.openni_depth = openni_depth
        self.mask_max_depth = mask_max_depth
        self.patch_height = patch_height
        self.overlap = overlap

        self.to_tensor = transforms.Compose([
            transforms.PILToTensor(),  # keep full resolution
        ])

    def __len__(self):
        return len(self.image_files)

    def _split_vertical_patches(self, tensor):
        _, H, W = tensor.shape
        step = self.patch_height - self.overlap
        starts = list(range(0, H - self.patch_height + 1, step))

        patches = []
        for start in starts:
            end = start + self.patch_height
            patches.append((tensor[:, start:end, :], start, end))
        
        # Handle bottom edge if it doesn’t align perfectly
        if patches[-1][2] < H:
            patch = tensor[:, -self.patch_height:, :]
            patches.append((patch, H - self.patch_height, H))
        
        return patches

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        depth_path = os.path.join(self.depth_dir, self.depth_files[index])
        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path)

        image_tensor = self.to_tensor(image).float().to(self.device) / 255.0
        depth_tensor = self.to_tensor(depth).float().to(self.device)

        if self.openni_depth:
            depth_tensor = depth_tensor / 1000.0

        if self.mask_max_depth:
            depth_tensor[depth_tensor == 0.] = depth_tensor.max()

        image_patches = self._split_vertical_patches(image_tensor)
        depth_patches = self._split_vertical_patches(depth_tensor)

        patch_list = []
        for i, ((img_patch, start, end), (d_patch, _, _)) in enumerate(zip(image_patches, depth_patches)):
            patch_list.append((img_patch, d_patch, self.image_files[index], i, start, end))

        return patch_list



class BackscatterNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=False).features
        first_conv = mobilenet[0][0]
        mobilenet[0][0] = nn.Conv2d(
            in_channels=1,       
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
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
    def __init__(self, in_channels=4):
        """
        Constructs a deattenuation network that produces a full-resolution (spatial) 
        attenuation map.

        Args:
            in_channels (int): Number of input channels after concatenating the direct 
                               output (from BackscatterNet) and the depth map.
                               For example, if direct is a 3-channel image (expanded) 
                               and depth is 1 channel, then in_channels should be 4.
        """
        super().__init__()
        # Load MobileNetV2 features
        mobilenet = models.mobilenet_v2(pretrained=False).features
        # Adjust the first convolution to accept `in_channels` instead of the default 3.
        first_conv = mobilenet[0][0]
        mobilenet[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )
        self.features = mobilenet
        
        # Instead of global pooling + fully-connected layer, build a deconvolutional head
        # that maintains spatial information. This head converts the backbones' 
        # lower-resolution feature map into a full-resolution per-pixel attenuation map.
        self.deconv = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Constrains the output to [0, 1]
        )

    def forward(self, direct, depth):
        """
        Forward pass of the deattenuation network.

        Args:
            direct: A tensor from BackscatterNet. If direct is 2D (B, C) as a global 
                    descriptor, it will be expanded to (B, C, H, W) using the spatial 
                    dimensions of the depth tensor.
            depth:  The depth map tensor of shape (B, 1, H, W).

        Returns:
            attenuation_map: A per-pixel attenuation map of shape (B, 1, H, W)
        """
        # If direct is a global descriptor (e.g. shape: (B, C)), expand it to 4D.
        if direct.dim() == 2:
            B = direct.shape[0]
            H, W = depth.shape[2], depth.shape[3]
            direct = direct.unsqueeze(-1).unsqueeze(-1).expand(B, direct.shape[1], H, W)
        
        # Concatenate the expanded direct output and the depth map along the channel dimension.
        # For instance, if direct has shape (B, 3, H, W) and depth is (B, 1, H, W),
        # then x will have shape (B, 4, H, W).
        x = torch.cat([direct, depth], dim=1)
        
        # Process the combined input with MobileNetV2 features. Note that MobileNetV2 will
        # reduce the spatial resolution (typically by a factor ~32).
        x = self.features(x)
        
        # Upsample the feature map back to the original spatial size of the depth map.
        x = F.interpolate(x, size=depth.shape[2:], mode='bilinear', align_corners=False)
        
        # Use the deconvolutional head to produce a per-pixel attenuation map.
        attenuation_map = self.deconv(x)
        return attenuation_map


class BackscatterLoss(nn.Module):
    def __init__(self, table_size=256, cost_ratio=1000.0):
        super().__init__()
        # A learnable 1D lookup table
        self.table = nn.Parameter(torch.rand(table_size))
        self.cost_ratio = cost_ratio
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)

    def forward(self, image_batch, depth):
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
        # Quantize J to indices and retrieve corresponding lookup table values.
        # This encourages a smooth, discretized mapping.
        indices = (J * (self.table.numel() - 1)).long().clamp(0, self.table.numel() - 1)
        table_values = self.table[indices]  # Same shape as J.
        lookup_loss = ((J - table_values) ** 2).mean()

        # Saturation loss penalizes values that exceed the expected [0, 1] range.
        saturation_loss = ((F.relu(-J) + F.relu(J - 1)) ** 2).mean()

        # Intensity loss: ensure the average intensity per channel is close to a desired target.
        channel_intensities = J.mean(dim=[2, 3], keepdim=True)
        intensity_loss = ((channel_intensities - self.target_intensity) ** 2).mean()

        # Spatial variation loss: if direct is the expanded backscatter correction, its
        # spatial standard deviation (if any) is compared with that of J.
        # Note: if direct was originally global and then expanded, it might show low variation.
        spatial_variation_loss = self.mse(J.std(dim=[2, 3]), direct.std(dim=[2, 3]))

        total_loss = lookup_loss + saturation_loss + intensity_loss + spatial_variation_loss
        return total_loss
