import os
import argparse
import torch
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms as T
from NowYouSeaMe import BackscatterNet, DeattenuateNet

def split_into_vertical_patches(tensor, patch_height, overlap):
    """
    tensor: Tensor[C, H, W]
    returns list of (patch_tensor[C, patch_H, W], start, end)
    """
    C, H, W = tensor.shape
    step = patch_height - overlap
    starts = list(range(0, H - patch_height + 1, step))
    patches = []
    for start in starts:
        end = start + patch_height
        patches.append((tensor[:, start:end, :].clone(), start, end))
    # Handle bottom‐edge remainder
    if not patches or patches[-1][2] < H:
        start = H - patch_height
        patches.append((tensor[:, start:H, :].clone(), start, H))
    return patches

def stitch_patches(patch_outputs, full_height, width):
    """
    patch_outputs: list of (patch_tensor[C, h_i, W], start_i, end_i)
    returns Tensor[C, full_height, width]
    """
    C = patch_outputs[0][0].shape[0]
    result = torch.zeros((C, full_height, width))
    count  = torch.zeros((1, full_height, width))
    for patch, start, end in patch_outputs:
        result[:, start:end, :] += patch
        count[:, start:end, :]  += 1
    return result / count

def load_rgb_depth_pair(rgb_path, depth_path, depth_16u, mask_max_depth, device):
    to_tensor = T.Compose([ T.PILToTensor() ])
    rgb   = Image.open(rgb_path).convert('RGB')
    depth = Image.open(depth_path)
    rgb_t   = to_tensor(rgb).float().to(device) / 255.0
    depth_t = to_tensor(depth).float().to(device)
    if depth_16u:
        depth_t /= 1000.0
    if mask_max_depth:
        depth_t[depth_t == 0.] = depth_t.max()
    return rgb_t, depth_t

def visualize_depth(depth_tensor):
    valid = depth_tensor > 0
    if not torch.any(valid):
        return torch.zeros_like(depth_tensor)
    mi = depth_tensor[valid].min()
    ma = depth_tensor[valid].max()
    return (depth_tensor - mi) / (ma - mi + 1e-8)

def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    # 1) load and save inputs
    rgb, depth = load_rgb_depth_pair(
        args.image, args.depth, args.depth_16u, args.mask_max_depth, device
    )
    save_image(rgb.unsqueeze(0), os.path.join(args.output, "original_rgb.png"))
    save_image(visualize_depth(depth).unsqueeze(0), os.path.join(args.output, "depth_visual.png"))

    # 2) load models
    checkpoint = torch.load(args.checkpoint, map_location=device)
    bs_model = BackscatterNet().to(device)
    da_model = DeattenuateNet().to(device)
    bs_model.load_state_dict(checkpoint['backscatter_model_state_dict'])
    da_model.load_state_dict(checkpoint['deattenuate_model_state_dict'])
    bs_model.eval()
    da_model.eval()

    H, W = depth.shape[-2], depth.shape[-1]
    patch_h = args.patch_height
    overlap = args.overlap

    with torch.no_grad():
        # --- Backscatter (global) ---
        direct_vec = bs_model(depth.unsqueeze(0))     # shape (1,3)
        direct_full = direct_vec.view(1,3,1,1).expand(1,3,H,W)
        save_image(torch.clamp(direct_full[0],0,1), os.path.join(args.output, "direct.png"))

        # --- Z-score + clamp (full) ---
        eps = 1e-5
        mean_full = direct_full.mean(dim=1, keepdim=True)
        std_full  = direct_full.std(dim=1, keepdim=True).clamp(min=eps)
        z_full    = (direct_full - mean_full)/std_full
        clamped_z = torch.clamp(z_full, -5, 5)
        save_image(torch.clamp(clamped_z[0],0,1), os.path.join(args.output, "clamped_z.png"))

        # --- final direct_no_grad (full) ---
        thr = torch.tensor([1.0/255], device=device)
        direct_no_grad = torch.clamp(clamped_z * std_full + torch.maximum(mean_full, thr), 0, 1)
        save_image(direct_no_grad[0], os.path.join(args.output, "direct_no_grad.png"))

        # --- Deattenuation via patches ---
        depth_patches = split_into_vertical_patches(depth,     patch_h, overlap)
        direct_patches= split_into_vertical_patches(direct_no_grad[0], patch_h, overlap)

        J_patches = []
        for (d_patch, start, end), (dir_patch, _, _) in zip(depth_patches, direct_patches):
            dp = d_patch.unsqueeze(0).unsqueeze(0)        # [1,1,patch_h,W]
            dr = dir_patch.unsqueeze(0)                  # [1,3,patch_h,W]
            Jp = da_model(dr, dp)                        # [1,1,patch_h,W]
            J_patches.append((Jp.squeeze(0).cpu(), start, end))

        # --- Stitch & save final result ---
        full_J = stitch_patches(J_patches, full_height=H, width=W)
        save_image(torch.clamp(full_J,0,1), os.path.join(args.output, "final_corrected.png"))

        print("✅ Debug outputs saved in", args.output)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=str, required=True, help='RGB file path')
    p.add_argument('--depth', type=str, required=True, help='Depth file path')
    p.add_argument('--checkpoint', type=str, required=True, help='Model .pth checkpoint')
    p.add_argument('--output', type=str, required=True, help='Where to save debug images')
    p.add_argument('--depth_16u', action='store_true', help='Convert mm→m for depth')
    p.add_argument('--mask_max_depth', action='store_true', help='Replace 0 with max depth')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--patch_height', type=int, default=400, help='Height of each vertical patch')
    p.add_argument('--overlap',      type=int, default=40, help='Overlap between patches')
    args = p.parse_args()
    main(args)
