import logging 
from datetime import datetime
import os
# Logging config stuff
logs = os.path.join(os.getcwd(), "logs")
os.makedirs(logs, exist_ok=True)
log_filename = os.path.join(logs, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

logging.info("Training started")


import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from time import time
from NowYouSeaMe import paired_rgb_depth_dataset, BackscatterNet, DeattenuateNet
from tqdm import trange
import torch.nn.functional as F

def stitch_patches(patch_outputs, full_height, width):
    """
    Stitch overlapping vertical patches by averaging overlaps.
    Args:
        patch_outputs: List of (patch_tensor[C, H, W], start, end)
        full_height: int, full image height
        width: int, full image width
    Returns:
        Tensor[C, full_height, width]
    """
    result = torch.zeros((3, full_height, width))
    count  = torch.zeros((1, full_height, width))

    for patch, start, end in patch_outputs:
        result[:, start:end, :] += patch
        count[:, start:end, :]  += 1

    return result / count

def main(args):
    seed = int(torch.randint(9223372036854775807, (1,))[0]) if args.seed is None else args.seed
    if args.seed is None:
        print("Seed:", seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(args.device)
    
    train_dataset = paired_rgb_depth_dataset(
        args.images, args.depth, args.depth_16u, args.mask_max_depth, device
    )
    os.makedirs(args.output, exist_ok=True)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x[0])
    logging.info("Training Dataset contains %d batches", len(dataloader))

    bs_model = BackscatterNet().to(device)
    da_model = DeattenuateNet().to(device)
    bs_criterion = nn.L1Loss().to(device) 
    da_criterion = nn.MSELoss().to(device)
    bs_optimizer = torch.optim.Adam(bs_model.parameters(), lr=args.init_lr)
    da_optimizer = torch.optim.Adam(da_model.parameters(), lr=args.init_lr)
    
    logging.info("Initializing BackscatterNet with architecture details: %s", bs_model)
    logging.info("Initializing DeattenuateNet with architecture details: %s", da_model)
    logging.info("Using Adam optimizer with learning rate %f", args.init_lr)

    scaler = torch.cuda.amp.GradScaler()
    
    total_bs_time = 0.0
    total_bs_evals = 0
    total_da_time = 0.0
    total_da_evals = 0

    for j, patch_list in enumerate(dataloader):
        logging.info("Processing image %d/%d: %s", j+1, len(dataloader), patch_list[0][2])
        patch_outputs_direct = []
        patch_outputs_J      = []

        # Training loop per patch
        for img_patch, depth_patch, fname, patch_idx, start, end in patch_list:
            rgb   = img_patch.unsqueeze(0).to(device)
            depth = depth_patch.unsqueeze(0).to(device)

            # Backscatter training on this patch
            bs_iters = args.init_iters if j == 0 else args.iters
            for _ in trange(bs_iters, desc="Backscatter Iter"):
                t0 = time()
                with torch.cuda.amp.autocast():
                    direct   = bs_model(depth)
                    bs_loss  = bs_criterion(torch.relu(direct), torch.zeros_like(direct))
                bs_loss.backward()
                bs_optimizer.step()
                bs_optimizer.zero_grad()
                total_bs_time  += (time() - t0)
                total_bs_evals += 1
                if bs_loss.item() == 0:
                    break

            # normalize + clamp for next stage
            eps = 1e-5
            direct_mean = direct.mean(dim=1, keepdim=True)
            direct_std  = direct.std(dim=1, keepdim=True, unbiased=False).clamp(min=eps)
            direct_z     = (direct - direct_mean) / direct_std
            clamped_z   = torch.clamp(direct_z, -5, 5)
            thr         = torch.tensor([1.0/255]).to(device)
            direct_no_grad = torch.clamp(clamped_z * direct_std + torch.maximum(direct_mean, thr), 0, 1).detach()

            patch_outputs_direct.append((direct_no_grad.squeeze(0).cpu(), start, end))

            # Deattenuation training on this patch
            da_iters = args.init_iters if j == 0 else args.iters
            for _ in trange(da_iters, desc="Deattenuate Iter"):
                t1 = time()
                with torch.cuda.amp.autocast():
                    J           = da_model(direct_no_grad, depth)
                    direct_glob = direct_no_grad.mean(dim=1, keepdim=True)
                    J_glob      = F.adaptive_avg_pool2d(J, (1,1)).view(J.size(0), -1)
                    da_loss     = da_criterion(direct_glob, J_glob)
                da_loss.backward()
                da_optimizer.step()
                da_optimizer.zero_grad()
                total_da_time  += (time() - t1)
                total_da_evals += 1
                if da_loss.item() == 0:
                    break

            patch_outputs_J.append((J.squeeze(0).cpu(), start, end))

        # stitch & save
        full_direct = stitch_patches(patch_outputs_direct, full_height=args.height, width=args.width)
        full_J      = stitch_patches(patch_outputs_J,      full_height=args.height, width=args.width)

        base_name = patch_list[0][2].rstrip('.png')
        if args.save_intermediates:
            save_image(full_direct, os.path.join(args.output, f"{base_name}-direct.png"))
        save_image(full_J, os.path.join(args.output, f"{base_name}-corrected.png"))


    if args.save is not None:
        # Move models to CPU and retrieve their state dictionaries.
        bs_model_cpu_state = bs_model.cpu().state_dict()
        da_model_cpu_state = da_model.cpu().state_dict()
        
        # Define the checkpoint save path.
        save_path = os.path.join(args.output, f"{args.save}.pth")
        
        # Save only the model state dictionaries and the args (for reproducibility).
        torch.save({
            'backscatter_model_state_dict': bs_model_cpu_state,
            'deattenuate_model_state_dict': da_model_cpu_state,
            'args': vars(args)
        }, save_path)
        
        print(f"Models saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help='Path to the images folder')
    parser.add_argument('--depth', type=str, required=True, help='Path to the depth folder')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--height', type=int, default=1242, help='Height of the image and depth files')
    parser.add_argument('--width', type=int, default=1952, help='Width of the image and depth')
    parser.add_argument('--depth_16u', action='store_true', help='True if depth images are 16-bit unsigned (millimetres), false if floating point (metres)')
    parser.add_argument('--mask_max_depth', action='store_true', help='If true, replace zeroes in depth files with max depth')
    parser.add_argument('--seed', type=int, default=None, help='Seed to initialize network weights')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing images')
    parser.add_argument('--save_intermediates', action='store_true', default=False, help='Set to True to save intermediate files')
    parser.add_argument('--init_iters', type=int, default=500, help='Iterations for the first image batch')
    parser.add_argument('--iters', type=int, default=50, help='Iterations for subsequent image batches')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save', type=str, default=None, help='Name to save the model checkpoint (e.g. 1.0)')
    
    args = parser.parse_args()
    main(args)
