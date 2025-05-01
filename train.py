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
    device = patch_outputs[0][0].device

    result = torch.zeros((3, full_height, width), device=device)
    count  = torch.zeros((3, full_height, width), device=device)

    for patch, start, end in patch_outputs:
        patch_height = patch.shape[1]
        expected_height = end - start

        if patch_height != expected_height:
            patch = patch[:, :expected_height, :]

        result[:, start:end, :] += patch
        count[:, start:end, :]  += 1
    count[count == 0] = 1
    return result / count

def collate_fn(batch):
    images, depths, filenames = zip(*batch)

    # Stack tensors normally
    image_batch = torch.stack(images)
    depth_batch = torch.stack(depths)
    return image_batch, depth_batch, list(filenames)

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
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn = collate_fn
    )
    logging.info("Training Dataset contains %d images", len(dataloader))

    bs_model = BackscatterNet().to(device)
    da_model = DeattenuateNet().to(device)
    bs_criterion = nn.L1Loss().to(device)
    da_criterion = nn.MSELoss().to(device)
    bs_optimizer = torch.optim.Adam(bs_model.parameters(), lr=args.init_lr)
    da_optimizer = torch.optim.Adam(da_model.parameters(), lr=args.init_lr)

    logging.info("Initializing BackscatterNet with architecture details: %s", bs_model)
    logging.info("Initializing DeattenuateNet with architecture details: %s", da_model)
    logging.info("Using Adam optimizer with learning rate %f", args.init_lr)

    scaler = torch.amp.GradScaler("cuda")

    total_bs_time = 0.0
    total_bs_evals = 0
    total_da_time = 0.0
    total_da_evals = 0

    for j, (rgb, depth, fname_list) in enumerate(dataloader):
        for i in range(rgb.size(0)):
            logging.info("Processing image %d/%d: %s", j + 1, len(dataloader), fname_list)

            rgb = rgb.to(device)
            depth = depth.to(device)
            depth = F.interpolate(depth, size=(args.height, args.width), mode='bilinear', align_corners=False)
            fname = fname_list[i]

            bs_iters = args.init_iters if j == 0 else args.iters
            for _ in trange(bs_iters, desc="Backscatter Iter"):
                t0 = time()
                with torch.amp.autocast('cuda'):
                    direct = bs_model(depth)
                    bs_loss = bs_criterion(direct, torch.zeros_like(direct))
                bs_loss.backward()
                bs_optimizer.step()
                bs_optimizer.zero_grad()
                total_bs_time += (time() - t0)
                total_bs_evals += 1
                if bs_loss.item() == 0:
                    break

            direct_no_grad = torch.clamp(direct.detach(), 0, 1)

            da_iters = args.init_iters if j == 0 else args.iters
            for _ in trange(da_iters, desc="Deattenuate Iter"):
                t1 = time()
                with torch.amp.autocast('cuda'):
                    J = da_model(direct_no_grad, depth)
                    direct_glob = F.adaptive_avg_pool2d(direct_no_grad, (1, 1)).view(J.size(0), -1)
                    J_glob = F.adaptive_avg_pool2d(J, (1, 1)).view(J.size(0), -1)
                    da_loss = da_criterion(direct_glob, J_glob)
                    J_glob = torch.clamp(J_glob, 0, 1)
                    direct_glob = torch.clamp(direct_glob, 0, 1)

                    if torch.isnan(direct_glob).any() or torch.isnan(J_glob).any():
                        logging.warning("NaNs detected in inputs to DA loss")
                        torch.cuda.empty_cache()
                        continue

                if torch.isnan(da_loss):
                    logging.error("NaN detected in da_loss - skipping backward pass")
                else:
                    scaler.scale(da_loss).backward()

                torch.nn.utils.clip_grad_norm_(da_model.parameters(), 1.0)

                scaler.step(da_optimizer)
                scaler.update()
                da_optimizer.zero_grad()
                if da_loss.item() == 0:
                    break

        base_name = fname.rstrip('.png')
        if args.save_intermediates:
            save_image(direct_no_grad.squeeze(0).cpu(), os.path.join(args.output, f"{base_name}-direct.png"))
        save_image(J.squeeze(0).cpu(), os.path.join(args.output, f"{base_name}-corrected.png"))

    if args.save is not None:
        bs_model_cpu_state = bs_model.cpu().state_dict()
        da_model_cpu_state = da_model.cpu().state_dict()
        save_path = os.path.join(args.output, f"{args.save}.pth")
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
