import logging 
from datetime import datetime
# Logging config stuff
log_filename = f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from time import time
from NowYouSeaMe import paired_rgb_depth_dataset, BackscatterNet, DeattenuateNet
from tqdm import trange

def main(args):
    # Set seed and device
    seed = int(torch.randint(9223372036854775807, (1,))[0]) if args.seed is None else args.seed
    if args.seed is None:
        print("Seed:", seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(args.device)
    
    # Create dataset and dataloader
    train_dataset = paired_rgb_depth_dataset(
        args.images, args.depth, args.depth_16u, args.mask_max_depth,
        args.height, args.width, device
    )
    os.makedirs(args.output, exist_ok=True)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
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

    # Create a GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler()
    
    total_bs_time = 0.0
    total_bs_evals = 0
    total_da_time = 0.0
    total_da_evals = 0

    for j, (left, depth, fnames) in enumerate(dataloader):
        print(f"Training batch {j}")
        logging.info("Starting training on batch %d with %d samples", j, batch_size)
        logging.debug("Processing files: %s", fnames)
        image_batch = left  # shape: (B, C, H, W)
        batch_size = image_batch.shape[0]
        logging.info("Batch size: %d", image_batch.shape[0])
        if j==0:
            for _ in trange(args.init_iters, desc="Backscatter Iterations"):
                start = time()
                with torch.cuda.amp.autocast():
                    direct = bs_model(depth)
                    # Using L1 loss on the rectified direct image as an example
                    bs_loss = bs_criterion(torch.relu(direct), torch.zeros_like(direct))
                scaler.scale(bs_loss).backward()
                scaler.step(bs_optimizer)
                scaler.update()
                iter_time = (time() - start)
                total_bs_time += iter_time
                total_bs_evals += batch_size
                logging.info("Iteration %d: Loss = %f, Iteration Time = %f sec", i, bs_loss.item(), iter_time)
        else:
            for _ in trange(args.iters, desc="Backscatter Iterations"):
                start = time()
                with torch.cuda.amp.autocast():
                    direct = bs_model(depth)
                    # Using L1 loss on the rectified direct image as an example
                    bs_loss = bs_criterion(torch.relu(direct), torch.zeros_like(direct))
                scaler.scale(bs_loss).backward()
                scaler.step(bs_optimizer)
                scaler.update()
                iter_time = (time() - start)
                total_bs_time += iter_time
                total_bs_evals += batch_size
                logging.info("Iteration %d: Loss = %f, Iteration Time = %f sec", i, bs_loss.item(), iter_time)
        
        direct_mean = direct.mean(dim=1, keepdim=True)
        direct_std = direct.std(dim=1, keepdim=True)
        direct_z = (direct - direct_mean) / direct_std
        clamped_z = torch.clamp(direct_z, -5, 5)
        threshold = torch.Tensor([1./255]).to(device)
        direct_no_grad = torch.clamp((clamped_z * direct_std) + torch.maximum(direct_mean, threshold), 0, 1).detach()
        logging.info("Statistics from last batch: direct_mean = %s, direct_std = %s", direct_mean.cpu().numpy(), direct_std.cpu().numpy())
        logging.info("Transitioning to deattenuation stage.")
        
        for _ in trange(args.init_iters if j == 0 else args.iters, desc="Deattenuate Iterations"):
            start = time()
            with torch.cuda.amp.autocast():
                f, J = da_model(direct_no_grad, depth)
                da_loss = da_criterion(direct_no_grad, J)
            scaler.scale(da_loss).backward()
            scaler.step(da_optimizer)
            scaler.update()
            total_da_time += (time() - start)
            total_da_evals += batch_size

        # Offload loss metrics to CPU for logging
        bs_loss_cpu = bs_loss.detach().cpu().item()
        da_loss_cpu = da_loss.detach().cpu().item()
        avg_bs_time = total_bs_time / total_bs_evals * 1000  
        avg_da_time = total_da_time / total_da_evals * 1000  
        avg_time = avg_bs_time + avg_da_time
        
        print(f"Losses: Backscatter: {bs_loss_cpu:.9f}, Deattenuation: {da_loss_cpu:.9f}")
        print(f"Avg time per eval: {avg_time:.6f} ms (BS: {avg_bs_time:.6f} ms, DA: {avg_da_time:.6f} ms)")
        
        # Save intermediate outputs if desired (moved to CPU)
        direct_img = torch.clamp(direct_no_grad, 0, 1).cpu()
        backscatter_img = torch.clamp(backscatter, 0, 1).cpu()
        f_img = f.cpu()
        f_img = f_img / f_img.max()  # Normalize for saving
        J_img = torch.clamp(J, 0, 1).cpu()
        
        for side in range(1):  # assuming single side here
            names = fnames[side]
            for n in range(batch_size):
                name = names[n].rstrip('.png')
                if args.save_intermediates:
                    save_image(direct_img[n], os.path.join(args.output, f"{name}-direct.png"))
                    save_image(backscatter_img[n], os.path.join(args.output, f"{name}-backscatter.png"))
                    save_image(f_img[n], os.path.join(args.output, f"{name}-f.png"))
                save_image(J_img[n], os.path.join(args.output, f"{name}-corrected.png"))

    if args.save is not None:
        bs_model_cpu = bs_model.cpu().state_dict() #offload saving to CPUs
        da_model_cpu = da_model.cpu().state_dict()
        save_path = os.path.join(args.output, f"{args.save}.pth")
        torch.save({
            'backscatter_model_state': bs_model_cpu,
            'deattenuate_model_state': da_model_cpu,
            'backscatter_optimizer_state': bs_optimizer.state_dict(),
            'deattenuate_optimizer_state': da_optimizer.state_dict(),
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
