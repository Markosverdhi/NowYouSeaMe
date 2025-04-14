import logging 
from datetime import datetime
import os
# Logging config stuff
logs_folder = os.path.join(os.getcwd(), "logs")
os.makedires(logs_folder, exist_ok=True)
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


import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from time import time
from NowYouSeaMe import paired_rgb_depth_dataset, BackscatterNet, DeattenuateNet
from tqdm import trange
import torch.nn.functional as F

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
        args.images, args.depth, args.depth_16u, args.mask_max_depth, device
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
        logging.debug("Processing files: %s", fnames)
        image_batch = left  # shape: (B, C, H, W)
        batch_size = image_batch.shape[0]
        logging.info("Batch size: %d", batch_size)
        
        if j==0:
            for _ in trange(args.init_iters, desc="Backscatter Iterations"):
                start = time()
                with torch.cuda.amp.autocast():
                    direct = bs_model(depth)
                    # Using L1 loss on the rectified direct image as an example
                    bs_loss = bs_criterion(torch.relu(direct), torch.zeros_like(direct))
                # scaler.scale(bs_loss).backward() #Uncomment for use with AMP
                # scaler.step(bs_optimizer)
                # scaler.update()
                bs_loss.backward() #Uncomment for use without AMP
                bs_optimizer.step() #""
                bs_optimizer.zero_grad() #""
                iter_time = (time() - start)
                total_bs_time += iter_time
                total_bs_evals += batch_size
                logging.info("Iteration %d: Loss = %f, Iteration Time = %f sec", j, bs_loss.item(), iter_time)

                if bs_loss.item() == 0:
                    logging.info("Loss reached zero. skipping the rest of training.")
                    break
        else:
            for _ in trange(args.iters, desc="Backscatter Iterations"):
                start = time()
                with torch.cuda.amp.autocast():
                    direct = bs_model(depth)
                    # Using L1 loss on the rectified direct image as an example
                    bs_loss = bs_criterion(torch.relu(direct), torch.zeros_like(direct))
                # scaler.scale(bs_loss).backward() #Uncomment for use with AMP
                # scaler.step(bs_optimizer)
                # scaler.update()
                bs_loss.backward() #Uncomment for use without AMP
                bs_optimizer.step() #""
                bs_optimizer.zero_grad() #""
                iter_time = (time() - start)
                total_bs_time += iter_time
                total_bs_evals += batch_size
                logging.info("Iteration %d: Loss = %f, Iteration Time = %f sec", j, bs_loss.item(), iter_time)

                if bs_loss.item() == 0:
                    logging.info("Loss reached zero. skipping the rest of training.")
                    break
        
        direct_mean = direct.mean(dim=1, keepdim=True)
        direct_std = direct.std(dim=1, keepdim=True)
        direct_z = (direct - direct_mean) / direct_std
        clamped_z = torch.clamp(direct_z, -5, 5)
        threshold = torch.Tensor([1./255]).to(device)
        direct_no_grad = torch.clamp((clamped_z * direct_std) + torch.maximum(direct_mean, threshold), 0, 1).detach()
        logging.info("Statistics from last batch: direct_mean = %s, direct_std = %s", direct_mean.detach().cpu().numpy(), direct_std.detach().cpu().numpy())
        logging.info("Transitioning to deattenuation stage.")

        if j == 0:
            for i in trange(args.init_iters, desc="Deattenuate Iterations"):
                start = time()
                with torch.cuda.amp.autocast():
                    # Forward pass: produce the spatial attenuation map.
                    J = da_model(direct_no_grad, depth)
                    # Aggregate the global signal from the direct output:
                    # direct_no_grad: (B, 3)  --> average over channels --> (B, 1)
                    direct_global = direct_no_grad.mean(dim=1, keepdim=True)
                    # Aggregate the spatial map J: (B, 1, H, W) --> pool to (B, 1, 1, 1)
                    J_global = F.adaptive_avg_pool2d(J, (1, 1)).view(J.size(0), -1)
                    # Compute the loss (e.g., MSELoss) between these global descriptors.
                    da_loss = da_criterion(direct_global, J_global)
                # For AMP, you would use scaler.scale(da_loss).backward(), etc.
                # Here, we use standard backward:
                da_loss.backward()
                da_optimizer.step()
                da_optimizer.zero_grad()
                iter_time = time() - start
                total_da_time += iter_time
                total_da_evals += batch_size
                logging.info("Iteration %d (Batch %d): Loss = %f, Iteration Time = %f sec", i, j, da_loss.item(), iter_time)
                
                if da_loss.item() == 0:
                    logging.info("Loss reached zero. Skipping the rest of training in this batch.")
                    break
            else:
                for i in trange(args.iters, desc="Deattenuate Iterations"):
                    start = time()
                    with torch.cuda.amp.autocast():
                        J = da_model(direct_no_grad, depth)
                        direct_global = direct_no_grad.mean(dim=1, keepdim=True)
                        J_global = F.adaptive_avg_pool2d(J, (1, 1)).view(J.size(0), -1)
                        da_loss = da_criterion(direct_global, J_global)
                    # For AMP, uncomment these lines instead:
                    # scaler.scale(da_loss).backward()
                    # scaler.step(da_optimizer)
                    # scaler.update()
                    da_loss.backward()
                    da_optimizer.step()
                    da_optimizer.zero_grad()
                    iter_time = time() - start
                    total_da_time += iter_time
                    total_da_evals += batch_size
                    logging.info("Iteration %d (Batch %d): Loss = %f, Iteration Time = %f sec", i, j, da_loss.item(), iter_time)
                
                if da_loss.item() == 0:
                    logging.info("Loss reached zero. Skipping the rest of training in this batch.")
                    break
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
        # Remove or comment out the backscatter line if not needed:
        # backscatter_img = torch.clamp(backscatter, 0, 1).cpu()
        J_img = torch.clamp(J, 0, 1).cpu()

        for side in range(1):  # assuming single side here
            names = fnames[side]
            for n in range(batch_size):
                name = names[n].rstrip('.png')
                if args.save_intermediates:
                    save_image(direct_img[n], os.path.join(args.output, f"{name}-direct.png"))
                    # Remove this save if not needed:
                    # save_image(backscatter_img[n], os.path.join(args.output, f"{name}-backscatter.png"))
                    save_image(f_img[n], os.path.join(args.output, f"{name}-f.png"))
                save_image(J_img[n], os.path.join(args.output, f"{name}-corrected.png"))

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
