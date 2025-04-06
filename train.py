import os
import argparse
import torch
from torch.utils.data import DataLoader
from NowYouSeaMe import paired_rgb_depth_dataset, BackscatterNet, DeattenuateNet

# (Other training code, including your loss definitions and main training loop)

def train(args):
    # Set seed, device, etc.
    seed = int(torch.randint(9223372036854775807, (1,))[0]) if args.seed is None else args.seed
    if args.seed is None:
        print('Seed:', seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    train_dataset = paired_rgb_depth_dataset(args.images, args.depth, args.depth_16u, args.mask_max_depth, args.height, args.width, args.device)
    os.makedirs(args.output, exist_ok=True)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    bs_model = BackscatterNet().to(device=args.device)
    da_model = DeattenuateNet().to(device=args.device)
    
    # (Initialize loss functions, optimizers, and the training loop here)

    # After training, optionally save the model:
    if args.save is not None:
        save_path = os.path.join(args.output, f"{args.save}.pth")
        torch.save({
            'backscatter_model_state': bs_model.state_dict(),
            'deattenuate_model_state': da_model.state_dict(),
            # (Optionally, also save optimizer states and args)
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
    parser.add_argument('--mask_max_depth', action='store_true', help='If true will replace zeroes in depth files with max depth')
    parser.add_argument('--seed', type=int, default=None, help='Seed to initialize network weights')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing images')
    parser.add_argument('--save_intermediates', action='store_true', default=False, help='Set to True to save intermediate files')
    parser.add_argument('--init_iters', type=int, default=500, help='Number of iterations for the first batch')
    parser.add_argument('--iters', type=int, default=50, help='Number of iterations for subsequent batches')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save', type=str, default=None, help='Name to save the model checkpoint (e.g. 1.0)')

    args = parser.parse_args()
    train(args)