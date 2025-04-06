import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from models import paired_rgb_depth_dataset, BackscatterNet, DeattenuateNet

def run(args):
    device = torch.device(args.device)
    dataset = paired_rgb_depth_dataset(args.images, args.depth, args.depth_16u, args.mask_max_depth, args.height, args.width, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    bs_model = BackscatterNet().to(device)
    da_model = DeattenuateNet().to(device)
    
    print(f"Loading model from {args.model} ...")
    checkpoint = torch.load(args.model, map_location=device)
    bs_model.load_state_dict(checkpoint['backscatter_model_state'])
    da_model.load_state_dict(checkpoint['deattenuate_model_state'])
    bs_model.eval()
    da_model.eval()
    
    os.makedirs(args.output, exist_ok=True)
    threshold = torch.Tensor([1./255]).to(device)
    
    with torch.no_grad():
        for image_batch, depth, fnames in tqdm(dataloader):
            direct, backscatter = bs_model(image_batch, depth)
            direct_mean = direct.mean(dim=[2, 3], keepdim=True)
            direct_std = direct.std(dim=[2, 3], keepdim=True)
            direct_z = (direct - direct_mean) / direct_std
            clamped_z = torch.clamp(direct_z, -5, 5)
            direct_no_grad = torch.clamp((clamped_z * direct_std) + torch.maximum(direct_mean, threshold), 0, 1).detach()
            f, J = da_model(direct_no_grad, depth)
            
            direct_img = torch.clamp(direct_no_grad, 0, 1).cpu()
            backscatter_img = torch.clamp(backscatter, 0, 1).cpu()
            f_img = f.cpu()
            J_img = torch.clamp(J, 0, 1).cpu()
            
            fnames = [name[0] for name in fnames]  # Flatten file names list
            for i, name in enumerate(fnames):
                base_name = os.path.splitext(name)[0]
                save_image(J_img[i], os.path.join(args.output, f"{base_name}-corrected.png"))
                save_image(backscatter_img[i], os.path.join(args.output, f"{base_name}-backscatter.png"))
                save_image(direct_img[i], os.path.join(args.output, f"{base_name}-direct.png"))
                save_image(f_img[i], os.path.join(args.output, f"{base_name}-f.png"))
                print(f"Saved outputs for {name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help='Path to the images folder')
    parser.add_argument('--depth', type=str, required=True, help='Path to the depth folder')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--height', type=int, default=1242, help='Height of the image and depth files')
    parser.add_argument('--width', type=int, default=1952, help='Width of the image and depth files')
    parser.add_argument('--depth_16u', action='store_true', help='True if depth images are 16-bit unsigned (millimetres), false if floating point (metres)')
    parser.add_argument('--mask_max_depth', action='store_true', help='If true, replace zeroes in depth files with max depth')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing images')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', type=str, required=True, help='Path to the saved model checkpoint (.pth)')
    
    args = parser.parse_args()
    run(args)