import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from NowYouSeaMe import paired_rgb_depth_dataset, BackscatterNet, DeattenuateNet

def main(args):
    # Set device (GPU if available, else CPU)
    device = torch.device(args.device)
    
    # Create the test dataset and data loader.
    # The dataset is assumed to load pairs of (image, depth, filename),
    # even if only the depth is used during processing.
    test_dataset = paired_rgb_depth_dataset(
        args.images, args.depth, args.depth_16u, args.mask_max_depth, device
    )
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load the saved checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Recreate the models and load their saved states.
    bs_model = BackscatterNet().to(device)
    da_model = DeattenuateNet().to(device)
    bs_model.load_state_dict(checkpoint['backscatter_model_state_dict'])
    da_model.load_state_dict(checkpoint['deattenuate_model_state_dict'])
    
    # Set models to evaluation mode.
    bs_model.eval()
    da_model.eval()
    
    # Ensure the output folder exists.
    os.makedirs(args.output, exist_ok=True)
    
    # Process each batch in the test dataset.
    with torch.no_grad():
        for i, (left, depth, fnames) in enumerate(dataloader):
            print(f"Processing batch {i}...")
            # Even though depth maps are the main input for training,
            # we follow the same pipeline as in training:
            # 1. Compute the backscatter-corrected output via BackscatterNet.
            direct = bs_model(depth)
            
            # 2. Post-process to compute a "global" corrected output.
            #    (This is the same as in your training loop: compute mean, std, z-scores, etc.)
            direct_mean = direct.mean(dim=1, keepdim=True)
            direct_std = direct.std(dim=1, keepdim=True)
            direct_z = (direct - direct_mean) / direct_std
            clamped_z = torch.clamp(direct_z, -5, 5)
            threshold = torch.Tensor([1. / 255]).to(device)
            # Compute a corrected version which we call direct_no_grad
            direct_no_grad = torch.clamp((clamped_z * direct_std) + torch.maximum(direct_mean, threshold), 0, 1).detach()
            
            # 3. Run the DeattenuateNet to obtain the full spatial attenuation map.
            #    (Assuming DeattenuateNet is designed to return a tensor of shape (B,1,H,W))
            J = da_model(direct_no_grad, depth)
            
            # 4. Save the output corrected images.
            # We'll use the filenames provided by the dataset; here we remove the ".png" extension.
            for b in range(left.size(0)):
                file_name = fnames[b].rstrip('.png')
                save_path = os.path.join(args.output, f"{file_name}-corrected.png")
                # Clamp values to [0, 1] and save.
                save_image(torch.clamp(J[b], 0, 1).cpu(), save_path)
                print(f"Saved {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Input folders for images and depth maps.
    parser.add_argument('--images', type=str, required=True, help='Path to the images folder')
    parser.add_argument('--depth', type=str, required=True, help='Path to the depth folder')
    
    # Checkpoint file containing the trained model states.
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint file (e.g. 1.0.pth)')
    
    # Output folder where corrected images will be saved.
    parser.add_argument('--output', type=str, required=True, help='Path to output folder for corrected images')
    
    # Other settings.
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference (usually 1 for low latency)')
    parser.add_argument('--depth_16u', action='store_true', help='True if depth images are 16-bit unsigned (millimetres)')
    parser.add_argument('--mask_max_depth', action='store_true', help='Replace zero values in depth with max depth')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    main(args)