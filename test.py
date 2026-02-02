import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import logging
import argparse
import numpy as np  
# Import your model and dataset classes
from generator import Generator
from datasets import MRIImageDataset
from metrics import denormalize # Assuming you have a denormalize function

# --- Setup Logging ---
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'inference.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

# --- Main Inference Function ---
def run_inference(model_path, field_strength, data_dir, output_dir):
    setup_logging(output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Model ---
    try:
        model_state = torch.load(model_path, map_location=device)
        G_T1_to_T2 = Generator().to(device)
        G_T1_to_T2.load_state_dict(model_state['G_T1_to_T2'])
        G_T1_to_T2.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        return

    # --- Setup Data and Metrics ---
    transform = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_t1_dir = os.path.join(data_dir, field_strength, 'T2')
    test_t2_dir = os.path.join(data_dir, field_strength, 'T1')

    test_dataset = MRIImageDataset(test_t1_dir, test_t2_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    num_samples = len(test_dataset)
    # Initialize metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device) # [-1, 1] range is 2.0
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    lpips_metric = lpips.LPIPS(net='alex').to(device) # Uses AlexNet as a perceptual feature extractor

    # --- Run Inference and Calculate Metrics ---
    psnr_values = []
    ssim_values = []
    lpips_values = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # if i >= num_samples:
            #     break
            
            real_T1 = batch["t1"].to(device)
            real_T2 = batch["t2"].to(device)

            # Generate fake T2 image
            fake_T2, _ = G_T1_to_T2(real_T1)
            
            # Prepare tensors for metrics: PSNR/SSIM need 1 channel, LPIPS needs 3
            # LPIPS expects images in [-1, 1] range, which matches your model's output
            # PSNR and SSIM are compatible with [-1, 1] as long as data_range is set correctly
            
            real_T2_rgb = real_T2.repeat(1, 3, 1, 1)
            fake_T2_rgb = fake_T2.repeat(1, 3, 1, 1)

            # Calculate metrics
            psnr_val = psnr_metric(fake_T2, real_T2)
            ssim_val = ssim_metric(fake_T2, real_T2)
            lpips_val = lpips_metric(fake_T2_rgb, real_T2_rgb)

            psnr_values.append(psnr_val.item())
            ssim_values.append(ssim_val.item())
            lpips_values.append(lpips_val.item())

            # --- Save Generated Images ---
            images_dir = os.path.join(output_dir, 'generated_images')
            os.makedirs(images_dir, exist_ok=True)
            
            # Denormalize for saving (from [-1, 1] to [0, 1])
            real_T1_denorm = denormalize(real_T1.cpu())
            real_T2_denorm = denormalize(real_T2.cpu())
            fake_T2_denorm = denormalize(fake_T2.cpu())

            save_image(real_T1_denorm, os.path.join(images_dir, f'sample_{i+1}_real_T1.png'))
            save_image(fake_T2_denorm, os.path.join(images_dir, f'sample_{i+1}_fake_T2.png'))
            save_image(real_T2_denorm, os.path.join(images_dir, f'sample_{i+1}_real_T2.png'))

            logging.info(f"Processed sample {i+1}/{num_samples}")

    # --- Display Results ---
    
    # The total number of processed samples is now the length of the lists
    processed_count = len(psnr_values)
    
    if processed_count == 0:
        logging.warning("No samples were processed.")
        return

    # Convert lists to numpy arrays for calculation
    psnr_array = np.array(psnr_values)
    ssim_array = np.array(ssim_values)
    lpips_array = np.array(lpips_values)

    # Calculate Average and Standard Deviation
    avg_psnr = np.mean(psnr_array)
    std_psnr = np.std(psnr_array)
    avg_ssim = np.mean(ssim_array)
    std_ssim = np.std(ssim_array)
    avg_lpips = np.mean(lpips_array)
    std_lpips = np.std(lpips_array)
    
    logging.info("\n--- Inference Results (Per Sample) ---")
    for i in range(processed_count):
        logging.info(f"Sample {i+1}: PSNR: {psnr_values[i]:.4f}, SSIM: {ssim_values[i]:.4f}, LPIPS: {lpips_values[i]:.4f}")
    
    logging.info("\n--- Average Metrics (Across All Samples) ---")
    logging.info(f"Total Samples Processed: {processed_count}")
    logging.info(f"Average PSNR: {avg_psnr:.4f} \u00b1 {std_psnr:.4f}")
    logging.info(f"Average SSIM: {avg_ssim:.4f} \u00b1 {std_ssim:.4f}")
    logging.info(f"Average LPIPS: {avg_lpips:.4f} \u00b1 {std_lpips:.4f}")
    
    logging.info(f"\nSaved generated images to: {os.path.abspath(images_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Script for CycleGAN MRI Image Translation")
    parser.add_argument('--model_path', type=str, default="best_model.pth",
                        help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--field_strength', type=str, default='3T',
                        help='Field strength (e.g., 1.5T or 3T)')
    parser.add_argument('--data_dir', type=str, default='/DATA/PMC_dataset/2D/test',
                        help='Directory containing the validation data (e.g., /DATA2/T1-T2-PMC/PMC dataset/2D/validation)')
    parser.add_argument('--output_dir', type=str, default='inference_output',
                        help='Directory to save generated images and logs')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate and evaluate')
    args = parser.parse_args()

    run_inference(args.model_path, args.field_strength, args.data_dir, args.output_dir)