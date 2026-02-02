import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def denormalize(img):
    """
    Denormalizes an image from [-1, 1] to [0, 1].
    """
    img = img * 0.5 + 0.5
    return img.clamp(0, 1)

def calculate_psnr(img1, img2):
    """
    Calculates PSNR between two images (assumed to be normalized to [-1, 1]).
    
    Args:
        img1 (Tensor): First image.
        img2 (Tensor): Second image.
        
    Returns:
        float: Average PSNR over the batch.
    """
    img1 = denormalize(img1)
    img2 = denormalize(img2)
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.mean().item()

def calculate_ssim(img1, img2):
    """
    Calculates SSIM between two images (assumed to be normalized to [-1, 1]).
    
    Args:
        img1 (Tensor): First image.
        img2 (Tensor): Second image.
        
    Returns:
        float: Average SSIM over the batch.
    """
    img1 = denormalize(img1)
    img2 = denormalize(img2)
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    ssim_values = []
    batch_size = img1_np.shape[0]
    for i in range(batch_size):
        img1_i = np.squeeze(img1_np[i])
        img2_i = np.squeeze(img2_np[i])
        ssim_value = ssim(img1_i, img2_i, data_range=1.0)
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)
