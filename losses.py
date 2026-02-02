# losses.py or added to train.py

import torch



def histogram_loss(generated_image, real_image, bins=256, device='cuda'):
    """
    Calculates the histogram loss (L1 distance) between the histograms of two images.
    
    Args:
        generated_image (Tensor): The image produced by the generator, shape (N, C, H, W).
        real_image (Tensor): The ground truth image, shape (N, C, H, W).
        bins (int): The number of bins for the histogram.
        device (str): The device on which to perform the calculation.
        
    Returns:
        Tensor: A scalar tensor representing the calculated loss.
    """
    # Clamp pixel values to the expected range for histogram calculation
    generated_img_flat = ((generated_image + 1) * 127.5).clamp(0, 255).view(-1)
    real_img_flat = ((real_image + 1) * 127.5).clamp(0, 255).view(-1)

    # --- CORRECTED FIX: Convert both HalfTensor to FloatTensor before calling torch.histc ---
    # Both generated_img_flat and real_img_flat must be converted.
    hist_gen = torch.histc(generated_img_flat.float(), bins=bins, min=0, max=255)
    hist_real = torch.histc(real_img_flat.float(), bins=bins, min=0, max=255)

    # Normalize the histograms to sum to 1 to make them comparable
    hist_gen_norm = hist_gen / hist_gen.sum()
    hist_real_norm = hist_real / hist_real.sum()

    # Calculate L1 distance between the normalized histograms
    loss = torch.sum(torch.abs(hist_gen_norm - hist_real_norm))
    
    return loss