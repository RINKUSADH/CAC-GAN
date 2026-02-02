import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import numpy as np
from losses import histogram_loss

# Custom modules for metrics and visualization
from metrics import calculate_psnr, calculate_ssim
from visualization import save_images, plot_losses_metrics


def calculate_combined_score(psnr, ssim, lpips):
    """Calculates a single score from three metrics."""
    
    # --- Step 1: Define Normalization Ranges and Weights ---
    # These are hyperparameters you can tune.
    TARGET_RANGES = {
        'psnr': {'min': 20.0, 'max': 35.0},
        'ssim': {'min': 0.7, 'max': 0.95},
        'lpips': {'min': 0.5, 'max': 0.95} # Corresponds to original LPIPS of 0.5 down to 0.05
    }
    WEIGHTS = {
        'psnr': 0.25,  # Pixel-level accuracy
        'ssim': 0.40,  # Structural similarity
        'lpips': 0.35  # Perceptual similarity
    }
    
    # --- Step 2: Invert LPIPS so higher is better ---
    perceptual_score = 1.0 - lpips

    # --- Step 3: Normalize all metrics to a [0, 1] scale ---
    def normalize(value, v_min, v_max):
        value = max(min(value, v_max), v_min)
        return (value - v_min) / (v_max - v_min)

    norm_psnr = normalize(psnr, TARGET_RANGES['psnr']['min'], TARGET_RANGES['psnr']['max'])
    norm_ssim = normalize(ssim, TARGET_RANGES['ssim']['min'], TARGET_RANGES['ssim']['max'])
    norm_lpips = normalize(perceptual_score, TARGET_RANGES['lpips']['min'], TARGET_RANGES['lpips']['max'])
    
    # --- Step 4: Calculate the final weighted score ---
    combined_score = (
        WEIGHTS['psnr'] * norm_psnr +
        WEIGHTS['ssim'] * norm_ssim +
        WEIGHTS['lpips'] * norm_lpips
    )
    
    return combined_score


def weights_init(m):
    """Initialize weights using He (Kaiming) initialization suitable for grayscale MRI images."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def initialize_loss_optimizers(G_T1_to_T2, G_T2_to_T1, D_T1, D_T2, lr_G, lr_D):
    """
    Initializes loss functions and optimizers.
    """
    # Basic LSGAN loss for adversarial training
    criterion_histogram = nn.L1Loss()
    criterion_GAN = nn.MSELoss()

    # Cycle-consistency and identity losses (L1 loss)
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Feature matching: L1 loss
    criterion_feature_matching = nn.L1Loss()

    optimizer_G = optim.Adam(
        list(G_T1_to_T2.parameters()) + list(G_T2_to_T1.parameters()),
        lr=lr_G, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        list(D_T1.parameters()) + list(D_T2.parameters()),
        lr=lr_D, betas=(0.5, 0.999)
    )

    # Learning rate schedulers (example: linear decay after epoch 200)
    scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    scheduler_D = optim.lr_scheduler.LambdaLR(
        optimizer_D, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )

    return (criterion_GAN, criterion_cycle, criterion_identity,
            criterion_feature_matching, criterion_histogram, # Add this
            optimizer_G, optimizer_D, scheduler_G, scheduler_D)


def validate(val_loader, 
             G_T1_to_T2, G_T2_to_T1, 
             D_T1, D_T2,
             criterion_GAN, criterion_cycle, criterion_identity, criterion_feature_matching, criterion_histogram, # Add this
             lambda_dict, device, images_dir, epoch, input_modality, target_modality):
    G_T1_to_T2.eval()
    G_T2_to_T1.eval()
    D_T1.eval()
    D_T2.eval()

    total_G_loss_val = 0.0
    total_D_loss_val = 0.0

    total_psnr_T1_val = 0.0
    total_ssim_T1_val = 0.0
    total_psnr_T2_val = 0.0
    total_ssim_T2_val = 0.0

    num_batches_val = 0

    lambda_feature_matching = lambda_dict.get('lambda_feature_matching', 10.0)
    lambda_histogram = lambda_dict.get('lambda_histogram', 1.0) # Add this
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if batch is None:
                continue

            real_T1 = batch[input_modality].to(device)
            real_T2 = batch[target_modality].to(device)

            # Generate images
            fake_T2, _ = G_T1_to_T2(real_T1)
            fake_T1, _ = G_T2_to_T1(fake_T2)

            # Compute cycle loss for T1 (T1→T2→T1) and direct reconstruction loss for T2
            loss_cycle_T1 = criterion_cycle(fake_T1, real_T1) * lambda_dict['lambda_cycle']
            loss_rec_T2 = criterion_cycle(fake_T2, real_T2) * lambda_dict['lambda_rec']

            # Identity losses for structure preservation
            idt_T1, _ = G_T2_to_T1(real_T1)
            loss_idt_T1 = criterion_identity(idt_T1, real_T1) * lambda_dict['lambda_identity']

            idt_T2, _ = G_T1_to_T2(real_T2)
            loss_idt_T2 = criterion_identity(idt_T2, real_T2) * lambda_dict['lambda_identity']
            loss_identity = loss_idt_T1 + loss_idt_T2

            # GAN losses
            pred_fake_T2, fake_features_T2 = D_T2(fake_T2)
            loss_GAN_T1_to_T2 = criterion_GAN(
                pred_fake_T2, torch.ones_like(pred_fake_T2, device=device)
            ) * lambda_dict['lambda_GAN']

            pred_fake_T1, fake_features_T1 = D_T1(fake_T1)
            loss_GAN_T2_to_T1 = criterion_GAN(
                pred_fake_T1, torch.ones_like(pred_fake_T1, device=device)
            ) * lambda_dict['lambda_GAN']

            loss_G = loss_GAN_T1_to_T2 + loss_GAN_T2_to_T1 + loss_cycle_T1 + loss_identity + loss_rec_T2

            # Feature Matching Loss
            pred_real_T2, real_features_T2 = D_T2(real_T2)
            pred_real_T1, real_features_T1 = D_T1(real_T1)

            loss_fm_T2 = sum(
                criterion_feature_matching(fake, real)
                for real, fake in zip(real_features_T2, fake_features_T2)
            )
            loss_fm_T1 = sum(
                criterion_feature_matching(fake, real)
                for real, fake in zip(real_features_T1, fake_features_T1)
            )
            loss_feature_matching = (loss_fm_T2 + loss_fm_T1) * lambda_feature_matching
            loss_G += loss_feature_matching

            loss_hist_T1 = histogram_loss(fake_T1, real_T1, device=device) * lambda_histogram
            loss_hist_T2 = histogram_loss(fake_T2, real_T2, device=device) * lambda_histogram
            loss_histogram = loss_hist_T1 + loss_hist_T2

            loss_G += loss_histogram

            # Discriminator losses
            pred_real_T2, _ = D_T2(real_T2)
            loss_D_real_T2 = criterion_GAN(pred_real_T2, torch.ones_like(pred_real_T2, device=device))
            pred_fake_T2_detach, _ = D_T2(fake_T2.detach())
            loss_D_fake_T2 = criterion_GAN(pred_fake_T2_detach, torch.zeros_like(pred_fake_T2_detach, device=device))
            loss_D_T2 = (loss_D_real_T2 + loss_D_fake_T2) * 0.5

            pred_real_T1, _ = D_T1(real_T1)
            loss_D_real_T1 = criterion_GAN(pred_real_T1, torch.ones_like(pred_real_T1, device=device))
            pred_fake_T1_detach, _ = D_T1(fake_T1.detach())
            loss_D_fake_T1 = criterion_GAN(pred_fake_T1_detach, torch.zeros_like(pred_fake_T1_detach, device=device))
            loss_D_T1 = (loss_D_real_T1 + loss_D_fake_T1) * 0.5

            loss_D = loss_D_T1 + loss_D_T2

            total_G_loss_val += loss_G.item()
            total_D_loss_val += loss_D.item()

            # Compute PSNR and SSIM metrics
            psnr_T1 = calculate_psnr(real_T1, fake_T1)
            ssim_T1 = calculate_ssim(real_T1, fake_T1)
            psnr_T2 = calculate_psnr(real_T2, fake_T2)
            ssim_T2 = calculate_ssim(real_T2, fake_T2)

            total_psnr_T1_val += psnr_T1
            total_ssim_T1_val += ssim_T1
            total_psnr_T2_val += psnr_T2
            total_ssim_T2_val += ssim_T2

            num_batches_val += 1

            # Save example images for qualitative inspection
            if i < 2:
                epoch_images_dir = os.path.join(images_dir, f'epoch_{epoch+1}', 'val')
                os.makedirs(epoch_images_dir, exist_ok=True)
                save_images(real_T1, fake_T2, fake_T1, real_T2, epoch_images_dir, epoch + 1, i, dataset_type='val')

    avg_G_loss_val = total_G_loss_val / num_batches_val if num_batches_val > 0 else 0
    avg_D_loss_val = total_D_loss_val / num_batches_val if num_batches_val > 0 else 0
    avg_psnr_T1_val = total_psnr_T1_val / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_T1_val = total_ssim_T1_val / num_batches_val if num_batches_val > 0 else 0
    avg_psnr_T2_val = total_psnr_T2_val / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_T2_val = total_ssim_T2_val / num_batches_val if num_batches_val > 0 else 0

    return (
        avg_G_loss_val, avg_D_loss_val,
        avg_psnr_T1_val, avg_ssim_T1_val,
        avg_psnr_T2_val, avg_ssim_T2_val
    )


def train_model(n_epochs, train_loader, val_loader, 
                G_T1_to_T2, G_T2_to_T1, 
                D_T1, D_T2, device,
                criterion_GAN, criterion_cycle, criterion_identity, criterion_feature_matching,criterion_histogram,
                optimizer_G, optimizer_D, scheduler_G, scheduler_D, 
                experiment_dir, lambda_dict,input_modality, target_modality):

    images_dir = os.path.join(experiment_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    plots_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    scaler = GradScaler()

    best_ssim = 0.0
    best_psnr = 0.0

    lambda_feature_matching = lambda_dict.get('lambda_feature_matching', 10.0)
    lambda_histogram = lambda_dict.get('lambda_histogram', 1.0) # Start with a lower value and tune


    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch + 1}/{n_epochs}")

        G_T1_to_T2.train()
        G_T2_to_T1.train()
        D_T1.train()
        D_T2.train()

        total_G_loss = 0.0
        total_D_loss = 0.0
        epoch_psnr_T1 = 0.0
        epoch_ssim_T1 = 0.0
        epoch_psnr_T2 = 0.0
        epoch_ssim_T2 = 0.0
        num_batches = 0

        for i, batch in enumerate(train_loader):
            if batch is None:
                continue

            real_T1 = batch[input_modality].to(device)
            real_T2 = batch[target_modality].to(device)

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            try:
                with autocast():
                    fake_T2, _ = G_T1_to_T2(real_T1)
                    fake_T1, _ = G_T2_to_T1(fake_T2)

                    # Compute cycle consistency loss for T1 and direct reconstruction loss for T2
                    loss_cycle_T1 = criterion_cycle(fake_T1, real_T1) * lambda_dict['lambda_cycle']
                    loss_rec_T2 = criterion_cycle(fake_T2, real_T2) * lambda_dict['lambda_rec']

                    # Identity losses
                    idt_T1, _ = G_T2_to_T1(real_T1)
                    loss_idt_T1 = criterion_identity(idt_T1, real_T1) * lambda_dict['lambda_identity']
                    idt_T2, _ = G_T1_to_T2(real_T2)
                    loss_idt_T2 = criterion_identity(idt_T2, real_T2) * lambda_dict['lambda_identity']
                    loss_identity = loss_idt_T1 + loss_idt_T2

                    # GAN losses for both generators
                    pred_fake_T2, fake_features_T2 = D_T2(fake_T2)
                    loss_GAN_T1_to_T2 = criterion_GAN(
                        pred_fake_T2, torch.ones_like(pred_fake_T2, device=device)
                    ) * lambda_dict['lambda_GAN']

                    pred_fake_T1, fake_features_T1 = D_T1(fake_T1)
                    loss_GAN_T2_to_T1 = criterion_GAN(
                        pred_fake_T1, torch.ones_like(pred_fake_T1, device=device)
                    ) * lambda_dict['lambda_GAN']

                    loss_G = loss_GAN_T1_to_T2 + loss_GAN_T2_to_T1 + loss_cycle_T1 + loss_identity + loss_rec_T2

                    # Feature Matching Loss
                    pred_real_T2, real_features_T2 = D_T2(real_T2)
                    pred_real_T1, real_features_T1 = D_T1(real_T1)

                    loss_fm_T2 = sum(
                        criterion_feature_matching(fake, real)
                        for real, fake in zip(real_features_T2, fake_features_T2)
                    )
                    loss_fm_T1 = sum(
                        criterion_feature_matching(fake, real)
                        for real, fake in zip(real_features_T1, fake_features_T1)
                    )
                    loss_feature_matching = (loss_fm_T2 + loss_fm_T1) * lambda_feature_matching

                    loss_G += loss_feature_matching

                    #  # --- Add the new Energy Minimization Loss ---
                    # # Calculate the energy minimization loss for T2
                    # # Note: This is applied to the generated T2 image against the real T2 image
                    # # You could choose to only apply it to one direction or a specific reconstruction.
                    # #loss_energy_T1 = energy_minimization_loss(fake_T1, real_T1) * lambda_energy
                    # loss_energy_T2 = energy_minimization_loss(fake_T2, real_T2) * lambda_dict.get('lambda_energy', 10.0)
                    # loss_energy =  loss_energy_T2
                    # loss_G += loss_energy
                    # --- Add the new Histogram Loss ---
                    loss_hist_T1 = histogram_loss(fake_T1, real_T1, device=device) * lambda_histogram
                    loss_hist_T2 = histogram_loss(fake_T2, real_T2, device=device) * lambda_histogram
                    loss_histogram = loss_hist_T1 + loss_hist_T2
                    
                    # Add to the total generator loss
                    loss_G =  loss_GAN_T1_to_T2 + loss_GAN_T2_to_T1 +\
                              loss_cycle_T1 + loss_identity + loss_rec_T2 +\
                              loss_feature_matching + loss_histogram
            except Exception as e:
                logging.error(f"Error during generator forward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            try:
                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)
                scaler.update()
            except Exception as e:
                logging.error(f"Error during generator backward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            try:
                with autocast():
                    # Discriminator loss for T2
                    pred_real_T2, _ = D_T2(real_T2)
                    loss_D_real_T2 = criterion_GAN(pred_real_T2, torch.ones_like(pred_real_T2, device=device))
                    pred_fake_T2_detach, _ = D_T2(fake_T2.detach())
                    loss_D_fake_T2 = criterion_GAN(pred_fake_T2_detach, torch.zeros_like(pred_fake_T2_detach, device=device))
                    loss_D_T2 = (loss_D_real_T2 + loss_D_fake_T2) * 0.5

                    # Discriminator loss for T1
                    pred_real_T1, _ = D_T1(real_T1)
                    loss_D_real_T1 = criterion_GAN(pred_real_T1, torch.ones_like(pred_real_T1, device=device))
                    pred_fake_T1_detach, _ = D_T1(fake_T1.detach())
                    loss_D_fake_T1 = criterion_GAN(pred_fake_T1_detach, torch.zeros_like(pred_fake_T1_detach, device=device))
                    loss_D_T1 = (loss_D_real_T1 + loss_D_fake_T1) * 0.5

                    loss_D = loss_D_T1 + loss_D_T2
            except Exception as e:
                logging.error(f"Error during discriminator forward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            try:
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                scaler.update()
            except Exception as e:
                logging.error(f"Error during discriminator backward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()

            psnr_T1 = calculate_psnr(real_T1, fake_T1)
            ssim_T1 = calculate_ssim(real_T1, fake_T1)
            psnr_T2 = calculate_psnr(real_T2, fake_T2)
            ssim_T2 = calculate_ssim(real_T2, fake_T2)

            epoch_psnr_T1 += psnr_T1
            epoch_ssim_T1 += ssim_T1
            epoch_psnr_T2 += psnr_T2
            epoch_ssim_T2 += ssim_T2

            num_batches += 1

            if i < 2:
                epoch_images_dir = os.path.join(images_dir, f'epoch_{epoch+1}', 'train')
                os.makedirs(epoch_images_dir, exist_ok=True)
                save_images(real_T1, fake_T2, fake_T1, real_T2, epoch_images_dir, epoch + 1, i, dataset_type='train')

        if num_batches > 0:
            avg_G_loss = total_G_loss / num_batches
            avg_D_loss = total_D_loss / num_batches
            avg_psnr_T1 = epoch_psnr_T1 / num_batches
            avg_ssim_T1 = epoch_ssim_T1 / num_batches
            avg_psnr_T2 = epoch_psnr_T2 / num_batches
            avg_ssim_T2 = epoch_ssim_T2 / num_batches
        else:
            avg_G_loss, avg_D_loss = 0, 0
            avg_psnr_T1, avg_ssim_T1 = 0, 0
            avg_psnr_T2, avg_ssim_T2 = 0, 0

        logging.info(f"Epoch {epoch + 1} Training Losses: Gen {avg_G_loss:.4f}, Dis {avg_D_loss:.4f}")
        logging.info(f"Epoch {epoch + 1} Training Metrics: PSNR_T1 {avg_psnr_T1:.4f}, SSIM_T1 {avg_ssim_T1:.4f}, "
                     f"PSNR_T2 {avg_psnr_T2:.4f}, SSIM_T2 {avg_ssim_T2:.4f}")

        train_losses.append({
                'gen_total_loss_train': avg_G_loss,
                'dis_total_loss_train': avg_D_loss,
                'cycle_loss_T1_train': loss_cycle_T1.item() if num_batches > 0 else 0,
                'rec_loss_T2_train': loss_rec_T2.item() if num_batches > 0 else 0,
                'identity_loss_train': loss_identity.item() if num_batches > 0 else 0,
                'feature_matching_loss_train': loss_feature_matching.item() if num_batches > 0 else 0,
                'gan_loss_T1_to_T2_train': loss_GAN_T1_to_T2.item() if num_batches > 0 else 0,
                'gan_loss_T2_to_T1_train': loss_GAN_T2_to_T1.item() if num_batches > 0 else 0,
                # 'energy_loss_train': loss_energy.item() if num_batches > 0 else 0, # Add this
                'histogram_loss_train': loss_histogram.item() if num_batches > 0 else 0, # Add this
            })
            

        train_metrics.append({
            'PSNR_T1_train': avg_psnr_T1,
            'SSIM_T1_train': avg_ssim_T1,
            'PSNR_T2_train': avg_psnr_T2,
            'SSIM_T2_train': avg_ssim_T2,
        })

        try:
            (avg_G_loss_val, avg_D_loss_val,
             avg_psnr_T1_val, avg_ssim_T1_val,
             avg_psnr_T2_val, avg_ssim_T2_val) = validate(
                val_loader, G_T1_to_T2, G_T2_to_T1, 
                D_T1, D_T2,
                criterion_GAN, criterion_cycle, criterion_identity, criterion_feature_matching, criterion_histogram, # Add this
                lambda_dict, device, images_dir, epoch # Pass the epoch variable here
            )
        except Exception as e:
            logging.error(f"Error during validation at epoch {epoch+1}: {e}", exc_info=True)
            raise

        logging.info(f"Epoch {epoch + 1} Validation Losses: Gen {avg_G_loss_val:.4f}, Dis {avg_D_loss_val:.4f}")
        logging.info(f"Epoch {epoch + 1} Validation Metrics: PSNR_T1 {avg_psnr_T1_val:.4f}, SSIM_T1 {avg_ssim_T1_val:.4f}, "
                     f"PSNR_T2 {avg_psnr_T2_val:.4f}, SSIM_T2 {avg_ssim_T2_val:.4f}")

        val_losses.append({
            'gen_total_loss_val': avg_G_loss_val,
            'dis_total_loss_val': avg_D_loss_val,
            'cycle_loss_T1_val': loss_cycle_T1.item() if num_batches > 0 else 0,
            'rec_loss_T2_val': loss_rec_T2.item() if num_batches > 0 else 0,
            'identity_loss_val': loss_identity.item() if num_batches > 0 else 0,
            'feature_matching_loss_val': loss_feature_matching.item() if num_batches > 0 else 0,
            'gan_loss_T1_to_T2_val': loss_GAN_T1_to_T2.item() if num_batches > 0 else 0,
            'gan_loss_T2_to_T1_val': loss_GAN_T2_to_T1.item() if num_batches > 0 else 0,
        })
        
        val_metrics.append({
            'PSNR_T1_val': avg_psnr_T1_val,
            'SSIM_T1_val': avg_ssim_T1_val,
            'PSNR_T2_val': avg_psnr_T2_val,
            'SSIM_T2_val': avg_ssim_T2_val,
        })

        current_avg_ssim = (avg_ssim_T1_val + avg_ssim_T2_val) / 2
        current_avg_psnr = (avg_psnr_T1_val + avg_psnr_T2_val) / 2
         # NEW: Calculate the combined score
        current_score = calculate_combined_score(
            current_avg_psnr, 
            current_avg_ssim, 
            
        )
        if current_score > best_combined_score:
            best_combined_score = current_score
            
            logging.info(f"New best model found at epoch {epoch + 1} with Combined Score: {best_combined_score:.4f}")
            logging.info(f"  (PSNR: {current_avg_psnr:.4f}, SSIM: {current_avg_ssim:.4f})")

            model_save_path = os.path.join(experiment_dir, 'best_model-combined.pth')
            torch.save({
                'G_T1_to_T2': G_T1_to_T2.state_dict(),
                'G_T2_to_T1': G_T2_to_T1.state_dict(),
                'D_T1': D_T1.state_dict(),
                'D_T2': D_T2.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch + 1,
                'best_ssim': best_ssim,
                'best_psnr': best_psnr,
            }, model_save_path)

        if current_avg_ssim > best_ssim:
            best_ssim = current_avg_ssim
            best_psnr = (avg_psnr_T1_val + avg_psnr_T2_val) / 2
            model_save_path = os.path.join(experiment_dir, 'best_model_ssim.pth')

            torch.save({
                'G_T1_to_T2': G_T1_to_T2.state_dict(),
                'G_T2_to_T1': G_T2_to_T1.state_dict(),
                'D_T1': D_T1.state_dict(),
                'D_T2': D_T2.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch + 1,
                'best_ssim': best_ssim,
                'best_psnr': best_psnr,
            }, model_save_path)

            logging.info(f"Best model saved at epoch {epoch + 1} with avg SSIM {best_ssim:.4f} and avg PSNR {best_psnr:.4f}")

        scheduler_G.step()
        scheduler_D.step()

        try:
            plot_losses_metrics(train_losses, val_losses, train_metrics, val_metrics, plots_dir)
            logging.info(f"Plots saved for epoch {epoch + 1}.")
        except Exception as e:
            logging.error(f"Error while plotting at epoch {epoch+1}: {e}", exc_info=True)

    return best_ssim, best_psnr
