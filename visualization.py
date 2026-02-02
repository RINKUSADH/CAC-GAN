# import os
import matplotlib.pyplot as plt
import logging
import os

def save_images(real_T1, fake_T2, fake_T1, real_T2, images_dir, epoch, batch_idx, dataset_type='train'):
    """
    Saves a set of generated images for visualization.
    """
    os.makedirs(images_dir, exist_ok=True)
    i = 0
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(real_T1[i].cpu().detach().squeeze(), cmap='gray')
    axes[0, 0].set_title('Real T1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(fake_T2[i].cpu().detach().squeeze(), cmap='gray')
    axes[0, 1].set_title('Fake T2')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(fake_T1[i].cpu().detach().squeeze(), cmap='gray')
    axes[1, 0].set_title('Reconstructed T1')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(real_T2[i].cpu().detach().squeeze(), cmap='gray')
    axes[1, 1].set_title('Real T2')
    axes[1, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(images_dir, f"{dataset_type}_epoch_{epoch}_batch_{batch_idx}.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Images saved at {save_path}")


def plot_losses_metrics(train_losses, val_losses, train_metrics, val_metrics, plots_dir):
    """
    Plots and saves loss and metric curves over training epochs.
    If certain keys are missing, the plot is still generated and an annotation is added to indicate which keys were not found.
    """
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Get available keys from the first element of each dictionary list
    available_train_loss_keys = train_losses[0].keys() if train_losses else []
    available_val_loss_keys = val_losses[0].keys() if val_losses else []
    available_train_metric_keys = train_metrics[0].keys() if train_metrics else []
    available_val_metric_keys = val_metrics[0].keys() if val_metrics else []

    logging.info(f"Available train loss keys: {available_train_loss_keys}")
    logging.info(f"Available val loss keys: {available_val_loss_keys}")
    logging.info(f"Available train metric keys: {available_train_metric_keys}")
    logging.info(f"Available val metric keys: {available_val_metric_keys}")

    # Define metric groups with the keys you expect
    metrics = {
        'Cycle Loss': ['cycle_loss_T1_train', 'cycle_loss_T1_val', 'rec_loss_T2_train', 'rec_loss_T2_val'],
        'Identity Loss': ['identity_loss_train', 'identity_loss_val'],
        'Feature Matching Loss': ['feature_matching_loss_train', 'feature_matching_loss_val'],
        'GAN Loss': ['gan_loss_T1_to_T2_train', 'gan_loss_T1_to_T2_val', 'gan_loss_T2_to_T1_train', 'gan_loss_T2_to_T1_val'],
        'Generator & Discriminator Loss': ['gen_total_loss_train', 'gen_total_loss_val', 'dis_total_loss_train', 'dis_total_loss_val'],
        'PSNR': ['PSNR_T1_train', 'PSNR_T1_val', 'PSNR_T2_train', 'PSNR_T2_val'],
        'SSIM': ['SSIM_T1_train', 'SSIM_T1_val', 'SSIM_T2_train', 'SSIM_T2_val']
    }

    

    for title, keys in metrics.items():
        plt.figure(figsize=(10, 6))
        missing_keys = []
        lines_plotted = False

        for key in keys:
            # Determine whether the key belongs to metrics (PSNR/SSIM) or losses
            is_metric = 'PSNR' in key or 'SSIM' in key
            dataset_type = 'train' if 'train' in key else 'val'
            exists = False

            if is_metric:
                if dataset_type == 'train' and key in available_train_metric_keys:
                    exists = True
                    values = [m[key] for m in train_metrics]
                elif dataset_type == 'val' and key in available_val_metric_keys:
                    exists = True
                    values = [m[key] for m in val_metrics]
            else:
                if dataset_type == 'train' and key in available_train_loss_keys:
                    exists = True
                    values = [loss[key] for loss in train_losses]
                elif dataset_type == 'val' and key in available_val_loss_keys:
                    exists = True
                    values = [loss[key] for loss in val_losses]

            if exists:
                plt.plot(epochs, values, label=key.replace('_', ' ').title())
                lines_plotted = True
            else:
                missing_keys.append(key)

        if missing_keys:
            logging.warning(f"Missing keys for {title}: {missing_keys}")
            missing_text = "Missing: " + ", ".join(missing_keys)
            # Add annotation for missing keys at the bottom center of the plot
            plt.text(0.5, 0.1, missing_text, transform=plt.gca().transAxes,
                     fontsize=10, color='red', ha='center')

        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.title(f'{title} Over Epochs')
        if lines_plotted:
            plt.legend()
        plt.grid(True)
        save_path = os.path.join(plots_dir, f"{title.lower().replace(' ', '_')}.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Plot saved for {title} at {save_path}")

    logging.info("All loss and metric plots saved successfully!")
