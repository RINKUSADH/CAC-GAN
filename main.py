import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import logging
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
import argparse

from train import train_model, weights_init, initialize_loss_optimizers
from generator import Generator
from discriminator import Discriminator
from datasets import MRIImageDataset


# Fixed batch size
BATCH_SIZE = 16

def setup_logging(output_dir):
    """
    Sets up logging to file and console.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def train_with_best_hyperparameters(field_strength, output_dir, n_epochs, input_modality, target_modality):
    """
    Trains the model using manually selected best hyperparameters.
    """
    # Best hyperparameters from Optuna trial 1
    best_hyperparams = {
       
        "lr_G": 0.0005785498689372762,
        "lr_D": 1.031748861146634e-06,
        "lambda_GAN": 3.7788037724018286,
        "lambda_cycle": 5.171861502689588,
        "lambda_identity": 4.895291043453484,
        "lambda_feature_matching": 11.98742415562527,
        "lambda_rec": 5.170542264699621,
        'lambda_histogram': 0.15352440582440083  
        
          
    }
   

    lambda_dict = {
        'lambda_GAN': best_hyperparams["lambda_GAN"],
        'lambda_cycle': best_hyperparams["lambda_cycle"],
        'lambda_identity': best_hyperparams["lambda_identity"],
        'lambda_feature_matching': best_hyperparams["lambda_feature_matching"],
        'lambda_rec': best_hyperparams["lambda_rec"],
        'lambda_histogram': best_hyperparams["lambda_histogram"]  
    }

    today = datetime.now().strftime('%Y-%m-%d')
    experiment_dir = os.path.join(output_dir, f"best_trial_{today}")
    os.makedirs(experiment_dir, exist_ok=True)

    setup_logging(experiment_dir)

    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_input_dir = f'/DATA/PMC_dataset/2D/train/{field_strength}/{input_modality.upper()}'
    train_target_dir = f'/DATA/PMC_dataset/2D/train/{field_strength}/{target_modality.upper()}'
    test_input_dir = f'/DATA/PMC_dataset/2D/valid/{field_strength}/{input_modality.upper()}'
    test_target_dir = f'/DATA/PMC_dataset/2D/valid/{field_strength}/{target_modality.upper()}'
    

    train_dataset = MRIImageDataset(train_input_dir, train_target_dir, input_modality, target_modality, transform)
    test_dataset = MRIImageDataset(test_input_dir, test_target_dir, input_modality, target_modality, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize generators and discriminators
    G_T1_to_T2 = Generator().to(device)
    G_T2_to_T1 = Generator().to(device)
    D_T1 = Discriminator().to(device)
    D_T2 = Discriminator().to(device)

    G_T1_to_T2.apply(weights_init)
    G_T2_to_T1.apply(weights_init)
    D_T1.apply(weights_init)
    D_T2.apply(weights_init)

    (criterion_GAN, criterion_cycle, criterion_identity,
     criterion_feature_matching,criterion_histogram,
     optimizer_G, optimizer_D, scheduler_G, scheduler_D) = initialize_loss_optimizers(
        G_T1_to_T2, G_T2_to_T1, D_T1, D_T2,
        lr_G=best_hyperparams["lr_G"], lr_D=best_hyperparams["lr_D"]
    )

    best_ssim, best_psnr = train_model(
        n_epochs=n_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        G_T1_to_T2=G_T1_to_T2,
        G_T2_to_T1=G_T2_to_T1,
        D_T1=D_T1,
        D_T2=D_T2,
        device=device,
        criterion_GAN=criterion_GAN,
        criterion_cycle=criterion_cycle,
        criterion_identity=criterion_identity,
        criterion_feature_matching=criterion_feature_matching,
        criterion_histogram=criterion_histogram,  
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        scheduler_G=scheduler_G,
        scheduler_D=scheduler_D,
        experiment_dir=experiment_dir,
        lambda_dict=lambda_dict,
        input_modality=input_modality,
        target_modality=target_modality
    )

    logging.info(f"Training completed with Best SSIM: {best_ssim:.4f}, Best PSNR: {best_psnr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CycleGAN with best hyperparameters for MRI Image Translation")
    parser.add_argument('--field_strength', type=str, default='3T',
                        help='Field strength (e.g., 1.5T or 3T)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output models and logs')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='Number of epochs for training')
    parser.add_argument('--input_modality', type=str, default='t1',
                        help='Input modality (e.g., t1)')
    parser.add_argument('--target_modality', type=str, default='t2',
                        help='Target modality (e.g., t2)')
    args = parser.parse_args()

    train_with_best_hyperparameters(args.field_strength, args.output_dir, args.n_epochs, args.input_modality, args.target_modality)
