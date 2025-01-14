import torch
from torch.utils.data import DataLoader
from config.path import VERVET_DATA
import random
#from dataloader_ex4 import create_dataloader
import os


import matplotlib.pyplot as plt
from dataloader_ex4 import create_paired_dataloader
import torch

def visualize_batch(batch, num_samples=4):
    """
    Visualize pairs of transmittance and FOM images from a batch.
    
    Args:
        batch (dict): Dictionary containing 'input' (transmittance) and 'target' (FOM) tensors
        num_samples (int): Number of pairs to visualize
    """
    transmittance = batch['input']
    fom = batch['target']
    
    num_samples = min(num_samples, transmittance.shape[0])
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        trans_img = transmittance[i, 0].numpy()  # Remove channel dimension
        axes[i, 0].imshow(trans_img, cmap='gray')
        axes[i, 0].set_title(f'Transmittance {i+1}')
        axes[i, 0].axis('off')
        
        # Plot FOM
        fom_img = fom[i].permute(1, 2, 0).numpy()  # Change to (H, W, C) for plotting
        axes[i, 1].imshow(fom_img)
        axes[i, 1].set_title(f'FOM {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def print_batch_info(batch):
    """Print information about the batch shapes and value ranges."""
    transmittance = batch['input']
    fom = batch['target']
    
    print("\nBatch Information:")
    print(f"Transmittance shape: {transmittance.shape}")
    print(f"Transmittance value range: [{transmittance.min():.3f}, {transmittance.max():.3f}]")
    print(f"FOM shape: {fom.shape}")
    print(f"FOM value range: [{fom.min():.3f}, {fom.max():.3f}]")

def test_dataloader():
    data_dir = os.path.join(os.getcwd(), "data")
    
    print("\nChecking data directory structure:")
    print(f"Data directory: {data_dir}")
    print("Contents of fom folder:", os.listdir(os.path.join(data_dir, "fom")))
    print("Contents of transmittance folder:", os.listdir(os.path.join(data_dir, "transmittance")))
    
    # Create dataloader
    dataloader = create_paired_dataloader(
        data_dir=data_dir,
        brain="Vervet1818",
        patch_size=64,
        batch_size=8,
        tiles_per_epoch=100,
        num_workers=0
    )
    
    print(f"\nNumber of batches in dataloader: {len(dataloader)}")
    
    # Test a few batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nProcessing batch {batch_idx + 1}")
        print_batch_info(batch)
        visualize_batch(batch)
        
        # Only first 3 batches
        if batch_idx >= 2:
            break

if __name__ == "__main__":
    torch.manual_seed(42)
    
    try:
        test_dataloader()
    except Exception as e:
        print(f"Error occurred: {str(e)}")