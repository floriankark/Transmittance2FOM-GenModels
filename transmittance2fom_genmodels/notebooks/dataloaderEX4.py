import os
import torch
import h5py
import random
import glob
from collections import namedtuple
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np

Tile = namedtuple('Tile', 'brain, section, region, map_type, row, column, patch_size')
resolution_level = '06' # TODO: Make this part of a config file


class PairedHDF5Dataset(Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        self.transform = transform
        self.data_dir = data_dir
        self._fom_hf = None
        self._trans_hf = None
        
    def _open_hdf5(self, loc: Tile, map_type: str):
        if map_type == "FOM":
            file_name = '_'.join(filter(None, [loc.brain, loc.section, loc.region, "FOM_HSV"]))
            file_path = os.path.join(self.data_dir, "fom", f"{file_name}.h5")
        else:
            
            file_name = '_'.join(filter(None, [loc.brain, loc.section, loc.region]))
            file_path = os.path.join(self.data_dir, "transmittance", f"{file_name}.h5")
            
        # Print debug information
        print(f"Looking for {map_type} file at: {file_path}")
        
        if map_type == "FOM" and self._fom_hf is None:
            if os.path.exists(file_path):
                self._fom_hf = h5py.File(file_path, "r")
            else:
                raise FileNotFoundError(f"No FOM HDF5 file found at {file_path}")
                
        elif map_type == "transmittance" and self._trans_hf is None:
            if os.path.exists(file_path):
                self._trans_hf = h5py.File(file_path, "r")
            else:
                raise FileNotFoundError(f"No transmittance HDF5 file found at {file_path}")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, loc: Tile) -> tuple:
        # Load both FOM and transmittance patches
        self._open_hdf5(loc, "FOM")
        self._open_hdf5(loc, "transmittance")
        
        # Get FOM patch
        fom_image = self._fom_hf["pyramid"][resolution_level]
        fom_patch = fom_image[loc.row:loc.row + loc.patch_size, 
                             loc.column:loc.column + loc.patch_size]
        
        # Get transmittance patch
        trans_image = self._trans_hf["pyramid"][resolution_level]
        trans_patch = trans_image[loc.row:loc.row + loc.patch_size, 
                                loc.column:loc.column + loc.patch_size]
        
        # Handle padding if needed
        if fom_patch.shape[:2] != (loc.patch_size, loc.patch_size):
            fom_patch = pad_patch(fom_patch, loc.patch_size)
        if trans_patch.shape != (loc.patch_size, loc.patch_size):
            trans_patch = pad_patch(trans_patch, loc.patch_size)
        
        # Process FOM patch
        fom_patch = fom_patch.transpose(2, 0, 1)
        fom_patch = torch.tensor(fom_patch, dtype=torch.float32) / 255.0
        
        # Process transmittance patch
        trans_patch = trans_patch[None, :, :]
        trans_patch = torch.tensor(trans_patch, dtype=torch.float32) / 255.0
        
        if self.transform:
            fom_patch = self.transform(fom_patch)
            trans_patch = self.transform(trans_patch)
            
        return {"input": trans_patch, "target": fom_patch}

class PairedHDF5Sampler(Sampler):
    def __init__(self, data_dir: str, brain: str, patch_size: int, tiles_per_epoch: int = 1000) -> None:
        self.data_dir = data_dir
        self.brain = brain
        self.patch_size = patch_size
        self.tiles_per_epoch = tiles_per_epoch
        
        # Get all FOM files for the specified brain
        self.fom_files = glob.glob(os.path.join(self.data_dir, "fom", f"{self.brain}*.h5"))
        if not self.fom_files:
            raise FileNotFoundError(f"No FOM files found for brain {self.brain}")

    def __len__(self) -> int:
        return self.tiles_per_epoch

    def __iter__(self):
        tiles = []
        
        for _ in range(self.tiles_per_epoch):
            file_path = random.choice(self.fom_files)
            
            file_name = os.path.basename(file_path)
            parts = file_name.split("_")
            section = parts[1]
            
            if len(parts) > 3 and parts[2] not in ["FOM"]:
                region = parts[2]
            else:
                region = "none"
                
            # Open FOM file to get dimensions
            with h5py.File(file_path, "r") as hf:
                shape = hf["pyramid"][resolution_level].shape
                row_len, col_len = shape[:2]
                
            row_range = row_len - self.patch_size
            col_range = col_len - self.patch_size
            row, col = random.randint(0, row_range), random.randint(0, col_range)
            
            tiles.append(Tile(self.brain, section, region, "FOM", row, col, self.patch_size))
            
        return iter(tiles)
    
def pad_patch(patch, target_size, is_transmittance=False):
    if is_transmittance:
        return np.pad(patch, ((0, max(0, target_size - patch.shape[0])), 
                             (0, max(0, target_size - patch.shape[1]))), 
                     mode='constant')
    else:
        return np.pad(patch, ((0, max(0, target_size - patch.shape[0])), 
                             (0, max(0, target_size - patch.shape[1])),
                             (0, 0)), 
                     mode='constant')

def create_paired_dataloader(data_dir: str,
                           brain: str,
                           patch_size: int = 64,
                           batch_size: int = 8,
                           tiles_per_epoch: int = 1000,
                           num_workers: int = 0,
                           transform=None) -> DataLoader:
    
    dataset = PairedHDF5Dataset(data_dir=data_dir, transform=transform)
    sampler = PairedHDF5Sampler(data_dir=data_dir, brain=brain, patch_size=patch_size, 
                               tiles_per_epoch=tiles_per_epoch)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                          num_workers=num_workers)
    return dataloader

def pad_patch(patch, target_size):
    pad_height = max(0, target_size - patch.shape[0])
    pad_width = max(0, target_size - patch.shape[1])
    return np.pad(patch, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')

