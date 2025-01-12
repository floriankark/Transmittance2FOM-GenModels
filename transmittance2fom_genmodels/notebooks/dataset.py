import os
import torch
import h5py
import random
import glob
from collections import namedtuple
from torch.utils.data import Dataset, Sampler, DataLoader
from config.path import VERVET_DATA
import numpy as np

Tile = namedtuple('Tile', 'brain, section, region, map_type, row, column, patch_size')
resolution_level = '05' # TODO: Make this part of a config file

class HDF5Dataset(Dataset):
    def __init__(self, transform=None) -> None:
        self.transform = transform
        
    def _open_hdf5(self, loc: Tile):
        file_name = '_'.join(filter(None, loc[:4]))
        file_path = glob.glob(f"{VERVET_DATA}/{file_name}*.h5")
        if not hasattr(self, "_hf"):
            if file_path not in [None, []]:
                self._hf = h5py.File(file_path[0], "r")
            else:
                raise FileNotFoundError(f"No HDF5 file found for {file_path[0]}")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, loc: Tile) -> torch.Tensor:
        self._open_hdf5(loc)
        image = self._hf["pyramid"][resolution_level]
        patch = image[loc.row:loc.row + loc.patch_size, loc.column:loc.column + loc.patch_size]
        if patch.shape[:2] != (loc.patch_size, loc.patch_size):
            patch = pad_patch(patch, loc.patch_size)  
        if loc.map_type == "FOM":
            patch = patch.transpose(2, 0, 1)
        else:
            patch = patch[None, :, :]

        patch = torch.tensor(patch, dtype=torch.float32) / 255.0

        if self.transform:
            patch = self.transform(patch)

        return patch


class HDF5Sampler(Sampler):
    def __init__(self, brain: str, map_type: str, patch_size: int, tiles_per_epoch: int = 1000) -> None:
        self.brain = brain
        self.map_type = map_type
        self.patch_size = patch_size
        self.tiles_per_epoch = tiles_per_epoch

    def _open_hdf5(self, file_path: str):
        if not hasattr(self, "_hf"):
            self._hf = h5py.File(file_path, "r")

    def __len__(self) -> int:
        return self.tiles_per_epoch

    def __iter__(self):
        tiles = []
        
        for _ in range(self.tiles_per_epoch):
            files = glob.glob(f"{VERVET_DATA}/{self.brain}*{self.map_type}*.h5")
            file_path = random.choice(files)
            #section, region = file_path.split("_")[1:3] 

            file_name = os.path.basename(file_path)
                        
            parts = file_name.split("_")
            section = parts[1]  # Second part is always the section
            if parts[2] not in ["FOM", "NTransmittance"]:
                region = parts[2]  # Region is present
            else:
                region = "none"   
            self._open_hdf5(file_path)
            #row_len, col_len = self._hf["pyramid"][resolution_level].shape
            shape = self._hf["pyramid"][resolution_level].shape
            row_len, col_len = shape[:2]  # Extract the first two dimensions
            row_range = row_len - self.patch_size
            col_range = col_len - self.patch_size
            row, col = random.randint(0, row_range), random.randint(0, col_range)
            
            tiles.append(Tile(self.brain, section, region, self.map_type, row, col, self.patch_size))
            
        return iter(tiles)


def create_dataloader(brain: str, 
                      map_type: str, 
                      patch_size: int = 64, 
                      batch_size: int = 8, 
                      tiles_per_epoch: int = 1000, 
                      num_workers: int = 0, 
                      transform=None,
                      ) -> DataLoader:
    
    dataset = HDF5Dataset(transform=transform)
    sampler = HDF5Sampler(brain=brain, map_type=map_type, patch_size=patch_size, tiles_per_epoch=tiles_per_epoch) 
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return dataloader

def pad_patch(patch, target_size):
    pad_height = max(0, target_size - patch.shape[0])
    pad_width = max(0, target_size - patch.shape[1])
    return np.pad(patch, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
