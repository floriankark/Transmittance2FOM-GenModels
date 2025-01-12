import torch
import h5py
import random
import glob
from collections import namedtuple
from torch.utils.data import Dataset, Sampler, DataLoader
from config.path import VERVET_DATA

Tile = namedtuple('Tile', 'brain, section, region, map_type, row, column, patch_size')

class HDF5Dataset(Dataset):
    """
    Args:
        resolution_level (str): "01", "02", ..., "10" downsampled version of the original image by a factor of 2^n
    
    Returns:
        patch (torch.Tensor): A tensor of shape (C, H, W) 
        
    Example usage:
    from dataset import HDF5Dataset, HDF5Sampler, Tile
    from config.path import VERVET_DATA

    tile = Tile(
        brain='Vervet1818',
        section='s0759',
        region='left',
        map_type='NTransmittance',
        row=0,
        column=0,
        patch_size=256
    )

    dataset = HDF5Dataset(resolution_level='05')
    print(dataset[tile])
    """
    def __init__(self, resolution_level: str, transform=None) -> None:
        self.resolution_level = resolution_level
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
        image = self._hf["pyramid"][self.resolution_level]
        patch = image[loc.row:loc.row + loc.patch_size, loc.column:loc.column + loc.patch_size]
        
        if loc.map_type == "FOM":
            patch = patch.transpose(2, 0, 1)
        else:
            patch = patch[None, :, :]

        patch = torch.tensor(patch, dtype=torch.float32) / 255.0

        if self.transform:
            patch = self.transform(patch)

        return patch


class HDF5Sampler(Sampler):
    def __init__(self, file_path: str, brains: list[str], tile_size: int, tiles_per_epoch: int) -> None:
        self.file_path = file_path
        self.brains = brains
        self.tile_size = tile_size
        self.tiles_per_epoch = tiles_per_epoch
        self._hf = None

    def _open_hdf5(self):
        if not self._hf:
            self._hf = h5py.File(self.file_path, "r")

    def __len__(self) -> int:
        return self.tiles_per_epoch

    def __iter__(self):
        self._open_hdf5()
        samples = []

        for _ in range(self.tiles_per_epoch):
            brain = random.choice(self.brains)
            section = random.choice(list(self._hf[brain].keys()))
            region = random.choice(["left", "right", "cerebellum", "none"])
            map_type = random.choice(["fom", "transmittance"])

            dataset_name = f"{map_type}/{section}/{region}"
            if dataset_name not in self._hf[brain]:
                continue

            image = self._hf[brain][dataset_name]
            rows, cols = image.shape[:2]

            if rows < self.tile_size or cols < self.tile_size:
                continue

            row = random.randint(0, rows - self.tile_size)
            column = random.randint(0, cols - self.tile_size)

            samples.append((brain, section, region, map_type, row, column, self.tile_size))

        return iter(samples)


def create_dataloader(hdf5_file_path: str, brains: list[str], tile_size: int = 64, batch_size: int = 8, tiles_per_epoch: int = 1000, num_workers: int = 0):
    dataset = HDF5Dataset(hdf5_file_path)
    sampler = HDF5Sampler(hdf5_file_path, brains, tile_size, tiles_per_epoch)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return dataloader
