import torch
import h5py
import random
from collections import namedtuple
from torch.utils.data import Dataset, Sampler, DataLoader
from config.path import VERVET_DATA

Tile = namedtuple('Tile', 'brain, section, region, map_type, row, column, patch_size')

class HDF5Dataset(Dataset):
    def __init__(self, file_path: str, transform=None) -> None:
        self.file_path = file_path
        self.transform = transform

    def _open_hdf5(self, loc: Tile):#
        file_path = '_'.join(filter(loc[:4]))
        print(file_path)
        """if not hasattr(self, "_hf"):
            self._hf = h5py.File(VERVET_DATA / f"{file_path}.h5", "r")"""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, loc: Tile) -> torch.Tensor:

        self._open_hdf5(loc)

        dataset_name = f"{map_type}/{section}/{region}"
        image = self._hf[brain][dataset_name]
        patch = image[row : row + patch_size, column : column + patch_size]
        if map_type == "fom":
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
