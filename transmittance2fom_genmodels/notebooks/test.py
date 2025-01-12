from dataset import HDF5Dataset, HDF5Sampler, create_dataloader
from torch.utils.data import DataLoader
from config.path import VERVET_DATA

"""dataset = HDF5Dataset()
sampler = HDF5Sampler(brain='Vervet1818', map_type='NTransmittance', patch_size=256)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
for batch in dataloader:
    print(batch)
    break"""
    
dataloader = create_dataloader(brain='Vervet1818', map_type='NTransmittance', patch_size=256)
for batch in dataloader:
    print(batch)
    break