from dataset import HDF5Dataset, HDF5Sampler
from config.path import VERVET_FOM_DATA, VERVET_TRANS_DATA
from collections import namedtuple

Tile = namedtuple('Tile', 'brain, section, region, map_type, row, column, patch_size')

tile = Tile(
    brain='brain1',
    section='section1',
    region='region1',
    map_type='fom',
    row=0,
    column=0,
    patch_size=256
)
print(HDF5Dataset._open_hdf5(tile))