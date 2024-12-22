from pathlib import Path
import os

project_root = Path(os.path.realpath(__file__)).parent.parent / "transmittance2fom_genmodels"
VERVET_TRANS_DATA = project_root / "data" / "transmittance"
VERVET_FOM_DATA = project_root / "data" / "fom"