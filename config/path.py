from pathlib import Path
import os

project_root = (
    Path(os.path.realpath(__file__)).parent.parent / "transmittance2fom_genmodels"
)
VERVET_DATA = project_root / "data"
