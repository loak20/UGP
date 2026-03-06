import os
import warnings

warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from mace.calculators import mace_mp
from ase.build import molecule

atoms = molecule('H2O')

calc = mace_mp(
    model="medium",
    device="cuda",
    default_dtype="float64" 
)

atoms.calc = calc

print("Energy:", atoms.get_potential_energy())
print("Forces:\n", atoms.get_forces())