import os, warnings
import numpy as np
import torch
from ase.io import read, write
from ase.optimize import LBFGS
from mace.calculators import mace_mp
from ase.visualize import view

# 1. Environment Setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

# 2. Load Structure
# Ensure 'SiO2-1.cif' is in the same folder
atoms = read("SiO2-1.cif")

# 3. Configure Physics
# Periodic in X and Y (sheet), Vacuum in Z
atoms.set_pbc([True, True, False])

# MACE Calculator
atoms.calc = mace_mp(
    model="medium",
    device=device,
    default_dtype="float64"
)

print(f"System: {len(atoms)} atoms")
print("Constraints: None (All atoms free)")
print("Optimizer: LBFGS (Polishing Mode)")

# 4. Initial State Analysis
E_initial = atoms.get_potential_energy()
forces_initial = atoms.get_forces()
F_initial = np.max(np.abs(forces_initial))

print("\n===== INITIAL STATE =====")
print(f"Energy:    {E_initial:.5f} eV")
print(f"Max Force: {F_initial:.5f} eV/Å")

# 5. Run Optimization
print("\n===== OPTIMIZING... =====")

# LBFGS Configuration:
# memory=100: Remembers last 100 steps to build a better map of the energy surface
# This helps it converge much faster and smoother than FIRE at this stage.
opt = LBFGS(
    atoms, 
    logfile="opt_lbfgs.log", 
    trajectory="relax_lbfgs.traj",
    memory=100 
)

# Run until the max force on ALL atoms is <= 0.02 eV/Å
opt.run(fmax=0.02)

# 6. Final State Analysis
E_final = atoms.get_potential_energy()
forces_final = atoms.get_forces()
F_final = np.max(np.abs(forces_final))

print("\n===== FINAL STATE =====")
print(f"Energy:    {E_final:.5f} eV")
print(f"Max Force: {F_final:.5f} eV/Å")

print("\n===== CHANGES =====")
print(f"Delta E:         {E_final - E_initial:.5f} eV")
print(f"Force Reduction: {F_initial:.4f} -> {F_final:.4f}")

# 7. Save Results
write("SiO2-1_opt_lbfgs.cif", atoms)
print("\nSaved structure to: SiO2-1_opt_lbfgs.cif")

# 8. Viewer
print("\nOpening viewer...")
try:
    view(read("relax_lbfgs.traj", ":"))
except Exception as e:
    print("Could not open viewer (likely no GUI backend found). File saved as .traj")