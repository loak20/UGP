import os, warnings
import numpy as np
import torch
from ase.build import molecule
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

# 2. Load/Build Structure
# Create standard H2O molecule
atoms = molecule('H2O')

# 3. Configure Physics
# Set a large box (20 Å) to simulate isolation (vacuum)
atoms.set_cell([20.0, 20.0, 20.0])
atoms.center() # Centers the H2O in the middle of the box

# Periodic in X and Y, but Vacuum in Z (as requested)
atoms.set_pbc([True, True, False])

# MACE Calculator
atoms.calc = mace_mp(
    model="medium",
    device=device,
    default_dtype="float64"
)

print(f"System: {atoms.get_chemical_formula()} (Isolated in 20Å Box)")
print("Constraints: None")
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
# memory=100: Remembers last 100 steps for smoother convergence
opt = LBFGS(
    atoms, 
    logfile="h2o_opt.log", 
    trajectory="h2o_relax.traj",
    memory=100 
)

# Run until the max force is <= 0.02 eV/Å
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
# Save the final optimized structure as a CIF file
write("H2O_opt.cif", atoms)
print("\nSaved structure to: H2O_opt.cif")
print("Trajectory saved as: h2o_relax.traj")

# 8. Viewer
print("\nOpening viewer...")
try:
    view(read("h2o_relax.traj", ":"))
except Exception as e:
    print("Could not open viewer (likely no GUI backend found). File saved as .traj")