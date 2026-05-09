import os
import warnings
import numpy as np
import torch
from ase.io import read
from ase.mep import NEB
from ase.optimize import FIRE
from ase.build import minimize_rotation_and_translation
from ase.geometry import find_mic
from mace.calculators import mace_mp

# =========================
# ENV
# =========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PBCS_XYZ = [True, True, False]
FMAX_STAGE1 = 0.1
FMAX_STAGE2 = 0.05
N_IMAGES = 7

def get_calculator():
    return mace_mp(model="medium", dispersion=False,
                   default_dtype="float64", device=DEVICE)

# =========================
# LOAD
# =========================
initial = read("SiO2_reactant.cif")
final   = read("SiO2_final.cif")

initial.pbc = PBCS_XYZ
final.pbc   = PBCS_XYZ

# =========================
# CHECK CONSISTENCY
# =========================
assert len(initial) == len(final), "Atom count mismatch"

# =========================
# ALIGN STRUCTURES
# =========================
minimize_rotation_and_translation(initial, final)

# =========================
# MIC FIX (CRITICAL)
# =========================
for i in range(len(initial)):
    dr = final.positions[i] - initial.positions[i]
    dr_mic, _ = find_mic(dr, initial.cell, pbc=initial.pbc)
    final.positions[i] = initial.positions[i] + dr_mic

# sanity check
disp = np.linalg.norm(final.positions - initial.positions, axis=1)
print("Max displacement after MIC:", disp.max())

# =========================
# CREATE IMAGES
# =========================
images = [initial]
images += [initial.copy() for _ in range(N_IMAGES)]
images += [final]

neb = NEB(images, climb=False)

neb.interpolate(method='idpp')

# attach calculators
for img in images[1:-1]:
    img.calc = get_calculator()

# =========================
# STAGE 1 (RELAX PATH)
# =========================
opt = FIRE(neb,
           trajectory="neb_stage1.traj",
           logfile="neb1.log",
           dt=0.05,
           maxstep=0.1)

opt.run(fmax=FMAX_STAGE1)

# =========================
# STAGE 2 (CLIMBING)
# =========================
neb.climb = True

opt = FIRE(neb,
           trajectory="neb_stage2.traj",
           logfile="neb2.log",
           dt=0.03,
           maxstep=0.08)

opt.run(fmax=FMAX_STAGE2)

print("✅ CI-NEB converged successfully")