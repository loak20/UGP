import os
import warnings
import numpy as np
import torch
from ase.io import read, write
from ase.optimize import LBFGS
from mace.calculators import mace_mp

# =========================
# ENV
# =========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PBCS_XYZ = [True, True, False]
FMAX = 0.02

def get_calculator():
    return mace_mp(model="medium", dispersion=False,
                   default_dtype="float64", device=DEVICE)

# =========================
# HELPERS (FIXED)
# =========================
def get_index_by_tag(atoms, tag):
    for i, a in enumerate(atoms):
        if a.tag == tag:
            return i
    raise ValueError(f"Tag {tag} not found")

def find_attached_hydrogen(atoms, o_idx):
    # deterministic: closest H
    dists = atoms.get_distances(o_idx, range(len(atoms)), mic=True)
    h_candidates = [(i, d) for i, d in enumerate(dists)
                    if atoms[i].symbol == 'H']
    if len(h_candidates) == 0:
        raise ValueError(f"No H attached to O index {o_idx}")
    return min(h_candidates, key=lambda x: x[1])[0]

# =========================
# MAIN
# =========================
atoms = read("SiO2-1.cif")
atoms.pbc = PBCS_XYZ

# tag atoms
atoms.set_tags(range(len(atoms)))

# ---- SELECT PAIR ----
Si1_tag, O1_tag = 64, 330
Si2_tag, O2_tag = 246, 336

idx_si1 = get_index_by_tag(atoms, Si1_tag)
idx_o1  = get_index_by_tag(atoms, O1_tag)
idx_si2 = get_index_by_tag(atoms, Si2_tag)
idx_o2  = get_index_by_tag(atoms, O2_tag)

idx_h1 = find_attached_hydrogen(atoms, idx_o1)
idx_h2 = find_attached_hydrogen(atoms, idx_o2)

pos = atoms.positions.copy()

# =========================
# 1. LOCAL BRIDGE FORMATION
# =========================
bridge_target = 0.5 * (pos[idx_si1] + pos[idx_si2])

# move O1 partially (NOT full overwrite)
atoms.positions[idx_o1] += 0.4 * (bridge_target - pos[idx_o1])

# =========================
# 2. LOCAL WATER FORMATION
# =========================
local_pos = bridge_target + np.array([0.0, 0.0, 1.2])

# move O2 partially (smooth displacement)
atoms.positions[idx_o2] += 0.4 * (local_pos - pos[idx_o2])

# place Hs relative to O2
o2_new = atoms.positions[idx_o2]

atoms.positions[idx_h1] = o2_new + np.array([0.96, 0.0, 0.0])
atoms.positions[idx_h2] = o2_new + np.array([-0.32, 0.9, 0.0])

# =========================
# 3. SANITY CHECK (important)
# =========================
disp = np.linalg.norm(atoms.positions - pos, axis=1)
print("Max displacement:", disp.max())

if disp.max() > 2.0:
    raise RuntimeError("Displacement too large — NEB will likely fail")

# =========================
# 4. RELAX FINAL STATE
# =========================
atoms.calc = get_calculator()

opt = LBFGS(atoms, trajectory="final_relax.traj", logfile="final.log")
opt.run(fmax=FMAX)

write("SiO2_final.cif", atoms)

print("✅ Final state ready (NEB-safe)")