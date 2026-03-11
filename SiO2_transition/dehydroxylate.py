import os
import warnings
import numpy as np
import torch
from ase.io import read, write
from ase.optimize import LBFGS
from ase.neighborlist import neighbor_list
from mace.calculators import mace_mp

# Suppress warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# Simulation Parameters
PBCS_XYZ = [True, True, False]
FMAX = 0.02
COLLISION_CUTOFF = 0.7 

# Output Directories
OPT_DIR = "optimised_neb_states"
os.makedirs(OPT_DIR, exist_ok=True)

# Target Pair (Pair 1)
PAIR_1 = {"id": 1,  "g1": {"Si_tag": 64,  "O_tag": 330}, "g2": {"Si_tag": 246, "O_tag": 336}}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_calculator():
    return mace_mp(model="medium", dispersion=False, default_dtype="float64", device=DEVICE)

def has_collisions(atoms, cutoff=COLLISION_CUTOFF):
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, 10.0) 
    min_dist = np.min(dists)
    if min_dist < cutoff:
        return True, f"Collision detected! Min dist {min_dist:.3f} Å < {cutoff} Å"
    return False, "No collisions"

def run_optimization(atoms, label, folder):
    atoms.calc = get_calculator()
    traj_path = os.path.join(folder, f"{label}.traj")
    log_path = os.path.join(folder, f"{label}_opt.log")
    
    opt = LBFGS(atoms, trajectory=traj_path, logfile=log_path)
    opt.run(fmax=FMAX)
    
    cif_path = os.path.join(folder, f"{label}.cif")
    write(cif_path, atoms)
    return atoms.get_potential_energy()

def get_index_by_tag(atoms, tag):
    indices = [i for i, atom in enumerate(atoms) if atom.tag == tag]
    return indices[0] if len(indices) > 0 else None

def find_attached_hydrogen(atoms, o_idx):
    i_indices, j_indices = neighbor_list('ij', atoms, cutoff=1.2, self_interaction=False)
    for k, central_index in enumerate(i_indices):
        if central_index == o_idx:
            neighbor_index = j_indices[k]
            if atoms[neighbor_index].symbol == 'H':
                return neighbor_index, atoms[neighbor_index].tag
    return None, None

# =============================================================================
# MAIN WORKFLOW
# =============================================================================
def main():
    print("\n--- Step 1: Optimising Reactant SiO2 Surface ---")
    cif_file = "SiO2-1.cif"
    if not os.path.exists(cif_file):
        raise FileNotFoundError(f"Please provide {cif_file} in the working directory.")
    
    reactant = read(cif_file)
    reactant.pbc = PBCS_XYZ
    
    # Assign tags to track atoms
    reactant.set_tags(range(len(reactant)))
    
    # Optimise and save the initial state for NEB
    print("Optimising initial state...")
    run_optimization(reactant, "SiO2_reactant", OPT_DIR)
    
    print("\n--- Step 2: Preparing Final State (Atom-Conserved) ---")
    current_atoms = reactant.copy() # Start from the optimised reactant
    
    g1, g2 = PAIR_1['g1'], PAIR_1['g2']
    
    # 1. Locate indices
    idx_si1 = get_index_by_tag(current_atoms, g1['Si_tag'])
    idx_o1  = get_index_by_tag(current_atoms, g1['O_tag'])
    idx_si2 = get_index_by_tag(current_atoms, g2['Si_tag'])
    idx_o2  = get_index_by_tag(current_atoms, g2['O_tag'])
    
    idx_h1, _ = find_attached_hydrogen(current_atoms, idx_o1)
    idx_h2, _ = find_attached_hydrogen(current_atoms, idx_o2)

    # 2. Geometry Manipulation
    pos_o1 = current_atoms.positions[idx_o1]
    pos_o2 = current_atoms.positions[idx_o2]
    
    # Handle PBC for vector calculation
    cell = current_atoms.get_cell()
    pbc = current_atoms.get_pbc()
    diff_o = pos_o2 - pos_o1
    if pbc.any():
        rec_cell = current_atoms.get_reciprocal_cell()
        scaled = np.dot(diff_o, rec_cell.T)
        scaled -= np.rint(scaled) * pbc
        diff_o = np.dot(scaled, cell)
        
    # Move O1 to midpoint to form the bridge
    target_pos = pos_o1 + 0.5 * diff_o
    current_atoms.positions[idx_o1] = target_pos
    
    # Construct detached water molecule in the vacuum space
    # Assuming Z is the vacuum axis. We place it 4.0 Å above the reaction site.
    vac_pos = target_pos + np.array([0.0, 0.0, 4.0])
    
    # Rough geometry for H2O to help the optimizer (O-H length ~0.96 A)
    current_atoms.positions[idx_o2] = vac_pos
    current_atoms.positions[idx_h1] = vac_pos + np.array([0.76, 0.59, 0.0])
    current_atoms.positions[idx_h2] = vac_pos + np.array([-0.76, 0.59, 0.0])
    
    # 3. Final Polish and Optimization
    col, msg = has_collisions(current_atoms)
    if col:
         print(f" -> Warning: {msg}. The optimizer will attempt to resolve this.")
         
    print("Optimising final state with detached H2O...")
    run_optimization(current_atoms, "SiO2_final", OPT_DIR)
    
    print("\n--- Success! ---")
    print(f"Initial state saved as: {OPT_DIR}/SiO2_reactant.cif")
    print(f"Final state saved as: {OPT_DIR}/SiO2_final.cif")
    print("Both files have matching indices and are ready for CI-NEB.")

if __name__ == "__main__":
    main()