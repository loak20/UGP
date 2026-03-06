import os
import warnings
import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from ase.build import molecule
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
WATER_CELL = [20.0, 20.0, 20.0]
COLLISION_CUTOFF = 0.7 

# Output Directories
OPT_DIR = "optimised"
os.makedirs(OPT_DIR, exist_ok=True)

# =============================================================================
# HARDCODED PAIRS FROM ENERGIES.TXT
# =============================================================================
# Format: Step Number, Group 1 Tags (Si, O), Group 2 Tags (Si, O)
# Note: Group 1 Oxygen is moved (bridging), Group 2 Oxygen is deleted.
PAIRS_TO_PROCESS = [
    {"id": 1,  "g1": {"Si_tag": 64,  "O_tag": 330}, "g2": {"Si_tag": 246, "O_tag": 336}},
    {"id": 2,  "g1": {"Si_tag": 161, "O_tag": 340}, "g2": {"Si_tag": 132, "O_tag": 354}},
    {"id": 3,  "g1": {"Si_tag": 225, "O_tag": 332}, "g2": {"Si_tag": 246, "O_tag": 338}},
    {"id": 4,  "g1": {"Si_tag": 74,  "O_tag": 228}, "g2": {"Si_tag": 202, "O_tag": 35}},
    {"id": 5,  "g1": {"Si_tag": 10,  "O_tag": 144}, "g2": {"Si_tag": 173, "O_tag": 283}},
    {"id": 6,  "g1": {"Si_tag": 291, "O_tag": 382}, "g2": {"Si_tag": 288, "O_tag": 396}},
    {"id": 7,  "g1": {"Si_tag": 214, "O_tag": 358}, "g2": {"Si_tag": 291, "O_tag": 381}},
    {"id": 8,  "g1": {"Si_tag": 33,  "O_tag": 342}, "g2": {"Si_tag": 98,  "O_tag": 344}},
    {"id": 9,  "g1": {"Si_tag": 160, "O_tag": 320}, "g2": {"Si_tag": 97,  "O_tag": 323}},
    {"id": 10, "g1": {"Si_tag": 172, "O_tag": 325}, "g2": {"Si_tag": 258, "O_tag": 250}},
    {"id": 11, "g1": {"Si_tag": 192, "O_tag": 294}, "g2": {"Si_tag": 56,  "O_tag": 185}},
    {"id": 12, "g1": {"Si_tag": 217, "O_tag": 36},  "g2": {"Si_tag": 74,  "O_tag": 19}}
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_calculator():
    """Returns a fresh instance of the MACE calculator."""
    return mace_mp(model="medium", dispersion=False, default_dtype="float64", device=DEVICE)

def has_collisions(atoms, cutoff=COLLISION_CUTOFF):
    """Simple check to ensure no two atoms are dangerously close."""
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, 10.0) 
    min_dist = np.min(dists)
    if min_dist < cutoff:
        return True, f"Collision detected! Min dist {min_dist:.3f} Å < {cutoff} Å"
    return False, "No collisions"

def run_optimization(atoms, label, folder):
    """Run LBFGS optimization and save artifacts."""
    atoms.calc = get_calculator()
    
    traj_path = os.path.join(folder, f"{label}.traj")
    log_path = os.path.join(folder, f"{label}_opt.log")
    
    opt = LBFGS(atoms, trajectory=traj_path, logfile=log_path)
    opt.run(fmax=FMAX)
    
    cif_path = os.path.join(folder, f"{label}.cif")
    write(cif_path, atoms)
    
    return atoms.get_potential_energy()

def get_index_by_tag(atoms, tag):
    """Helper to find the current index of an atom given its unique tag."""
    indices = [i for i, atom in enumerate(atoms) if atom.tag == tag]
    if len(indices) == 0:
        return None
    return indices[0]

def find_attached_hydrogen(atoms, o_idx):
    """
    Given an Oxygen index, find the attached Hydrogen index.
    Assumes standard OH bond length < 1.2 A.
    """
    # Fix: Ask for 'ij' to get both central (i) and neighbor (j) indices
    i_indices, j_indices = neighbor_list('ij', atoms, cutoff=1.2, self_interaction=False)
    
    # Iterate through the results to find neighbors of our specific Oxygen (o_idx)
    for k, central_index in enumerate(i_indices):
        if central_index == o_idx:
            neighbor_index = j_indices[k]
            # Check if this neighbor is a Hydrogen
            if atoms[neighbor_index].symbol == 'H':
                return neighbor_index, atoms[neighbor_index].tag
            
    return None, None
# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    # ---------------------------------------------------------
    # STEP 1: Calculate E[H2O]
    # ---------------------------------------------------------
    print("\n--- Step 1: Optimising isolated Water molecule ---")
    water = molecule("H2O")
    water.set_cell(WATER_CELL)
    water.center()
    water.pbc = PBCS_XYZ
    
    e_water = run_optimization(water, "water", OPT_DIR)
    print(f"E[H2O] = {e_water:.5f} eV")

    # ---------------------------------------------------------
    # STEP 2: Optimise Reactant Surface E[SiO2(r)]
    # ---------------------------------------------------------
    print("\n--- Step 2: Optimising Reactant SiO2 Surface ---")
    cif_file = "SiO2-1.cif"
 
    if not os.path.exists(cif_file):
        raise FileNotFoundError(f"Please provide {cif_file} in the working directory.")
    
    reactant = read(cif_file)
    reactant.pbc = PBCS_XYZ
    
    # *** CRITICAL: Assign unique tags to track atoms through deletions ***
    reactant.set_tags(range(len(reactant)))
    
    e_reactant_initial = run_optimization(reactant, "SiO2_reactant", OPT_DIR)
    print(f"E[SiO2(r)] = {e_reactant_initial:.5f} eV")
    
    # Save this optimized reactant as our immutable BASE structure
    # We will copy from this for every single pair calculation
    base_atoms = reactant.copy()

    # ---------------------------------------------------------
    # STEP 3: Independent Pair Dehydroxylation
    # ---------------------------------------------------------
    print("\n--- Step 3: Independent Pair Dehydroxylation (Restoring Base after each) ---")
    
    for pair_data in PAIRS_TO_PROCESS:
        step_id = pair_data['id']
        g1 = pair_data['g1']
        g2 = pair_data['g2']
        
        print(f"\n[Pair ID {step_id}] Processing specified pair...")
        
        # 1. Always start from the fresh, optimized BASE structure
        current_atoms = base_atoms.copy()
        
        # 2. Locate the atoms by TAG in the current structure
        idx_si1 = get_index_by_tag(current_atoms, g1['Si_tag'])
        idx_o1  = get_index_by_tag(current_atoms, g1['O_tag'])
        idx_si2 = get_index_by_tag(current_atoms, g2['Si_tag'])
        idx_o2  = get_index_by_tag(current_atoms, g2['O_tag'])
        
        if None in [idx_si1, idx_o1, idx_si2, idx_o2]:
            print(f"  -> Error: Could not find one or more atoms by tag for Pair {step_id}. Skipping.")
            continue

        # 3. Find the Hydrogens attached to these Oxygens
        idx_h1, tag_h1 = find_attached_hydrogen(current_atoms, idx_o1)
        idx_h2, tag_h2 = find_attached_hydrogen(current_atoms, idx_o2)
        
        if idx_h1 is None or idx_h2 is None:
            print(f"  -> Error: Could not find attached Hydrogens for Oxygen tags {g1['O_tag']} or {g2['O_tag']}. Skipping.")
            continue

        # Calculate initial distance for logging (sanity check vs energies.txt)
        dist_o1_h2 = current_atoms.get_distance(idx_o1, idx_h2, mic=True)
        dist_o2_h1 = current_atoms.get_distance(idx_o2, idx_h1, mic=True)
        metric_dist = min(dist_o1_h2, dist_o2_h1)

        print(f"  -> Found Tags: Si1:{g1['Si_tag']} O1:{g1['O_tag']} H1:{tag_h1} -- Si2:{g2['Si_tag']} O2:{g2['O_tag']} H2:{tag_h2}")
        print(f"  -> Initial Pair Dist (Calculated): {metric_dist:.3f} Å")

        # 4. Prepare folder
        folder_name = f"pair_{step_id}"
        os.makedirs(folder_name, exist_ok=True)
        
        # 5. Reaction Geometry Manipulation
        # -- Move O1 to midpoint --
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
            
        target_pos = pos_o1 + 0.5 * diff_o
        current_atoms.positions[idx_o1] = target_pos
        
        # -- Delete H1, H2, O2 --
        # Important: Calculate indices to remove immediately before deletion
        indices_to_remove = sorted([idx_h1, idx_h2, idx_o2], reverse=True)
        
        reacting_info_str = (
            f"Group 1 (Kept O): Si(Tag:{g1['Si_tag']}) - O(Tag:{g1['O_tag']}) - H(Tag:{tag_h1})\n"
            f"Group 2 (Del O):  Si(Tag:{g2['Si_tag']}) - O(Tag:{g2['O_tag']}) - H(Tag:{tag_h2})\n"
        )
        
        del current_atoms[indices_to_remove]
        
        # 6. Optimization
        col, msg = has_collisions(current_atoms)
        if col:
             print(f"  -> Warning: {msg}. Optimizing...")

        try:
            # Optimize and save results in pair_X folder
            e_product = run_optimization(current_atoms, f"pair_{step_id}_indep", folder_name)
            
            # 7. Calculate Energies (Independent Step Logic)
            # Since we always start from reactant, dE is simply (Product + Water) - Reactant
            d_e_step = (e_product + e_water) - e_reactant_initial
            
            result_txt = (
                f"Step ID: {step_id}\n"
                f"Reaction: Si-OH + HO-Si -> Si-O-Si + H2O\n"
                f"Initial Pair Dist (Calc): {metric_dist:.3f} Å\n"
                f"--- Reacting Atom Tags ---\n"
                f"{reacting_info_str}"
                f"------------------------------------------------------\n"
                f"E_Base_Reactant: {e_reactant_initial:.5f} eV\n"
                f"E_Product:       {e_product:.5f} eV\n"
                f"E_H2O:           {e_water:.5f} eV\n"
                f"Step dE:         {d_e_step:.5f} eV\n"
                f"(Note: dE is calculated independently from the base structure)\n"
            )
            
            with open(os.path.join(folder_name, "results.txt"), "w") as f:
                f.write(result_txt)
                
            print(f"  -> Done. dE = {d_e_step:.3f} eV")
            
        except Exception as e:
            print(f"  -> Optimization Failed: {str(e)}")
            with open(os.path.join(folder_name, "error.txt"), "w") as f:
                f.write(str(e))
            # Continue to next pair even if this one fails
            continue

if __name__ == "__main__":
    main()