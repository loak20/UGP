import os
import json
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
FMAX = 1.0  # Relaxed FMAX as requested
WATER_CELL = [20.0, 20.0, 20.0]
SEARCH_CUTOFF = 3.0  # Å (Criteria from paper: SiOH...OHSi interaction)
COLLISION_CUTOFF = 0.7 # Å (Safety check)

# Output Directories
ROOT_DIR = "Tree_Search"
os.makedirs(ROOT_DIR, exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_calculator():
    """Returns a fresh instance of the MACE calculator."""
    return mace_mp(model="medium", dispersion=False, default_dtype="float64", device=DEVICE)

def has_collisions(atoms, cutoff=COLLISION_CUTOFF):
    """
    Checks for dangerous atomic overlaps.
    Returns: (bool, message)
    """
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, 10.0) # Ignore self-interaction
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
    
    # Save CIF
    cif_path = os.path.join(folder, f"{label}.cif")
    write(cif_path, atoms)
    
    return atoms.get_potential_energy()

def find_silanol_pairs(atoms):
    """
    Identifies all valid non-geminal silanol pairs in the given structure.
    Returns a list of tuples: (s1_dict, s2_dict, distance)
    """
    cutoffs_oh = [1.2 if s == 'H' else 0.0 for s in atoms.get_chemical_symbols()]
    nl_oh = neighbor_list('ij', atoms, cutoffs_oh)
    
    silanols = [] # List of dicts: {'H': idx, 'O': idx, 'Si': idx}
    h_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'H']
    
    for h_idx in h_indices:
        # Find O bonded to H
        neighbors_of_h = nl_oh[1][nl_oh[0] == h_idx]
        
        if len(neighbors_of_h) == 1:
            o_idx = neighbors_of_h[0]
            
            # Find Si bonded to this O (Si-O dist ~ 1.6)
            dists = atoms.get_distances(o_idx, range(len(atoms)), mic=True)
            # Filter for Si within 1.9A
            si_candidates = [i for i, d in enumerate(dists) 
                             if atoms.symbols[i] == 'Si' and d < 1.9]
            
            if len(si_candidates) == 1:
                si_idx = si_candidates[0]
                silanols.append({'H': h_idx, 'O': o_idx, 'Si': si_idx})

    # Find pairs based on H-Bond proximity
    valid_pairs = []
    
    for i in range(len(silanols)):
        for j in range(i + 1, len(silanols)):
            s1 = silanols[i]
            s2 = silanols[j]

            # PREVENT GEMINAL: If both silanols share the same Si atom, skip.
            if s1['Si'] == s2['Si']:
                continue
            
            # Calculate cross distances for H-bonding
            d_o1_h2 = atoms.get_distance(s1['O'], s2['H'], mic=True)
            d_o2_h1 = atoms.get_distance(s2['O'], s1['H'], mic=True)
            
            if d_o1_h2 < SEARCH_CUTOFF or d_o2_h1 < SEARCH_CUTOFF:
                valid_pairs.append((s1, s2, min(d_o1_h2, d_o2_h1)))
                
    return valid_pairs

# =============================================================================
# RECURSIVE TREE SEARCH LOGIC
# =============================================================================

def process_node(current_atoms, current_path, parent_energy, e_water, node_label):
    """
    Recursive function to explore the reaction tree.
    1. Finds pairs in current_atoms.
    2. Branches for each pair.
    3. Optimizes and recurses.
    """
    print(f"\n[{node_label}] Analyzing structure for reactive pairs...")
    
    # Find candidates in this current structure
    pairs = find_silanol_pairs(current_atoms)
    
    if not pairs:
        print(f"[{node_label}] No valid pairs found. Leaf node reached.")
        return

    print(f"[{node_label}] Found {len(pairs)} candidate pairs. Branching...")

    for idx, (s1, s2, init_dist) in enumerate(pairs):
        pair_id = idx + 1
        # Construct folder name: e.g. pair_1, pair_2 inside the current folder
        branch_folder_name = f"pair_{pair_id}"
        full_branch_path = os.path.join(current_path, branch_folder_name)
        os.makedirs(full_branch_path, exist_ok=True)
        
        # New label for the recursion: e.g. R_1 -> R_1_1
        new_label = f"{node_label}_{pair_id}"
        
        print(f"  > Processing Branch: {new_label} (Si{s1['Si']}-Si{s2['Si']})")
        
        # --- EXECUTE REACTION ---
        # Create a copy for this specific branch
        sys = current_atoms.copy()
        
        # 1. Geometry Move: Move O1 to O1-O2 midpoint (with PBC)
        pos_o1 = sys.get_positions()[s1['O']]
        pos_o2 = sys.get_positions()[s2['O']]
        
        cell = sys.get_cell()
        pbc = sys.get_pbc()
        diff_o = pos_o2 - pos_o1
        
        if pbc.any():
            rec_cell = sys.get_reciprocal_cell()
            scaled = np.dot(diff_o, rec_cell.T)
            scaled -= np.rint(scaled) * pbc
            diff_o = np.dot(scaled, cell)
            
        target_pos = pos_o1 + 0.5 * diff_o
        sys.positions[s1['O']] = target_pos
        
        # 2. Delete Atoms (O2, H1, H2)
        indices_to_remove = [s1['H'], s2['H'], s2['O']]
        indices_to_remove.sort(reverse=True)
        del sys[indices_to_remove]
        
        # 3. Collision Check
        collision, msg = has_collisions(sys)
        if collision:
            print(f"    x Skipped {new_label}: {msg}")
            with open(os.path.join(full_branch_path, "skipped.txt"), "w") as f:
                f.write(msg)
            continue # Do not recurse on this invalid branch
            
        # 4. Optimization
        try:
            # Optimize the new node
            e_product = run_optimization(sys, f"structure_{new_label}", full_branch_path)
            
            # Calc Energy
            d_e = (e_product + e_water) - parent_energy
            
            # Save Results
            result_txt = (
                f"Node Label: {new_label}\n"
                f"Parent Path: {current_path}\n"
                f"Reaction: Si{s1['Si']}/Si{s2['Si']} dehydroxylation\n"
                f"Initial H-bond dist: {init_dist:.3f} A\n"
                f"E_parent:   {parent_energy:.5f} eV\n"
                f"E_product:  {e_product:.5f} eV\n"
                f"E_H2O:      {e_water:.5f} eV\n"
                f"Reaction Energy (dE): {d_e:.5f} eV\n"
            )
            
            with open(os.path.join(full_branch_path, "results.txt"), "w") as f:
                f.write(result_txt)
                
            print(f"    v Done {new_label}. dE = {d_e:.3f} eV. Recursing...")
            
            # 5. RECURSION STEP
            # Use the newly optimized product as the root for the next depth
            process_node(sys, full_branch_path, e_product, e_water, new_label)
            
        except Exception as e:
            print(f"    ! Failed {new_label}: {str(e)}")
            with open(os.path.join(full_branch_path, "error.txt"), "w") as f:
                f.write(str(e))

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    # ---------------------------------------------------------
    # STEP 1: Calculate E[H2O]
    # ---------------------------------------------------------
    print("\n--- Step 1: Optimising isolated Water molecule ---")
    water_dir = os.path.join(ROOT_DIR, "Water_Ref")
    os.makedirs(water_dir, exist_ok=True)
    
    water = molecule("H2O")
    water.set_cell(WATER_CELL)
    water.center()
    water.pbc = PBCS_XYZ
    
    e_water = run_optimization(water, "water", water_dir)
    print(f"E[H2O] = {e_water:.5f} eV")

    # ---------------------------------------------------------
    # STEP 2: Optimise Root SiO2 Surface
    # ---------------------------------------------------------
    print("\n--- Step 2: Optimising Root SiO2 Surface ---")
    cif_file = "SiO2-1.cif"
    if not os.path.exists(cif_file):
        raise FileNotFoundError(f"Please provide {cif_file} in the working directory.")
    
    reactant = read(cif_file)
    reactant.pbc = PBCS_XYZ
    
    # Create Root folder inside Tree Search
    root_folder = os.path.join(ROOT_DIR, "Root")
    os.makedirs(root_folder, exist_ok=True)
    
    e_root = run_optimization(reactant, "root_structure", root_folder)
    print(f"E[Root] = {e_root:.5f} eV")
    
    # Reload optimized root to ensure clean state
    root_opt = read(os.path.join(root_folder, "root_structure.cif"))
    
    # ---------------------------------------------------------
    # STEP 3: Start Recursive Tree Search
    # ---------------------------------------------------------
    print("\n--- Step 3: Starting Recursive Tree Search ---")
    process_node(
        current_atoms=root_opt,
        current_path=root_folder,
        parent_energy=e_root,
        e_water=e_water,
        node_label="R"
    )
    
    print("\n--- Tree Search Complete ---")

if __name__ == "__main__":
    main()