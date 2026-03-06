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
SEARCH_CUTOFF = 3.0  # Å (Interaction search radius)
COLLISION_CUTOFF = 0.7 # Å (Threshold for "bad" overlap)

# Output Directories
OPT_DIR = "optimised_structure"
os.makedirs(OPT_DIR, exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_calculator():
    """Returns a fresh instance of the MACE calculator."""
    return mace_mp(model="medium", dispersion=False, default_dtype="float64", device=DEVICE)

def has_collisions(atoms, cutoff=COLLISION_CUTOFF):
    """
    Simple check to ensure no two atoms are dangerously close.
    Returns: (bool, message)
    """
    # Get all distances (with mic=True for PBC)
    # We want to avoid self-distance (0.0), so we add identity matrix or filter
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
    
    # Use LBFGS (standard for relaxation)
    opt = LBFGS(atoms, trajectory=traj_path, logfile=log_path)
    opt.run(fmax=FMAX)
    
    # Save CIF
    cif_path = os.path.join(folder, f"{label}.cif")
    write(cif_path, atoms)
    
    return atoms.get_potential_energy()

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
    
    e_reactant = run_optimization(reactant, "SiO2_reactant", OPT_DIR)
    print(f"E[SiO2(r)] = {e_reactant:.5f} eV")
    
    # Reload optimized structure to ensure clean state
    reactant_opt = read(os.path.join(OPT_DIR, "SiO2_reactant.cif"))
    
    # ---------------------------------------------------------
    # STEP 3: Identify Silanol Pairs
    # ---------------------------------------------------------
    print("\n--- Step 3: Finding Silanol Pairs ---")
    
    # Build connectivity using NeighborList
    cutoffs_oh = [1.2 if s == 'H' else 0.0 for s in reactant_opt.get_chemical_symbols()]
    nl_oh = neighbor_list('ij', reactant_opt, cutoffs_oh)
    
    silanols = [] # List of dicts: {'H': idx, 'O': idx, 'Si': idx}
    h_indices = [i for i, s in enumerate(reactant_opt.get_chemical_symbols()) if s == 'H']
    
    for h_idx in h_indices:
        # Find O bonded to H
        neighbors_of_h = nl_oh[1][nl_oh[0] == h_idx]
        
        if len(neighbors_of_h) == 1:
            o_idx = neighbors_of_h[0]
            
            # Find Si bonded to this O (manual distance check < 1.9 A)
            dists = reactant_opt.get_distances(o_idx, range(len(reactant_opt)), mic=True)
            si_candidates = [i for i, d in enumerate(dists) 
                             if reactant_opt.symbols[i] == 'Si' and d < 1.9]
            
            if len(si_candidates) == 1:
                si_idx = si_candidates[0]
                silanols.append({'H': h_idx, 'O': o_idx, 'Si': si_idx})

    print(f"Found {len(silanols)} silanol groups.")
    
    # Find interacting pairs
    pairs_to_process = []
    
    for i in range(len(silanols)):
        for j in range(i + 1, len(silanols)):
            s1 = silanols[i]
            s2 = silanols[j]
            
            # Check proximity (O-H...O or O...O)
            # Checking cross distance O1...O2 or O1...H2
            d_o1_o2 = reactant_opt.get_distance(s1['O'], s2['O'], mic=True)
            
            if d_o1_o2 < SEARCH_CUTOFF:
                pairs_to_process.append((s1, s2, d_o1_o2))

    print(f"Found {len(pairs_to_process)} candidate pairs for dehydroxylation.")

    # ---------------------------------------------------------
    # STEP 4: Dehydroxylation Loop
    # ---------------------------------------------------------
    
    for idx, (s1, s2, init_dist) in enumerate(pairs_to_process):
        pair_id = idx + 1
        folder_name = f"pair_{pair_id}"
        os.makedirs(folder_name, exist_ok=True)
        
        print(f"\nProcessing Pair {pair_id}: Si{s1['Si']}-O{s1['O']} ... O{s2['O']}-Si{s2['Si']}")
        
        # Create work copy
        sys = reactant_opt.copy()
        
        # --- GEOMETRY MANIPULATION ---
        
        # 1. Get positions of the two Oxygens
        pos_o1 = sys.get_positions()[s1['O']]
        pos_o2 = sys.get_positions()[s2['O']]
        
        # 2. Calculate the "Ideal" Bridge Position
        # Instead of Si-Si midpoint (which yields 180 deg angle), 
        # we use the O-O midpoint. Since silanol Os stick out from surface,
        # their midpoint naturally creates a "bent" bridge above the Si-Si axis.
        
        # Need to handle PBC for the midpoint calculation
        cell = sys.get_cell()
        pbc = sys.get_pbc()
        diff_o = pos_o2 - pos_o1
        
        # Apply MIC wrap to finding the difference vector
        if pbc.any():
            rec_cell = sys.get_reciprocal_cell()
            scaled = np.dot(diff_o, rec_cell.T)
            scaled -= np.rint(scaled) * pbc
            diff_o = np.dot(scaled, cell)
            
        target_pos = pos_o1 + 0.5 * diff_o
        
        # 3. Move O1 to this target position (O1 becomes the bridge)
        sys.positions[s1['O']] = target_pos
        
        # 4. Identify atoms to remove (H1, H2, O2)
        # We keep O1 as the bridge. We remove the rest of the water components.
        indices_to_remove = [s1['H'], s2['H'], s2['O']]
        indices_to_remove.sort(reverse=True)
        
        # 5. Check for Collisions BEFORE deletion? 
        # No, check AFTER deletion/move, because we might have moved O1 into a bad spot
        # relative to neighbors, or the deletion might clear space.
        
        del sys[indices_to_remove]
        
        # --- PRE-OPTIMIZATION CHECK ---
        collision, msg = has_collisions(sys)
        if collision:
            print(f"  -> Skipping Pair {pair_id}: {msg}")
            with open(os.path.join(folder_name, "skipped.txt"), "w") as f:
                f.write(msg)
            continue

        # --- OPTIMIZATION ---
        try:
            e_product = run_optimization(sys, f"pair_{pair_id}_relaxed", folder_name)
            
            # dE = (E_product + E_water) - E_reactant
            d_e = (e_product + e_water) - e_reactant
            
            result_txt = (
                f"Pair ID: {pair_id}\n"
                f"Reaction: Si{s1['Si']}-OH + HO-Si{s2['Si']} -> Si-O-Si + H2O\n"
                f"E_reactant: {e_reactant:.5f} eV\n"
                f"E_product:  {e_product:.5f} eV\n"
                f"E_H2O:      {e_water:.5f} eV\n"
                f"Reaction Energy (dE): {d_e:.5f} eV\n"
            )
            
            with open(os.path.join(folder_name, "results.txt"), "w") as f:
                f.write(result_txt)
                
            print(f"  -> Done. dE = {d_e:.3f} eV")
            
        except Exception as e:
            print(f"  -> Failed: {str(e)}")
            with open(os.path.join(folder_name, "error.txt"), "w") as f:
                f.write(str(e))

if __name__ == "__main__":
    main()