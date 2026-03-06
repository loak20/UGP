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
SEARCH_CUTOFF = 3.0  
COLLISION_CUTOFF = 0.7 

# Output Directories
# OPT_DIR = "optimised_structure"
OPT_DIR = "optimised_mod"
os.makedirs(OPT_DIR, exist_ok=True)

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

def find_silanol_pairs(atoms, cutoff=SEARCH_CUTOFF):
    """
    Identifies all Silanol (Si-OH) groups in the provided atoms object
    and returns a list of candidate pairs sorted by O-H distance.
    """
    # 1. Identify OH groups
    # Use symbol-based cutoffs for neighbor list
    cutoffs_oh = [1.2 if s == 'H' else 0.0 for s in atoms.get_chemical_symbols()]
    nl_oh = neighbor_list('ij', atoms, cutoffs_oh)
    
    silanols = [] 
    h_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'H']
    
    for h_idx in h_indices:
        neighbors_of_h = nl_oh[1][nl_oh[0] == h_idx]
        
        if len(neighbors_of_h) == 1:
            o_idx = neighbors_of_h[0]
            
            # Find Si attached to O (dist < 1.9 A)
            dists = atoms.get_distances(o_idx, range(len(atoms)), mic=True)
            si_candidates = [i for i, d in enumerate(dists) 
                             if atoms.symbols[i] == 'Si' and d < 1.9]
            
            if len(si_candidates) == 1:
                si_idx = si_candidates[0]
                # Store TAGS (persistent IDs) and current indices
                silanols.append({
                    'H_tag': atoms[h_idx].tag,
                    'O_tag': atoms[o_idx].tag,
                    'Si_tag': atoms[si_idx].tag,
                    'H_idx': h_idx, 'O_idx': o_idx, 'Si_idx': si_idx
                })

    # 2. Find interacting pairs
    pairs = []
    for i in range(len(silanols)):
        for j in range(i + 1, len(silanols)):
            s1 = silanols[i]
            s2 = silanols[j]
            
            # Check Geminal (same Si)
            if s1['Si_tag'] == s2['Si_tag']:
                continue
            
            # Cross distances
            d_o1_h2 = atoms.get_distance(s1['O_idx'], s2['H_idx'], mic=True)
            d_o2_h1 = atoms.get_distance(s2['O_idx'], s1['H_idx'], mic=True)
            
            metric_dist = min(d_o1_h2, d_o2_h1)
            
            if metric_dist < cutoff:
                pairs.append({
                    's1': s1,
                    's2': s2,
                    'dist': metric_dist
                })
    
    # Sort by distance (closest first)
    pairs.sort(key=lambda x: x['dist'])
    return pairs

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
    
    # ---------------------------------------------------------
    # STEP 3 & 4: Dynamic Sequential Dehydroxylation
    # ---------------------------------------------------------
    print("\n--- Step 3: Dynamic Sequential Dehydroxylation ---")
    
    # Initialize current state with the optimized reactant
    current_atoms = read(os.path.join(OPT_DIR, "SiO2_reactant.cif"))
    # Ensure tags are restored (ASE usually preserves them in CIF if written correctly, 
    # but to be safe we re-assign if missing, though typically we rely on memory copy)
    # Better: Use the in-memory object 'reactant' which definitely has tags.
    # However, 'run_optimization' modifies 'reactant' in place, so we are good.
    current_atoms = reactant.copy()
    
    current_energy = e_reactant_initial
    
    # We want to perform exactly 3 steps
    MAX_STEPS = 3
    
    for step_num in range(1, MAX_STEPS + 1):
        print(f"\n[Iteration {step_num}/{MAX_STEPS}] Scanning for best silanol pair...")
        
        # 1. Recalculate pairs on the CURRENT geometry
        candidate_pairs = find_silanol_pairs(current_atoms, SEARCH_CUTOFF)
        
        if not candidate_pairs:
            print("  -> No valid silanol pairs found within cutoff. Stopping early.")
            break
            
        # 2. Choose the one with min dist (first in sorted list)
        best_pair = candidate_pairs[0]
        s1 = best_pair['s1']
        s2 = best_pair['s2']
        
        print(f"  -> Selected Pair: Dist {best_pair['dist']:.3f} Å")
        print(f"     Si(tag {s1['Si_tag']}) ... Si(tag {s2['Si_tag']})")
        
        # 3. Prepare folder
        folder_name = f"pair_mod_{step_num}"
        os.makedirs(folder_name, exist_ok=True)
        
        # 4. Reaction Geometry Manipulation
        # Re-find indices using TAGS because indices shift after every deletion
        idx_h1 = get_index_by_tag(current_atoms, s1['H_tag'])
        idx_o1 = get_index_by_tag(current_atoms, s1['O_tag'])
        idx_h2 = get_index_by_tag(current_atoms, s2['H_tag'])
        idx_o2 = get_index_by_tag(current_atoms, s2['O_tag'])
        
        if None in [idx_h1, idx_o1, idx_h2, idx_o2]:
            print("  -> Error: Could not locate atoms by tag. Skipping this step.")
            continue

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
        indices_to_remove = sorted([idx_h1, idx_h2, idx_o2], reverse=True)
        del current_atoms[indices_to_remove]
        
        # 5. Optimization
        col, msg = has_collisions(current_atoms)
        if col:
             print(f"  -> Warning: {msg}. Optimizing...")

        try:
            # Optimize and save results in pair_X folder
            e_product = run_optimization(current_atoms, f"pair_{step_num}_sequential", folder_name)
            
            # 6. Calculate Energies
            # dE relative to the PREVIOUS step
            d_e_step = (e_product + e_water) - current_energy
            
            # dE relative to INITIAL reactant
            total_water_e = step_num * e_water
            d_e_total = (e_product + total_water_e) - e_reactant_initial

            result_txt = (
                f"Step: {step_num}\n"
                f"Reaction: Si-OH + HO-Si -> Si-O-Si + H2O\n"
                f"Initial Pair Dist: {best_pair['dist']:.3f} Å\n"
                f"E_prev_step: {current_energy:.5f} eV\n"
                f"E_product:   {e_product:.5f} eV\n"
                f"E_H2O:       {e_water:.5f} eV\n"
                f"Step dE:     {d_e_step:.5f} eV\n"
                f"Cumulative dE vs Initial: {d_e_total:.5f} eV\n"
            )
            
            with open(os.path.join(folder_name, "results.txt"), "w") as f:
                f.write(result_txt)
                
            print(f"  -> Done. Step dE = {d_e_step:.3f} eV")
            
            # Update energy for the next iteration
            current_energy = e_product
            
        except Exception as e:
            print(f"  -> Optimization Failed: {str(e)}")
            with open(os.path.join(folder_name, "error.txt"), "w") as f:
                f.write(str(e))
            break

if __name__ == "__main__":
    main()