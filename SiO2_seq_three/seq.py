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
OPT_DIR = "optimised_structure"
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
    
    # Reload optimized structure
    reactant_opt = read(os.path.join(OPT_DIR, "SiO2_reactant.cif"))
    # Ensure tags are preserved/restored. 
    reactant_opt.set_tags(range(len(reactant_opt)))

    # ---------------------------------------------------------
    # STEP 3: Identify Silanol Pairs
    # ---------------------------------------------------------
    print("\n--- Step 3: Finding Silanol Pairs ---")
    
    cutoffs_oh = [1.2 if s == 'H' else 0.0 for s in reactant_opt.get_chemical_symbols()]
    nl_oh = neighbor_list('ij', reactant_opt, cutoffs_oh)
    
    # List of dicts: stores TAGS instead of indices for robustness
    silanols = [] 
    h_indices = [i for i, s in enumerate(reactant_opt.get_chemical_symbols()) if s == 'H']
    
    for h_idx in h_indices:
        neighbors_of_h = nl_oh[1][nl_oh[0] == h_idx]
        
        if len(neighbors_of_h) == 1:
            o_idx = neighbors_of_h[0]
            
            dists = reactant_opt.get_distances(o_idx, range(len(reactant_opt)), mic=True)
            si_candidates = [i for i, d in enumerate(dists) 
                             if reactant_opt.symbols[i] == 'Si' and d < 1.9]
            
            if len(si_candidates) == 1:
                si_idx = si_candidates[0]
                # Store TAGS
                silanols.append({
                    'H_tag': reactant_opt[h_idx].tag,
                    'O_tag': reactant_opt[o_idx].tag,
                    'Si_tag': reactant_opt[si_idx].tag,
                    # Keep indices just for immediate geometric check
                    'H_idx': h_idx, 'O_idx': o_idx, 'Si_idx': si_idx
                })

    print(f"Found {len(silanols)} silanol groups.")
    
    # Find interacting pairs
    candidate_pairs = []
    
    for i in range(len(silanols)):
        for j in range(i + 1, len(silanols)):
            s1 = silanols[i]
            s2 = silanols[j]
            
            # --- CRITERIA 2: NO GEMINALS ---
            # Check if they share the same Silicon atom tag
            if s1['Si_tag'] == s2['Si_tag']:
                continue
            
            # --- CRITERIA 1: CROSS DISTANCE CHECK ---
            # Distance between O of first and H of second
            d_o1_h2 = reactant_opt.get_distance(s1['O_idx'], s2['H_idx'], mic=True)
            # Distance between O of second and H of first
            d_o2_h1 = reactant_opt.get_distance(s2['O_idx'], s1['H_idx'], mic=True)
            
            # Take the minimum of the two cross interactions
            metric_dist = min(d_o1_h2, d_o2_h1)
            
            if metric_dist < SEARCH_CUTOFF:
                candidate_pairs.append({
                    's1': s1,
                    's2': s2,
                    'dist': metric_dist
                })

    # --- CRITERIA 3: SORT AND PRINT ---
    # Sort by the calculated metric distance
    candidate_pairs.sort(key=lambda x: x['dist'])
    
    print(f"Found {len(candidate_pairs)} valid pairs (non-geminal).")
    print("Top 3 Candidates:")
    for k, p in enumerate(candidate_pairs[:3]):
        print(f"  {k+1}. Dist: {p['dist']:.3f} Å | Si{p['s1']['Si_idx']}-OH ... HO-Si{p['s2']['Si_idx']}")

    # ---------------------------------------------------------
    # STEP 4: Sequential Dehydroxylation
    # ---------------------------------------------------------
    print("\n--- Step 4: Sequential Dehydroxylation ---")
    
    # We maintain a 'current_structure' that evolves
    current_atoms = reactant_opt.copy()
    current_energy = e_reactant_initial
    
    # Keep track of reacted tags to ensure we don't try to react the same group twice
    reacted_tags = set()
    
    processed_count = 0
    max_process = 3
    
    for pair_data in candidate_pairs:
        if processed_count >= max_process:
            break
            
        s1 = pair_data['s1']
        s2 = pair_data['s2']
        
        # Check if these atoms have already been removed
        if (s1['Si_tag'] in reacted_tags) or (s2['Si_tag'] in reacted_tags):
            continue
            
        processed_count += 1
        pair_id = processed_count
        folder_name = f"pair_{pair_id}"
        os.makedirs(folder_name, exist_ok=True)
        
        print(f"\nProcessing Pair {pair_id} (Sequential)")
        print(f"  Reaction: Si(tag {s1['Si_tag']}) ... Si(tag {s2['Si_tag']})")
        
        # --- FIND CURRENT INDICES USING TAGS ---
        idx_h1 = get_index_by_tag(current_atoms, s1['H_tag'])
        idx_o1 = get_index_by_tag(current_atoms, s1['O_tag'])
        idx_h2 = get_index_by_tag(current_atoms, s2['H_tag'])
        idx_o2 = get_index_by_tag(current_atoms, s2['O_tag'])
        
        # Safety check
        if None in [idx_h1, idx_o1, idx_h2, idx_o2]:
            print("  -> Error: Atoms for this pair not found (unexpected overlap). Skipping.")
            continue
            
        # --- GEOMETRY MANIPULATION ---
        
        # 1. Get positions
        pos_o1 = current_atoms.positions[idx_o1]
        pos_o2 = current_atoms.positions[idx_o2]
        
        # 2. Calculate Midpoint of O1 and O2 (Criteria 5)
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
        
        # 3. Move O1 to target (O1 becomes the bridge)
        current_atoms.positions[idx_o1] = target_pos
        
        # 4. Delete H1, H2, O2 (keep O1 as bridge)
        # Important: Delete by index, highest first to preserve lower indices during this specific deletion
        indices_to_remove = sorted([idx_h1, idx_h2, idx_o2], reverse=True)
        del current_atoms[indices_to_remove]
        
        # 5. Mark tags as reacted so we don't reuse them
        reacted_tags.add(s1['Si_tag'])
        reacted_tags.add(s2['Si_tag'])
        
        # --- OPTIMIZATION ---
        
        # Check collisions before running expensive calc
        col, msg = has_collisions(current_atoms)
        if col:
             print(f"  -> Warning: {msg}. Attempting optimization anyway.")

        try:
            # We optimize 'current_atoms' in place
            e_product = run_optimization(current_atoms, f"pair_{pair_id}_sequential", folder_name)
            
            # dE relative to the PREVIOUS step
            d_e_step = (e_product + e_water) - current_energy
            
            # dE relative to INITIAL reactant
            total_water_e = processed_count * e_water
            d_e_total = (e_product + total_water_e) - e_reactant_initial

            result_txt = (
                f"Sequence Step: {pair_id}\n"
                f"Reaction: Si-OH + HO-Si -> Si-O-Si + H2O\n"
                f"Initial Pair Dist: {pair_data['dist']:.3f} Å\n"
                f"E_prev_step: {current_energy:.5f} eV\n"
                f"E_product:   {e_product:.5f} eV\n"
                f"E_H2O:       {e_water:.5f} eV\n"
                f"Step dE:     {d_e_step:.5f} eV\n"
                f"Cumulative dE vs Initial: {d_e_total:.5f} eV\n"
            )
            
            with open(os.path.join(folder_name, "results.txt"), "w") as f:
                f.write(result_txt)
                
            print(f"  -> Optimization Done. Step dE = {d_e_step:.3f} eV")
            
            # Update energy for next step
            current_energy = e_product
            
        except Exception as e:
            print(f"  -> Failed: {str(e)}")
            with open(os.path.join(folder_name, "error.txt"), "w") as f:
                f.write(str(e))
            break

if __name__ == "__main__":
    main()