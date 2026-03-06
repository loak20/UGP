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
FMAX = 0.02
WATER_CELL = [20.0, 20.0, 20.0]
SEARCH_CUTOFF = 3.0  # Å (Criteria from paper: SiOH...OHSi interaction)
COLLISION_CUTOFF = 0.7 # Å (Safety check)

# Output Directories
OPT_DIR = "optimised_structure"
os.makedirs(OPT_DIR, exist_ok=True)

# Validation Thresholds
# Note: We define distinct cutoffs for what counts as a "bond"
# Si-O bond is typically 1.61 Å -> Cutoff 1.9 covers slight stretch
# O-H bond is typically 0.96 Å -> Cutoff 1.2 covers slight stretch
COORD_CUTOFF_SI_O = 1.9 
COORD_CUTOFF_H_O = 1.2 

MIN_DIST_CRITERIA = {
    "H-H": 0.5, "H-O": 0.8, "H-Si": 1.4,
    "O-O": 1.5, "O-Si": 1.4, "Si-Si": 2.3
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_calculator():
    """Returns a fresh instance of the MACE calculator."""
    return mace_mp(model="medium", dispersion=False, default_dtype="float64", device=DEVICE)

def has_collisions(atoms, cutoff=COLLISION_CUTOFF):
    """
    Checks for dangerous atomic overlaps before optimization starts.
    Returns: (bool, message)
    """
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, 10.0) # Ignore self-interaction
    min_dist = np.min(dists)
    
    if min_dist < cutoff:
        return True, f"Collision detected! Min dist {min_dist:.3f} Å < {cutoff} Å"
    return False, "No collisions"

def validate_structure(atoms):
    """
    Checks for minimum distances and strict coordination numbers.
    Si must have 4 bonds.
    O must have 2 bonds.
    H must have 1 bond.
    Returns (bool, str): (Valid?, Reason)
    """
    # 1. Minimum Distance Check (PBC aware)
    dists = atoms.get_all_distances(mic=True)
    syms = atoms.get_chemical_symbols()
    n = len(atoms)
    
    for i in range(n):
        for j in range(i+1, n):
            pair = "-".join(sorted([syms[i], syms[j]]))
            if pair in MIN_DIST_CRITERIA:
                if dists[i,j] < MIN_DIST_CRITERIA[pair]:
                    return False, f"Clash {pair}: {dists[i,j]:.3f} < {MIN_DIST_CRITERIA[pair]}"

    # 2. Coordination Check
    # We build a neighbor list using species-specific cutoffs to count bonds accurately.
    cutoffs = []
    for s in syms:
        if s == 'Si':
            cutoffs.append(COORD_CUTOFF_SI_O)
        elif s == 'H':
            cutoffs.append(COORD_CUTOFF_H_O)
        elif s == 'O':
            # O needs to see Si (1.9) or H (1.2). We take the max to catch both.
            # However, neighbor_list uses (r_i + r_j)/2 logic or straight cutoffs.
            # To simply check "is bonded", we usually define a global list or specific pair checks.
            # Here we assume a radius such that radius_Si + radius_O approx 1.9.
            # For simplicity in this function, we rely on the specific neighbor_list dictionary method
            # or we pass a single list of radii. 
            # BUT: The most robust way is to use 'neighbor_list' with a dictionary of pair cutoffs 
            # or handle it manually.
            # Let's use the standard "natural cutoffs" multiplied by a small factor if we want, 
            # but here we will manually check neighbors for each atom to be precise.
            cutoffs.append(1.0) # Dummy, we won't use this list directly for everything

    # Better approach: Get full connectivity list with a generous cutoff, then filter by bond type.
    nl = neighbor_list('ij', atoms, cutoff=2.0) # 2.0 covers Si-O (1.6) but avoids Si-Si (3.0)
    
    # Count bonds
    coordination_counts = np.zeros(n, dtype=int)
    
    for k in range(len(nl[0])):
        idx_i = nl[0][k]
        idx_j = nl[1][k]
        
        s_i = syms[idx_i]
        s_j = syms[idx_j]
        
        dist = atoms.get_distance(idx_i, idx_j, mic=True)
        
        is_bonded = False
        
        # Check if this pair constitutes a bond based on our strict cutoffs
        # Si-O bond
        if 'Si' in (s_i, s_j) and 'O' in (s_i, s_j):
            if dist < COORD_CUTOFF_SI_O: is_bonded = True
        # O-H bond
        elif 'H' in (s_i, s_j) and 'O' in (s_i, s_j):
            if dist < COORD_CUTOFF_H_O: is_bonded = True
        
        if is_bonded:
            coordination_counts[idx_i] += 1

    # Verify counts
    for i, sym in enumerate(syms):
        if sym == 'Si':
            if coordination_counts[i] != 4:
                return False, f"Si atom {i} has {coordination_counts[i]} bonds (expected 4)"
        elif sym == 'O':
            if coordination_counts[i] != 2:
                return False, f"O atom {i} has {coordination_counts[i]} bonds (expected 2)"
        elif sym == 'H':
            if coordination_counts[i] != 1:
                return False, f"H atom {i} has {coordination_counts[i]} bonds (expected 1)"
                
    return True, "Valid"

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
    # STEP 3: Identify Silanol Pairs (Modified for Non-Geminal)
    # ---------------------------------------------------------
    print("\n--- Step 3: Finding Silanol Pairs ---")
    
    # Logic: Find H atoms -> find bonded O -> find bonded Si -> Group as Silanol (Si-O-H)
    cutoffs_oh = [1.2 if s == 'H' else 0.0 for s in reactant_opt.get_chemical_symbols()]
    nl_oh = neighbor_list('ij', reactant_opt, cutoffs_oh)
    
    silanols = [] # List of dicts: {'H': idx, 'O': idx, 'Si': idx}
    h_indices = [i for i, s in enumerate(reactant_opt.get_chemical_symbols()) if s == 'H']
    
    for h_idx in h_indices:
        # Find O bonded to H
        neighbors_of_h = nl_oh[1][nl_oh[0] == h_idx]
        
        if len(neighbors_of_h) == 1:
            o_idx = neighbors_of_h[0]
            
            # Find Si bonded to this O (Si-O dist ~ 1.6)
            dists = reactant_opt.get_distances(o_idx, range(len(reactant_opt)), mic=True)
            # Filter for Si within 1.9A
            si_candidates = [i for i, d in enumerate(dists) 
                             if reactant_opt.symbols[i] == 'Si' and d < 1.9]
            
            if len(si_candidates) == 1:
                si_idx = si_candidates[0]
                silanols.append({'H': h_idx, 'O': o_idx, 'Si': si_idx})

    print(f"Found {len(silanols)} silanol groups.")
    
    # Find pairs of silanols based on H-Bond proximity (O...H < Cutoff)
    pairs_to_process = []
    
    for i in range(len(silanols)):
        for j in range(i + 1, len(silanols)):
            s1 = silanols[i]
            s2 = silanols[j]

            # --- NEW CONDITION: PREVENT GEMINAL DEHYDROXYLATION ---
            # If both silanols share the same Si atom, skip them.
            if s1['Si'] == s2['Si']:
                continue
            # ------------------------------------------------------
            
            # Calculate cross distances for H-bonding: O1-H2 and O2-H1
            d_o1_h2 = reactant_opt.get_distance(s1['O'], s2['H'], mic=True)
            d_o2_h1 = reactant_opt.get_distance(s2['O'], s1['H'], mic=True)
            
            if d_o1_h2 < SEARCH_CUTOFF or d_o2_h1 < SEARCH_CUTOFF:
                pairs_to_process.append((s1, s2, min(d_o1_h2, d_o2_h1)))

    print(f"Found {len(pairs_to_process)} candidate pairs for dehydroxylation (excluding geminal pairs).")

    # ---------------------------------------------------------
    # STEP 4 & 5: Simulation Loop
    # ---------------------------------------------------------
    
    for idx, (s1, s2, init_dist) in enumerate(pairs_to_process):
        pair_id = idx + 1
        folder_name = f"pair_{pair_id}"
        os.makedirs(folder_name, exist_ok=True)
        
        print(f"\nProcessing Pair {pair_id}: Si{s1['Si']}-O{s1['O']}H{s1['H']} ... H{s2['H']}O{s2['O']}-Si{s2['Si']}")
        
        # Create a clean copy for this reaction
        sys = reactant_opt.copy()
        
        # --- MODIFIED GEOMETRY LOGIC (FROM CODE 2) ---
        # Instead of placing O at the Si-Si midpoint, we place it at the O-O midpoint.
        # This creates a naturally "bent" bridge.
        
        # 1. Get positions of the two Oxygens
        pos_o1 = sys.get_positions()[s1['O']]
        pos_o2 = sys.get_positions()[s2['O']]
        
        # 2. Calculate the "Ideal" Bridge Position with PBC
        cell = sys.get_cell()
        pbc = sys.get_pbc()
        diff_o = pos_o2 - pos_o1
        
        # Apply MIC wrap
        if pbc.any():
            rec_cell = sys.get_reciprocal_cell()
            scaled = np.dot(diff_o, rec_cell.T)
            scaled -= np.rint(scaled) * pbc
            diff_o = np.dot(scaled, cell)
            
        target_pos = pos_o1 + 0.5 * diff_o
        
        # 3. Move O1 to this target position (O1 becomes the bridge)
        sys.positions[s1['O']] = target_pos
        
        # 4. Identify atoms to remove
        # We keep O1 as the bridge. We remove O2, H1, H2.
        indices_to_remove = [s1['H'], s2['H'], s2['O']]
        indices_to_remove.sort(reverse=True) # Sort reverse to delete safely
        
        del sys[indices_to_remove]
        
        # --- PRE-OPTIMIZATION SAFETY CHECK ---
        collision, msg = has_collisions(sys)
        if collision:
            print(f"  -> Skipping Pair {pair_id}: {msg}")
            with open(os.path.join(folder_name, "skipped.txt"), "w") as f:
                f.write(msg)
            continue
        
        # --- RUN OPTIMIZATION ---
        try:
            e_product = run_optimization(sys, f"pair_{pair_id}_relaxed", folder_name)
            
            # Validation (Now checks Si=4, O=2, H=1)
            is_valid, reason = validate_structure(sys)
            
            # Energy Calculation
            d_e = (e_product + e_water) - e_reactant
            
            result_txt = (
                f"Pair ID: {pair_id}\n"
                f"Reaction: Si{s1['Si']}/Si{s2['Si']} dehydroxylation\n"
                f"Initial H-bond dist: {init_dist:.3f} A\n"
                f"Validation: {is_valid} ({reason})\n"
                f"E_reactant: {e_reactant:.5f} eV\n"
                f"E_product:  {e_product:.5f} eV\n"
                f"E_H2O:      {e_water:.5f} eV\n"
                f"Reaction Energy (dE): {d_e:.5f} eV\n"
            )
            
            with open(os.path.join(folder_name, "results.txt"), "w") as f:
                f.write(result_txt)
                
            print(f"  -> Done. dE = {d_e:.3f} eV. Valid: {is_valid}")
            
        except Exception as e:
            print(f"  -> Failed: {str(e)}")
            with open(os.path.join(folder_name, "error.txt"), "w") as f:
                f.write(str(e))

if __name__ == "__main__":
    main()