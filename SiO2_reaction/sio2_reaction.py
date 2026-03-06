import os
import warnings
import numpy as np
import torch
from ase.io import read, write
from ase.build import molecule
from ase.optimize import LBFGS
from ase.neighborlist import neighbor_list
from mace.calculators import mace_mp

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# User Parameters
START_FILE = "SiO2-1_opt_lbfgs.cif"     # Starting Structure
MAX_SEARCH_RADIUS = 5.0           # Max distance to look for pairs
TARGET_BOND_LENGTH = 1.65         # Ideal Si-O bond length
MIN_SAFE_DISTANCE = 1.3           # CRITICAL: Min distance to avoid fusion/crash
FMAX_TARGET = 0.02                # Strict convergence criteria
MAX_ATTEMPTS = 50                 # How many pairs to check per cycle before stopping

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

def get_calculator():
    # float64 is required for strict fmax=0.02
    return mace_mp(model="medium", device=device, default_dtype="float64")

# Helper: Check for collisions
def check_overlap(structure, position, ignored_indices, min_dist=1.2):
    """Returns True if 'position' is too close to any valid atom."""
    valid_indices = [i for i in range(len(structure)) if i not in ignored_indices]
    if not valid_indices: return False
    
    valid_positions = structure.positions[valid_indices]
    dists = np.linalg.norm(valid_positions - position, axis=1)
    
    if np.min(dists) < min_dist:
        return True # Crash detected
    return False

# ==========================================
# 2. REFERENCE CALCULATIONS
# ==========================================
print("\n--- Step 1: Water Reference Optimization ---")

# Build and optimize isolated Water
water = molecule('H2O')
water.set_cell([20.0, 20.0, 20.0]); water.center(); water.set_pbc([True, True, False])
water.calc = get_calculator()

# Strict optimization for water
opt_w = LBFGS(water, logfile="h2o_opt.log", memory=100)
opt_w.run(fmax=FMAX_TARGET)
E_h2o = water.get_potential_energy()
print(f"Reference Energy (H2O): {E_h2o:.5f} eV")

print("\n--- Step 2: Loading SiO2 Surface ---")

if not os.path.exists(START_FILE):
    # Fallback to try finding a generic file if specific name missing
    import glob
    files = glob.glob("*.cif")
    if files:
        print(f"Warning: {START_FILE} not found. Using {files[0]} instead.")
        START_FILE = files[0]
    else:
        raise FileNotFoundError(f"Missing input file: {START_FILE}")

current_structure = read(START_FILE)
current_structure.calc = get_calculator()
E_prev = current_structure.get_potential_energy()
print(f"Initial Surface Energy: {E_prev:.5f} eV")


# ==========================================
# 3. ITERATIVE DEHYDROXYLATION LOOP
# ==========================================
print("\n--- Step 3: Starting Smart Dehydroxylation ---")

step_count = 0
results = []

while True:
    step_count += 1
    print(f"\n[Cycle {step_count}] Scanning for interacting pairs...")

    # A. IDENTIFY SILANOL GROUPS (H -> O -> Si)
    # -----------------------------------------
    nl = neighbor_list('ij', current_structure, {('O', 'H'): 1.2, ('Si', 'O'): 2.0})
    adj = [[] for _ in range(len(current_structure))]
    for i, j in zip(nl[0], nl[1]):
        adj[i].append(j)

    silanols = []
    for idx_h in range(len(current_structure)):
        if current_structure[idx_h].symbol == 'H':
            connected_o = [n for n in adj[idx_h] if current_structure[n].symbol == 'O']
            if len(connected_o) == 1:
                idx_o = connected_o[0]
                connected_si = [n for n in adj[idx_o] if current_structure[n].symbol == 'Si']
                if len(connected_si) == 1:
                    idx_si = connected_si[0]
                    silanols.append({'H': idx_h, 'O': idx_o, 'Si': idx_si})

    if len(silanols) < 2:
        print("Stopping: Fewer than 2 silanols remaining.")
        break

    # B. GENERATE & SORT CANDIDATES
    # -----------------------------
    candidates = []
    positions = current_structure.get_positions()

    for i in range(len(silanols)):
        for j in range(i + 1, len(silanols)):
            g1 = silanols[i]; g2 = silanols[j]
            
            # Filter: Must be different Silicons (No Geminal on same Si)
            if g1['Si'] == g2['Si']: continue 

            # Metric: O-O Distance
            dist = np.linalg.norm(positions[g1['O']] - positions[g2['O']])
            if dist < MAX_SEARCH_RADIUS:
                candidates.append((dist, g1, g2))
    
    # Sort closest first
    candidates.sort(key=lambda x: x[0])

    if not candidates:
        print(f"Stopping: No valid pairs found within {MAX_SEARCH_RADIUS} Å.")
        break

    # C. SMART SELECTION (Check Overlaps)
    # -----------------------------------
    selected_pair = None
    best_new_pos = None
    final_dist = 0
    
    print(f"  > Checking safety of {len(candidates)} candidates...")
    
    # Loop through top candidates to find one that isn't blocked
    for dist, g1, g2 in candidates[:MAX_ATTEMPTS]:
        
        # Atoms to be deleted (ignore them in collision check)
        ignore_indices = [g1['H'], g2['H'], g2['O'], g1['O']] # Temp ignore O1 to check its new spot

        # Calc Midpoint
        p1 = positions[g1['Si']]
        p2 = positions[g2['Si']]
        midpoint = (p1 + p2) / 2.0
        vec = p2 - p1
        half_dist = np.linalg.norm(vec) / 2.0
        
        # Calc Height
        if half_dist < TARGET_BOND_LENGTH:
            h = np.sqrt(TARGET_BOND_LENGTH**2 - half_dist**2)
        else:
            h = 0.5 

        # Strategy 1: Standard Placement
        
        pos_std = midpoint.copy()
        pos_std[2] += h
        
        if not check_overlap(current_structure, pos_std, ignore_indices, MIN_SAFE_DISTANCE):
            selected_pair = (g1, g2)
            best_new_pos = pos_std
            final_dist = dist
            print(f"  > Selected Pair: Dist {dist:.3f} Å (Safe)")
            break
            
        # Strategy 2: High Lift (Avoid "Mountain" Obstruction)
        
        pos_high = pos_std.copy()
        pos_high[2] += 0.8 # Lift extra 0.8 Å
        
        if not check_overlap(current_structure, pos_high, ignore_indices, MIN_SAFE_DISTANCE):
            selected_pair = (g1, g2)
            best_new_pos = pos_high
            final_dist = dist
            print(f"  > Selected Pair: Dist {dist:.3f} Å (Nudged High to avoid clash)")
            break
    
    if selected_pair is None:
        print("Stopping: All reachable pairs are obstructed/unsafe.")
        break

    # D. APPLY REACTION
    # -----------------
    g1, g2 = selected_pair
    # Delete H1, H2, O2
    atoms_to_delete = sorted([g1['H'], g2['H'], g2['O']], reverse=True)
    
    # Move O1 to new safe spot
    current_structure.positions[g1['O']] = best_new_pos
    del current_structure[atoms_to_delete]
    
    # E. OPTIMIZE STRUCTURE
    # ---------------------
    print(f"  > Optimizing Step {step_count} (fmax={FMAX_TARGET})...")
    current_structure.calc = get_calculator()
    
    # Save trajectory for movie making later
    opt = LBFGS(current_structure, 
                logfile=f"opt_step_{step_count}.log", 
                trajectory=f"opt_step_{step_count}.traj", # <--- NEW: Enables Smooth Movie
                maxstep=0.05, # Conservative step size
                memory=100)
    
    try:
        opt.run(fmax=FMAX_TARGET, steps=500)
    except Exception as e:
        print(f"  ! Crash at Step {step_count}: {e}")
        write(f"SiO2_crash_step_{step_count}.cif", current_structure)
        break

    # F. STORE RESULTS
    # ----------------
    E_curr = current_structure.get_potential_energy()
    dE_step = E_curr + E_h2o - E_prev
    
    print(f"  > Done. Energy: {E_curr:.5f} eV | dE: {dE_step:.5f} eV")
    
    results.append({
        "step": step_count,
        "Si1": g1['Si'],
        "Si2": g2['Si'],
        "dist_OO": final_dist,
        "dE_reaction": dE_step
    })
    
    E_prev = E_curr
    write(f"SiO2_step_{step_count}.cif", current_structure)


# ==========================================
# 4. FINAL REPORT
# ==========================================
print("\n=======================================================")
print("             DEHYDROXYLATION SUMMARY                   ")
print("=======================================================")
print(f"{'Step':<5} | {'Si-Si Pair':<15} | {'Dist (Å)':<10} | {'dE (eV)':<10}")
print("-" * 55)

total_dE = 0.0
if len(results) > 0:
    for res in results:
        pair_str = f"{res['Si1']}-{res['Si2']}"
        print(f"{res['step']:<5} | {pair_str:<15} | {res['dist_OO']:<10.3f} | {res['dE_reaction']:<10.5f}")
        total_dE += res['dE_reaction']

    avg_dE = total_dE / len(results)
    print("-" * 55)
    print(f"Total H2O Removed:   {len(results)}")
    print(f"Total Energy Cost:   {total_dE:.5f} eV")
    print(f"Average dE per H2O:  {avg_dE:.5f} eV")
    
    write("SiO2_final_dehydroxylated.cif", current_structure)
else:
    print("No reactions occurred.")
print("=======================================================")