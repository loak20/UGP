import os
import glob
import re
import warnings
import numpy as np
import torch
from ase.io import read, write
from ase.build import molecule
from ase.optimize import LBFGS, FIRE
from ase.neighborlist import neighbor_list
from mace.calculators import mace_mp

# ==========================================
# 1. CONFIGURATION
# ==========================================
RESUME_FROM_FILE = "SiO2_step_13.cif"      # Set to "SiO2_step_13.cif" to force a specific step, or None to auto-detect
MAX_SEARCH_RADIUS = 5.0       # Max distance to look for pairs (Å)
TARGET_BOND_LENGTH = 1.65     # Ideal Si-O bond length
MIN_SAFE_DISTANCE = 1.3       # CRITICAL: Min distance to avoid overlapping atoms
FMAX_TARGET = 0.05            # Convergence target (eV/Å) - 0.05 is standard for surfaces
MAX_ATTEMPTS = 50             # How many pairs to check before giving up on a cycle

# Suppress Warnings & Setup Device
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Running on {device.upper()} ---")

def get_calculator():
    # Use float64 for stability in tight optimization
    return mace_mp(model="medium", device=device, default_dtype="float64")

def check_overlap(structure, position, ignored_indices, min_dist=1.2):
    """Returns True if position is too close to any existing atom."""
    valid_indices = [i for i in range(len(structure)) if i not in ignored_indices]
    if not valid_indices: return False
    
    # Vectorized distance check
    dists = np.linalg.norm(structure.positions[valid_indices] - position, axis=1)
    if np.min(dists) < min_dist:
        return True 
    return False

# ==========================================
# 2. INITIALIZATION
# ==========================================
print("\n--- Step 1: Initialization ---")

# A. Water Reference (for dE calculation)
water = molecule('H2O')
water.set_cell([20.0, 20.0, 20.0]); water.center(); water.set_pbc([True, True, False])
water.calc = get_calculator()
if not os.path.exists("h2o_reference.log"):
    opt_w = LBFGS(water, logfile="h2o_reference.log") 
    opt_w.run(fmax=FMAX_TARGET)
E_h2o = water.get_potential_energy()

# B. Load Simulation State
if RESUME_FROM_FILE and os.path.exists(RESUME_FROM_FILE):
    # Manual Override
    print(f"  > Manual Resume: Loading {RESUME_FROM_FILE}")
    current_structure = read(RESUME_FROM_FILE)
    match = re.search(r'step_(\d+)', RESUME_FROM_FILE)
    step_count = int(match.group(1)) if match else 0
else:
    # Auto-Detect Latest File
    files = glob.glob("SiO2_step_*.cif")
    if files:
        files.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
        latest_file = files[-1]
        print(f"  > Auto-Resume: Found latest checkpoint {latest_file}")
        current_structure = read(latest_file)
        step_count = int(re.search(r'step_(\d+)', latest_file).group(1))
    else:
        # Fallback to initial reactant
        starters = ["SiO2-1_opt.cif", "SiO2-1_opt_lbfgs.cif"]
        start_file = next((f for f in starters if os.path.exists(f)), None)
        if start_file:
            print(f"  > Starting from Scratch: {start_file}")
            current_structure = read(start_file)
            step_count = 0
        else:
            raise FileNotFoundError("No input files found! Put SiO2-1_opt.cif in this folder.")

# FORCE SAFETY: Explicitly set 2D Surface Boundary Conditions
current_structure.set_pbc([True, True, False]) 
print(f"  > PBC enforced: {current_structure.get_pbc()}")

current_structure.calc = get_calculator()
E_prev = current_structure.get_potential_energy()

# ==========================================
# 3. MAIN SIMULATION LOOP
# ==========================================
print("\n--- Step 2: Robust Dehydroxylation Loop ---")

while True:
    step_count += 1
    print(f"\n[Cycle {step_count}] Scanning surface...")

    # A. Identify Silanols (H -> O -> Si)
    # ------------------------------------
    nl = neighbor_list('ij', current_structure, {('O', 'H'): 1.2, ('Si', 'O'): 2.0})
    adj = [[] for _ in range(len(current_structure))]
    for i, j in zip(nl[0], nl[1]): adj[i].append(j)

    silanols = []
    for idx_h in range(len(current_structure)):
        if current_structure[idx_h].symbol == 'H':
            connected_o = [n for n in adj[idx_h] if current_structure[n].symbol == 'O']
            if len(connected_o) == 1:
                connected_si = [n for n in adj[connected_o[0]] if current_structure[n].symbol == 'Si']
                if len(connected_si) == 1:
                    silanols.append({'H': idx_h, 'O': connected_o[0], 'Si': connected_si[0]})

    if len(silanols) < 2:
        print("Stopping: Surface is dry (< 2 silanols).")
        break

    # B. Generate Candidates
    # ----------------------
    candidates = []
    pos = current_structure.get_positions()
    for i in range(len(silanols)):
        for j in range(i + 1, len(silanols)):
            g1 = silanols[i]; g2 = silanols[j]
            # Constraint: Must act on different Silicons
            if g1['Si'] == g2['Si']: continue
            
            dist = np.linalg.norm(pos[g1['O']] - pos[g2['O']])
            if dist < MAX_SEARCH_RADIUS: 
                candidates.append((dist, g1, g2))
    
    candidates.sort(key=lambda x: x[0])
    
    if not candidates:
        print("Stopping: No reachable pairs found.")
        break

    # C. Select Safe Pair (Collision Check)
    # -------------------------------------
    selected_pair = None
    best_new_pos = None
    
    print(f"  > Checking {len(candidates)} pairs for safety...")
    for dist, g1, g2 in candidates[:MAX_ATTEMPTS]:
        ignore = [g1['H'], g2['H'], g2['O'], g1['O']]
        mid = (pos[g1['Si']] + pos[g2['Si']]) / 2.0
        
        # Calculate Lift Height (Geometry)
        si_vec = pos[g2['Si']] - pos[g1['Si']]
        half_dist = np.linalg.norm(si_vec) / 2.0
        if half_dist < TARGET_BOND_LENGTH:
            h = np.sqrt(TARGET_BOND_LENGTH**2 - half_dist**2)
        else:
            h = 0.5
        
        # Strategy 1: Standard Placement
        pos_try = mid.copy(); pos_try[2] += h
        if not check_overlap(current_structure, pos_try, ignore, MIN_SAFE_DISTANCE):
            selected_pair = (g1, g2); best_new_pos = pos_try
            print(f"    -> Accepted Pair: Dist {dist:.3f} Å (Standard)")
            break
        
        # Strategy 2: High Lift (+0.8 Å)
        pos_try[2] += 0.8
        if not check_overlap(current_structure, pos_try, ignore, MIN_SAFE_DISTANCE):
            selected_pair = (g1, g2); best_new_pos = pos_try
            print(f"    -> Accepted Pair: Dist {dist:.3f} Å (High Lift)")
            break

    if not selected_pair:
        print("Stopping: All pairs obstructed/unsafe.")
        break

    # D. Execute Reaction
    # -------------------
    g1, g2 = selected_pair
    current_structure.positions[g1['O']] = best_new_pos
    del current_structure[sorted([g1['H'], g2['H'], g2['O']], reverse=True)]

    # E. 3-STAGE ROBUST OPTIMIZATION
    # ------------------------------
    print(f"  > Optimizing Step {step_count}...")
    current_structure.calc = get_calculator()
    
    # STAGE 1: Fast LBFGS (300 steps)
    try:
        opt = LBFGS(current_structure, 
                    logfile=f"opt_step_{step_count}.log", 
                    trajectory=f"opt_step_{step_count}.traj", 
                    maxstep=0.04, 
                    memory=100)
        opt.run(fmax=FMAX_TARGET, steps=300)
    except: pass

    # STAGE 2: Fine-Tuning LBFGS (If needed)
    fmax_now = np.max(np.linalg.norm(current_structure.get_forces(), axis=1))
    if fmax_now > FMAX_TARGET:
        print(f"    -> Stage 1 incomplete ({fmax_now:.3f}). Switching to Fine Tuning...")
        opt_fine = LBFGS(current_structure, 
                         logfile=f"opt_step_{step_count}_fine.log", 
                         trajectory=f"opt_step_{step_count}.traj", 
                         maxstep=0.01,  # Smaller steps
                         append_trajectory=True,
                         memory=100)
        opt_fine.run(fmax=FMAX_TARGET, steps=300)

    # STAGE 3: FIRE Polish (The "Grinder" for stuck steps)
    fmax_now = np.max(np.linalg.norm(current_structure.get_forces(), axis=1))
    if fmax_now > FMAX_TARGET:
        print(f"    -> Stage 2 incomplete ({fmax_now:.3f}). Switching to FIRE (Robust)...")
        opt_fire = FIRE(current_structure, 
                        logfile=f"opt_step_{step_count}_fire.log", 
                        trajectory=f"opt_step_{step_count}.traj",
                        append_trajectory=True)
        opt_fire.run(fmax=FMAX_TARGET, steps=500)

    # F. Save & Report
    # ----------------
    E_curr = current_structure.get_potential_energy()
    dE_step = E_curr + E_h2o - E_prev
    print(f"  > Done. dE = {dE_step:.3f} eV")
    
    write(f"SiO2_step_{step_count}.cif", current_structure)
    E_prev = E_curr

print("\n=== Simulation Finished Successfully ===")