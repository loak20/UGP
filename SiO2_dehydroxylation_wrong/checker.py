import os
import sys
import numpy as np
import warnings
from ase.io import read
from ase.neighborlist import neighbor_list

# Suppress warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# =========================================================
# MACE IMPORT SETUP
# =========================================================
HAS_MACE = False
try:
    from mace.calculators import MACECalculator, mace_mp
    HAS_MACE = True
except ImportError:
    try:
        # Fallback for older MACE versions
        from mace.calculators import MACECalculator
        mace_mp = None 
        HAS_MACE = True
    except ImportError:
        print("Warning: MACE not found. Energy calculation will be skipped.")

# =========================================================
# 1. CONFIGURATION RULES
# =========================================================

# A. Collision Rules (Minimum allowed distance between pairs)
MIN_DIST_CRITERIA = {
    "H-H": 0.5, "H-O": 0.8, "H-Pt": 1.4, "H-Si": 1.4,
    "O-O": 1.5, "O-Pt": 1.8, "O-Si": 1.4,
    "Pt-Pt": 2.2, "Pt-Si": 1.5,
    "Si-Si": 2.2, 
}

# B. Bonding Definitions (Max distance to consider a bond exists)
BOND_CUTOFFS = {
    frozenset(["Si", "O"]): 1.9,
    frozenset(["Si", "Si"]): 2.6, 
    frozenset(["H", "O"]): 1.2,
}

# C. Coordination Requirements
COORDINATION_RULES = {
    "Si": {
        "target_cn": [4],             
        "valid_neighbors": ["O", "Si"] 
    },
    "H": {
        "target_cn": [1],             
        "valid_neighbors": ["O"]      
    }
}

# D. MACE Model Configuration
# Use "small", "medium", "large" for pre-trained models, OR a direct path to a .model file
MACE_MODEL_PATH = "medium"  
DEVICE = "cpu"              

# =========================================================
# 2. VALIDATION LOGIC
# =========================================================

def get_bond_cutoff(symbol1, symbol2):
    """Retrieve bond cutoff for a specific pair from config."""
    pair = frozenset([symbol1, symbol2])
    return BOND_CUTOFFS.get(pair, 0.0)

def validate_collisions(atoms):
    """Checks if any two atoms are closer than the allowed minimum distance."""
    symbols = atoms.get_chemical_symbols()
    distances = atoms.get_all_distances(mic=True)
    n = len(atoms)

    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = symbols[i], symbols[j]
            dist = distances[i, j]

            key = f"{s1}-{s2}"
            if key not in MIN_DIST_CRITERIA:
                key = f"{s2}-{s1}"

            if key in MIN_DIST_CRITERIA:
                min_allowed = MIN_DIST_CRITERIA[key]
                if dist < min_allowed:
                    return False, f"Collision: {s1}#{i} - {s2}#{j} dist {dist:.3f} < {min_allowed}"
    
    return True, "Passed"

def validate_coordination(atoms):
    """Checks if atoms meet their defined coordination numbers (CN) and neighbor types."""
    symbols = atoms.get_chemical_symbols()
    distances = atoms.get_all_distances(mic=True)
    n = len(atoms)

    for i in range(n):
        center_sym = symbols[i]
        
        if center_sym not in COORDINATION_RULES:
            continue
            
        rule = COORDINATION_RULES[center_sym]
        neighbors = []

        for j in range(n):
            if i == j: continue
            neigh_sym = symbols[j]
            cutoff = get_bond_cutoff(center_sym, neigh_sym)
            
            if cutoff > 0 and distances[i, j] < cutoff:
                neighbors.append(neigh_sym)

        cn = len(neighbors)
        if cn not in rule["target_cn"]:
            return False, f"Bad CN: {center_sym}#{i} has {cn} bonds {neighbors}, expected {rule['target_cn']}"

        if "valid_neighbors" in rule:
            for neigh_sym in neighbors:
                if neigh_sym not in rule["valid_neighbors"]:
                    return False, f"Bad Bond: {center_sym}#{i} bonded to forbidden {neigh_sym}"

    return True, "Passed"

def validate_energy(atoms):
    """
    Runs MACE to calculate energy. 
    Intelligently switches between mace_mp (for keywords) and MACECalculator (for paths).
    """
    if not HAS_MACE:
        return True, "Skipped (MACE not installed)"

    try:
        # Only initialize if calculator is missing
        if atoms.calc is None:
            # Check if user is asking for a pre-trained foundation model
            if MACE_MODEL_PATH in ["small", "medium", "large"] and mace_mp is not None:
                calc = mace_mp(model=MACE_MODEL_PATH, device=DEVICE, default_dtype="float64")
            else:
                # Assume it is a specific file path
                calc = MACECalculator(model_paths=MACE_MODEL_PATH, device=DEVICE, default_dtype="float64")
            
            atoms.calc = calc
        
        energy = atoms.get_potential_energy()
        return True, f"Energy: {energy:.3f} eV"
        
    except Exception as e:
        # Provide a clearer error if it looks like a path issue
        err_str = str(e)
        if "Couldn't find MACE model" in err_str:
             return False, f"MACE Error: Could not load model '{MACE_MODEL_PATH}'. Check file path or install mace-torch."
        return False, f"MACE Error: {err_str}"

# =========================================================
# 3. MAIN RUNNER
# =========================================================

def validate_file(filepath):
    print(f"\n{'='*60}")
    print(f"Checking: {filepath}")
    
    if not os.path.exists(filepath):
        print("❌ File not found.")
        return

    try:
        atoms = read(filepath)
    except Exception as e:
        print(f"❌ Failed to parse CIF: {e}")
        return

    # 1. Collision Check
    ok, msg = validate_collisions(atoms)
    if not ok:
        print(f"❌ Geometric Failure: {msg}")
        return
    print("✅ Collision Check Passed")

    # 2. Coordination Check
    ok, msg = validate_coordination(atoms)
    if not ok:
        print(f"❌ Bonding Failure: {msg}")
        return
    print("✅ Coordination/Bonding Check Passed")

    # 3. MACE Energy Check
    ok, msg = validate_energy(atoms)
    if not ok:
        print(f"❌ Energy Check Failed: {msg}")
        return
    print(f"✅ {msg}")
    
    print(f"🎉 STRUCTURE VALID")

if __name__ == "__main__":
    files = [
        'SiO2-1.cif'
    ]

    # CLI Support
    if not files and len(sys.argv) > 1:
        files = sys.argv[1:]

    for f in files:
        validate_file(f)