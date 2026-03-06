import os
import numpy as np
from ase.io import read

# =========================================================
# 1. CONFIGURATION (From your snippet)
# =========================================================

MIN_DIST_CRITERIA = {
    "H-H": 0.5, "H-O": 0.8, "H-Pt": 1.4, "H-Si": 1.4,
    "O-O": 1.5, "O-Pt": 1.8, "O-Si": 1.4,
    "Pt-Pt": 2.2, "Pt-Si": 1.5, "Si-Si": 2.3,
}

BOND_CUTOFF_SI_O = 1.9
BOND_CUTOFF_H_O  = 1.2

# =========================================================
# 2. EXTRACTED FILTER LOGIC
# =========================================================

def check_min_distance(atoms):
    """
    Filter-1: Minimum interatomic distance filter (collision check).
    Uses Periodic Boundary Conditions (mic=True).
    """
    symbols = atoms.get_chemical_symbols()
    # mic=True ensures we check distances across periodic boundaries
    dist_matrix = atoms.get_all_distances(mic=True) 
    n = len(atoms)
    collisions = []

    for i in range(n):
        for j in range(i + 1, n):
            # Construct key (e.g., "O-Si" sorted alphabetically)
            s1, s2 = sorted([symbols[i], symbols[j]])
            key = f"{s1}-{s2}"

            if key in MIN_DIST_CRITERIA:
                limit = MIN_DIST_CRITERIA[key]
                d = dist_matrix[i, j]
                if d < limit:
                    collisions.append(f"Collision: {symbols[i]}{i}-{symbols[j]}{j} (Dist: {d:.3f} < {limit})")
    
    return collisions

def check_si_o_bonding(atoms):
    """
    Filter-2: Si coordination filter.
    Si must have exactly 4 O neighbors within 1.9 Angstroms.
    """
    symbols = atoms.get_chemical_symbols()
    dist_matrix = atoms.get_all_distances(mic=True)
    n = len(atoms)
    violations = []

    for i in range(n):
        if symbols[i] != "Si": continue
        
        count_O = 0
        for j in range(n):
            if symbols[j] == "O":
                if dist_matrix[i, j] < BOND_CUTOFF_SI_O:
                    count_O += 1
        
        if count_O != 4:
            violations.append(f"Si atom {i} has {count_O} O-bonds (Expected 4)")

    return violations

def check_h_o_bonding(atoms):
    """
    Filter-3: H coordination filter.
    H must have exactly 1 O neighbor within 1.2 Angstroms.
    """
    symbols = atoms.get_chemical_symbols()
    dist_matrix = atoms.get_all_distances(mic=True)
    n = len(atoms)
    violations = []

    for i in range(n):
        if symbols[i] != "H": continue
        
        count_O = 0
        for j in range(n):
            if symbols[j] == "O":
                if dist_matrix[i, j] < BOND_CUTOFF_H_O:
                    count_O += 1
        
        if count_O != 1:
            violations.append(f"H atom {i} has {count_O} O-bonds (Expected 1)")

    return violations

# =========================================================
# 3. MAIN RUNNER
# =========================================================

def validate_file(filename):
    print(f"\n{'-'*60}")
    print(f"VALIDATING FILE: {filename}")
    print(f"{'-'*60}")

    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    try:
        atoms = read(filename)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # --- Run Filter 1: Collisions ---
    print("Filter 1 (Collisions):", end=" ")
    col_errors = check_min_distance(atoms)
    if not col_errors:
        print("PASSED ✅")
    else:
        print(f"FAILED ❌ ({len(col_errors)} issues)")
        for err in col_errors[:5]: print("  " + err) # Print only first 5
        if len(col_errors) > 5: print(f"  ...and {len(col_errors)-5} more")

    # --- Run Filter 2: Si-O Bonding ---
    print("Filter 2 (Si-O Bonds):", end=" ")
    si_errors = check_si_o_bonding(atoms)
    if not si_errors:
        print("PASSED ✅")
    else:
        print(f"FAILED ❌ ({len(si_errors)} issues)")
        for err in si_errors[:5]: print("  " + err)

    # --- Run Filter 3: H-O Bonding ---
    # Only run if H exists in the system
    if 'H' in atoms.get_chemical_symbols():
        print("Filter 3 (H-O Bonds): ", end=" ")
        h_errors = check_h_o_bonding(atoms)
        if not h_errors:
            print("PASSED ✅")
        else:
            print(f"FAILED ❌ ({len(h_errors)} issues)")
            for err in h_errors[:5]: print("  " + err)
    else:
        print("Filter 3 (H-O Bonds):  SKIPPED (No Hydrogen)")

    # Final Result
    if not col_errors and not si_errors and (not 'H' in atoms.get_chemical_symbols() or not h_errors):
        print("\n>>> OVERALL STATUS: VALID STRUCTURE 🟢")
    else:
        print("\n>>> OVERALL STATUS: INVALID STRUCTURE 🔴")

if __name__ == "__main__":
    # Check your specific files
    validate_file('SiO2-1.cif')  # Original
    # validate_file('./pair_2/pair_2_relaxed.cif')  # Original
    # validate_file('./pair_3/pair_3_relaxed.cif')  # Original
    # validate_file('./pair_4/pair_4_relaxed.cif')  # Original
    # validate_file('./pair_5/pair_5_relaxed.cif')  # Original
    # validate_file('./pair_6/pair_6_relaxed.cif')  # Original
    # validate_file('./pair_7/pair_7_relaxed.cif')  # Original
    # validate_file('./pair_8/pair_8_relaxed.cif')  # Original
    # validate_file('./pair_9/pair_9_relaxed.cif')  # Original

