import os
import warnings
import torch
from ase.io import read
from ase.mep import NEB
from ase.optimize import FIRE
from mace.calculators import mace_mp

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
# Suppress warnings for cleaner output
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# =============================================================================
# CONFIGURATION
# =============================================================================
PBCS_XYZ = [True, True, False]
FMAX = 0.05   # Standard force tolerance for NEB 
N_IMAGES = 5  # Number of intermediate images

# Input and Output Paths based on your directory structure
INITIAL_STATE_PATH = os.path.join("optimised_neb_states", "SiO2_reactant.cif")
FINAL_STATE_PATH = os.path.join("optimised_neb_states", "SiO2_final.cif")
OUTPUT_DIR = "transition_states"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_calculator():
    """Returns a fresh instance of the MACE calculator."""
    return mace_mp(model="medium", dispersion=False, default_dtype="float64", device=DEVICE)

# =============================================================================
# MAIN CI-NEB WORKFLOW
# =============================================================================
def main():
    print(f"\n--- Loading States ---")
    if not os.path.exists(INITIAL_STATE_PATH) or not os.path.exists(FINAL_STATE_PATH):
        raise FileNotFoundError(f"Missing states. Could not find {INITIAL_STATE_PATH} or {FINAL_STATE_PATH}.")

    # Read the highly optimized states
    initial = read(INITIAL_STATE_PATH)
    final = read(FINAL_STATE_PATH)
    
    # Apply your specific PBCs
    initial.pbc = PBCS_XYZ
    final.pbc = PBCS_XYZ

    # Strict check: NEB will crash if atom counts/indices don't match exactly
    if len(initial) != len(final):
        raise ValueError(f"Atom count mismatch! Initial: {len(initial)}, Final: {len(final)}.")

    print(f"\n--- Setting up the NEB Band ({N_IMAGES} Intermediate Images) ---")
    # Must use .copy() to ensure distinct memory references for each image
    images = [initial]
    images += [initial.copy() for _ in range(N_IMAGES)]
    images += [final]
    
    # Instantiate the NEB object with Climbing Image enabled
    neb = NEB(images, climb=True)
    
    print("\n--- Interpolating the Initial Guess Path ---")
    # IDPP creates a much safer, physically realistic initial guess than linear interpolation
    neb.interpolate('idpp') 
    
    print("\n--- Attaching MACE Calculators ---")
    # Set calculators ONLY for the intermediate images.
    # Initial and final states remain frozen as endpoints.
    for image in images[1:-1]:
        image.calc = get_calculator()
        
    print(f"\n--- Running CI-NEB Optimization with FIRE ---")
    traj_file = os.path.join(OUTPUT_DIR, 'cneb_path.traj')
    log_file = os.path.join(OUTPUT_DIR, 'cneb_opt.log')
    
    # FIRE is highly resistant to the unstable spring forces in NEB
    optimizer = FIRE(neb, trajectory=traj_file, logfile=log_file)
    
    # Run the optimization
    optimizer.run(fmax=FMAX)
    
    print(f"\n--- CI-NEB Complete! ---")
    print(f"Results successfully saved in the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()