import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
FILE_NAME = "final_results.npz"

def main():
    if not os.path.exists(FILE_NAME):
        print(f"Error: '{FILE_NAME}' not found in the current directory.")
        return

    # Load the data
    try:
        data = np.load(FILE_NAME, allow_pickle=True)
        print(f"Successfully loaded '{FILE_NAME}'\n")
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # Extract arrays
    # (These keys match what we saved in the previous script)
    labels = data['node_labels']
    parents = data['parent_paths']
    reactions = data['reactions']
    dE = data['reaction_energies']
    init_dists = data['init_h_dists']

    # ---------------------------------------------------------
    # 1. PRINT ALL RESULTS (Tabular Format)
    # ---------------------------------------------------------
    print(f"{'Label':<15} | {'Reaction Energy (eV)':<20} | {'Init Dist (A)':<15} | {'Reaction Type'}")
    print("-" * 80)

    for i in range(len(labels)):
        print(f"{labels[i]:<15} | {dE[i]:<20.5f} | {init_dists[i]:<15.3f} | {reactions[i]}")

    print("-" * 80)
    print(f"Total Structures: {len(labels)}")

    # ---------------------------------------------------------
    # 2. ANALYSIS EXAMPLES (Optional)
    # ---------------------------------------------------------
    print("\n--- Analysis Stats ---")

    # Example A: Find the Minimum Energy Structure (Most Stable)
    if len(dE) > 0:
        min_idx = np.argmin(dE)
        print(f"Most Stable Structure: {labels[min_idx]} (dE = {dE[min_idx]:.5f} eV)")
        print(f"Location: {os.path.join(parents[min_idx], 'pair_' + labels[min_idx].split('_')[-1])}")

    # Example B: Find highly probable structures (small positive dE)
    # Let's say we want structures where 0 < dE < 0.05 eV
    stable_indices = np.where((dE > 0) & (dE < 0.05))[0]
    print(f"\nStructures with small dE (0 < dE < 0.05 eV): {len(stable_indices)} found")
    for idx in stable_indices:
        print(f" -> {labels[idx]}: {dE[idx]:.5f} eV")

if __name__ == "__main__":
    main()