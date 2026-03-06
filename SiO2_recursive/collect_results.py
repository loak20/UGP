import os
import numpy as np
import re

# =============================================================================
# CONFIGURATION
# =============================================================================
SEARCH_DIR = "Tree_Search"
OUTPUT_FILE = "final_results.npz"

def parse_value(line, dtype=str):
    """Helper to extract value after the first colon."""
    parts = line.split(":", 1)
    if len(parts) < 2:
        return None
    val_str = parts[1].strip()
    
    # Remove units like 'eV' or 'A' if present for float conversion
    if dtype == float:
        val_str = val_str.replace("eV", "").replace("A", "").strip()
        try:
            return float(val_str)
        except ValueError:
            return np.nan
    return val_str

def main():
    if not os.path.exists(SEARCH_DIR):
        print(f"Error: Directory '{SEARCH_DIR}' not found.")
        return

    # Data containers
    data = {
        "node_labels": [],
        "parent_paths": [],
        "reactions": [],
        "init_h_dists": [],
        "e_parents": [],
        "e_products": [],
        "e_h2o": [],
        "reaction_energies": []
    }

    print(f"Scanning '{SEARCH_DIR}' for results.txt files...")
    count = 0

    # Walk through the directory tree
    for root, dirs, files in os.walk(SEARCH_DIR):
        if "results.txt" in files:
            file_path = os.path.join(root, "results.txt")
            
            try:
                with open(file_path, "r") as f:
                    lines = f.readlines()
                
                # Temporary storage for this file
                entry = {}
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("Node Label:"):
                        entry["node_labels"] = parse_value(line, str)
                    elif line.startswith("Parent Path:"):
                        entry["parent_paths"] = parse_value(line, str)
                    elif line.startswith("Reaction:"):
                        entry["reactions"] = parse_value(line, str)
                    elif line.startswith("Initial H-bond dist:"):
                        entry["init_h_dists"] = parse_value(line, float)
                    elif line.startswith("E_parent:"):
                        entry["e_parents"] = parse_value(line, float)
                    elif line.startswith("E_product:"):
                        entry["e_products"] = parse_value(line, float)
                    elif line.startswith("E_H2O:"):
                        entry["e_h2o"] = parse_value(line, float)
                    elif line.startswith("Reaction Energy (dE):"):
                        entry["reaction_energies"] = parse_value(line, float)

                # Ensure all fields are present (handle partial files safely)
                if "node_labels" in entry: # Only add if we at least found a label
                    for key in data.keys():
                        # Append found value or None/NaN if missing
                        val = entry.get(key)
                        if val is None:
                            val = np.nan if key != "node_labels" and key != "reactions" else "Unknown"
                        data[key].append(val)
                    count += 1
                    
            except Exception as e:
                print(f"Warning: Could not parse {file_path}. Reason: {e}")

    if count == 0:
        print("No results found.")
        return

    # Convert lists to numpy arrays
    np_data = {}
    for key, val in data.items():
        np_data[key] = np.array(val)

    # Save to NPZ in the current directory (Project Root)
    np.savez(OUTPUT_FILE, **np_data)
    
    print(f"\nSuccess! Extracted {count} entries.")
    print(f"Saved data to: {os.path.abspath(OUTPUT_FILE)}")
    
    # Validation print
    print("\n--- Summary of Extracted Data ---")
    print(f"{'Node Label':<15} | {'dE (eV)':<10}")
    print("-" * 30)
    for i in range(min(5, count)): # Print first 5
        print(f"{np_data['node_labels'][i]:<15} | {np_data['reaction_energies'][i]:.4f}")
    if count > 5:
        print(f"... and {count - 5} more.")

if __name__ == "__main__":
    main()