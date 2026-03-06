from ase.io import read, write

# ================= SETTINGS =================
input_filename = "relax_reaction.traj"  # The file you already made
output_filename = "h2_Only.traj"
# ============================================

print(f"Reading {input_filename}...")
try:
    # Read all frames
    frames = read(input_filename, index=':')
    
    h_only_frames = []
    
    for i, atoms in enumerate(frames):
        # Create a new Atoms object with ONLY Hydrogen
        # We assume Hydrogen has the chemical symbol 'H'
        h_atoms = atoms[[atom.index for atom in atoms if atom.symbol == 'H']]
        
        # Keep the cell dimensions so they don't look like they are floating in void
        h_atoms.set_cell(atoms.get_cell())
        h_atoms.set_pbc(atoms.get_pbc())
        
        h_only_frames.append(h_atoms)
        
        # Optional: Print count to terminal so you can check numbers
        print(f"Frame {i+1}: {len(h_atoms)} Hydrogens")

    if h_only_frames:
        write(output_filename, h_only_frames)
        print("\n" + "="*50)
        print(f" DONE! Saved: {output_filename}")
        print(f" Watch it: ase gui {output_filename}")
        print("="*50)

except Exception as e:
    print(f"Error: {e}")
    print("Make sure you run the previous script to generate the input file first.")