import os
from ase.io import read, write

# ================= USER SETTINGS =================
input_prefix = "SiO2_step_"
input_extension = ".cif"
output_filename = "relax_reaction.traj"

# Range of steps to include
start_step = 1
end_step = 14
# =================================================

movie_frames = []

print(f"Collecting final structures from Step {start_step} to {end_step}...")

for i in range(start_step, end_step + 1):
    # specific filename format: SiO2_step_1.cif
    filename = f"{input_prefix}{i}{input_extension}"
    
    if os.path.exists(filename):
        try:
            # reading a cif without index=':' defaults to the final structure
            structure = read(filename)
            movie_frames.append(structure)
            print(f" > Frame {i}: Added final structure from {filename}")
        except Exception as e:
            print(f"    ! Error reading {filename}: {e}")
    else:
        print(f"    ! Warning: File not found: {filename}")

# Save the movie
if movie_frames:
    write(output_filename, movie_frames)
    print("\n" + "="*50)
    print(f" DONE! Movie saved to: {output_filename}")
    print(f" Total Frames: {len(movie_frames)}")
    print(f" Watch it by running: ase gui {output_filename}")
    print("="*50)
else:
    print("No valid files found.")