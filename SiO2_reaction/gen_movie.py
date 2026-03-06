import os
import glob
import re
from ase.io import read, write

# ================= USER SETTINGS =================
# We use the .cif checkpoints since standard .log files don't have coordinates
file_pattern = "SiO2_step_*.cif" 
output_filename = "SiO2_dehydroxylation_movie.traj"
skip_failed_step = 17  # We exclude step 17 because it crashed
# =================================================

print(f"Looking for files matching: {file_pattern} ...")
files = glob.glob(file_pattern)

# Helper function to sort files numerically (1, 2, 10) instead of alphabetically (1, 10, 2)
def get_step_num(filename):
    # Regex to find the number in "SiO2_step_12.cif"
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

# Sort the files so the movie plays in the correct order
files.sort(key=get_step_num)

if not files:
    print("ERROR: No .cif files found. Make sure you are in the correct folder.")
else:
    print(f"Found {len(files)} checkpoint files.")
    
    movie_frames = []
    
    # 1. Try to add the very first starting structure (if available)
    start_file = "SiO2-1_opt_lbfgs.cif"
    if os.path.exists(start_file):
        print(f"  > Frame 0: Adding initial reactant ({start_file})")
        movie_frames.append(read(start_file))
    
    # 2. Add each successful step
    for f in files:
        step = get_step_num(f)
        
        if step < skip_failed_step:
            print(f"  > Frame {step}: Adding {f}")
            try:
                struct = read(f)
                movie_frames.append(struct)
            except Exception as e:
                print(f"    ! Error reading {f}: {e}")
        else:
            print(f"  > Skipping Step {step} (Excluded due to crash)")

    # 3. Save the movie
    if movie_frames:
        write(output_filename, movie_frames)
        print("\n" + "="*50)
        print(f" DONE! Movie saved to: {output_filename}")
        print(f" Total Frames: {len(movie_frames)}")
        print(f" Watch it by running: ase gui {output_filename}")
        print("="*50)
    else:
        print("No valid frames could be loaded.")