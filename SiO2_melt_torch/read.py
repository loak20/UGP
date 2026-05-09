import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch_sim as ts
from ase.io import write

# Make sure this matches the exact extension you saved it as (.h5md or .hdf5)
hdf5_file = "npt_mod.h5md" 
traj_file = "npt_mod.traj"

print(f"\nConverting '{hdf5_file}' to ASE .traj format...")

with ts.TorchSimTrajectory(hdf5_file, mode="r") as traj:
    # First, determine the total number of frames saved in the HDF5 file.
    try:
        num_frames = len(traj)
    except TypeError:
        # Safe fallback if len(traj) isn't supported directly
        num_frames = len(traj.get_array("positions"))
        
    print(f"Found {num_frames} frames in the trajectory. Extracting...")
    
    # Loop through and extract the Atoms object for every single frame
    ase_trajectory = []
    for i in range(num_frames):
        ase_trajectory.append(traj.get_atoms(i))

# Write the full list of frames to the new .traj file
write(traj_file, ase_trajectory)

print(f"\nConversion complete! Successfully saved {len(ase_trajectory)} images/frames.")
print(f"You can now view the full animation by running:")
print(f"ase gui {traj_file}")