import h5py

file = "sio2_melt_quench_supercell.h5md"

with h5py.File(file, "r") as f:
    print("---- Shapes ----")

    positions = f["steps/positions"]
    print("positions:", positions.shape)

    cell = f["steps/cell"]
    print("cell:", cell.shape)

    temp = f["steps/temperature"]
    print("temperature:", temp.shape)

    # static info
    Z = f["data/atomic_numbers"]
    print("atomic_numbers:", Z.shape)

    # example frame
    frame = 0
    pos_frame = positions[frame]
    print("\nFrame 0 positions:", pos_frame.shape)