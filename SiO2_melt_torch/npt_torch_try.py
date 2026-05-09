import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
import torch
import torch_sim as ts
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

# Silence MACE and PyTorch Autograd spam
warnings.filterwarnings("ignore", category=UserWarning, module="mace")
warnings.filterwarnings("ignore", message="The .grad attribute of a Tensor that is not a leaf Tensor is being accessed")


device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

print(f"Using device: {device_str}")

# ---------------------------------------------------------
# 1. System Setup
# ---------------------------------------------------------
atoms = read("SiO2.cif")

# Initialize velocities
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Load MACE
mace_calc = mace_mp(model="medium", dispersion=False,
                    default_dtype="float32", device=device_str)

model = MaceModel(model=mace_calc.models[0], device=device)

state = ts.initialize_state([atoms], device=device, dtype=torch.float32)

# ---------------------------------------------------------
# 2. Simulation Parameters
# ---------------------------------------------------------
timestep_ps = 0.001  # 1 fs

# More stable thermostat/barostat
npt_kwargs = dict(
    integrator=ts.Integrator.npt_nose_hoover_isotropic,
    external_pressure=1.01325,
    init_kwargs=dict(
        tau_t=0.02,   # stronger thermostat coupling
        tau_p=2.0,
        compressibility=1e-5
    )
)

traj_file = "fixed_traj.h5md"
log_file = "fixed_log.txt"

if os.path.exists(traj_file):
    os.remove(traj_file)
if os.path.exists(log_file):
    os.remove(log_file)

# ---------------------------------------------------------
# 3. Logging
# ---------------------------------------------------------
def log_T(state, t):
    T = float(state.calc_temperature().cpu()[0])
    print(f"{t:8.3f} ps  |  {T:8.2f} K")
    with open(log_file, "a") as f:
        f.write(f"{t:.4f} {T:.2f}\n")

# ---------------------------------------------------------
# 4. Smooth Ramp Function (CRITICAL FIX)
# ---------------------------------------------------------
def smooth_ramp(state, T_start, T_end, time_ps, current_time,
                steps_per_update=10, write_every=10):

    total_steps = int(time_ps / timestep_ps)
    n_updates = total_steps // steps_per_update

    dT = (T_end - T_start) / n_updates
    T = T_start

    for i in range(n_updates):
        T += dT
        current_time += steps_per_update * timestep_ps

        state = ts.integrate(
            system=state,
            model=model,
            n_steps=steps_per_update,
            timestep=timestep_ps,
            temperature=T,
            trajectory_reporter=dict(
                filenames=[traj_file],
                state_frequency=write_every,
                trajectory_kwargs=dict(mode="a")
            ),
            **npt_kwargs
        )

        if i % 10 == 0:
            log_T(state, current_time)

    return state, current_time

# ---------------------------------------------------------
# 5. Equilibration
# ---------------------------------------------------------
print("\n--- Equilibrating (300 K, 5 ps) ---")
time_ps = 0.0

state, time_ps = smooth_ramp(state, 300, 300, 5.0, time_ps)

# ---------------------------------------------------------
# 6. Melt (IMPORTANT: hold at high T)
# ---------------------------------------------------------
print("\n--- Heating to 2000 K ---")
state, time_ps = smooth_ramp(state, 300, 2000, 5.0, time_ps)

print("\n--- Holding melt (2000 K, 10 ps) ---")
state, time_ps = smooth_ramp(state, 2000, 2000, 10.0, time_ps)

# ---------------------------------------------------------
# 7. Controlled Quench (FIXED)
# ---------------------------------------------------------
print("\n--- Slow quench (2000 → 300 K) ---")

# 10 K/ps instead of 100
quench_time = (2000 - 300) / 10.0   # = 170 ps

state, time_ps = smooth_ramp(state, 2000, 300, quench_time, time_ps)

# ---------------------------------------------------------
# 8. Post-quench relaxation (MOST IMPORTANT)
# ---------------------------------------------------------
print("\n--- Relaxing at 300 K (20 ps) ---")
state, time_ps = smooth_ramp(state, 300, 300, 20.0, time_ps)

print("\nSimulation complete.")

# ---------------------------------------------------------
# 9. Convert trajectory
# ---------------------------------------------------------
print("\nConverting trajectory...")

with ts.TorchSimTrajectory(traj_file) as traj:
    atoms_list = traj.to_atoms()

write("fixed_traj.traj", atoms_list)

print("Done: fixed_traj.traj ready.")