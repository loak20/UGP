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

# ---------------------------------------------------------
# 1. System & Hardware Setup
# ---------------------------------------------------------
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
print(f"Using device: {device_str}")

# Load your single unit cell
atoms = read('SiO2.cif')

# Initialize velocities using ASE at 298 K BEFORE converting to TorchSim
MaxwellBoltzmannDistribution(atoms, temperature_K=298.0)

# Load the MACE model
mace_calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device=device_str)
raw_pytorch_model = mace_calc.models[0]

# Wrap the raw PyTorch model in TorchSim
mace_model = MaceModel(model=raw_pytorch_model, device=device)

# Convert ASE Atoms to TorchSim's SimState
system_state = ts.initialize_state([atoms], device=device, dtype=torch.float32)

# ---------------------------------------------------------
# 2. Simulation Parameters
# ---------------------------------------------------------
timestep_ps = 0.001  # 1 fs = 0.001 ps

# Convert Berendsen Barostat Modulus to Compressibility
modulus_atm = 360000
compressibility_bar_inv = 1.0 / (modulus_atm * 1.01325)

npt_kwargs = dict(
    integrator=ts.Integrator.npt_nose_hoover_isotropic,
    external_pressure=1.01325,
    init_kwargs=dict(
        tau_t=0.05,
        tau_p=1.0,  
        compressibility=compressibility_bar_inv
    )
)

h5md_file = "npt_mod.h5md"
log_filename = "md_nptmod.log"

# Clean up old trajectory and log files to prevent append errors on rerun
if os.path.exists(h5md_file):
    os.remove(h5md_file)
if os.path.exists(log_filename):
    os.remove(log_filename)

# ---------------------------------------------------------
# 3. Enhanced Logging Functions
# ---------------------------------------------------------
def init_logger(filename):
    """Initializes the log file with a clean header."""
    with open(filename, "w") as logfile:
        header = f"{'Time[ps]':>10} {'T[K]':>10} {'PE[eV]':>15} {'KE[eV]':>15} {'TotE[eV]':>15} {'Vol[A^3]':>12}\n"
        logfile.write(header)
        print(header.strip())

def log_state(state, current_time_ps, filename):
    """Extracts system parameters from the TorchSim state and logs them."""
    T = float(state.temperature.cpu()[0])
    PE = float(state.potential_energy.cpu()[0])
    KE = float(state.kinetic_energy.cpu()[0])
    TotE = PE + KE
    
    cell = state.cell.cpu()
    if cell.dim() == 3:
        vol = float(torch.linalg.det(cell[0]).abs())
    else:
        vol = float(torch.linalg.det(cell).abs())

    log_line = f"{current_time_ps:10.4f} {T:10.1f} {PE:15.4f} {KE:15.4f} {TotE:15.4f} {vol:12.4f}"
    
    print(log_line)
    with open(filename, "a") as logfile:
        logfile.write(log_line + "\n")

def ramp_temperature(state, start_T, end_T, time_ps, current_time_ps, steps_per_update=100):
    """Gradually changes the thermostat temperature while logging."""
    total_steps = int(time_ps / timestep_ps)
    num_updates = total_steps // steps_per_update
    
    delta_T = (end_T - start_T) / num_updates
    current_T = start_T
    
    print(f"\n--- Ramping from {start_T}K to {end_T}K over {time_ps} ps ---")
    
    for _ in range(num_updates):
        current_T += delta_T
        current_time_ps += (steps_per_update * timestep_ps)
        
        # Sequentially update the temperature and integrate
        state = ts.integrate(
            system=state,
            model=mace_model,
            n_steps=steps_per_update,
            timestep=timestep_ps,
            temperature=current_T,
            trajectory_reporter=dict(
                filenames=[h5md_file], 
                state_frequency=100, 
                trajectory_kwargs=dict(mode="a") # THE FIX FOR VERSION 0.5.2
            ),
            **npt_kwargs
        )
        
        log_state(state, current_time_ps, log_filename)
        
    print(f"Reached target temperature: {current_T:.1f} K")
    return state, current_time_ps

# ---------------------------------------------------------
# Initialize Global Time & Logger
# ---------------------------------------------------------
global_time_ps = 0.0
init_logger(log_filename)

# ---------------------------------------------------------
# 4. Stage 0: Equilibration (Now Chunked for Logging)
# ---------------------------------------------------------
print("\n--- Equilibrating at 298 K to stabilize velocities ---")
eq_time_ps = 1.0  # 1000 steps = 1.0 ps
eq_steps_per_update = 100
num_eq_updates = int((eq_time_ps / timestep_ps) / eq_steps_per_update)

for _ in range(num_eq_updates):
    global_time_ps += (eq_steps_per_update * timestep_ps)
    system_state = ts.integrate(
        system=system_state,
        model=mace_model,
        n_steps=eq_steps_per_update,
        timestep=timestep_ps,
        temperature=298.0,
        trajectory_reporter=dict(
            filenames=[h5md_file], 
            state_frequency=100, 
            trajectory_kwargs=dict(mode="a") # THE FIX FOR VERSION 0.5.2
        ), 
        **npt_kwargs
    )
    log_state(system_state, global_time_ps, log_filename)

# ---------------------------------------------------------
# 5. Stage 1: Melting
# ---------------------------------------------------------
melt_time_ps = 1.0
system_state, global_time_ps = ramp_temperature(
    system_state, 
    start_T=298.0, 
    end_T=2000.0, 
    time_ps=melt_time_ps, 
    current_time_ps=global_time_ps
)

# ---------------------------------------------------------
# 6. Stage 2: Quenching
# ---------------------------------------------------------
cooling_rate_K_per_ps = 100.0
temp_drop_K = 2000.0 - 298.0
quench_time_ps = temp_drop_K / cooling_rate_K_per_ps

system_state, global_time_ps = ramp_temperature(
    system_state, 
    start_T=2000.0, 
    end_T=298.0, 
    time_ps=quench_time_ps, 
    current_time_ps=global_time_ps
)

print(f"\nSimulation complete. Data saved to '{h5md_file}' and '{log_filename}'.")

# ---------------------------------------------------------
# 7. Convert H5MD back to standard ASE Trajectory (.traj)
# ---------------------------------------------------------
print("\nConverting .h5md trajectory to ASE .traj format...")

with ts.TorchSimTrajectory(h5md_file) as traj:
    ase_trajectory = traj.to_atoms()

write("npt_mod.traj", ase_trajectory)

print("Conversion complete! You now have 'npt_mod.traj' ready for post-processing.")
