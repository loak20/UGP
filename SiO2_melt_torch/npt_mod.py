import numpy as np
import os 
import time
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=UserWarning, module="mace")
import numpy as np
from ase.io import read
from ase import units
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import mace_mp
# from ase.optimize import LBFGS

# ---------------------------------------------------------
# 1. System Setup
# ---------------------------------------------------------
# Load your single unit cell 
atoms = read('SiO2.cif')

# Attach the MACE calculator
# Using 'large' model and default float64 for MD stability
calculator = mace_mp(model="large", dispersion=False, default_dtype="float64", device='cuda')
atoms.calc = calculator

# ---------------------------------------------------------
# 2. Simulation Parameters
# ---------------------------------------------------------
timestep_fs = 1.0  # 1 fs time step
dt = timestep_fs * units.fs

# Initialize velocities at 298 K
MaxwellBoltzmannDistribution(atoms, temperature_K=298)

# Convert Berendsen Barostat Modulus to Compressibility in ASE units
modulus_atm = 360000
modulus_bar = modulus_atm * 1.01325
compressibility_bar_inv = 1.0 / modulus_bar

# Set up the NPT Berendsen Dynamics
dyn = NPTBerendsen(
    atoms,
    timestep=dt,
    temperature_K=298.0,                   # Starting temperature
    pressure_au=1.01325 * units.bar,       # 1.0 atm
    taut= 50 * units.fs,                  # 1 ps temperature coupling
    taup=1000 * units.fs,                  # 1 ps pressure coupling
    compressibility_au=compressibility_bar_inv / units.bar,
    trajectory='npt_mod.traj',
    logfile='md_nptmod.log',
    loginterval=100                        # Log every 100 steps
)

# ---------------------------------------------------------
# 3. Helper Function for Temperature Ramping
# ---------------------------------------------------------
def ramp_temperature(dynamics, start_T, end_T, time_ps, steps_per_update=10):
    """Gradually changes the thermostat temperature over a specified time."""
    total_fs = time_ps * 1000
    total_steps = int(total_fs / timestep_fs)
    num_updates = total_steps // steps_per_update
    
    delta_T = (end_T - start_T) / num_updates
    current_T = start_T
    
    print(f"Ramping temperature from {start_T}K to {end_T}K over {time_ps} ps...")
    
    for _ in range(num_updates):
        current_T += delta_T
        # Update the target temperature in the NPTBerendsen object (stored in eV)
        dynamics.temperature = current_T
        dynamics.run(steps_per_update)
        
    print(f"Reached target temperature: {current_T:.1f} K")

# Equilibirating before starting
print("Equilibrating at 298 K to stabilize velocities...")
dyn.run(1000)  

# ---------------------------------------------------------
# 4. Stage 1: Melting
# ---------------------------------------------------------
# Heat from 298 K to 8000 K over 1 ps
melt_time_ps = 1.0
ramp_temperature(dyn, start_T=298.0, end_T=2000.0, time_ps=melt_time_ps)

# ---------------------------------------------------------
# 5. Stage 2: Quenching
# ---------------------------------------------------------
# Cool from 2000 K to 298 K at a rate of 100 K / ps
cooling_rate_K_per_ps = 100.0
temp_drop_K = 2000.0 - 298.0
quench_time_ps = temp_drop_K / cooling_rate_K_per_ps

ramp_temperature(dyn, start_T=2000.0, end_T=298.0, time_ps=quench_time_ps)

print("Melt and quench simulation complete. Trajectory saved to 'npt_mod.traj'.")