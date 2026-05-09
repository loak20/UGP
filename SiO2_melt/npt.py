import numpy as np
import os 
import time
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=UserWarning, module="mace")

from ase.io import read
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.constraints import FixCom
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.logger import MDLogger
from mace.calculators import mace_mp

def run_md_stage(dyn, steps, T_start, T_end, step_interval=10):
    """
    Executes an MD stage with robust linear temperature ramping.
    """
    if T_start == T_end:
        dyn.set_temperature(temperature_K=T_start)
        dyn.run(steps)
    else:
        for i in range(0, steps, step_interval):
            frac = i / steps
            T_curr = T_start + frac * (T_end - T_start)
            dyn.set_temperature(temperature_K=T_curr)
            
            steps_to_run = min(step_interval, steps - i)
            dyn.run(steps_to_run)

def main():
    # ---------------------------------------------------------
    # 1. System Initialization & Constraints
    # ---------------------------------------------------------
    # Using a single unit cell (no replication/cleaving)
    atoms = read('SiO2.cif')
    
    # Modern approach to prevent the entire cell from drifting 
    atoms.set_constraint(FixCom())
    
    # Initialize MACE-MP 
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device='cuda')
    atoms.calc = calc

    # ---------------------------------------------------------
    # 2. Thermodynamics Parameters (Optimized for 1 ps stability)
    # ---------------------------------------------------------
    # Reduced timestep because atomic velocities at 8000 K are extreme
    dt = 0.5 * units.fs 
    steps_per_ps = int(1000 * units.fs / dt) # 2000 steps
    
    MaxwellBoltzmannDistribution(atoms, temperature_K=298)
    Stationary(atoms)

    # Convert paper's NPT parameters to ASE standard atomic units
    # Target Pressure: 1.0 atm 
    target_pressure = 1.0 * 1.01325 * units.bar
    
    # Target Modulus: 360,000 atm -> Compressibility = 1 / Modulus
    modulus_bar = 360000 * 1.01325
    target_compressibility = (1.0 / modulus_bar) / units.bar

    # Initialize Berendsen NPT Dynamics
    # taut and taup are significantly reduced from the paper's 1 ps 
    # to ensure the system actually couples to the bath within your 1 ps runtime.
    dyn = NPTBerendsen(
        atoms,
        timestep=dt,
        temperature_K=298,
        pressure_au=target_pressure,
        compressibility_au=target_compressibility,
        taut=20 * units.fs,   # Fast thermal coupling
        taup=100 * units.fs,  # Barostat coupling (slower than taut to avoid ringing)
        fixcm=False           # Disabled here since we applied FixCom() globally
    )

    # ---------------------------------------------------------
    # 3. Logging Output
    # ---------------------------------------------------------
    # 1 ps = 1000 fs = 2000 steps. We log every 20 steps (10 fs).
    traj = Trajectory('npt.traj', 'w', atoms)
    dyn.attach(traj.write, interval=20) 

    # Stress tracking enabled to monitor NPT barostat health
    logger = MDLogger(dyn, atoms, 'md_npt.log', header=True, stress=True, peratom=True, mode="w")
    dyn.attach(logger, interval=20)

    # ---------------------------------------------------------
    # 4. Melt-Quench Algorithm (Bulk, 1 ps / phase)
    # ---------------------------------------------------------
    print("\n--- Phase 1: Melting (298 K -> 8000 K over 1 ps) ---")
    run_md_stage(dyn, steps=steps_per_ps, T_start=298, T_end=8000, step_interval=10)
    
    print("\n--- Phase 2: Liquid Equilibration (1 ps at 8000 K) ---")
    run_md_stage(dyn, steps=steps_per_ps, T_start=8000, T_end=8000)
    
    print("\n--- Phase 3: Quenching (8000 K -> 298 K over 1 ps) ---")
    run_md_stage(dyn, steps=steps_per_ps, T_start=8000, T_end=298, step_interval=10)
    
    print("\n--- Phase 4: Room Temp Relaxation (1 ps at 298 K) ---")
    run_md_stage(dyn, steps=steps_per_ps, T_start=298, T_end=298)
    
    print("\nSimulation Complete. Trajectory saved to 'npt.traj'.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {(time.time() - start_time):.2f} seconds")