import numpy as np
import os 
import time
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=UserWarning, module="mace")

from ase.io import read
from ase.md.langevin import Langevin
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
    atoms = read('SiO2.cif')
    
    # Modern approach to prevent the entire cell from drifting 
    # (Addresses the fixcm=True deprecation warning)
    atoms.set_constraint(FixCom())
    
    # Initialize MACE-MP 
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device='cuda')
    atoms.calc = calc

    # ---------------------------------------------------------
    # 2. Thermodynamics Parameters (Optimized for 1 ps stability)
    # ---------------------------------------------------------
    # Reduced timestep because atomic velocities at 8000 K are extreme
    dt = 0.5 * units.fs 
    
    # Friction coefficient: 0.02 fs^-1 (damping time of 50 fs) ensures rapid 
    # thermal coupling necessary for a 1 ps phase timescale.
    friction = 0.02 / units.fs 

    MaxwellBoltzmannDistribution(atoms, temperature_K=298)
    Stationary(atoms)

    # Using NVT (Langevin) to prevent the 6-atom box from exploding under pressure
    dyn = Langevin(
        atoms,
        timestep=dt,
        temperature_K=298,
        friction=friction,
        fixcm=False  # Disabled here since we applied FixCom() globally above
    )

    # ---------------------------------------------------------
    # 3. Logging Output
    # ---------------------------------------------------------
    # 1 ps = 1000 fs = 2000 steps. We log every 20 steps (10 fs).
    traj = Trajectory('unit_cell_amorphous_silica.traj', 'w', atoms)
    dyn.attach(traj.write, interval=20) 

    logger = MDLogger(dyn, atoms, 'md_optimization.log', header=True, stress=False, peratom=True, mode="w")
    dyn.attach(logger, interval=20)

    # ---------------------------------------------------------
    # 4. Melt-Quench Algorithm (1 ps / phase)
    # ---------------------------------------------------------
    steps_per_ps = 2000

    print("\n--- Phase 1: Melting (298 K -> 8000 K over 1 ps) ---")
    run_md_stage(dyn, steps=steps_per_ps, T_start=298, T_end=8000, step_interval=10)
    
    print("\n--- Phase 2: Liquid Equilibration (1 ps at 8000 K) ---")
    run_md_stage(dyn, steps=steps_per_ps, T_start=8000, T_end=8000)
    
    print("\n--- Phase 3: Quenching (8000 K -> 298 K over 1 ps) ---")
    run_md_stage(dyn, steps=steps_per_ps, T_start=8000, T_end=298, step_interval=10)
    
    print("\n--- Phase 4: Room Temp Relaxation (1 ps at 298 K) ---")
    run_md_stage(dyn, steps=steps_per_ps, T_start=298, T_end=298)
    
    print("\nSimulation Complete. Trajectory saved to 'unit_cell_amorphous_silica.traj'.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {(time.time() - start_time):.2f} seconds")