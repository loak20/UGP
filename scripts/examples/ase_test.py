# Simple ASE test: optimize and run MD on a water molecule

from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.io import write

# 1. Create structure
atoms = molecule('H2O')

# 2. Attach a simple built-in calculator (no external code needed)
atoms.calc = EMT()

# 3. Geometry optimization
print("Starting optimization...")
opt = BFGS(atoms)
opt.run(fmax=0.05)

print("Optimized energy:", atoms.get_potential_energy())

# Save optimized structure
write("optimized_H2O.xyz", atoms)

# 4. Simple molecular dynamics
print("Starting MD...")

dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)

for i in range(10):
    dyn.run(1)
    print("Step", i, "Energy:", atoms.get_potential_energy())

# Save final structure
write("final_H2O.xyz", atoms)
