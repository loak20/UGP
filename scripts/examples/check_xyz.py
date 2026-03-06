from ase.io import read

atoms1 = read("optimized_H2O.xyz")
atoms2 = read("final_H2O.xyz")

print("Optimized atoms:", atoms1)
print("Final atoms:", atoms2)

print("Number of atoms:", len(atoms1))
print("Positions:\n", atoms1.get_positions())