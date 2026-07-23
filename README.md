# CHE-496 UGP

## Overview
This repository contains the codes, scripts, and simulation workflows developed during the CHE-496 Undergraduate Project (UGP) related to SiO₂ surface modeling, dehydroxylation studies, and melt–quench atomistic simulations using ASE and MACE-based workflows.

The repository includes:
- ASE + MACE example scripts
- SiO2 Surface dehydroxylation workflows done one by one, sequentially and recursively.
- Melt–quench molecular dynamic simulations
- Torch-Sim-atomsitic implementation and transition state exploration attempts (error prone and needs fixing)
- yaml files containing the packages used for the conda environment 

---

# Folder Descriptions
## `scripts/examples`
Contains sample scripts demonstrating:
- ASE workflows
- MACE calculator usage
- Basic atomistic simulation examples

---

## `SiO2_one_by_one`
Contains scripts for performing **dehydroxylation one site at a time** on the SiO₂ surface.

---

## `SiO2_recursive`
Contains scripts for performing **recursive dehydroxylation** on the SiO₂ surface.

---

## `SiO2_seq_all`
Contains scripts for **sequential dehydroxylation** workflows on the SiO₂ surface.

---

## `SiO2_melt`
Contains melt–quench simulation related codes.

Important file:
- `npt_mod.py`
  - Main script for performing melt–quench simulations using the **NPT Berendsen thermostat/barostat**.

---

## `SiO2_melt_torch`
Contains PyTorch-based atomistic simulation implementations.

### Notes
- Intended implementation of melt–quench workflow using Torch.
- Current implementation is **error-prone** and requires debugging/fixing.

---

## `SiO2_transition`
Contains transition state related codes.

### Notes
- Current implementation fails due to unresolved errors and requires debugging.

---

# Base Structure File

## `SiO2.cif`
The `SiO2.cif` file is used as the base structure for the simulations and surface modification workflows.

---

# Software and Packages

The project primarily uses:
- Python
- ASE (Atomic Simulation Environment)
- MACE calculators
- PyTorch
- Conda environments

---

# Conda Environments

Two Conda environments are provided:

| Environment | Purpose |
|---|---|
| `ugp_env` | Main environment for most simulations and workflows |
| `ugp_torch` | Environment for Torch/PyTorch-based atomistic simulations |

---

# Creating the Conda Environments

## 1. Create `ugp_env`

```bash
conda env create -f ugp_env.yml
```

Activate using:

```bash
conda activate ugp_env
```

---

## 2. Create `ugp_torch`

```bash
conda env create -f ugp_torch.yml
```

Activate using:

```bash
conda activate ugp_torch
```

---

# Typical Workflow

1. Create and activate the required Conda environment.
2. Use `SiO2.cif` as the starting structure.
3. Run:
   - Sequential dehydroxylation (`seq_all.py`)
   - One-by-one dehydroxylation (`SiO2_one_by_one.py`)
   - Recursive dehydroxylation (`recursive.py`)
4. Perform melt–quench simulations using:
   - `SiO2_melt/npt_mod.py`
5. Use ASE + MACE example scripts from:
   - `scripts/examples`

---

# Notes

- Some folders contain experimental or incomplete implementations.
- `SiO2_melt_torch` and `SiO2_transition` currently require debugging and further development.

---

# Acknowledgement

This work was carried out as part of the CHE-496 Undergraduate Project (UGP) under the guidance of Dr. Salman Ahmad Khan
