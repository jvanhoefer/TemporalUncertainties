# About

This is the code for the paper `Efficient Bayesian inference for ordinary differential equation models from experimental data with uncertain measurement times`. This repository is structured as follows:

- `./Application_didactic` contains all code for the exponential decay example (i.e. Section 3.1 and Figure 2)
- `./Application_ConversionReaction` contains all code for the conversion reaction example (i.e. Section 3.2 and Figures 3/4)
- `./Application_Bruno` contains all code for the Bruno et al. example (i.e. section 3.3. and Figures 5/6)

Throughout all examples `problem_wrong_tp` and `result_wrong_tp` corresponds to the original formulation, neglecting temporal uncertainties. Further details can be found in the `README.md` files in the folders of the different examples.

Figure 1 is a schematic and was not generated with code.

# Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for the paper (Python 3.11 on the cluster). AMICI needs a C++ compiler, BLAS and SWIG; see the [AMICI installation guide](https://amici.readthedocs.io/en/latest/python_installation.html). Keep `pandas<3`: with `amici==0.34.2`, pandas 3 silently fixes parameters from the condition table in the Bruno problem.

# Reproducing the figures

All scripts resolve their paths relative to their own location, so they can be run from any working directory. Figures are written to `Figures Paper/building_blocks/`.

## Figure 2 (didactic example)

```bash
python Application_didactic/simulations.py
python Application_didactic/visualizations.py
```

## Figures 3 and 4 (conversion reaction)

`simulation_cr.py` runs one of 156 measurement setups per call, selected by `PETAB_INDEX` (or `SLURM_ARRAY_TASK_ID`), and stores it in `res_<n_experiment>/`. Figure 3 and Figure 4 D use setups 4 and 150; Figure 4 A–C uses all 156.

```bash
for i in $(seq 0 155); do PETAB_INDEX=$i python Application_ConversionReaction/simulation_cr.py; done
python Application_ConversionReaction/postprocessing_cr.py   # -> per_experiment_df.csv, per_setup_df.csv
python Application_ConversionReaction/visualization_cr.py
```

On a SLURM cluster, `Application_ConversionReaction/submit_slurm_array.sh` submits the 156 setups as a job array followed by post-processing and plotting.

## Figures 5 and 6 (Bruno et al. model)

1. Generate the perturbed PEtab problems (19 replicates x 9 values of sigma_tau, plus the unperturbed `sigma_tau_0_0`). Replicate `r` uses random seed `r`, which reproduces the problems used in the paper exactly:

   ```bash
   for r in $(seq 1 19); do REPLICATE_ID=$r python Application_Bruno/generate_petabs.py; done
   ```

2. Fit all 172 problems with the three objectives. Each call with `PETAB_INDEX=i` processes the i-th problem (sorted by name):

   ```bash
   for m in wrong_tp full_multi_tau marginalized_multi_tau; do
     for i in $(seq 0 171); do PETAB_INDEX=$i python Application_Bruno/simulations/$m.py; done
   done
   ```

   On a SLURM cluster use `Application_Bruno/simulations/submit_all_bruno.sh`, which submits one job array per objective.

   The shared-tau variants (`full_shared_tau.py`, `marginalized_shared_tau.py`) are included but not used in the paper.

3. Plot:

   ```bash
   python Application_Bruno/simulations/visualizations_Fig05.py
   python Application_Bruno/simulations/visualizations_Fig06.py
   ```

## Cluster setup

`prerequisites.example.sh` shows the module loads and environment used on our cluster. Copy it to `prerequisites.sh` and adapt it; the submit scripts source it automatically. Optional settings for the submit scripts (`MAIL_USER`, `ALLOWED_NODES`, `CPUS_PER_TASK`, `MAX_CONCURRENT`, `ARRAY_TIME`) are documented at the top of each script.

# Reusable implementation

A general-purpose, installable implementation of the marginalized approach is available as the package
[`petab-temporal-uncertainty`](https://github.com/PaulJonasJost/petab-temporal-uncertainty)
(`pip install petab-temporal-uncertainty`). It defines a PEtab extension for uncertain measurement times
and builds the augmented AMICI model and the corresponding pyPESTO objective (with gradient) for
user-supplied PEtab problems. The package has the same functionality as the marginalized implementation
in this repository. The analyses in the paper were performed with the code in this repository, which
we keep unchanged for reproducibility.

# PEtab version

The PEtab problems in this repository are in PEtab v1 format, which we used throughout the study.
They can be converted to PEtab v2 with a single function call of libpetab-python
([`petab.v2.petab1to2`](https://petab.readthedocs.io/projects/libpetab-python/en/latest/build/_autosummary/petab.v2.petab1to2.html)):

```python
from petab.v2.petab1to2 import petab1to2
petab1to2("path/to/problem_v1.yaml", output_dir="path/to/problem_v2")
```

To avoid duplicating every problem in the deposit, we do not store separate v2 copies.
