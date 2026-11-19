
import pypesto
import libsbml as sbml
import petab


import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.visualize as visualize
import pypesto.store as store
from pypesto.petab import PetabImporter
from pypesto.engine import MultiProcessEngine
from amici.petab.simulator import PetabSimulator

"""
Generate petab versions, 
with different perturbation schemas for the time points.
"""
import numpy as np
import os
import shutil
import sys
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "Application_Bruno"

RANDOM_SEED = 42
# SIGMA_TAUS = [0.5, 1,bb 2, 3, 4, 5, 10, 15, 20]
# SIGMA_TAUS = [6, 7, 8, 9, 11, 12, 13, 14]
# SIGMA_TAUS = [0.0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
SIGMA_TAUS = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]


"""
RUN SIMULATIONS: A) SIMULATIONS USING THE WRONG TIME POINTS
"""

# check if the results subfolder exists, otherwise create it

result_folder = str(APP_DIR / "simulations" / "results" / "wrong_tp")
os.makedirs(result_folder, exist_ok=True)


# delete all amici scratch folders, but NOT the "amici_models" directories --
# this script has no explicit amici.SbmlImporter/sbml2amici call of its own;
# it relies on pypesto.petab.PetabImporter.create_problem() to compile
# internally, which caches the compiled model under a CWD-relative
# "amici_models/<amici-version>/<model-id>" directory (pyPESTO's own
# force_compile=False default already skips recompilation when that
# directory exists). Since this script's invocation CWD isn't fixed, that
# cache could resolve under any of APP_DIR, APP_DIR/simulations, or
# REPO_ROOT -- so none of those "amici_models" candidates may be wiped, or a
# concurrent array task's compiled model gets deleted out from under it.
amici_files = [
    str(APP_DIR / "simulations" / "ode_sbml"),
    str(APP_DIR / "amici"),
]

for amici_folder in amici_files:
    shutil.rmtree(amici_folder, ignore_errors=True)


"""
Select which PEtab variant this process runs.

PETAB_INDEX < 0 (the default): reproduce the original scan-and-pick
behavior exactly -- pick the first variant (after sorting) with no
result yet. This is what run_all.sh's local sequential loop still uses.

PETAB_INDEX >= 0 (set via SLURM_ARRAY_TASK_ID or the PETAB_INDEX env var):
claim that exact variant by position, and exit immediately if its result
already exists -- this is what makes concurrent SLURM array tasks safe:
each task claims a distinct, deterministic variant instead of racing to
scan the same "first missing" one.
"""
petab_dir = str(APP_DIR / "simulations" / "PEtab_versions")
petab_models = sorted(f for f in os.listdir(petab_dir) if os.path.isdir(os.path.join(petab_dir, f)))

exisiting_results = sorted(os.path.splitext(f)[0] for f in os.listdir(result_folder))

PETAB_INDEX = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("PETAB_INDEX", "-1")))

if PETAB_INDEX >= 0:
    model_to_run = petab_models[PETAB_INDEX]
    if model_to_run in exisiting_results:
        print(f"Result for {model_to_run} (index {PETAB_INDEX}) already exists, skipping.")
        sys.exit(0)
    models_to_run = [model_to_run]
else:
    # get the list of all models, for which no result exists yet:
    models_to_run = sorted([model for model in petab_models if model not in exisiting_results])
    if not models_to_run:
        if not os.environ.get("COMPILE_ONLY"):
            print("All variants already have results, nothing to run.")
            sys.exit(0)
        # prep job on a finished set: still compile, using the unperturbed variant
        models_to_run = ["sigma_tau_0_0"]
    model_to_run = models_to_run[0]


print(f"Running model {model_to_run} with wrong time points")
petab_yaml = str(APP_DIR / "simulations" / "PEtab_versions" / model_to_run / "Bruno_JExpBot2016.yaml")

"""
Do the actual optimization
"""

N_STARTS = 1_000
# N_SAMPLES = 200_000
N_SAMPLES = None
RANDOM_SEED = 0

fides_optimizer = optimize.FidesOptimizer(verbose=0)
fides_optimizer.options["maxiter"]=5_000

importer_wrong_tp = PetabImporter.from_yaml(petab_yaml)
problem_wrong_tp = importer_wrong_tp.create_problem(verbose=False)

# COMPILE_ONLY lets a short prep job compile the model and exit, without
# running any (multi-hour) optimization -- see submit_slurm_array.sh's prep
# stage, which depends the array job on this exiting 0 quickly rather than
# on a full variant finishing within the prep job's time limit.
if os.environ.get("COMPILE_ONLY"):
    print("COMPILE_ONLY set: model compiled, exiting without running any variant.")
    sys.exit(0)

# Under SLURM, os.cpu_count() reports the NODE's total cores, not this
# array task's allocation -- SLURM_CPUS_PER_TASK reflects what --cpus-per-task
# actually reserved for this task. Falls back to os.cpu_count() for local runs.
# method="fork": this script has no __main__ guard, so "spawn" (the macOS
# default) would re-execute the whole module in every worker.
N_PROCS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
engine = MultiProcessEngine(n_procs=N_PROCS, method="fork")

# your optimizer stays the same
result_wrong_tp = optimize.minimize(
    problem_wrong_tp,
    optimizer=fides_optimizer,
    n_starts=N_STARTS,
    engine=engine,
)

# sample the posterior
# sample.sample(problem_wrong_tp, n_samples=N_SAMPLES, result=result_wrong_tp)

# store the result
result_file_name = f"{result_folder}/{model_to_run}.hdf5"
store.write_result(result_wrong_tp, result_file_name, overwrite=True)

for i in range(13): 
    print(f"Paramater {i}")
    # print(f"Min: {10**np.min(result_wrong_tp.sample_result.trace_x[0, :, i])}")
    print(f"theta_opt: {10**result_wrong_tp.optimize_result[0].x[i]}")
    print(f"theta_pub: {importer_wrong_tp.petab_problem.parameter_df['nominalValue'].values[i]}")
    # print(f"Max: {10**np.max(result_wrong_tp.sample_result.trace_x[0, :, i])}")

    print("---------------")


print(f"{len(petab_models) - len(models_to_run) + 1}/{len(petab_models)} models analyzed.")