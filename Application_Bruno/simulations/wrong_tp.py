
import pypesto
import libsbml as sbml
import petab


import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.visualize as visualize
import pypesto.store as store
from pypesto.petab import PetabImporter
from amici.petab.simulator import PetabSimulator

"""
Generate petab versions, 
with different perturbation schemas for the time points.
"""
import numpy as np
import os
import shutil
import pandas as pd



RANDOM_SEED = 42
# SIGMA_TAUS = [0.5, 1,bb 2, 3, 4, 5, 10, 15, 20]
# SIGMA_TAUS = [6, 7, 8, 9, 11, 12, 13, 14]
# SIGMA_TAUS = [0.0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
SIGMA_TAUS = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]


"""
RUN SIMULATIONS: A) SIMULATIONS USING THE WRONG TIME POINTS
"""

# check if the results subfolder exists, otherwise create it

result_folder = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/results/wrong_tp"
os.makedirs(result_folder, exist_ok=True)


# delete all amici folders with content and subfolders:
amici_files = [
    "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/amici_models", 
    "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/amici_models", 
    "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/amici_models", 
    "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/ode_sbml",
    "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/amici"
]

for amici_folder in amici_files:
    shutil.rmtree(amici_folder, ignore_errors=True)


"""
Get the first model (after sorting) for which no result exists yet...
"""

petab_dir = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/PEtab_versions"
petab_models = sorted(f for f in os.listdir(petab_dir) if os.path.isdir(os.path.join(petab_dir, f)))

exisiting_results = sorted(os.path.splitext(f)[0] for f in os.listdir(result_folder))
# get the list of all models, for which no result exists yet:
models_to_run = sorted([model for model in petab_models if model not in exisiting_results])
model_to_run = models_to_run[0]


print(f"Running model {model_to_run} with wrong time points")
petab_yaml = f"/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/PEtab_versions/{model_to_run}/Bruno_JExpBot2016.yaml"

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

# your optimizer stays the same
result_wrong_tp = optimize.minimize(
    problem_wrong_tp,
    optimizer=fides_optimizer,
    n_starts=N_STARTS
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