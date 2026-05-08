"""
This folder contains all the code to run the simulations. 

It starts with the definition of the objective functions and/or the functions 
to generate the extended SBML models. Afterwards we run and store all 
simulations.

Definition of the objective functions:

- problem_wrong_tp: pypesto Problem, as given in the original paper

- problem_marginalized_multi_tau: 
    For this objective, we assume that every data point has a different temporal 
    uncertainty. 
    We further fit SIGMA_TAU to the data and implement the MARGINALIZED MODEL.
- problem_marginalized_shared_tau:
    For this objective, we assume that all data points measured at the same time 
    point are shifted by the same value for tau i.e. temporal uncertainty. 
    We further fit SIGMA_TAU to the data and implement the MARGINALIZED MODEL.

- problem_full_multi_tau:
    For this objective, we assume that every data point has a different temporal 
    uncertainty. 
    We further fit SIGMA_TAU to the data and implement the FULLY EXTENDED MODEL.
- problem_full_shared_tau:
    For this objective, we assume that all data points measured at the same time 
    point are shifted by the same value for tau i.e. temporal uncertainty. 
    We further fit SIGMA_TAU to the data and implement the FULLY EXTENDED MODEL.
"""
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


# SIGMA_TAUS = [0.5, 1, 2, 3, 4, 5, 10, 15, 20]
# SIGMA_TAUS = [6, 7, 8, 9, 11, 12, 13, 14]
# SIGMA_TAUS = [0.0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
# SIGMA_TAUS = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]
# SIGMA_TAUS = [0, 15]
SIGMA_TAUS = [0, 1, 3, 5, 7, 9, 11, 13, 15]

"""
Setup for the synthetic data generation.
"""
# if true, the same time points are taken for measurements taken at the same time in the original data
# if the measurements come from the same tp + measurement condition
SHARED_TIMEPOINTS = False 
# If true, the data will be generated synthetically, otherwise the original data will be used
# and only the time points will be perturbed
SYNTHETIC_DATA = False

REPLICATE_ID = 19
RANDOM_SEED = 19

np.random.seed(RANDOM_SEED)
# generate the folders:
ORIGINAL_PETAB_DIR = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/petab_bruno_original"

PETAB_FOLDER_BASE = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/PEtab_versions"
os.makedirs(PETAB_FOLDER_BASE, exist_ok=True)
os.makedirs(f"{PETAB_FOLDER_BASE}/sigma_tau_0_0", exist_ok=True)


# copy original files (unchanged)
for file in os.listdir(ORIGINAL_PETAB_DIR):
    src_file = os.path.join(ORIGINAL_PETAB_DIR, file)
    dst_file = os.path.join(f"{PETAB_FOLDER_BASE}/sigma_tau_0_0", file)
    if os.path.isfile(src_file) and src_file.endswith((".yaml", ".xml", ".tsv")):
        shutil.copy(src_file, dst_file)


def generate_timepoints_with_perturbation(
    sigma_tau: float, 
    shared_timepoints: bool = False
    ) -> np.ndarray:
    """
    Generates perturbed time points, either with shared or non-shared (i.e. multi) time points.
    
    Ensures non-negative time points (by resampling if necessary).
    """
    measurement_df = pd.read_csv(os.path.join(ORIGINAL_PETAB_DIR, "measurementData_Bruno_JExpBot2016.tsv"), sep="\t")
    times = measurement_df["time"].to_numpy(dtype=float)

    if shared_timepoints:
        # read in tsv file with time point information
        time_points_df = pd.read_csv(
            "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/shared_parametrization_df.tsv", 
            sep="\t"
            )
        delta_tau_reduced = np.random.normal(loc=0, scale=sigma_tau, size=(42, ))
        # map the delta_tau to the original time points, using the time_points_df
        delta_tau = np.array([delta_tau_reduced[idx] for idx in time_points_df["tau_number"].to_numpy(dtype=int)])
    else:
        delta_tau = np.random.normal(loc=0, scale=sigma_tau, size=times.shape)

    # perturb the time points and check, if they are still positive
    times += delta_tau
    while np.any(times < 0):
        print(f"Resample times for sigma_tau = {sigma_tau}")
        times = generate_timepoints_with_perturbation(sigma_tau, shared_timepoints)

    return times

def copy_all_petab_files(original_petab_dir: str, new_petab_dir: str):
    """
    Copies all PEtab files from the original directory to the new directory.
    """
    os.makedirs(new_petab_dir, exist_ok=True)
    # copy files from original source
    for file in os.listdir(original_petab_dir):
        src_file = os.path.join(original_petab_dir, file)
        dst_file = os.path.join(new_petab_dir, file)
        if os.path.isfile(src_file) and src_file.endswith((".yaml", ".xml", ".tsv")):
            shutil.copy(src_file, dst_file)

"""
Naming of the PEtab folders is
- for multi time points with copied data: copied_data_multi_timepoints_sigma_tau_{sigma_tau}
- for shared time points with copied data: copied_data_shared_timepoints_sigma_tau_{sigma_tau}

- for multi time points with synthetic data: synthetic_data_multi_timepoints_sigma_tau_{sigma_tau}
- for shared time points with synthetic data: synthetic_data_shared_timepoints_sigma_tau_{sigma_tau}


- for the original data (without perturbation): sigma_tau_0_042
"""

# copy the modified files
for sigma_tau in SIGMA_TAUS:

    print(f"Generated PEtab version with synthetic data with multi_tau for sigma_tau = {sigma_tau}")
    
    if SYNTHETIC_DATA:
        # test, if the "non synthetic" PEtab exist, as this should be taken as the basis
        if SHARED_TIMEPOINTS:
            template_petab_dir = f"{PETAB_FOLDER_BASE}/copied_data_shared_timepoints_{str(REPLICATE_ID)}_sigma_tau_{str(sigma_tau).replace('.', '_')}"
            new_petab_dir = f"{PETAB_FOLDER_BASE}/synthetic_data_shared_timepoints_{str(REPLICATE_ID)}_sigma_tau_{str(sigma_tau).replace('.', '_')}"
        else:
            template_petab_dir = f"{PETAB_FOLDER_BASE}/copied_data_multi_timepoints_{str(REPLICATE_ID)}_sigma_tau_{str(sigma_tau).replace('.', '_')}"
            new_petab_dir = f"{PETAB_FOLDER_BASE}/synthetic_data_multi_timepoints_{str(REPLICATE_ID)}_sigma_tau_{str(sigma_tau).replace('.', '_')}"
        
        # test if the template petab yaml exists
        template_petab_yaml_file = os.path.join(template_petab_dir, "Bruno_JExpBot2016.yaml")
        if not os.path.exists(template_petab_yaml_file):
            raise NotImplementedError("Synthetic data requires the non-synthetic PEtab to exist first.")
        
        # import and simulate data
        petab_problem_copied = petab.Problem.from_yaml(template_petab_yaml_file)

        # create "copied" pypesto problem
        pypesto_importer_copied = pypesto.petab.PetabImporter(petab_problem_copied)
        pypesto_problem_copied = pypesto_importer_copied.create_problem(
            verbose=False
        )

        # now simulate the data using the copied pypesto problem, which contains 
        # perturbed time points...       
        simulator = PetabSimulator(petab_problem_copied)
        petab_problem_copied.measurement_df = simulator.simulate(
            noise=True,
            amici_model=pypesto_problem_copied.objective.amici_model,
            as_measurement=True,
        )
        simulator.remove_working_dir()

        # copy all files from the original petab dir to the new petab dir
        copy_all_petab_files(ORIGINAL_PETAB_DIR, new_petab_dir)

        measurement_df_original = pd.read_csv(os.path.join(ORIGINAL_PETAB_DIR, "measurementData_Bruno_JExpBot2016.tsv"), sep="\t")
        petab_problem_copied.measurement_df["time"] = measurement_df_original["time"]
        
        # save the new measurement df with the original time points, but perturbed data
        petab_problem_copied.measurement_df.to_csv(os.path.join(new_petab_dir, "measurementData_Bruno_JExpBot2016.tsv"), sep="\t", index=False)
        

    if not SYNTHETIC_DATA:
        # generate new folder, where the modified files will be stored
        if SHARED_TIMEPOINTS:
            new_petab_dir = f"{PETAB_FOLDER_BASE}/copied_data_shared_timepoints_{str(REPLICATE_ID)}_sigma_tau_{str(sigma_tau).replace('.', '_')}"
        else:
            new_petab_dir = f"{PETAB_FOLDER_BASE}/copied_data_multi_timepoints_{str(REPLICATE_ID)}_sigma_tau_{str(sigma_tau).replace('.', '_')}"
  
        # copy all files from the original petab dir to the new petab dir
        copy_all_petab_files(ORIGINAL_PETAB_DIR, new_petab_dir)

        # (over-)write the measurement df
        measurement_df = pd.read_csv(os.path.join(ORIGINAL_PETAB_DIR, "measurementData_Bruno_JExpBot2016.tsv"), sep="\t")
        measurement_df["time"] = generate_timepoints_with_perturbation(sigma_tau, shared_timepoints=SHARED_TIMEPOINTS)
        measurement_df.to_csv(os.path.join(new_petab_dir, "measurementData_Bruno_JExpBot2016.tsv"), sep="\t", index=False)
    


