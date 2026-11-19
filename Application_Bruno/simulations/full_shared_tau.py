"""
The shared multi-tau model means, is the same as full_multi_tau, but with shared 
tau parameters for data points, that are at the same time point for the same condition.


theta_extended = [theta_estimate, taus, sigma_tau]

i.e. all taus are additional model parameters.


Further taus are shared.


Nomenclature: 
    All parameters (theta, taus, ...) and gradients are called either 
        - **multi** (if they are for the multi-tau-model, i.e. 77 tau paramerers) or 
        - **shared** (if they are for the shared-tau-model, i.e. 42 tau parameters).
"""
import amici
import numpy as np
import libsbml as sbml
import pypesto
from pypesto.petab import PetabImporter
from copy import deepcopy
import pandas as pd
from typing import List, Tuple
import shutil
import os
import sys
from pathlib import Path

from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "Application_Bruno"

TRUNCATED_GAUSSIAN = False
HIERARCHICAL_SIGMA_TAU = True # not TRUNCATED_GAUSSIAN

# All functions for simulations etc. can be reused, after mapping the parameters



"""
RUN SIMULATIONS: A) SIMULATIONS USING THE WRONG TIME POINTS
"""

# check if the results subfolder exists, otherwise create it

result_folder = str(APP_DIR / "simulations" / "results" / "full_shared_tau")
os.makedirs(result_folder, exist_ok=True)


def _compile_if_needed(sbml_path, model_name: str, model_dir) -> None:
    """Compile only if `model_dir` isn't already a compiled AMICI model for
    `model_name`. Safe to call from multiple concurrent processes as long as
    the FIRST caller wins the race (see submit_slurm_array.sh's prep step,
    which compiles once before any array task starts)."""
    model_dir = Path(model_dir)
    if not (model_dir / model_name / "__init__.py").exists():
        sbml_importer = amici.SbmlImporter(str(sbml_path))
        sbml_importer.sbml2amici(model_name, str(model_dir))


# delete only this script's own per-run scratch space. NOT "amici_models"
# (any of its APP_DIR/simulations/REPO_ROOT candidates -- see wrong_tp.py)
# or "amici" (see below): both now double as idempotent compiled-model
# caches that concurrent array tasks share read-only, first-caller-wins.
amici_files = [
    str(APP_DIR / "simulations" / "ode_sbml"),
]

for amici_folder in amici_files:
    shutil.rmtree(amici_folder, ignore_errors=True)


"""
Select which PEtab variant this process runs.

PETAB_INDEX < 0 (the default): reproduce the original scan-and-pick
behavior exactly -- pick the first variant (after sorting, and after
filtering out multi-timepoint-with-copied-data variants, which are not
usable by the shared-tau model) with no result yet. This is what
run_all.sh's local sequential loop still uses.

PETAB_INDEX >= 0 (set via SLURM_ARRAY_TASK_ID or the PETAB_INDEX env var):
claim that exact variant by position in the full (unfiltered) PEtab
variant listing -- matching submit_slurm_array.sh's array range, which is
sized from the raw variant count -- and exit immediately if that variant
is not applicable to this model (same filter as the default branch) or if
its result already exists. This is what makes concurrent SLURM array
tasks safe: each task claims a distinct, deterministic variant instead of
racing to scan the same "first missing" one.
"""
petab_dir = str(APP_DIR / "simulations" / "PEtab_versions")
petab_models = sorted(f for f in os.listdir(petab_dir) if os.path.isdir(os.path.join(petab_dir, f)))

exisiting_results = sorted(os.path.splitext(f)[0] for f in os.listdir(result_folder))

PETAB_INDEX = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("PETAB_INDEX", "-1")))

if PETAB_INDEX >= 0:
    model_to_run = petab_models[PETAB_INDEX]
    if "copied_data_multi_timepoints" in model_to_run:
        print(f"Variant {model_to_run} (index {PETAB_INDEX}) is not applicable to the shared-tau model, skipping.")
        sys.exit(0)
    if model_to_run in exisiting_results:
        print(f"Result for {model_to_run} (index {PETAB_INDEX}) already exists, skipping.")
        sys.exit(0)
    models_to_run = [model_to_run]
else:
    # get the list of all models, for which no result exists yet
    # filter for multi-timepoint models with copied data, as they can not be
    # used for the shared tau model
    models_to_run = sorted([
        model for model in petab_models
        if model not in exisiting_results
        and "copied_data_multi_timepoints" not in model
    ])
    if not models_to_run:
        if not os.environ.get("COMPILE_ONLY"):
            print("All variants already have results, nothing to run.")
            sys.exit(0)
        # prep job on a finished set: still compile, using the unperturbed variant
        models_to_run = ["sigma_tau_0_0"]
    model_to_run = models_to_run[0]

petab_yaml = str(APP_DIR / "simulations" / "PEtab_versions" / model_to_run / "Bruno_JExpBot2016.yaml")

print(f"Running model {model_to_run} with wrong time points")


"""
From here, the copy past of dependencies from full_multi_tau.py starts, but with the shared tau mapping.
"""

# Set up the model import -- compiled once into a stable, per-script cache
# dir; reused as-is by every concurrent array task processing a different
# PEtab variant, since this model is identical across all variants.
model_name = "ModelSharedlMultiTau"
model_dir = str(APP_DIR / "_amici_cache" / "full_shared_tau")
_compile_if_needed(APP_DIR / "petab_bruno_original" / "model_Bruno_JExpBot2016.xml", model_name, model_dir)
model_module = amici.import_model_module(model_name, model_dir)
amici_model = model_module.getModel()

# create PEtab model, in order to get informations from there
importer = PetabImporter.from_yaml(petab_yaml)
problem_wrong_tp = importer.create_problem(verbose=False)

# COMPILE_ONLY lets a short prep job compile the model and exit, without
# running any (multi-hour) optimization -- see submit_slurm_array.sh's prep
# stage, which depends the array job on this exiting 0 quickly rather than
# on a full variant finishing within the prep job's time limit.
if os.environ.get("COMPILE_ONLY"):
    print("COMPILE_ONLY set: model compiled, exiting without running any variant.")
    sys.exit(0)

# note: The time points are corrected below, when T_PETAB is deinfed
tau_shared_df = pd.read_csv(
    str(APP_DIR / "simulations" / "shared_parametrization_df.tsv"), sep="\t")


# the i-th data point is measured at time point tau[data_index_to_tau_shared_index[i]]
tau_shared_index_to_tau_multi_index = tau_shared_df["tau_number"].values

# the i-th tau parameter coresponds to the measurements  at index reduced_taus_to_data_index[i]
tau_multi_index_to_tau_shared_index = dict()
for i in range(len(tau_shared_index_to_tau_multi_index)):
    idx = int(tau_shared_index_to_tau_multi_index[i])
    if idx not in tau_multi_index_to_tau_shared_index:
        tau_multi_index_to_tau_shared_index[idx] = [i]
    else:
        tau_multi_index_to_tau_shared_index[idx].append(i)

T_PETAB = importer.petab_problem.measurement_df['time'].values
T_PETAB_shared = [float(T_PETAB[tau_multi_index_to_tau_shared_index[i][0]]) for i in tau_multi_index_to_tau_shared_index]

tau_shared_df["time"] = T_PETAB


DELTA_TIME_BOUND = 10 # the time variables are bound by +/- delta time

LB_FULL_SHARED_TAU = np.concatenate(
    [np.log10(importer.petab_problem.parameter_df['lowerBound'].values),
    np.zeros((max(tau_shared_df["tau_number"])+1,)),
    [0]])

UB_FULL_SHARED_TAU = np.concatenate(
    [np.log10(importer.petab_problem.parameter_df['upperBound'].values),
    tau_shared_df.drop_duplicates("tau_number").set_index("tau_number")["time"].reindex(range(42)).to_numpy()+DELTA_TIME_BOUND,
    [5]])

theta_shared_true = np.concatenate(
    [np.log10(importer.petab_problem.parameter_df['nominalValue'].values), 
    tau_shared_df.drop_duplicates("tau_number").set_index("tau_number")["time"].reindex(range(42)).to_numpy()]
    )


if TRUNCATED_GAUSSIAN:
    raise NotImplementedError("Truncated Gaussian not implemented for shared tau model yet. Please set TRUNCATED_GAUSSIAN to False.")


def _theta_shared_to_thetas(
    theta_extended: np.array, 
    log_trafo: bool = True,
    hierarchical_sigma_tau: bool = HIERARCHICAL_SIGMA_TAU
    ):
    """
    unpack the thetas.

    Not that by default, the dynamic parameters (i.e. theta_estimate) are log10 transformed.
    => returns 10**theta_estimate

    Input:
        theta_extended = [theta_estimate, taus_shared, sigma_tau (in case it is hierarchical)]
    Output:
        Three separate variables:
        - theta_estimate (the original model parameters, see petab parameter df for the mapping there)
        - taus_shared: the time point parameters (the i-th entry is for the i-th data point as given by the petab measurement df)
        - sigma_tau: the uncertainty w.r.t time points

    NOTE: the taus are shared, i.e. there are only 42 taus, that are shared between data points at the same time point for the same condition. 
    The mapping of which tau is for which data point is given by the shared_parametrization_df.tsv.
    """
    if log_trafo:
        theta_estimate = 10 ** theta_extended[:13]
    else:
        theta_estimate = theta_extended[:13]
    
    
    if hierarchical_sigma_tau:
        taus_shared = theta_extended[13:]
        sigma_tau = np.sqrt(np.sum((taus_shared-T_PETAB_shared)**2)/len(taus_shared))
    else:
        taus_shared = theta_extended[13:-1]
        sigma_tau = theta_extended[-1]
    return theta_estimate, taus_shared, sigma_tau


PARAMETER_IDS = importer.petab_problem.get_x_ids()
def _theta_estimate_to_dict(theta_estimate: np.array) -> dict:
    """
    Maps the theta_estimate to a dictionary, where the keys are the parameter ids
    and the values are the corresponding parameter values.
    
    Args:
        theta_estimate: np.array
            The parameters to map.
    
    Returns:
        dict: A dictionary mapping parameter ids to their values.
    """
    return {PARAMETER_IDS[i]: float(theta_estimate[i]) for i in range(len(theta_estimate))}


def taus_shared_to_taus_multi(taus_shared: np.array) -> np.array:
    """
    Maps the taus_shared to taus_multi, by copying the shared tau values to the corresponding multi tau values.
    """
    taus_multi = np.nan*np.ones((77, ))
    # i.e. tau_shared_index_to_tau_multi_index[i] is the index of the tau in taus_multi
    
    for i, tau_idx in enumerate(tau_shared_index_to_tau_multi_index):
        taus_multi[i] = taus_shared[tau_idx]

    return taus_multi


"""
Some more copy paste of stuff, that is needed in neg_log_post_shared_with_grad...
"""


"""
Compute the needed condition df's once, in order to get a faster function eval later...
"""
conditions_as_dict = dict()
condition_df = deepcopy(importer.petab_problem.condition_df).reset_index()
    
for condition in ["model1_data1", "model1_data2", "model1_data3", "model1_data4", "model1_data5", "model1_data6"]:
    condition_as_dict = condition_df[condition_df["conditionId"] == condition].iloc[0].to_dict()
    conditions_as_dict[condition] = condition_as_dict

condition_df_keys = condition_df.keys()

def set_parameters_for_condition(
    condition: str,
    theta_estimate: dict,
    amici_model: amici.Model,
):
    """
    Set the parameters for a given condition.

    Args:
        condition: str
            The condition to set parameters for.
        theta_extended: np.array
            The parameters to use for the simulation.
        amici_model: amici.Model
            The AMICI model to set parameters for.
    """
    theta_estimate_as_dict = _theta_estimate_to_dict(theta_estimate)

    # CONDITION INFOS
    condition_as_dict = conditions_as_dict[condition]
 
    # SET THE PARAMETERS THAT ARE NOT CONDITION SPECIFIC   
    for parameter_id in amici_model.getParameterIds():
        if parameter_id not in condition_df_keys:
            value = theta_estimate_as_dict[parameter_id]
            amici_model.setParameterById(parameter_id, value)

    # SET THE PARAMETERS THAT ARE CONDITION SPECIFIC
    for parameter_id, value in condition_as_dict.items():
        if parameter_id not in ["conditionId", "conditionName"]:
            # check if the value is a string => parameter id...
            try:
                value = float(value)
            except ValueError:
                value = theta_estimate_as_dict[value]
            amici_model.setParameterById(parameter_id, value)
    
    return amici_model



def _simulate_single_data_point_dicts_for_condition(
    condition: str,
    theta_extended: np.array,
    amici_model: amici.Model,
) -> List[dict]:
    """
    Simulates the dict, that contains _all_ the relevant infos for a single datapoint.

    Returns a list with all dicts for a condition

    The dict contains the following keys:
    - "sim": The simulated value.
    - "observable_name": The name of the observable.
    - "tau": The time point corresponding to the simulated values.
    - "data": The experimental data corresponding to the simulated values.
    - "t": The "assumed" time points corresponding to the experimental data, as in the PeTab.
    - "sigma_y": The standard deviation of the experimental data, as measured/given in the PeTab.
    - "theta_extended_index": The index of tau, as in theta_extended (i.e. IN THE MULTI FORMULATION!!!!).
    - "measurement_df_index": The index of the measurement in the measurement_df.
    - "dsim_dtau": The derivative of the simulated values with respect to tau.


    Args:
        condition: str
            The condition to simulate for.
        theta_extended: np.array
            The parameters to use for the simulation.
        amici_model: amici.Model
            The AMICI model to simulate.

    Returns:
        dict: A dictionary containing the simulation results.
    """
    # Set parameters for the condition
    theta_estimate, taus_shared, _ = _theta_shared_to_thetas(theta_extended, hierarchical_sigma_tau=HIERARCHICAL_SIGMA_TAU)
    taus_multi = taus_shared_to_taus_multi(taus_shared)
    
    amici_model = set_parameters_for_condition(condition, theta_estimate, amici_model)

    # Get all the relevant infos from a slightly extended Parameter df
    df_for_condition = tau_shared_df[
        tau_shared_df["simulationConditionId"] == condition]

    # get the values from the condition df. data_indices correspond to taus_shared!!!
    data_indices = df_for_condition["tau_number"].to_list()  # tau index of the data
    
    # get values, that will be part of the return dict:
    taus = taus_shared[data_indices].reshape(-1)      # tau values for the condition
    data = df_for_condition["measurement"].to_list()  # data values for the condition
    t = df_for_condition["time"].to_list()          # time values for the condition
    sigma_y = df_for_condition["noiseParameters"].to_list()          # sigma_y values for the condition
    measurement_df_index = data_indices
    tau_extended_indices = [idx + 13 for idx in measurement_df_index]


    # sort the taus, as AMICI needs sorted time points
    indices_sorted = np.argsort(taus)
    taus_sorted = taus[indices_sorted]
    amici_model.setTimepoints(taus_sorted.reshape(-1))

    # simulate the model, return the correctly ordered data (i.e. the i-th entry corresponds to the i-th data point)
    solver = amici_model.getSolver()
    solver.setSensitivityOrder(1)
    rdata = amici.runAmiciSimulation(amici_model, solver)

    # now we need to extract the right entries from rdata.x, so that the i-th entry corresponds to the i-th data point
    # NOTE: The ObservableIds as in PeTab start with an "o" (i.e. "o<insertObservable>")
    # The Observables as in AMICI start with "y" (i.e. "y<insertObservable>")
    # we Need to account for this, as observable_to_index should take the AMICI observable ids as keys

    observable_ids = df_for_condition["observableId"].to_list()
    observable_to_index = {observable_id: i for i, observable_id in enumerate(amici_model.getObservableIds())}

    # Inverse permutation: for each ORIGINAL row i, where does it land in the SORTED order?
    inv = np.empty_like(indices_sorted)
    inv[indices_sorted] = np.arange(len(indices_sorted))

    # Now pick rdata.y at (time_idx_in_sorted_order, observable_idx) for each original row i
    
    return_list = []

    for i in range(len(taus)):
        observable_i = observable_ids[i]
        
        # please have a look at xdot.cpp from AMICI, which contains the mapping of xdot -> w
        flux_v1_ReactionName = rdata.w[inv[i], 0]
        flux_v2_ReactionName = rdata.w[inv[i], 1]
        flux_v3_ReactionName = rdata.w[inv[i], 2]
        flux_v4_ReactionName = rdata.w[inv[i], 3]
        flux_v5_ReactionName = rdata.w[inv[i], 4]
        flux_v6_ReactionName = rdata.w[inv[i], 5]


        if observable_i == "yb10":
            ddtau = flux_v1_ReactionName - flux_v2_ReactionName + flux_v3_ReactionName
        elif observable_i == "ybcar":
            ddtau = -flux_v1_ReactionName
        elif observable_i == "ybcry":
            ddtau = -flux_v3_ReactionName - flux_v4_ReactionName
        elif observable_i == "yohb10":
            ddtau = flux_v4_ReactionName - flux_v5_ReactionName + flux_v6_ReactionName
        elif observable_i == "yzea":
            ddtau = -flux_v6_ReactionName

        return_list.append(
            {
                # Get the inv[i]s (i.e. undo sorting) entry, and there the right observable
                "sim": float(rdata.y[inv[i], observable_to_index[observable_i]]),
                "observable_name": observable_i,
                "tau": taus[i],
                "data": data[i],
                "t": t[i],
                "sigma_y": sigma_y[i],
                "theta_extended_index": tau_extended_indices[i],
                "measurement_df_index": measurement_df_index[i],
                "dsim_dtau": ddtau,
                "dsim_dtheta": _get_dsim_dtheta_estimate(
                    rdata, 
                    condition,
                    inv, 
                    observable_ids, 
                    observable_to_index,
                    i)
            }
        )

    return return_list

def _get_dsim_dtheta_estimate(
    rdata, 
    condition,
    inv, 
    observable_ids,
    observable_to_index,
    idx,
    ):
    """
    Compute the derivative of the simulated data with respect to the parameters.

    Returns derivatives of sim w.r.t theta_estimate.


    Input:
    rdata: AMICI rdata 
    inv: inverts the sorting of the data points
    observable_ids: list of observable IDs
    idx: index of the data point, as inv[idx] and observable_ids[idx]
    Dimensions:
    - dsim_d_theta_estimate: (n_params_theta, n_t)
    - dsim_d_tau: (n_params_tau, n_t)
    - d_sim_d_sigma_tau: (1, n_t)
    """
    d_sim = np.full((13, ), 0.0)

    # sim[idx] = rdata.y[inv[idx], observable_to_index[observable_ids[idx]]]

    # n_params_in_amici x n_t -> needs to be mapped tp PeTab Parameters
    # -> Row = parameter
    # -> Column = timepoint
    _before_parameter_mapping = rdata.sy[inv[idx], :, observable_to_index[observable_ids[idx]]]

    # init_b10_1 is only used in model1_data1
    if condition == "model1_data1":
        d_sim[0] = _before_parameter_mapping[0]
    # init_bcar1 is only used in model1_data2
    if condition == "model1_data2":
        d_sim[1] = _before_parameter_mapping[1]
    # init_bcar2 is only used in model1_data3
    if condition == "model1_data3":
        d_sim[2] = _before_parameter_mapping[1]
    # init_bcry_1 is only used in model1_data4
    if condition == "model1_data4":
        d_sim[3] = _before_parameter_mapping[2]
    # init_ohb10_1 is only used in model1_data5
    if condition == "model1_data5":
        d_sim[4] = _before_parameter_mapping[3]
    # init_zea_1 is only used in model1_data6
    if condition == "model1_data6":
        d_sim[5] = _before_parameter_mapping[4]

    # k5, kb1, kb2, kc1, kc2, kc4
    d_sim[6:12] = _before_parameter_mapping[5:11]

    # szea
    if condition == "model1_data3":
        # in the condition, all the multipliers are set to szea
        d_sim[12] = np.sum(_before_parameter_mapping[11:])
    elif condition == "model1_data6":
        # only the kc5 & kc4 multiplier are set to szea
        d_sim[12] = _before_parameter_mapping[11] + _before_parameter_mapping[16]

    return d_sim



def neg_log_post_shared_with_grad(
    theta_extended: np.ndarray, 
    amici_model
    ) -> Tuple[float, np.ndarray]:
    """
    Negative log-posterior:
        nlp = -log p(data | theta, taus) - log p(taus | times, sigma_tau)
    where
        data likelihood: y_i ~ N(sim_i(theta, tau_i), sigma_y_i^2)
        tau prior:       tau_i ~ N(t_i (from PEtab), sigma_tau^2)

    theta_extended layout is defined in _theta_extended_to_thetas().

    Returns a posterior + gradient.
    """
    # get theta_estimate (in linear space), taus_multi and sigma_tau
    # i.e [theta_log, taus_shared, sigma_tau (in case it is hierarchical)] 
    # ->  [[theta_lin, taus_multi, sigma_tau]. 
    theta_estimate, taus_shared, sigma_tau = _theta_shared_to_thetas(
        theta_extended, hierarchical_sigma_tau=HIERARCHICAL_SIGMA_TAU)

    _simulated_datapoints_list = []
    for condition in ["model1_data1", "model1_data2", "model1_data3", "model1_data4", "model1_data5", "model1_data6"]:
        _simulated_datapoints_for_condition = _simulate_single_data_point_dicts_for_condition(condition, theta_extended, amici_model)
        
        for e in _simulated_datapoints_for_condition:
            _simulated_datapoints_list.append(e) 

    # correct measurement_df_index and theta_extended_index
    tau_shared_index_to_tau_multi_index = tau_shared_df['tau_number'].values
    for i, sim_dict in enumerate(_simulated_datapoints_list):
        sim_dict["measurement_df_index"] = tau_shared_index_to_tau_multi_index[i]
        sim_dict["theta_extended_index"] = tau_shared_index_to_tau_multi_index[i] + 13

    # From here: unpack them by looping over the list
    # update/add to log_post + gradient.
    neg_log_post = 0.0
    grad = np.zeros_like(theta_extended)

    # list of tau indices, that already are in the posterior
    evaluated_tau_indices = []
    
    for sim_dict in _simulated_datapoints_list:
        sim = float(sim_dict["sim"])
        data = float(sim_dict["data"])
        tau = float(sim_dict["tau"])
        t = float(sim_dict["t"])
        sigma_y = float(sim_dict["sigma_y"])
        theta_extended_index = int(sim_dict["theta_extended_index"])

        # check if sim is NaN
        if np.isnan(sim):
            return np.inf, np.zeros_like(theta_extended)

        # calculate the obj function
        resid = sim - data

        neg_log_likelihood = 0.5 * (np.log(2*np.pi*sigma_y**2) + (resid**2) / (sigma_y**2))
    
        # adapt the way in which the posterior is evaluated to 
        # only count for every time point index once
        if theta_extended_index not in evaluated_tau_indices:
            
            neg_log_prior_tau = 0.5 * (np.log(2*np.pi*sigma_tau**2) + ((tau - t)**2) / (sigma_tau**2))        
            
            # grad of neg log post w.r.t. tau
            
            # gradient w.r.t sigma_tau
            if HIERARCHICAL_SIGMA_TAU:
                # sigma_tau is the profiled optimum sqrt(S/N); by the envelope theorem
                # d/dtau of the profiled prior is (tau - t) / sigma_tau^2 = N (tau - t) / S.
                d_tau = (tau - t) / sigma_tau**2
            else:
                d_sigma_tau = 1 / sigma_tau - ((tau - t)**2) / (sigma_tau**3)
                d_tau = (tau - t) / (sigma_tau**2)
            
            evaluated_tau_indices.append(theta_extended_index)
        else:
            # The posterior is zero, the posterior part of the gradient w.r.t tau is zero.
            neg_log_prior_tau = 0.0
            d_tau = 0.0
            d_sigma_tau = 0.0

        neg_log_post += neg_log_likelihood + neg_log_prior_tau
        # calculate the gradient
        # w.r.t theta_estimate => resid/sigma^2 * "grad_theta_estimate of sim"
        # w.r.t tau => resid/sigma^2 * "grad_tau of sim" for the corresponding tau index
        # w.r.t sigma_tau => derivative of neg log prior w.r.t sigma_tau
        grad[:len(theta_estimate)] += (resid / sigma_y**2) * sim_dict["dsim_dtheta"]
        
        if HIERARCHICAL_SIGMA_TAU:
            # w.r.t taus_shared => resid/sigma^2 * "grad_tau of sim" for the corresponding tau index + derivative of neg log prior w.r.t tau_shared
            grad[theta_extended_index] += (resid / sigma_y**2) * sim_dict["dsim_dtau"]  + d_tau
        else:
            # w.r.t taus_shared => resid/sigma^2 * "grad_tau of sim" for the corresponding tau index + derivative of neg log prior w.r.t tau_shared
            grad[theta_extended_index] += (resid / sigma_y**2) * sim_dict["dsim_dtau"]  + d_tau
            grad[-1] += d_sigma_tau
    # do the log-trafo chain rule for theta_estimate
    grad[:len(theta_estimate)] *= 10 ** theta_extended[:13] * np.log(10)

    return float(neg_log_post), grad



"""
pypesto Optimization FTW :D
"""
import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.visualize as visualize
import pypesto.store as store
from pypesto.petab import PetabImporter
import fides

"""
Import Multi-Threading
"""
import threading
from pypesto.engine import MultiThreadEngine
# Under SLURM, os.cpu_count() reports the NODE's total cores, not this
# array task's allocation -- SLURM_CPUS_PER_TASK reflects what --cpus-per-task
# actually reserved for this task. Falls back to os.cpu_count() for local runs
# (no SLURM_CPUS_PER_TASK set).
N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
engine = MultiThreadEngine(n_threads=N_THREADS)

_thread_local = threading.local()

def _get_thread_local_model():
    if not hasattr(_thread_local, 'amici_model'):
        _thread_local.amici_model = model_module.getModel()
    return _thread_local.amici_model

def obj_full_shared_tau(theta_shared):
    return neg_log_post_shared_with_grad(theta_shared, _get_thread_local_model())


objective_full_shared_tau = pypesto.Objective(
    fun=obj_full_shared_tau,
    grad=True,
)

if HIERARCHICAL_SIGMA_TAU:
    problem_full_shared_tau = pypesto.Problem(
        objective=objective_full_shared_tau,
        lb=LB_FULL_SHARED_TAU[:-1],
        ub=UB_FULL_SHARED_TAU[:-1]
    )
else:
    problem_full_shared_tau = pypesto.Problem(
        objective=objective_full_shared_tau,
        lb=LB_FULL_SHARED_TAU,
        ub=UB_FULL_SHARED_TAU
    )


fides_optimizer = optimize.FidesOptimizer(
    verbose=0, 
    hessian_update=fides.hessian_approximation.BFGS()
    )

fides_optimizer.options["maxiter"]=2_500


N_STARTS = 1_000

# your optimizer stays the same
result_full_shared_tau = optimize.minimize(
    problem_full_shared_tau,
    optimizer=fides_optimizer,
    engine=engine,
    n_starts=N_STARTS,
)

result_file_name = f"{result_folder}/{model_to_run}.hdf5"
store.write_result(result_full_shared_tau, result_file_name, overwrite=True)



for i in range(13): 
    print(f"Paramater {i}")
    # print(f"Min: {10**np.min(result_wrong_tp.sample_result.trace_x[0, :, i])}")
    print(f"theta_opt: {10**result_full_shared_tau.optimize_result[0].x[i]}")
    print(f"theta_pub: {importer.petab_problem.parameter_df['nominalValue'].values[i]}")
    # print(f"Max: {10**np.max(result_wrong_tp.sample_result.trace_x[0, :, i])}")

    print("---------------")
