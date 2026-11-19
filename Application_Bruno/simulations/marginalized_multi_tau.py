"""
Marginalized Objective for the Multi-Tau version of the Model

We have the generate_sbml_for_condition function, that creates the SBML models, 
further we have `neg_log_post_multi_tau_marginalized` Function, that computes the 
marginalized negative log posterior for the multi-tau version of the model.

Here the SBML contains a z_i state, with right hand side set to a rate rule
the RHS is equal to llh*posterior (NOT LOG-Likelihood PLUS LOG POSTERIOR)

The Objective is then given by sum log(z(t_end)).
"""
import libsbml as sbml
from typing import List
import pandas as pd
import pypesto
import petab
import amici
import warnings
import numpy as np
import os
import sys
from copy import deepcopy
from scipy.stats import norm
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "Application_Bruno"

"""
Set variables for the simulation/analysis e.g. for the "right" PEtab
"""
INCLUDE_SAMPLING = False
INCLUDE_PROFILES = False
INCLUDE_GRAD_CHECK = False

RANDOM_SEED = 42
# SIGMA_TAUS = [0.0, 0.5, 1, 2, 3, 4, 5, 10, 15, 20]

TRUNCATED_GAUSSIAN = True
"""
Initialize the folder, to automatically choose the "next" model
"""
result_folder = str(APP_DIR / "simulations" / "results" / "marginalized_multi_tau")
os.makedirs(result_folder, exist_ok=True)

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

petab_yaml = str(APP_DIR / "simulations" / "PEtab_versions" / model_to_run / "Bruno_JExpBot2016.yaml")
print(f"Running model {model_to_run} with wrong time points")



sigma_tau_parts = model_to_run.split("_")
if sigma_tau_parts[-2].isdigit():
    # decimal case: e.g. "..._0_5" -> "0.5" or "..._0_0" -> "0.0"
    SIGMA_TAU = float(f"{sigma_tau_parts[-2]}.{sigma_tau_parts[-1]}")
else:
    # integer case: e.g. "..._5" -> "5.0"
    SIGMA_TAU = float(sigma_tau_parts[-1])
del sigma_tau_parts

ORIGINAL_PETAB_DIR = str(APP_DIR / "petab_bruno_original")

np.random.seed(RANDOM_SEED)


ORIGINAL_SBML_DIR = str(APP_DIR / "simulations" / "PEtab_versions" / model_to_run / "model_Bruno_JExpBot2016.xml")

reader = sbml.SBMLReader()
document = reader.readSBML(ORIGINAL_SBML_DIR)

# create PEtab model, in order to get informations from there
importer = petab.v1.Problem.from_yaml(petab_yaml)

# Marginalization integration bounds: every z_k state is integrated over the
# SAME shared window [T_0, T_MAX] (rather than a per-data-point local window),
# matching the paper's Section 2.4 definition of a single (t0, tmax) support
# for the latent measurement-time distribution. Measurement times in this
# dataset range from 5 to 180, so T_MAX = 240 comfortably covers all of them.
T_0 = 0
T_MAX = 240
CONDITIONS = ["model1_data1", "model1_data2", "model1_data3", "model1_data4", "model1_data5", "model1_data6"]

"""
Delete all previous files, in order to e.g. recompile the AMICI model
"""
import shutil
# Delete only THIS script's own (script, variant)-scoped scratch space --
# NOT the shared "amici"/"sbml_models" roots (a concurrent task processing
# a different variant, or marginalized_shared_tau.py processing ANY
# variant, would have its own compiled model/SBML wiped out from under it)
# and NOT the "amici_models"/"ode_sbml" paths (unused by this script; they
# are pyPESTO's/full_*.py's own compiled-model caches, shared across the
# OTHER Bruno fitting scripts -- see Task 2's fix for those).
amici_files = [
    str(APP_DIR / "amici" / "marginalized_multi_tau" / model_to_run),
    str(APP_DIR / "simulations" / "sbml_models" / "marginalized_multi_tau" / model_to_run),
]

for amici_folder in amici_files:
    shutil.rmtree(amici_folder, ignore_errors=True)


def _create_z_states(
    sbml_model, 
    observable_ids: List[str],
    data_vals: List[float],
    ts_true: List[float],
    sigmas_x: float, 
    sigma_tau_id: str,
    data_point_indices: List[int],
    ts_start: List[float] = None,
    ts_end: List[float] = None
    ):
    """
    Create a new z_i state in the SBML model with id "<z_id>".
    The right hand side is given by

        exp[-1/2 * ((x(tau) - data_val)/sigma_x)^2 + (t_true - tau)/sigma_tau)^2]

    In the end we consider

            (1/(2*pi*sigma_x * sigma_tau)) * z(t_end)

    Expects that the lists are all the same length. 
    The state will have the id z_<i> for some integer i.

    Parameters:
    - sbml_model: The SBML model to modify.
    - observable_ids: The IDs of the observables to create.
    - data_vals: The data values to use.
    - ts_true: The true time points.
    - sigmas_x: The standard deviation for the observable (measured in our case).
    - sigmas_tau_id: The ID of sigma_tau (which is a parameter in the model).
    - data_point_indices: The indices of the data points to use. i.e. the index of the PEtab data table.
    - ts_start: The prior is truncated to be zero for time < t_start.
    - ts_end: The prior is truncated to be zero for time > t_end.
    """
    for observable_id, data_val, t_true, sigma_x, i in zip(observable_ids, data_vals, ts_true, sigmas_x, data_point_indices):
        # create z_id = name/id for the state
        z_id = f"z_{i}"
        
        # create z state
        z_state = sbml_model.createSpecies()
        z_state.setId(z_id)
        z_state.setName(z_id)
        z_state.setCompartment("cyt")
        z_state.setConstant(False)
        z_state.setInitialConcentration(0)

        # create rate rule for z state
        lsq_data = f"pow(({observable_id} - {data_val})/{sigma_x}, 2)"
        # lsq_time = f"pow(({t_true} - time)/{sigma_tau_id}, 2)"
        lsq_time = f"pow(({t_true} - t)/{sigma_tau_id}, 2)"
        rhs_string = f"exp(-0.5 * ({lsq_data} + {lsq_time}))"
        
        # add the piecewise function, in case the ts_start/end are given.
        if (ts_start is not None) or (ts_end is not None):
            raise NotImplementedError("we don't want this...")
            t_start = ts_start[i]
            t_end = ts_end[i]
            rhs_string = f"piecewise(0, time < {t_start}, exp((-0.5) * ({lsq_data} + {lsq_time})), time < {t_end}, 0)"
            # rhs_string = f"piecewise(1, time < {t_start}, 0)"

        rate_rule = sbml_model.createRateRule()
        rate_rule.setId(f"d_dt_{z_id}")
        rate_rule.setVariable(z_id)

        math_ml_formula = sbml.parseFormula(rhs_string)
        if math_ml_formula:
            rate_rule.setMath(math_ml_formula)
        else:
            raise ValueError(f"Could not parse SBML formula: {rhs_string}")
    return sbml_model

def create_adapted_sbml(
    measurement_df: pd.DataFrame,
    condition: str,
    ):
    """
    Parameters:
    - sbml_dir: Directory of the ORIGINAL SBML model
    - measurement_df: DataFrame containing the measurement data
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Invalid condition {condition}")
    
    reader = sbml.SBMLReader()
    document = reader.readSBML(ORIGINAL_SBML_DIR)
    sbml_model = document.getModel()

    sbml_model.setId(f"Bruno_JExpBot2016_marginalized_{condition}")
    sbml_model.setName(f"Bruno_JExpBot2016_marginalized_{condition}")

    # add sigma_tau parameter
    k = sbml_model.createParameter()
    k.setId("sigma_tau")
    k.setName("sigma_tau")
    k.setValue(0)
    k.setConstant(True)

    # add a time parameter
    time_parameter = sbml_model.createParameter()

    time_parameter.setName("t")
    time_parameter.setId("t")
    time_parameter.setConstant(False)

    time_assignment = sbml_model.createAssignmentRule()
    time_assignment.setVariable("t")
    time_assignment.setMath(sbml.parseL3Formula('time'))

    # filter the measurement df for the given condition to get the inputs for _create_z_states:
    condition_df = measurement_df[measurement_df["simulationConditionId"] == condition]
    observable_ids = [obs_id[1:] for obs_id in condition_df["observableId"].tolist()]
    data_vals = condition_df["measurement"].tolist()
    ts_true = condition_df["time"].tolist()
    sigmas_x = condition_df["noiseParameters"].tolist()
    sigma_tau_id = "sigma_tau"
    data_point_indices = condition_df.index.tolist()

    # ts_start = [t_true - DELTA_TIME_BOUND for t_true in ts_true]
    # ts_end = [t_true + DELTA_TIME_BOUND for t_true in ts_true]

    sbml_model = _create_z_states(
        sbml_model, 
        observable_ids, 
        data_vals, 
        ts_true, 
        sigmas_x, 
        sigma_tau_id, 
        data_point_indices, 
        # ts_start, 
        # ts_end
    )
    # sbml_as_string = sbml.writeSBMLToString(document)
    # sbml_as_string = sbml_as_string.replace("<ci> time </ci>", '<csymbol encoding=\"text\" definitionURL=\"http://www.sbml.org/sbml/symbols/time\"> t </csymbol>')
    
    # check for consistency
    document.setConsistencyChecks(sbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)
    if document.checkConsistency():

        for error_num in range(document.getErrorLog().getNumErrors()):
            if not document.getErrorLog().getError(error_num).isWarning():
                warnings.warn(
                    document.getErrorLog().getError(error_num).getMessage(),
                    RuntimeWarning)

    
    return sbml.writeSBMLToString(document)

# load the data df from petab
measurement_df = importer.measurement_df


# create all models for all Conditions:
from pathlib import Path

for condition in CONDITIONS:
    # Update the SBML model
    sbml_as_string = create_adapted_sbml(
        measurement_df=measurement_df,
        condition=condition
    )

    # create the directory for the SBML file
    sbml_dir = APP_DIR / "simulations" / "sbml_models" / "marginalized_multi_tau" / model_to_run / f"{condition}.xml"
    sbml_dir.parent.mkdir(parents=True, exist_ok=True)

    # save the model
    with open(sbml_dir, 'w') as f_out:
        f_out.write(sbml_as_string)
    print(f"create SBML model for condition: {condition}")

# create AMICI models
amici_models = []
for condition in CONDITIONS:

    # create an AMICI importer and compile/import the amici file
    sbml_importer = amici.SbmlImporter(str(APP_DIR / "simulations" / "sbml_models" / "marginalized_multi_tau" / model_to_run / f"{condition}.xml"))
    # model_to_run is this file's existing unique-per-variant key (also used
    # for sbml_dir above and for result_file_name) -- this model's ODEs
    # embed model_to_run's specific perturbed measurement times, so it needs
    # its own compiled module per (script, variant, condition) triple or a
    # concurrent array task processing a different variant crashes AMICI's
    # staleness guard on import (same failure mode fixed for
    # Application_ConversionReaction). The "multi" tag also distinguishes
    # this from marginalized_shared_tau.py's compiled models for the SAME
    # variant/condition, whose SBML content differs (multi- vs shared-tau
    # parameter mapping) despite sharing model_to_run/condition.
    model_name = f"Bruno_JExpBot2016_marginalized_multi_{model_to_run}_{condition}"
    model_dir = str(APP_DIR / "amici" / "marginalized_multi_tau" / model_to_run / f"amici_{condition}")
    sbml_importer.sbml2amici(model_name, model_dir)
    model_module = amici.import_model_module(model_name, model_dir)
    amici_model = model_module.getModel()


    # Append the AMICI model to the list
    amici_models.append(amici_model)
    print(f"create AMICI model for condition: {condition}")

# COMPILE_ONLY lets a short prep job compile this variant's per-condition
# models and exit, without running any (multi-hour) optimization -- see
# submit_slurm_array.sh's prep stage, which depends the array job on this
# exiting 0 quickly rather than on a full variant finishing within the prep
# job's time limit. Note: unlike full_multi_tau.py/full_shared_tau.py, these
# per-condition models are NOT shared across variants (Task 3), so this
# doesn't save recompilation for the array's other variants -- it only
# guards against the prep job timing out and blocking --dependency=afterok.
if os.environ.get("COMPILE_ONLY"):
    print("COMPILE_ONLY set: models compiled, exiting without running any variant.")
    sys.exit(0)

# create an objective function, where the z's are evaluated at z(t_end) - z(t_start)...
"""
from here on, all functions are part of the objective function:
"""

def _theta_to_dict(theta_log: np.array):
    """
    Returns the parameters in linear scale as dict.

    NOTE: The last parameter (sigma_tau) is not transformed.
    """
    theta_as_dict = {
        'init_b10_1': 10**theta_log[0],
        'init_bcar1': 10**theta_log[1],
        'init_bcar2': 10**theta_log[2],
        'init_bcry_1': 10**theta_log[3],
        'init_ohb10_1': 10**theta_log[4],
        'init_zea_1': 10**theta_log[5],
        'k5': 10**theta_log[6],
        'kb1': 10**theta_log[7],
        'kb2': 10**theta_log[8],
        'kc1': 10**theta_log[9],
        'kc2': 10**theta_log[10],
        'kc4': 10**theta_log[11],
        'szea': 10**theta_log[12],
        'sigma_tau': theta_log[13]
    }

    return theta_as_dict

def set_parameters_for_condition(
    condition: str,
    theta: np.array,
    amici_model: amici.Model,
):
    """
    Set the parameters for a given condition.

    Args:
        condition: str
            The condition to set parameters for.
        theta: np.array
            The parameters to use for the simulation.
        amici_model: amici.Model
            The AMICI model to set parameters for.
    """
    theta_as_dict = _theta_to_dict(theta)

    # CONDITION INFOS
    condition_df = deepcopy(importer.condition_df).reset_index()
    condition_as_dict = condition_df[condition_df['conditionId'] == condition].iloc[0].to_dict()
 
    # SET THE PARAMETERS THAT ARE NOT CONDITION SPECIFIC   
    for parameter_id in amici_model.getParameterIds():
        if parameter_id not in condition_df.keys():
            value = theta_as_dict[parameter_id]
            amici_model.setParameterById(parameter_id, value)

    # SET THE PARAMETERS THAT ARE CONDITION SPECIFIC
    for parameter_id, value in condition_as_dict.items():
        if parameter_id not in ["conditionId", "conditionName"]:
            # check if the value is a string => parameter id...
            try:
                value = float(value)
            except ValueError:
                value = theta_as_dict[value]
            amici_model.setParameterById(parameter_id, value)
    
    return amici_model

def simulate_amici_models(
    amici_models: List[amici.Model],
    theta: np.array
):
    """
    Simulate the AMICI models over a given time span.

    Args:
        amici_models: list
            A list of AMICI models to simulate.
        theta: np.array
            The parameters to use for the simulation.

    Returns:
        list
            A list of simulation results (rdata) for each model.
    """
    rdatas = []
    
    # simulate for individual conditions
    for amici_model, condition in zip(amici_models, CONDITIONS):
        # Set parameters
        amici_model = set_parameters_for_condition(condition, theta, amici_model)

        # every z_k shares the same global integration window [T_0, T_MAX]
        amici_model.setTimepoints([T_0, T_MAX])

        # get solver and simulate
        solver = amici_model.getSolver()
        solver.setSensitivityOrder(1)
        solver.setMaxSteps(50_000)
        rdata = amici.runAmiciSimulation(amici_model, solver)
        rdatas.append(rdata)

    return rdatas

def get_z_values_from_rdatas(
    rdatas: List,
    amici_models: List[amici.Model]
):
    """
    Extract the z-values from the simulation results.

    Args:
        rdata: The simulation results.

    Returns:
    z_value_dict: A dictionary with values for every z_i.
         
         obs_id: observable_name
         condition_id: 
         t_true: t according to PEtab...
         t_start: t_start
         t_end: t_end

         z_start: z(t_start)
         z_end: z(t_end)

         grad_z_start: d/d theta z(t_start)
         grad_z_end: d/d theta z(t_end)
    """
    z_value_dict = dict()
    for i in range(77):
        # measurement as dict 
        measurement_dict = importer.measurement_df.iloc[i].to_dict()
        z_value_dict[f"z_{i}"] = {
            "obs_id": measurement_dict["observableId"][1:],
            "condition_id": measurement_dict["simulationConditionId"],
            "t_true": measurement_dict["time"],
            "t_start": T_0,
            "t_end": T_MAX,
            "sigma_x": measurement_dict["noiseParameters"],
            "z_start": None,
            "z_end": None,
            "grad_z_start": None,
            "grad_z_end": None
        }

    for rdata, amici_model in zip(rdatas, amici_models):
        for i_state, state_id in enumerate(amici_model.getObservableIds()):
            # filter out the non-z states of the simulation...
            state_id = state_id[1:]
            if state_id not in z_value_dict:
                # print(f"Warning: {state_id} not found in z_value_dict")
                continue

            # get the time points, that we need to extract the z-states for
            t_start = z_value_dict[state_id]["t_start"]
            i_t_start = int(np.where(rdata.ts == t_start)[0][0])

            t_end = z_value_dict[state_id]["t_end"]
            i_t_end = int(np.where(rdata.ts == t_end)[0][0])

            z_value_dict[state_id]["z_start"] = rdata.y[i_t_start, i_state]
            z_value_dict[state_id]["z_end"] = rdata.y[i_t_end, i_state]
            z_value_dict[state_id]["grad_z_start"] = rdata.sy[i_t_start, :, i_state]
            z_value_dict[state_id]["grad_z_end"] = rdata.sy[i_t_end, :, i_state]

    return z_value_dict


def _amici_parametrization_to_petab_parametrization_for_gradient(
    grad_z_amici: np.ndarray,
    condition: str
    ) -> np.ndarray:
    """
    Convert AMICI parameterization to PEtab parameterization for gradient computation.

    This is necessary, as the AMICI parametrization corresponds to the SBML, 
    which aslo encodes the mapping of condition specific parameters, which are 
    not fitted, but assigned as in the condition DF.


    Args:
    grad_z_amici: gradient of the observable w.r.t SBML=AMICI Parameters
    condition: str a condition name, that tells, how the condition specific 
        parameters are set (e.g. the k<b1, b2, c1, c2, c4, 5>_multiplier parameters)    

    Returns:
    grad_petab: gradient of the observable w.r.t PEtab parameters (i.e. (14, )-dimensional)
    """
    grad_z_petab = np.full((14, ), 0.0)

    # init_b10_1 is only used in model1_data1
    if condition == "model1_data1":
        grad_z_petab[0] = grad_z_amici[0]
    # init_bcar1 is only used in model1_data2
    if condition == "model1_data2":
        grad_z_petab[1] = grad_z_amici[1]
    # init_bcar2 is only used in model1_data3
    if condition == "model1_data3":
        grad_z_petab[2] = grad_z_amici[1]
    # init_bcry_1 is only used in model1_data4
    if condition == "model1_data4":
        grad_z_petab[3] = grad_z_amici[2]
    # init_ohb10_1 is only used in model1_data5
    if condition == "model1_data5":
        grad_z_petab[4] = grad_z_amici[3]
    # init_zea_1 is only used in model1_data6
    if condition == "model1_data6":
        grad_z_petab[5] = grad_z_amici[4]

    # k5, kb1, kb2, kc1, kc2, kc4
    grad_z_petab[6:12] = grad_z_amici[5:11]

    # szea
    if condition == "model1_data3":
        # in the condition, all the multipliers are set to szea
        grad_z_petab[12] = np.sum(grad_z_amici[11:-1])
    elif condition == "model1_data6":
        # only the kc5 & kc4 multiplier are set to szea
        grad_z_petab[12] = grad_z_amici[11] + grad_z_amici[16]

    # parameter w.r.t. sigma_tau
    grad_z_petab[-1] = grad_z_amici[-1]

    return grad_z_petab


def neg_log_post_multi_tau_marginalized(
    theta: np.array,
    amici_models: List[amici.Model],
    ):
    """
    Compute the negative log posterior for the multi-tau model, marginalized over the tau_i.

    Args:
        theta: np.array
            The parameters to use for the simulation.
        amici_models: list
            A list of AMICI models to simulate.

    Returns:
        float
            The negative log posterior.
        np.array
            The gradient of the negative log posterior.
    """
    sigma_tau = theta[-1]

    # simulate the models
    rdatas = simulate_amici_models(amici_models, theta)

    # extract z values from the simulation results
    z_values_dict = get_z_values_from_rdatas(rdatas, amici_models)

    # compute the negative log posterior
    neg_log_post = 0
    grad_neg_log_post = np.zeros(theta.shape)

    # See the doc string of _create_z_states
    # for details on how to compute the z states and their gradients.
    """
    In the end we consider

        (1/(2*pi*sigma_x * sigma_tau)) * (z(t_end) - z(t_start))
    """
    for z_id in z_values_dict:
        z_value_dict = z_values_dict[z_id]

        sigma_x = z_value_dict["sigma_x"]

        z_start = z_value_dict["z_start"]
        z_end = z_value_dict["z_end"]

        z = 1/(2*np.pi*sigma_x * sigma_tau) * (z_end - z_start)

        grad_z_start = _amici_parametrization_to_petab_parametrization_for_gradient(
            z_value_dict["grad_z_start"], 
            z_value_dict["condition_id"]
            )
        grad_z_end = _amici_parametrization_to_petab_parametrization_for_gradient(
            z_value_dict["grad_z_end"], 
            z_value_dict["condition_id"]
            )

        grad_z = np.full(theta.shape, np.nan)
        grad_z = 1/(2*np.pi*sigma_x * sigma_tau) * (grad_z_end - grad_z_start)
        # for sigma_tau, the gradient of (1/...)*z needs some "u'v + uv'"
        # Here the uv' part is given above, while the derivative of (1/...) w.r.t. sigma_tau
        # remains to be added...
        grad_z[-1] += 1/(2*np.pi*sigma_x) * (z_end - z_start) * (-1/(sigma_tau**2))
        if z <= 0:
            return np.inf, np.full(theta.shape, np.nan)
        
        neg_log_post += -np.log(z) 

        grad_neg_log_post += -1/z * grad_z

    # normalize the Gaussian prior on tau to the SAME window [T_0, T_MAX] the
    # z states are integrated over (truncated normal, paper Section 2.4):
    # -log p(tau) gains + log(Phi(b) - Phi(a)), a = (T_0-t)/sigma_tau, b = (T_MAX-t)/sigma_tau
    if TRUNCATED_GAUSSIAN:
        for z_id in z_values_dict:
            t_true = z_values_dict[z_id]["t_true"]
            a = (T_0 - t_true) / sigma_tau
            b = (T_MAX - t_true) / sigma_tau
            prior_mass = norm.cdf(b) - norm.cdf(a)
            neg_log_post += np.log(prior_mass)
            # d/dsigma_tau log(Phi(b) - Phi(a)) = (a*phi(a) - b*phi(b)) / (sigma_tau * prior_mass)
            grad_neg_log_post[-1] += (a * norm.pdf(a) - b * norm.pdf(b)) / (sigma_tau * prior_mass)
 

    # undo the chain rule for lag-transformed parameters for the gradient:
    grad_neg_log_post[:13] *= 10 ** theta[:13] * np.log(10)

    return neg_log_post, grad_neg_log_post


"""
Now, the pyPESTO Fun begins...
"""

import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.visualize as visualize
import pypesto.store as store
from pypesto.petab import PetabImporter


N_STARTS=1_000

LB_MARGINALIZED = np.concatenate(
    [np.log10(importer.get_lb()), [0]])
UB_MARGINALIZED = np.concatenate(
    [np.log10(importer.get_ub()), [25]])
"""
LB_MARGINALIZED_TIGHT = np.concatenate(
    [np.log10(importer.get_x_nominal())-2, [0]])
UB_MARGINALIZED_TIGHT = np.concatenate(
    [np.log10(importer.get_x_nominal())+3, [30]])
"""


import threading
from pypesto.engine import MultiThreadEngine

_thread_local = threading.local()

def _get_thread_local_models():
    """
    Returns a thread-local list of AMICI models (one per condition).
    Each thread gets its own copies to avoid race conditions.
    """
    if not hasattr(_thread_local, 'amici_models'):
        _thread_local.amici_models = []
        for condition in CONDITIONS:
            model_name = f"Bruno_JExpBot2016_marginalized_multi_{model_to_run}_{condition}"
            model_dir = str(APP_DIR / "amici" / "marginalized_multi_tau" / model_to_run / f"amici_{condition}")
            model_module = amici.import_model_module(model_name, model_dir)
            _thread_local.amici_models.append(model_module.getModel())
    
    return _thread_local.amici_models

def obj_fun_marginalized_multi_tau(theta):
    return neg_log_post_multi_tau_marginalized(theta, _get_thread_local_models())


"""
def obj_fun_marginalized_multi_tau(theta):
    return neg_log_post_multi_tau_marginalized(theta, amici_models)
"""



objective_marginalized_multi_tau = pypesto.Objective(
    fun=obj_fun_marginalized_multi_tau,
    grad=True,
)

problem_marginalized_multi_tau = pypesto.Problem(
        objective=objective_marginalized_multi_tau,
        lb=LB_MARGINALIZED,
        ub=UB_MARGINALIZED
        )

# gradient check
if INCLUDE_GRAD_CHECK:
    theta_true_marginal = np.concatenate([np.log10(importer.parameter_df['nominalValue'].values), [SIGMA_TAU]])
    print(obj_fun_marginalized_multi_tau(theta_true_marginal))
    problem_marginalized_multi_tau.objective.check_grad(theta_true_marginal)
    # problem_marginalized_multi_tau.objective.check_grad(theta_true_marginal + np.random.randn(*theta_true_marginal.shape))

import fides

fides_optimizer = optimize.FidesOptimizer(verbose=0, hessian_update=fides.hessian_approximation.BFGS())
fides_optimizer.options["maxiter"]=2_500

# Under SLURM, os.cpu_count() reports the NODE's total cores, not this
# array task's allocation -- SLURM_CPUS_PER_TASK reflects what --cpus-per-task
# actually reserved for this task. Falls back to os.cpu_count() for local runs
# (no SLURM_CPUS_PER_TASK set).
N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
engine = MultiThreadEngine(n_threads=N_THREADS)

# your optimizer stays the same
result_marginalized_multi_tau = optimize.minimize(
    problem_marginalized_multi_tau,
    optimizer=fides_optimizer,
    engine=engine,
    n_starts=N_STARTS
)

# profile likelihoods
import pypesto.profile as profile

if INCLUDE_PROFILES:

    # this leads to exploring larger areas
    profile_options = profile.ProfileOptions(ratio_min=0.01)
    result_marginalized_multi_tau = profile.parameter_profile(
        problem=problem_marginalized_multi_tau,
        result=result_marginalized_multi_tau,
        optimizer=fides_optimizer,
        # profile_index=[13],
        profile_options=profile_options
    )
    visualize.profiles(result_marginalized_multi_tau, show_bounds=True)

# sampling
if INCLUDE_SAMPLING:
    N_SAMPLES = 200_000
    sample.sample(problem_marginalized_multi_tau, n_samples=N_SAMPLES, result=result_marginalized_multi_tau)


# store the result
result_file_name = f"{result_folder}/{model_to_run}.hdf5"
store.write_result(result_marginalized_multi_tau, result_file_name, overwrite=True)

for i in range(13): 
    print(f"Paramater {i}")

    if INCLUDE_SAMPLING:
        print(f"Min: {10**np.min(result_marginalized_multi_tau.sample_result.trace_x[0, :, i])}")
        print(f"theta_opt: {10**result_marginalized_multi_tau.optimize_result[0].x[i]}")
        print(f"theta_pub: {importer.parameter_df['nominalValue'].values[i]}")
        print(f"Max: {10**np.max(result_marginalized_multi_tau.sample_result.trace_x[0, :, i])}")
    else:
        print(f"theta_opt: {10**result_marginalized_multi_tau.optimize_result[0].x[i]}")
        print(f"theta_pub: {importer.parameter_df['nominalValue'].values[i]}")

    print("---------------")

print(f"Paramater {13} (sigma_tau)")
if INCLUDE_SAMPLING:
    print(f"Min: {result_marginalized_multi_tau.sample_result.trace_x[0, :, -1].min()}")
    print(f"theta_opt: {result_marginalized_multi_tau.optimize_result[0].x[-1]}")
    print(f"theta_true: {SIGMA_TAU}")
    print(f"Max: {result_marginalized_multi_tau.sample_result.trace_x[0, :, -1].max()}")
else:
    print(f"theta_opt: {result_marginalized_multi_tau.optimize_result[0].x[-1]}")
    print(f"theta_true: {SIGMA_TAU}")

visualize.waterfall(result_marginalized_multi_tau)