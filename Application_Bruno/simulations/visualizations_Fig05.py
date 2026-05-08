"""
Figure 5B: 
Waterfall-plots for the different objective functions and on the true data.
"""
import pypesto.visualize as visualize
import pypesto.store as store
import matplotlib.pyplot as plt
from pypesto.petab import PetabImporter
import pandas as pd
import numpy as np
import amici
from scipy.stats import chi2

RESULTS_DIR = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/results"
FIGURE_STORE_DIR = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Figures Paper/building_blocks"


COLOR_STANDARD     = [144/255, 144/255, 133/255, 1]  # grey
# COLOR_MARGINALIZED = [252/255, 186/255, 0] # orange
COLOR_MARGINALIZED = [243/255, 118/255, 40/255, 1] # orange
COLOR_JOINT        = [0, 78/255, 159/255, 1]          # blue

FONT_SIZE_TICKS = 10
FONT_SIZE_AXES  = 14
FONT_SIZE_TITLE = 16

LINE_WIDTH = 3
CAP_SIZE   = 6
CAP_THICK  = LINE_WIDTH

DATAFORMAT = "svg"

result_wrong_tp = store.read_result(f"{RESULTS_DIR}/wrong_tp/copied_data_multi_timepoints_1_sigma_tau_0.hdf5")
result_full_multi_tau = store.read_result(f"{RESULTS_DIR}/full_multi_tau/copied_data_multi_timepoints_1_sigma_tau_0.hdf5")
result_marginalized_multi_tau = store.read_result(f"{RESULTS_DIR}/marginalized_multi_tau/copied_data_multi_timepoints_1_sigma_tau_0.hdf5")

result_full_multi_tau.problem.x_names = result_wrong_tp.problem.x_names + ["tau_{i}" for i in range(77)] + ["sigma_tau"]
result_marginalized_multi_tau.problem.x_names = result_wrong_tp.problem.x_names  + ["sigma_tau"]

# DELTA_OBJECTIVE = 10 # only plot up to 10 units away from the best objective function...
DELTA_OBJECTIVE = chi2.ppf(1 - 0.95, df=13)


def get_multi_tau_sigma_from_theta(
    theta,
    petab_dir: str
    ):
    """
    petab_dir: the folder, where the petab is stored. We need this for T_PETAB
    """
    T_PETAB = pd.read_csv(f"{petab_dir}/measurementData_Bruno_JExpBot2016.tsv", sep="\t")["time"].values
    taus = theta[13:]
    return np.sqrt(np.sum((taus-T_PETAB)**2)/len(taus))


# read in tsv data for the reference point
parameter_df = pd.read_csv("/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/PEtab_versions_backup/sigma_tau_0_0/parameters_Bruno_JExpBot2016.tsv", sep="\t")
measurement_df = pd.read_csv("/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/PEtab_versions_backup/sigma_tau_0_0/measurementData_Bruno_JExpBot2016.tsv", sep="\t")

reference_point = visualize.ReferencePoint(
    x=np.log10(parameter_df["nominalValue"].values),
    fval=float("nan"), 
    color=[0, 0, 0, 1]
    )

"""
Figure 5 B+C: Waterfall plots (left, stacked) + Runtime bar chart (right, y-axis aligned)
"""

def _finish_up_plotting(ax):
    ax.plot([-1, 100], [DELTA_OBJECTIVE, DELTA_OBJECTIVE], color="black", linestyle='--', label="Convergence", linewidth=LINE_WIDTH)
    ax.set_xlim(-1, 101)
    ax.set_xlabel("ordered optimizer\nruns", fontsize=FONT_SIZE_AXES)
    # ax.set_ylabel("sub-optimality + 1", fontsize=FONT_SIZE_AXES)
    ax.set_title(None)
    ax.legend(fontsize=FONT_SIZE_AXES, frameon=False)
    

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylim((0.5, 7_000))

from matplotlib.gridspec import GridSpec

# Heights of the two waterfall plots
waterfall_h = 2.3  # inches each
gap_h = 0.3        # gap between waterfalls
bar_w = 3.5        # width of bar chart column
waterfall_w = 4.5  # width of waterfall column

fig_w = waterfall_w + bar_w
fig_h = 2 * waterfall_h + gap_h

fig = plt.figure(figsize=(fig_w, fig_h))

# GridSpec: 2 rows (for waterfalls) × 2 cols (waterfall | bar)
# The bar chart spans both rows
gs = GridSpec(
    2, 2,
    figure=fig,
    width_ratios=[waterfall_w, bar_w],
    height_ratios=[waterfall_h, waterfall_h],
    hspace=gap_h / (2 * waterfall_h),  # relative gap
    wspace=0.4
)

ax_wf1 = fig.add_subplot(gs[0, 0])  # top waterfall
ax_wf2 = fig.add_subplot(gs[1, 0])  # bottom waterfall
ax_bar = fig.add_subplot(gs[:, 1])  # bar chart spans both rows

# ── Waterfall top: Standard + Marginalized ───────────────────────────────────
visualize.waterfall(
    [result_wrong_tp, result_marginalized_multi_tau],
    ax=ax_wf1,
    colors=[COLOR_STANDARD, COLOR_MARGINALIZED],
    legends=["Standard", "Marginalized"],
    start_indices=100
)
_finish_up_plotting(ax_wf1)
ax_wf1.set_title(None)
ax_wf1.set_xticklabels([])
ax_wf1.set_xlabel("")
ax_wf1.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)

# ── Waterfall bottom: Joint ──────────────────────────────────────────────────
visualize.waterfall(
    [result_full_multi_tau],
    ax=ax_wf2,
    colors=[COLOR_JOINT],
    legends=["Joint"],
    start_indices=100
)
_finish_up_plotting(ax_wf2)
ax_wf2.set_title(None)
ax_wf2.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)

ax_wf1.set_ylabel("")
ax_wf2.set_ylabel("")

# Add shared y label centered between the two waterfall axes
fig.text(
    0.04,        # x position (adjust leftward if clipped)
    0.5,        # y center of the figure
    "sub-optimality + 1",  # replace with your actual label
    va='center',
    ha='center',
    rotation='vertical',
    fontsize=FONT_SIZE_AXES
)

# ── Bar chart: Runtime per converged start ───────────────────────────────────
number_wrong_tp = int(np.sum(
    result_wrong_tp.optimize_result.fval - result_wrong_tp.optimize_result.fval[0] < DELTA_OBJECTIVE))
number_full_multi_tau = int(np.sum(
    result_full_multi_tau.optimize_result.fval - result_full_multi_tau.optimize_result.fval[0] < DELTA_OBJECTIVE))
number_marginalized_multi_tau = int(np.sum(
    result_marginalized_multi_tau.optimize_result.fval - result_marginalized_multi_tau.optimize_result.fval[0] < DELTA_OBJECTIVE))

run_time_wrong_tp = np.sum(result_wrong_tp.optimize_result.time)
run_time_full_multi_tau = np.sum(result_full_multi_tau.optimize_result.time)
run_time_marginalized_multi_tau = np.sum(result_marginalized_multi_tau.optimize_result.time)

ax_bar.bar(
    ["Standard", "Joint", "Marginalized"],
    [run_time_wrong_tp / number_wrong_tp,
     run_time_full_multi_tau / number_full_multi_tau,
     run_time_marginalized_multi_tau / number_marginalized_multi_tau],
    color=[COLOR_STANDARD, COLOR_JOINT, COLOR_MARGINALIZED]
)
ax_bar.set_yscale("log")
ax_bar.set_ylabel("runtime per converged start [s]", fontsize=FONT_SIZE_AXES)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['left'].set_linewidth(LINE_WIDTH)
ax_bar.spines['bottom'].set_linewidth(LINE_WIDTH)

ax_bar.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)
ax_bar.set_xticklabels(["Standard", "Joint", "Marginalized"], rotation=45, ha="right", fontsize=FONT_SIZE_TICKS)

fig.savefig(f"{FIGURE_STORE_DIR}/figure_5_bc.{DATAFORMAT}", bbox_inches="tight")
plt.show()


"""
Figure 5E: Parallel coordinate plot
"""
x_wrong_tp = result_wrong_tp.optimize_result.x[0]
x_full_multi_tau = result_full_multi_tau.optimize_result.x[0]
x_marginalized_multi_tau = result_marginalized_multi_tau.optimize_result.x[0]

fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=(9, 2), gridspec_kw={'width_ratios': [7, 2]})

ax.plot(x_wrong_tp,                    color=COLOR_STANDARD,     label="Standard",     linewidth=LINE_WIDTH, alpha=0.5, linestyle="--")
ax.plot(x_full_multi_tau[:13],         color=COLOR_JOINT,        label="Joint",        linewidth=LINE_WIDTH, alpha=0.5, linestyle="--")
ax.plot(x_marginalized_multi_tau[:13], color=COLOR_MARGINALIZED, label="Marginalized", linewidth=LINE_WIDTH, alpha=0.5, linestyle="--")

ax.plot([0, 12], [-5, -5], color="black", linestyle="--", linewidth=LINE_WIDTH)
ax.plot([0, 12], [3,  3],  color="black", linestyle="--", linewidth=LINE_WIDTH)

ax.set_ylim(-5.2, 3.2)
ax.set_xlim(0, 12)
ax.set_xticks(range(13))
ax.set_xticklabels(result_wrong_tp.problem.x_names[:13], rotation=75, ha="right", fontsize=FONT_SIZE_TICKS)
ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(LINE_WIDTH)
ax.spines['bottom'].set_linewidth(LINE_WIDTH)
ax.set_ylabel("Log10(Param)", fontsize=FONT_SIZE_AXES)

# Legend-only axis
handles, labels = ax.get_legend_handles_labels()
ax_leg.legend(handles, labels, fontsize=FONT_SIZE_AXES, frameon=False, loc='center left', ncols=1)
ax_leg.axis('off')

fig.savefig(f"{FIGURE_STORE_DIR}/figure_5_e.{DATAFORMAT}", bbox_inches="tight")
plt.show()


"""
Figure 5D: Trajectories of the different approaches
"""
theta_standard_lin = 10 ** result_wrong_tp.optimize_result.x[0][:13]
theta_joint_lin = 10 ** result_full_multi_tau.optimize_result.x[0][:13]
theta_marginalized_lin = 10 ** result_marginalized_multi_tau.optimize_result.x[0][:13]

import pypesto.visualize as visualize
import pypesto.store as store
import matplotlib.pyplot as plt
from pypesto.petab import PetabImporter
import pandas as pd
import numpy as np
import amici
from copy import deepcopy
from typing import List

COLOR_STANDARD     = [144/255, 144/255, 133/255, 1]  # grey
COLOR_MARGINALIZED = [252/255, 186/255, 0, 1]        # orange
COLOR_JOINT        = [0, 78/255, 159/255, 1]          # blue
LINE_WIDTH = 3



import os

model_name = "ModelFullMultiTau"
model_dir = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/amici"

# Only compile if not already compiled
if not os.path.exists(os.path.join(model_dir, model_name)):
    sbml_importer = amici.SbmlImporter("/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/petab_bruno_original/model_Bruno_JExpBot2016.xml")
    sbml_importer.sbml2amici(model_name, model_dir)

model_module = amici.import_model_module(model_name, model_dir)
amici_model = model_module.getModel()

importer = PetabImporter.from_yaml("/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/petab_bruno_original/Bruno_JExpBot2016.yaml")
problem_wrong_tp = importer.create_problem(verbose=False)


# CONSTANTS
TRUNCATED_GAUSSIAN = False
HIERARCHICAL_SIGMA_TAU = not TRUNCATED_GAUSSIAN
T_PETAB = importer.petab_problem.measurement_df['time'].values

conditions_as_dict = dict()
condition_df = deepcopy(importer.petab_problem.condition_df).reset_index()
    
for condition in ["model1_data1", "model1_data2", "model1_data3", "model1_data4", "model1_data5", "model1_data6"]:
    condition_as_dict = condition_df[condition_df["conditionId"] == condition].iloc[0].to_dict()
    conditions_as_dict[condition] = condition_as_dict

condition_df_keys = condition_df.keys()
PARAMETER_IDS = importer.petab_problem.get_x_ids()

tau_df = pd.read_csv(
        "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/full_parametrization_df.tsv", 
        sep="\t"
        ).reset_index()

def _sigma_tau_for_full_modell(
    theta_extended: np.array, 
    log_trafo: bool = True,
    hierarchical_sigma_tau: bool = HIERARCHICAL_SIGMA_TAU
    ):
    """
    unpack the thetas.

    Not that by default, the dynamic parameters (i.e. theta_estimate) are log10 transformed.
    => returns 10**theta_estimate

    Maps theta_extended to 
        - theta_estimate (the original model parameters, see petab parameter df for the mapping there)
        - tau: the time point parameters (the i-th entry is for the i-th data point as given by the petab measurement df)
        - sigma_tau: the uncertainty w.r.t time points
    """
    if hierarchical_sigma_tau:
        if TRUNCATED_GAUSSIAN:
            raise NotImplementedError("Hierarchical sigma tau not implemented for truncated Gaussian yet.")
        taus = theta_extended[13:]
        sigma_tau = np.sqrt(np.sum((taus-T_PETAB)**2)/len(taus))
    else:
        sigma_tau = theta_extended[-1]
    return sigma_tau

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
    - "theta_extended_index": The index of tau, as in theta_extended.
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
    theta_estimate = 10 ** theta_extended[:13]
    taus = np.array(T_PETAB, dtype=float)

    amici_model = set_parameters_for_condition(condition, theta_estimate, amici_model)

    # Get all the relevant infos from a slightly extended Parameter df
    df_for_condition = tau_df[
        tau_df["simulationConditionId"] == condition]

    # get the values from the condition df
    data_indices = df_for_condition["tau_number"].to_list()  # tau index of the data
    
    # get values, that will be part of the return dict:
    taus = taus[data_indices].reshape(-1)      # tau values for the condition
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
            }
        )

    return return_list


def visualize_for_condition(
    condition: str, 
    amici_model: amici.Model,
    thetas: List[np.ndarray],
    plot_errorbars: bool = False,
    save_in_dir: str = None,
    sigma_tau: float = 0.0
):
    # Pre-simulate all thetas first
    simulations = []
    for theta, label, color in zip(
        thetas, 
        ["standard", "joint", "marginalized"], 
        [COLOR_STANDARD, COLOR_JOINT, COLOR_MARGINALIZED]):

        theta_estimate = 10 ** theta[:13]
        amici_model = set_parameters_for_condition(condition, theta_estimate, amici_model)
        dp_dicts = _simulate_single_data_point_dicts_for_condition(condition, theta, amici_model)
        
        amici_model.setTimepoints(list(range(185)))
        solver = amici_model.getSolver()
        rdata = amici.runAmiciSimulation(amici_model, solver)

        simulations.append((theta, label, color, rdata, dp_dicts))

    # One figure per observable
    df_for_condition = tau_df[tau_df["simulationConditionId"] == condition]
    observables_for_condition = set(df_for_condition["observableId"])

    for observable in observables_for_condition:
        fig, ax = plt.subplots(figsize=(3.500, 2.325))
        data_plotted = False

        for theta, label, color, rdata, dp_dicts in simulations:

            if observable == "ybcar":
                x_trajectors = rdata.x[:, 0]
            elif observable == "ybcry":
                x_trajectors = rdata.x[:, 1]
            elif observable == "yb10":
                x_trajectors = rdata.x[:, 2]
            elif observable == "yohb10":
                x_trajectors = rdata.x[:, 4]
            elif observable == "yzea":
                x_trajectors = rdata.x[:, 6]

            ax.plot(rdata.t, x_trajectors, alpha=0.7, label=label, color=color, linewidth=LINE_WIDTH)

        # Plot data last (so it appears last in legend)
        _, _, _, _, dp_dicts = simulations[0]  # data is the same for all thetas
        for dp in dp_dicts:
            if dp["observable_name"] == observable:
                if plot_errorbars:
                    ax.errorbar(
                        x=dp["t"], y=dp["data"],
                        yerr=dp["sigma_y"], xerr=sigma_tau,
                        fmt='o', color="black", ecolor="gray",
                        elinewidth=1, capsize=3)
                else:
                    ax.scatter(x=dp["t"], y=dp["data"], color="black")

        ax.set_xlabel("Time", fontsize=FONT_SIZE_AXES)
        ax.set_ylabel(observable, fontsize=FONT_SIZE_AXES)
        ax.set_title(f"{observable} - {condition}", fontsize=FONT_SIZE_TITLE)
        # ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(LINE_WIDTH)
        ax.spines['bottom'].set_linewidth(LINE_WIDTH)
        ax.set_xlim((0, 185))
        ax.set_ylim((0, np.max(x_trajectors) * 1.2))

        if save_in_dir is not None:
            plt.savefig(f"{save_in_dir}/{condition}_{observable}.png", bbox_inches="tight")
        plt.show()
        plt.close()

def save_legend_figure(
    thetas: List[np.ndarray],
    save_in_dir: str = None
):
    """Standalone legend figure — no axes, just the legend handles."""
    labels = ["standard", "joint", "marginalized"]
    colors = [COLOR_STANDARD, COLOR_JOINT, COLOR_MARGINALIZED]

    fig, ax = plt.subplots(figsize=(3, 1))
    ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=color, linewidth=LINE_WIDTH, alpha=0.7, label=label)
        for label, color in zip(labels, colors)
    ]
    # Add data point handle to match the scatter in the plots
    handles.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=5, label='data')
    )

    ax.legend(handles=handles, loc="center", frameon=False, ncol=1, fontsize=FONT_SIZE_AXES)

    if save_in_dir is not None:
        plt.savefig(f"{save_in_dir}/legend.png", bbox_inches="tight")
    plt.show()
    plt.close()

"""
Figure 5D in one run:
"""
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Layout definition: (condition, observables_in_order)
CONDITION_OBSERVABLES = {
    "model1_data1": ["yb10"],
    "model1_data2": ["yb10", "ybcar"],
    "model1_data3": ["yb10", "ybcar"],   # adjust to actual observables
    "model1_data4": ["yb10", "ybcry", "yohb10"],
    "model1_data5": ["yohb10"],
    "model1_data6": ["yohb10", "yzea"],
}

CONDITION_LABELS = {
    "model1_data1": "Data Set 1",
    "model1_data2": "Data Set 2",
    "model1_data3": "Data Set 3",
    "model1_data4": "Data Set 4",
    "model1_data5": "Data Set 5",
    "model1_data6": "Data Set 6",
}

# Column layout: each column is a list of (condition, [observables]) groups
# Matches the visual layout in the target image
COLUMN_LAYOUT = [
    ["model1_data1", "model1_data2"],   # col 0
    ["model1_data3"],                    # col 1  (+ legend at bottom)
    ["model1_data4"],                    # col 2
    ["model1_data5", "model1_data6"],   # col 3
]

def _simulate_for_assembly(condition, theta, amici_model):
    """Simulate trajectory for a single condition/theta, returns (rdata, dp_dicts)."""
    theta_estimate = 10 ** theta[:13]
    amici_model = set_parameters_for_condition(condition, theta_estimate, amici_model)
    dp_dicts = _simulate_single_data_point_dicts_for_condition(condition, theta, amici_model)
    amici_model.setTimepoints(list(range(185)))
    solver = amici_model.getSolver()
    rdata = amici.runAmiciSimulation(amici_model, solver)
    return rdata, dp_dicts


def _get_trajectory(rdata, observable):
    obs_map = {
        "ybcar":  0,
        "ybcry":  1,
        "yb10":   2,
        "yohb10": 4,
        "yzea":   6,
    }
    return rdata.x[:, obs_map[observable]]


def assemble_figure(
    thetas,
    amici_model,
    condition_observables=CONDITION_OBSERVABLES,
    column_layout=COLUMN_LAYOUT,
    save_path=None,
    plot_errorbars=False,
    sigma_tau=0.0,
):
    """
    Assemble all trajectory subplots into a single structured figure,
    with boxes drawn around each dataset group and axes vertically aligned.

    Parameters
    ----------
    thetas : list of np.ndarray
        [theta_standard, theta_joint, theta_marginalized]
    amici_model : amici.Model
    condition_observables : dict  condition -> list of observable ids (in display order)
    column_layout : list of lists   which conditions go in which column
    save_path : str or None
    """
    labels = ["Standard Approach", "Joint Approach", "Marginalized Approach"]
    colors = [COLOR_STANDARD, COLOR_JOINT, COLOR_MARGINALIZED]

    # ------------------------------------------------------------------
    # 1.  Pre-simulate everything
    # ------------------------------------------------------------------
    sim_cache = {}   # condition -> list of (label, color, rdata, dp_dicts)
    all_conditions = [c for col in column_layout for c in col]
    for condition in all_conditions:
        entries = []
        for theta, label, color in zip(thetas, labels, colors):
            rdata, dp_dicts = _simulate_for_assembly(condition, theta, amici_model)
            entries.append((label, color, rdata, dp_dicts))
        sim_cache[condition] = entries

    # ------------------------------------------------------------------
    # 2.  Compute figure dimensions
    #     max_rows = tallest column (in number of observable subplots)
    #     +1 for the legend slot in col 1
    # ------------------------------------------------------------------
    n_cols = len(column_layout)
    col_heights = []
    for col_conditions in column_layout:
        h = sum(len(condition_observables[c]) for c in col_conditions)
        col_heights.append(h)

    # legend occupies 1 row-slot at the bottom of col 1
    col_heights[1] += 1
    max_rows = max(col_heights)

    subplot_w = 3.5
    subplot_h = 2.325
    fig_w = n_cols * subplot_w + 0.6 * (n_cols - 1)   # gaps between cols
    fig_h = max_rows * subplot_h + 0.4 * (max_rows - 1)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # ------------------------------------------------------------------
    # 3.  Build a uniform GridSpec (max_rows x n_cols)
    # ------------------------------------------------------------------
    gs = gridspec.GridSpec(
        max_rows, n_cols,
        figure=fig,
        hspace=0.55,
        wspace=0.45,
    )

    # ------------------------------------------------------------------
    # 4.  Fill axes column by column
    # ------------------------------------------------------------------
    all_axes = {}   # (condition, observable) -> ax

    for col_idx, col_conditions in enumerate(column_layout):
        row_cursor = 0
        for condition in col_conditions:
            observables = condition_observables[condition]
            for obs in observables:
                ax = fig.add_subplot(gs[row_cursor, col_idx])
                all_axes[(condition, obs)] = ax
                row_cursor += 1

        # legend placeholder in col 1
        if col_idx == 1:
            ax_legend = fig.add_subplot(gs[row_cursor: row_cursor + 1, col_idx])
            ax_legend.axis("off")

    # ------------------------------------------------------------------
    # 5.  Draw trajectories + data into each axis
    # ------------------------------------------------------------------
    for col_idx, col_conditions in enumerate(column_layout):
        for condition in col_conditions:
            observables = condition_observables[condition]
            entries = sim_cache[condition]

            for obs in observables:
                ax = all_axes[(condition, obs)]

                for label, color, rdata, dp_dicts in entries:
                    traj = _get_trajectory(rdata, obs)
                    ax.plot(rdata.t, traj,
                            alpha=0.7, color=color, linewidth=LINE_WIDTH, label=label)

                # scatter data (same for all thetas, use first entry)
                _, _, _, dp_dicts_0 = entries[0][0], entries[0][1], entries[0][2], entries[0][3]
                for dp in dp_dicts_0:
                    if dp["observable_name"] == obs:
                        if plot_errorbars:
                            ax.errorbar(dp["t"], dp["data"],
                                        yerr=dp["sigma_y"], xerr=sigma_tau,
                                        fmt='o', color="black", ecolor="gray",
                                        elinewidth=1, capsize=3)
                        else:
                            ax.scatter(dp["t"], dp["data"], color="black", s=20, zorder=5)

                # cosmetics
                ax.set_xlim((0, 185))
                traj_max = _get_trajectory(entries[0][2], obs).max()
                ax.set_ylim((0, traj_max * 1.2))
                ax.set_ylabel(obs[1:], fontsize=FONT_SIZE_AXES)  # remove leading 'y' from observable name
                ax.set_xlabel("Time", fontsize=FONT_SIZE_AXES)
                ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(LINE_WIDTH)
                ax.spines['bottom'].set_linewidth(LINE_WIDTH)
    # 7.  Legend
    # ------------------------------------------------------------------
    handles = [
        plt.Line2D([0], [0], color=c, linewidth=LINE_WIDTH, alpha=0.7, label=l)
        for l, c in zip(labels, colors)
    ]
    handles.append(
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='black', markersize=5, label='Data')
    )
    ax_legend.legend(handles=handles, loc="center", frameon=True,
                     fontsize=FONT_SIZE_AXES, ncol=1)

    # ------------------------------------------------------------------
    # 8.  Save / show
    # ------------------------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    return fig

theta_theta_wrong_tp = result_wrong_tp.optimize_result.x[0]
theta_full_multi_tau = result_full_multi_tau.optimize_result.x[0]
theta_marginalized_multi_tau = result_marginalized_multi_tau.optimize_result.x[0]
thetas = [theta_theta_wrong_tp, theta_full_multi_tau, theta_marginalized_multi_tau]

assemble_figure(
    thetas=thetas,
    amici_model=amici_model,
    save_path=f"{FIGURE_STORE_DIR}/figure_5_d.{DATAFORMAT}",
)


"""
Now, let's actually plot some trajectories
"""
RESULTS_DIR = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/results"
result_wrong_tp = store.read_result(f"{RESULTS_DIR}/wrong_tp/sigma_tau_0_0.hdf5")
result_full_multi_tau = store.read_result(f"{RESULTS_DIR}/full_multi_tau/sigma_tau_0_0.hdf5")
result_marginalized_multi_tau = store.read_result(f"{RESULTS_DIR}/marginalized_multi_tau/sigma_tau_0_0.hdf5")

theta_theta_wrong_tp = result_wrong_tp.optimize_result.x[0]
theta_full_multi_tau = result_full_multi_tau.optimize_result.x[0]
theta_marginalized_multi_tau = result_marginalized_multi_tau.optimize_result.x[0]
thetas = [theta_theta_wrong_tp, theta_full_multi_tau, theta_marginalized_multi_tau]

save_dir = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/figures"

for condition in ["model1_data1", "model1_data2", "model1_data3", "model1_data4", "model1_data5", "model1_data6"]:
    visualize_for_condition(condition=condition, amici_model=amici_model, thetas=thetas, save_in_dir=save_dir)

save_legend_figure(thetas=thetas, save_in_dir=save_dir)



for sigma_tau in [str(i) for i in range(1, 16, 2)]:
    result_wrong_tp = store.read_result(f"{RESULTS_DIR}/wrong_tp/synthetic_data_multi_timepoints_sigma_tau_{sigma_tau}.hdf5", optimize=True)
    result_full_multi_tau = store.read_result(f"{RESULTS_DIR}/full_multi_tau/synthetic_data_multi_timepoints_sigma_tau_{sigma_tau}.hdf5", optimize=True)
    result_marginalized_multi_tau = store.read_result(f"{RESULTS_DIR}/marginalized_multi_tau/synthetic_data_multi_timepoints_sigma_tau_{sigma_tau}.hdf5", optimize=True)

    print("--------------------------------")
    print(np.sum(np.abs(result_full_multi_tau.optimize_result.x[0][13:90] -  measurement_df["time"].values)))
    print(np.linalg.norm(result_wrong_tp.optimize_result.x[0][:13]-reference_point.x, ord=float("inf")))    
    print(np.linalg.norm(result_marginalized_multi_tau.optimize_result.x[0][:13]-reference_point.x, ord=float("inf")))
