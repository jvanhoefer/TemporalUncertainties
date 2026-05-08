import pypesto
import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.visualize as visualize
import matplotlib.pyplot as plt
import pypesto.store as store
import numpy as np
import yaml2sbml
import amici
import pandas as pd
import seaborn as sns

N_EXPERIMENT = 4
# RANDOM_SEED = 0
STORAGE_DIR_BASE = f"/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_ConversionReaction/res_{N_EXPERIMENT}"
STORAGE_FOR_REPEATED_EXPERIMENTS = f"/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_ConversionReaction/"

T_PLOT_MAX = 3.5
CI_VAL = 0.95

FONT_SIZE_TICKS = 14
FONT_SIZE_AXES  = 16
FONT_SIZE_TITLE = 20

LINE_WIDTH = 3
CAP_SIZE   = 6
CAP_THICK  = LINE_WIDTH 

FIGURE_STORE_DIR = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Figures Paper/building_blocks"

DATAFORMAT = "svg"

# Load results
result_wrong_tp = store.read_result(f"{STORAGE_DIR_BASE}/result_wrong_tp.hdf5")
result_full = store.read_result(f"{STORAGE_DIR_BASE}/result_full.hdf5")
result_marginalized = store.read_result(f"{STORAGE_DIR_BASE}/result_marginalized.hdf5")

x_data = np.loadtxt(f'{STORAGE_DIR_BASE}/x_data.csv', delimiter=',')
t_true = np.loadtxt(f'{STORAGE_DIR_BASE}/t_true.csv', delimiter=',')

THETA_TRUE = np.array([0.5, 0, 1, 1], dtype=float)
SIGMA_X = 0.1
SIGMA_TAU = 1

"""
Some Copy Paste from simulation_cr.py
"""
# theta lin is defined in Log10 space
THETA_TRUE = np.array([0.5, 0, 1, 1], dtype=float)
SIGMA_X = 0.1
SIGMA_TAU = 1

# 
# N_TIME = 5
N_TIME = len(t_true)
T_MAX = 8
T_WRONG = np.linspace(0, T_MAX, N_TIME+1, dtype=float)[1:] 

# OBS = ['A', 'B']
OBS = ['B']

T_SIM_END = T_WRONG[-1] + 3 * SIGMA_TAU # the end of the integration for the marginal approach

Z_MIN = 1e-300


yaml_dir = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_ConversionReaction/models/conversion_reaction_base.yaml"
sbml_dir = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_ConversionReaction/models/conversion_reaction_base.xml"

yaml2sbml.yaml2sbml(yaml_dir,sbml_dir)

# generate the AMICI model
model_name_base = "Conversion_Reaction_Base"
amici_model_dir = 'ode_sbml/amici_files'

amici_importer = amici.SbmlImporter(sbml_dir)
amici_importer.sbml2amici(model_name_base, amici_model_dir)

amici_module = amici.import_model_module(model_name_base, amici_model_dir)
amici_model_base = amici_module.getModel()

def _set_parameter_base(amici_model, theta_log: np.array):
    """
    Set the parameters of the AMICI model based on the provided log10-transformed values.
    """
    # Transform the log10 values back to the original scale
    theta = 10 ** theta_log

    # Set the parameters in the AMICI model
    amici_model.setParameterById('k_1', float(theta[0]))
    amici_model.setParameterById('k_2', float(theta[1]))
    amici_model.setParameterById('k_3', float(theta[2]))
    amici_model.setParameterById('k_4', float(theta[3]))

    return amici_model

def simulate_amici_base(
    amici_model_base, 
    theta_log: np.array, 
    t_sim: np.array):
    """
    Simulate the AMICI model for the base case.
    
    Returns:
        - x (for all observables)
        - sx (for all observables)
        - x (for all states!!! (for gradient computation in objective full...))
    
    """
    amici_model_base = _set_parameter_base(amici_model_base, theta_log)

    # sort t_sim, keep track of permutation
    sort_idx = np.argsort(t_sim)
    t_sim_sorted = t_sim[sort_idx]

    amici_model_base.setTimepoints(np.array(t_sim_sorted, dtype=float))

    # Run the simulation
    solver = amici_model_base.getSolver()
    solver.setSensitivityOrder(1)

    rdata = amici.runAmiciSimulation(amici_model_base, solver)
    # Reorder results back to original input order
    x_out = np.empty_like(rdata.x)
    sx_out = np.empty_like(rdata.sx)

    if rdata.x is None:
        return None, None, None
    
    x_out[sort_idx] = rdata.x
    sx_out[sort_idx, :, :] = rdata.sx

    if "A" in OBS and "B" in OBS:
        return x_out, sx_out, x_out
    elif "A" in OBS:
        return x_out[:, 0], sx_out[:, :, 0], x_out
    elif "B" in OBS:
        return x_out[:, 1], sx_out[:, :, 1], x_out
    return None, None, None



DELTA_T_OPT = 2

LB_WRONG_TP = np.array([-1, -1, -1, -1], dtype=float)
UB_WRONG_TP = np.array([1, 1, 2, 2], dtype=float)

# make sure, that LB < THETA TRUE < UB
assert np.all(LB_WRONG_TP < THETA_TRUE)
assert np.all(THETA_TRUE < UB_WRONG_TP)

LB_FULL = np.concatenate([LB_WRONG_TP, np.maximum(T_WRONG - DELTA_T_OPT, 0)], dtype=float)
UB_FULL = np.concatenate([UB_WRONG_TP, T_WRONG + DELTA_T_OPT], dtype=float)

LB_MARGINAL = LB_WRONG_TP
UB_MARGINAL = UB_WRONG_TP

OPTIMIZER = optimize.FidesOptimizer(verbose=0)
# OPTIMIZER = None
SAMPLER = None
STARTPOINT_METHOD = pypesto.startpoint.UniformStartpoints(check_fval=True)

N_STARTS = 100
N_SAMPLES = 500_000


"""
Styling of the plots
"""
COLOR_FULL = [0, 78/255, 159/255] # blue
# COLOR_MARGINALIZED = [252/255, 186/255, 0] # orange
COLOR_MARGINALIZED = [243/255, 118/255, 40/255] # orange
COLOR_WRONG_TP = [144/255, 144/255, 133/255] # grey
COLOR_GROUND_TRUTH = [0, 0, 0] # black

COLORS = [COLOR_WRONG_TP, COLOR_FULL, COLOR_MARGINALIZED]


FONT_SIZE_TICKS = 14
FONT_SIZE_AXES  = 16
FONT_SIZE_TITLE = 20

LINE_WIDTH = 3
CAP_SIZE   = 6
CAP_THICK  = LINE_WIDTH 

"""
Figure 3B: Time courses of data + model simulations for 
different objectives for observable.
Figure 3C: Marginalized posterior distributions for the parameters of 
interest (theta_1, theta_2, theta_3, theta_4) for both approaches.
"""
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
from tqdm import tqdm

result_full = store.read_result(f"{STORAGE_FOR_REPEATED_EXPERIMENTS}res_4/result_full.hdf5")
result_marginalized = store.read_result(f"{STORAGE_FOR_REPEATED_EXPERIMENTS}res_4/result_marginalized.hdf5")

# samples_wrong_tp = result_wrong_tp.sample_result.trace_x[0, :, :4]
samples_full = result_full.sample_result.trace_x[0, :, :4]
samples_marginalized = result_marginalized.sample_result.trace_x[0, :, :4]

parameter_dicts = {
    "Joint Approach": {
        "theta": result_full.optimize_result.x[0],
        "samples": samples_full,
        "color": COLOR_FULL
    },
    "Marginalized Approach": {
        "theta": result_marginalized.optimize_result.x[0],
        "samples": samples_marginalized,
        "color": COLOR_MARGINALIZED
    }
}

H = 0.05
T_VIS = np.array([H * i for i in range(int((T_WRONG[-1] + 2) / H))])

# Pre-compute trajectories (every 10th sample)
for label, val in parameter_dicts.items():
    x_sims = []
    for theta in tqdm(val["samples"][::10], desc=f"Simulating {label}"):
        x, _, _ = simulate_amici_base(
            amici_model_base, theta_log=theta[:4], t_sim=T_VIS
        )
        x_sims.append(x)
    val["x_sims"] = x_sims

"""
Start from here to avoid rerunning simulations
"""
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.6)

# Each trajectory plot spans 2 columns
ax_traj = [
    fig.add_subplot(gs[0, 0:2]),  # cols 0-1
    fig.add_subplot(gs[0, 2:4]),  # cols 2-3
]

# Four marginal plots, one column each
ax_marg = [fig.add_subplot(gs[1, i]) for i in range(4)]

# ---- Figure 3B: Trajectories ----
class HandlerLineInPatch(HandlerTuple):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        line, patch = orig_handle
        patch_artist = mpatches.Rectangle(
            (xdescent, ydescent), width, height,
            facecolor=patch.get_facecolor(), alpha=0.4,
            transform=trans, lw=0
        )
        line_artist = mlines.Line2D(
            [xdescent, xdescent + width],
            [ydescent + height / 2, ydescent + height / 2],
            color=line.get_color(), lw=line.get_linewidth(),
            transform=trans
        )
        return [patch_artist, line_artist]

legend_handles_traj = []
legend_labels_traj = []

for ax, (label, val) in zip(ax_traj, parameter_dicts.items()):
    color = val["color"]
    x_sims = val["x_sims"]
    
    # plot the 90% CI + median
    ax.fill_between(
        T_VIS,
        np.percentile(x_sims, 5, axis=0),
        np.percentile(x_sims, 95, axis=0),
        color=color, alpha=0.4
    )

    ax.plot(T_VIS, np.median(x_sims, axis=0), color=color, linewidth=LINE_WIDTH)
    
    # plot data and ground truth
    data_err = ax.errorbar(
        T_WRONG, x_data,
        yerr=SIGMA_X, xerr=SIGMA_TAU,
        fmt='o', markersize=8, capsize=CAP_SIZE,
        color=COLOR_WRONG_TP, elinewidth=LINE_WIDTH, capthick=CAP_THICK
    )

    x_gt, _, _ = simulate_amici_base(amici_model_base, theta_log=THETA_TRUE, t_sim=T_VIS)
    ax.plot(T_VIS, x_gt, color=COLOR_GROUND_TRUTH, linestyle='--', linewidth=LINE_WIDTH)



    ax.set_xlim((0, T_VIS[-1]))
    ax.set_ylim((0, np.max(x_data) + SIGMA_X + 0.01))
    ax.set_xlabel('Time', fontsize=FONT_SIZE_AXES)
    ax.set_ylabel('Observable' if ax == ax_traj[0] else '', fontsize=FONT_SIZE_AXES)
    # ax.set_title(label, fontsize=FONT_SIZE_TITLE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)

    ci_patch = mpatches.Patch(color=color, alpha=0.6)
    median_line_handle = mlines.Line2D([], [], color=color, linewidth=LINE_WIDTH)
    legend_handles_traj.append((median_line_handle, ci_patch))
    legend_labels_traj.append(f'Median + 90% CI\n{label}')

ground_truth_handle = mlines.Line2D(
    [], [], color=COLOR_GROUND_TRUTH, linestyle='--', linewidth=LINE_WIDTH
)
legend_handles_traj = [ground_truth_handle, data_err] + legend_handles_traj
legend_labels_traj = ['Ground Truth', 'Data'] + legend_labels_traj

# Place trajectory legend to the right of ax_traj[1]
ax_traj[1].legend(
    handles=legend_handles_traj,
    labels=legend_labels_traj,
    handler_map={tuple: HandlerLineInPatch()},
    loc='center left',
    bbox_to_anchor=(1.05, 0.6),
    fontsize=FONT_SIZE_AXES,
    frameon=False
)

# ---- Figure 3C: Marginals ----
for i, ax in enumerate(ax_marg):
    df = pd.DataFrame({
        "Joint\nApproach": samples_full[:, i],
        "Marginalized\nApproach": samples_marginalized[:, i]
    })
    df_melted = df.melt(var_name='Model', value_name='theta')

    sns.kdeplot(
        data=df_melted,
        x='theta',
        hue='Model',
        fill=False,
        legend=False,
        common_norm=False,
        palette=[COLOR_FULL, COLOR_MARGINALIZED],
        linewidth=LINE_WIDTH,
        ax=ax
    )

    y_max = ax.get_ylim()[1]
    ax.plot(
        [THETA_TRUE[i], THETA_TRUE[i]], [0, y_max],
        color=COLOR_GROUND_TRUTH, linestyle='--', linewidth=LINE_WIDTH
    )
    ax.set_title(r"$p(\theta_{" + str(i+1) + r"}\mid \mathcal{D})$", fontsize=FONT_SIZE_TITLE)
    
    ax.set_xlabel(r"$\theta_{" + str(i+1) + r"}$", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel(
        r"Posterior" if i == 0 else "",
        fontsize=FONT_SIZE_AXES
    )
    ax.set_xlim(LB_WRONG_TP[i] - .1, UB_WRONG_TP[i] + .1)
    ax.set_ylim(0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)

# Place marginals legend to the right of ax_marg[3]
joint_handle = mlines.Line2D([], [], color=COLOR_FULL, linewidth=LINE_WIDTH)
marg_handle = mlines.Line2D([], [], color=COLOR_MARGINALIZED, linewidth=LINE_WIDTH)
gt_handle = mlines.Line2D([], [], color=COLOR_GROUND_TRUTH, linestyle='--', linewidth=LINE_WIDTH)

ax_marg[3].legend(
    handles=[gt_handle, joint_handle, marg_handle],
    labels=['Ground Truth', 'Joint Approach', 'Marginalized Approach'],
    loc='center left',
    bbox_to_anchor=(1.05, 0.6),
    fontsize=FONT_SIZE_AXES,
    frameon=False
)

# fig.suptitle("Marginalized Posterior Distributions", fontsize=FONT_SIZE_TITLE)
plt.tight_layout()
plt.savefig(f"{FIGURE_STORE_DIR}/figure_3_bc.{DATAFORMAT}", bbox_inches="tight")  
plt.show()

"""
FIGURE 4:

"""
# save the run dict to a csv file
import pandas as pd
import matplotlib.pyplot as plt
"""
Figure 4 A:  Number of parameters vs. number of data points
"""
# now we want to plot the number of parameters for both approaches
"""
plt.figure(figsize=(8, 6))
n_params_full = [4 + n_time for n_time in n_to_plot]
n_params_marginalized = [4 for _ in n_to_plot]


plt.plot(
    n_to_plot,
    n_params_full,
    'o-',
    label=r"$J_{full}$",
    color=COLOR_FULL,
    linewidth=LINE_WIDTH)

plt.plot(
    n_to_plot, 
    n_params_marginalized, 
    'o-',
    label=r"$J_{marginalized}$", 
    color=COLOR_MARGINALIZED, 
    linewidth=LINE_WIDTH)


plt.xlabel("# data points", fontsize=FONT_SIZE_AXES)
plt.ylabel("# parameters", fontsize=FONT_SIZE_AXES)
plt.title("#parameters vs. # data points", fontsize=FONT_SIZE_TITLE)

plt.xlim((4, 51))
plt.ylim((0, 55))

plt.legend(fontsize=FONT_SIZE_AXES)


ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(LINE_WIDTH)
ax.spines['bottom'].set_linewidth(LINE_WIDTH)
ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)
"""
"""
Figure 4 TBD: ESS
"""
"""
n_to_plot = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 30, 40, 50]

plt.figure(figsize=(8, 6))

for approach, color in zip(["full", "marginalized"], [COLOR_FULL, COLOR_MARGINALIZED]):
    n_data = []
    ess = []
    for run, vals in run_dict.items():
        if vals["converged_" + approach] and vals["n_data"] in n_to_plot:
            n_data.append(vals["n_data"])
            ess.append(vals["ESS_" + approach]/N_SAMPLES)

    # do a bar plot, where on the x axis you have n_data and on the y axis you have ess
    # plt.scatter(n_data, ess, label=approach, color=color)
    plt.plot(n_data, ess, 'o-', color=color, linewidth=LINE_WIDTH, label=approach)
    plt.xlabel("Number of data points", fontsize=FONT_SIZE_AXES)
    plt.ylabel("ESS/#Sample", fontsize=FONT_SIZE_AXES)
    plt.title("ESS/#Sample", fontsize=FONT_SIZE_TITLE)
    plt.xlim((4, 51))
    plt.ylim(0, 0.04)

    plt.legend(fontsize=FONT_SIZE_AXES)
    
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)
"""


"""
All the plots for Figure 4 are done from two functions 
(one for line-plots, one for violins)
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba


# X_FIXED = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 30, 40, 50]
X_FIXED = [5, 10, 15, 20, 30, 40, 50]

def plot_rowise_mean_two_methods(
    full_arr: np.ndarray,
    marg_arr: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    legend_labels=(r"Full", r"Marginalized"),
    ax = None
):
    # row-wise means, ignoring NaNs
    mean_full = np.mean(np.nan_to_num(full_arr, nan=0.0), axis=1)
    mean_marg = np.mean(np.nan_to_num(marg_arr, nan=0.0), axis=1)
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(X_FIXED, mean_full, "o-", color=COLOR_FULL, linewidth=LINE_WIDTH, label=legend_labels[0])
    ax.plot(X_FIXED, mean_marg, "o-", color=COLOR_MARGINALIZED, linewidth=LINE_WIDTH, label=legend_labels[1])

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_AXES)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_AXES)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)

    ax.set_xlim((4, 51))
    ax.set_ylim((0, None))
    ax.legend(fontsize=FONT_SIZE_AXES)

    # style like your example
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)
    return ax

def plot_rowwise_median_two_methods(
    full_arr: np.ndarray,
    marg_arr: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    legend_labels=(r"Full", r"Marginalized"),
    ax = None
):
    # row-wise medians, ignoring NaNs
    med_full = np.nanmedian(full_arr, axis=1)
    med_marg = np.nanmedian(marg_arr, axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))


    ax.plot(X_FIXED, med_full, "o-", color=COLOR_FULL, linewidth=LINE_WIDTH, label=legend_labels[0])
    ax.plot(X_FIXED, med_marg, "o-", color=COLOR_MARGINALIZED, linewidth=LINE_WIDTH, label=legend_labels[1])

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_AXES)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_AXES)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)

    ax.set_xlim((4, 51))
    ax.set_ylim((0, None))
    ax.legend(fontsize=FONT_SIZE_AXES)

    # style like your example
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)

    return ax

def plot_line_two_methods(
    full_arr: np.ndarray,
    marg_arr: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    legend_labels=(r"Joint", r"Marginalized"), 
    ax = None
):
    """
    Plots a line plot for both methods
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(X_FIXED, full_arr, "o-", color=COLOR_FULL, linewidth=LINE_WIDTH, label=legend_labels[0])
    ax.plot(X_FIXED, marg_arr, "o-", color=COLOR_MARGINALIZED, linewidth=LINE_WIDTH, label=legend_labels[1])

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_AXES)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_AXES)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)

    ax.set_xlim((4, 51))
    ax.set_ylim((0, None))
    # ax.legend(fontsize=FONT_SIZE_AXES)

    # style like your example
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)

    return ax


def plot_split_violin_two_methods(
    full_arr: np.ndarray,
    marg_arr: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    legend_labels=(r"$J_{full}$", r"$J_{marginalized}$")
):
    """
    Split violin per row (evenly spaced x positions 1..14):
      - left half = full_arr row distribution
      - right half = marg_arr row distribution
    Treats ±inf as NaN, ignores NaNs, and scales width by valid fraction.
    """
    # --- Treat inf as NaN (without mutating caller's arrays) ---
    full = np.array(full_arr, copy=True, dtype=float)
    marg = np.array(marg_arr, copy=True, dtype=float)
    full[~np.isfinite(full)] = np.nan
    marg[~np.isfinite(marg)] = np.nan

    # --- Build per-row samples & width scaling (valid fraction) ---
    n_rows, n_cols_full = full.shape
    _,        n_cols_marg = marg.shape

    pos_idx = np.arange(1, len(X_FIXED) + 1)

    data_full, pos_full, widths_full = [], [], []
    data_marg, pos_marg, widths_marg = [], [], []

    MAX_WIDTH = 0.8  # base width for a row with 100% valid values

    for i in range(len(X_FIXED)):
        # Full
        row_f = full[i, :]
        mask_f = ~np.isnan(row_f)
        n_valid_f = int(mask_f.sum())
        if n_valid_f > 0:
            data_full.append(row_f[mask_f])
            pos_full.append(pos_idx[i])
            widths_full.append(MAX_WIDTH * (n_valid_f / n_cols_full))

        # Marginalized
        row_m = marg[i, :]
        mask_m = ~np.isnan(row_m)
        n_valid_m = int(mask_m.sum())
        if n_valid_m > 0:
            data_marg.append(row_m[mask_m])
            pos_marg.append(pos_idx[i])
            widths_marg.append(MAX_WIDTH * (n_valid_m / n_cols_marg))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot violins only where we have data; widths shrink with valid fraction
    v_full = ax.violinplot(
        data_full, positions=pos_full,
        widths=widths_full,
        showmeans=False, showmedians=False, showextrema=False
    )
    v_marg = ax.violinplot(
        data_marg, positions=pos_marg,
        widths=widths_marg,
        showmeans=False, showmedians=False, showextrema=False
    )

    for body, pos in zip(v_full.get('bodies', []), pos_full):
        body.set_facecolor(to_rgba(COLOR_FULL, 0.8))
        # body.set_edgecolor(to_rgba(COLOR_FULL, 0))
        body.set_linewidth(0)
        body.set_alpha(0.8)  # ensure no transparency
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.minimum(verts[:, 0], pos)  # left half only

    for body, pos in zip(v_marg.get('bodies', []), pos_marg):
        body.set_facecolor(to_rgba(COLOR_MARGINALIZED, 0.8))
        # body.set_edgecolor(to_rgba(COLOR_MARGINALIZED, 0))
        body.set_linewidth(0)
        body.set_alpha(0.8)  # ensure no transparency
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.maximum(verts[:, 0], pos)  # right half only

    # Labels / title
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_AXES)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_AXES)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)

    # Even spacing with your X as tick labels
    ax.set_xticks(pos_idx)
    ax.set_xticklabels(X_FIXED)
    ax.set_xlim(0.5, len(X_FIXED) + 0.5)
    ax.set_ylim((0, None))

    # Legend (proxy patches)
    handles = [
        Patch(facecolor=COLOR_FULL, edgecolor=COLOR_FULL, linewidth=LINE_WIDTH, label=legend_labels[0]),
        Patch(facecolor=COLOR_MARGINALIZED, edgecolor=COLOR_MARGINALIZED, linewidth=LINE_WIDTH, label=legend_labels[1]),
    ]
    ax.legend(handles=handles, fontsize=FONT_SIZE_AXES)

    # Match your line-plot styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)

    plt.tight_layout()
    return ax

def plot_box_two_methods(
    full_list: list[np.ndarray],
    marg_list: list[np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    legend_labels=(r"Joint", r"Marginalized"), 
    ax = None
):
    """
    Plots side-by-side boxplots for both methods.

    Parameters
    ----------
    full_list : list of arrays, one entry per x-tick (n_data value),
                each array contains the values across random seeds.
    marg_list : same for marginalized.
    """
    if isinstance(full_list, np.ndarray):
        full_list = [full_list[i] for i in range(full_list.shape[0])]
    if isinstance(marg_list, np.ndarray):
        marg_list = [marg_list[i] for i in range(marg_list.shape[0])]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    n_groups = len(X_FIXED)
    x = np.arange(n_groups)
    width = 0.35  # half-spacing between the two boxes

    def make_boxplot(ax, data_list, positions, color):
        bp = ax.boxplot(
            data_list,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            boxprops=dict(facecolor=color, color=color, alpha=1),
            medianprops=dict(color="black", linewidth=LINE_WIDTH),
            whiskerprops=dict(color=color, linewidth=LINE_WIDTH),
            capprops=dict(color=color, linewidth=LINE_WIDTH),
            flierprops=dict(marker='o', color=color, markersize=4, alpha=0.5),
        )
        return bp

    bp_full = make_boxplot(ax, full_list, x - width / 2, COLOR_FULL)
    bp_marg = make_boxplot(ax, marg_list, x + width / 2, COLOR_MARGINALIZED)

    # Legend proxies
    """
    from matplotlib.patches import Patch
    ax.legend(
        [Patch(facecolor=COLOR_FULL, alpha=0.6), Patch(facecolor=COLOR_MARGINALIZED, alpha=0.6)],
        legend_labels,
        fontsize=FONT_SIZE_AXES
    )
    """

    ax.set_xticks(x)
    ax.set_xticklabels(X_FIXED)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_AXES)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_AXES)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.set_xlim((-0.5, n_groups - 0.5))
    ax.set_ylim((0, None))
    
    y_max = ax.get_ylim()[1]
    ax.set_ylim((-0.01 * y_max, y_max))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)

    return ax



"""
Figure 4 A: Number of parameters vs. number of data points 
"""
x_full = np.array([4 + n_time for n_time in X_FIXED], dtype=float).reshape(-1, 1)
x_marg = np.array([4 for _ in X_FIXED], dtype=float).reshape(-1, 1)

plot_line_two_methods(
    full_arr=x_full,
    marg_arr=x_marg,
    xlabel=r"Number of Time Points $n_t$",
    ylabel="Problem Dimensionality",
    title=r"Problem Dimensionality vs. $n_t$"
)


"""
Figure 4 Prep DF & Playground for plotting ESS and Sampling Walltime
"""
# 1st: ESS per second vs. n_data for all three approaches (only for converged sampling runs)
# read in pandas df
per_experiment_df = pd.read_csv("/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_ConversionReaction/per_experiment_df.csv", index_col=0)
per_setup_df = pd.read_csv("/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_ConversionReaction/per_setup_df.csv", index_col=0)

"""
Set ESS_per_second + ESS to 0, and run_time_sampling to inf, 
iff the sampling (visiually) did not converge
"""
random_seeds = set(per_experiment_df["random_seed"].to_numpy(dtype=int))

# loop over all rows in per_setup_df
"""
for idx, row in per_setup_df.iterrows():
    for random_seed in random_seeds:
        converged_full = row[f"converged_full_{random_seed}"]
        converged_marg = row[f"converged_marginalized_{random_seed}"]

        if pd.isna(converged_full) or str(converged_full).strip() == "True":
            pass  # treat NaN and True as converged
        else:
            per_setup_df.at[idx, f"ESS_per_second_full_{random_seed}"] = 0
            per_setup_df.at[idx, f"ESS_full_{random_seed}"] = 0
            per_setup_df.at[idx, f"run_time_sampling_full_{random_seed}"] = float("inf")

        if pd.isna(converged_marg) or str(converged_marg).strip() == "True":
            pass
        else:
            per_setup_df.at[idx, f"ESS_per_second_marginalized_{random_seed}"] = 0
            per_setup_df.at[idx, f"ESS_marginalized_{random_seed}"] = 0
            per_setup_df.at[idx, f"run_time_sampling_marginalized_{random_seed}"] = float("inf")

# filter the DataFrame to keep only rows where the index "n_data" is in X_FIXED
"""
per_setup_df = per_setup_df[per_setup_df.index.isin(X_FIXED)]

if True:
    """
    Figure 4: Line Plot Median for ESS/SAMPLE
    """


    # now let's go for ESS (effective sample size per sample, i.e. ESS/N_SAMPLES)
    ess_cols_full = [col for col in per_setup_df.columns if col.startswith("ESS_full")]
    ess_array_full = per_setup_df[ess_cols_full].to_numpy()/N_SAMPLES


    ess_cols_marg = [col for col in per_setup_df.columns if col.startswith("ESS_marginalized")]
    ess_array_marg = per_setup_df[ess_cols_marg].to_numpy()/N_SAMPLES

    plot_rowise_mean_two_methods(
        full_arr=ess_array_full,
        marg_arr=ess_array_marg,
        xlabel="Number of Data Points",
        ylabel="ESS/Sample",
        title="Mean ESS/Sample"
    )

    plot_box_two_methods(
        full_list=ess_array_full,
        marg_list=ess_array_marg,
        xlabel="Number of Data Points",
        ylabel="ESS/Sample",
        title="Mean ESS/Sample"
    )

    """
    plot_rowwise_median_two_methods(
        full_arr=ess_array_full,
        marg_arr=ess_array_marg,
        xlabel="Number of data points",
        ylabel="ESS/Sample",
        title="Median ESS/Sample"
    )

    plot_split_violin_two_methods(
        full_arr=ess_array_full,
        marg_arr=ess_array_marg,
        xlabel="Number of data points",
        ylabel="ESS/Sample",
        title="ESS/Sample"
    )
    """

    """
    Figure 4: Line Plot Median Sampling Walltime
    """

    # now let's go for total sampling run time
    ess_cols_full = [col for col in per_setup_df.columns if col.startswith("run_time_sampling_full_")]
    ess_array_full = per_setup_df[ess_cols_full].to_numpy()


    ess_cols_marg = [col for col in per_setup_df.columns if col.startswith("run_time_sampling_marginalized_")]
    ess_array_marg = per_setup_df[ess_cols_marg].to_numpy()

    plot_rowise_mean_two_methods(
        full_arr=ess_array_full,
        marg_arr=ess_array_marg,
        xlabel="Number of data points",
        ylabel="Sampling Walltime [s]",
        title="Mean Sampling Walltime"
    )

    plot_rowwise_median_two_methods(
        full_arr=ess_array_full,
        marg_arr=ess_array_marg,
        xlabel="Number of data points",
        ylabel="Sampling Walltime [s]",
        title="Median Sampling Walltime"
    )

    plot_box_two_methods(
        full_list=[ess_array_full[i] for i in range(len(X_FIXED))],
        marg_list=[ess_array_marg[i] for i in range(len(X_FIXED))],
        xlabel="Number of Data Points",
        ylabel="Sampling Walltime [s]",
        title="Sampling Walltime"
    )


    plot_split_violin_two_methods(
        full_arr=ess_array_full,
        marg_arr=ess_array_marg,
        xlabel="Number of data points",
        ylabel="Sampling Walltime [s]",
        title="Sampling Walltime"
    )

    """
    Figure 4: Line Plot Median for ESS/second
    """

    # now let's go for ESS per seconds
    ess_cols_full = [col for col in per_setup_df.columns if col.startswith("ESS_per_second_full_")]
    ess_array_full = per_setup_df[ess_cols_full].to_numpy()


    ess_cols_marg = [col for col in per_setup_df.columns if col.startswith("ESS_per_second_marginalized_")]
    ess_array_marg = per_setup_df[ess_cols_marg].to_numpy()

    plot_rowise_mean_two_methods(
        full_arr=ess_array_full,
        marg_arr=ess_array_marg,
        xlabel="Number of data points",
        ylabel="ESS/Second",
        title="Mean ESS/Second"
    )

    plot_rowwise_median_two_methods(
        full_arr=ess_array_full,
        marg_arr=ess_array_marg,
        xlabel="Number of data points",
        ylabel="ESS/Second",
        title="Median ESS/Second"
    )

    plot_box_two_methods(
        full_list=[ess_array_full[i] for i in range(len(X_FIXED))],
        marg_list=[ess_array_marg[i] for i in range(len(X_FIXED))],
        xlabel="Number of Data Points",
        ylabel="ESS/Second",
        title="ESS/Second"
    )



    plot_split_violin_two_methods(
        full_arr=ess_array_full,
        marg_arr=ess_array_marg,
        xlabel="Number of data points",
        ylabel="ESS/Second",
        title="ESS/Second"
    )


"""
Figure 4 A-C:
"""
fig, axes = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={'wspace': 0.4})
"""
Panel A: Problem Dimensionality
"""
x_full = np.array([4 + n_time for n_time in X_FIXED], dtype=float).reshape(-1, 1)
x_marg = np.array([4 for _ in X_FIXED], dtype=float).reshape(-1, 1)

plot_line_two_methods(
    full_arr=x_full,
    marg_arr=x_marg,
    xlabel=r"$n_t$",
    ylabel="Problem Dimension",
    title=" ",
    ax=axes[0]
)

"""
Panel B: ESS/Sample
"""
"""
ess_cols_full = [col for col in per_setup_df.columns if col.startswith("ESS_full")]
ess_array_full = per_setup_df[ess_cols_full].to_numpy() / N_SAMPLES
ess_cols_marg = [col for col in per_setup_df.columns if col.startswith("ESS_marginalized")]
ess_array_marg = per_setup_df[ess_cols_marg].to_numpy() / N_SAMPLES

plot_box_two_methods(
    full_list=[ess_array_full[i] for i in range(len(X_FIXED))],
    marg_list=[ess_array_marg[i] for i in range(len(X_FIXED))],
    xlabel=r"$n_t$",
    ylabel="ESS/Sample",
    title="ESS/Sample",
    ax=axes[1]
)
"""
z_cols_full = [col for col in per_setup_df.columns if col.startswith("geweke_z_score_full")]
z_array_full = per_setup_df[z_cols_full].to_numpy() 
z_cols_marg = [col for col in per_setup_df.columns if col.startswith("geweke_z_score_marginalized")]
z_array_marg = per_setup_df[z_cols_marg].to_numpy() 

plot_box_two_methods(
    full_list=[z_array_full[i] for i in range(len(X_FIXED))],
    marg_list=[z_array_marg[i] for i in range(len(X_FIXED))],
    xlabel=r"$n_t$",
    ylabel="z-Score",
    title=" ",
    ax=axes[1]
)

axes[1].plot(
    axes[1].get_xlim(), 
    1.96*np.array([1, 1]),
    "k--",
    linewidth=LINE_WIDTH,
    label="95% threshold")

"""
Panel C: ESS/Second
"""
ess_cols_full = [col for col in per_setup_df.columns if col.startswith("ESS_per_second_full_")]
ess_array_full = per_setup_df[ess_cols_full].to_numpy()
ess_cols_marg = [col for col in per_setup_df.columns if col.startswith("ESS_per_second_marginalized_")]
ess_array_marg = per_setup_df[ess_cols_marg].to_numpy()

plot_box_two_methods(
    full_list=[ess_array_full[i] for i in range(len(X_FIXED))],
    marg_list=[ess_array_marg[i] for i in range(len(X_FIXED))],
    xlabel=r"$n_t$",
    ylabel="ESS/Second",
    title=" ",
    ax=axes[2]
)

plt.tight_layout()

# Shared legend to the right
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=COLOR_FULL, label="Joint Approach"),
    Patch(facecolor=COLOR_MARGINALIZED, label="Marginalized Approach"),
]
fig.legend(
    handles=legend_handles,
    loc="center right",
    bbox_to_anchor=(1.05, 0.77),  
    fontsize=FONT_SIZE_AXES,
    frameon=False,  # box around legend
)
# save as vecotr graphic
plt.savefig(f"{FIGURE_STORE_DIR}/figure_4_abc.{DATAFORMAT}", bbox_inches="tight")  
# plt.savefig("figure.pdf", bbox_inches="tight")  # bbox_inches="tight" ensures legend isn't clipped
plt.show()




"""
Figure 4 D: Parameter traces: Comparison of 
two methods & two runs (one, where "full" did not converged, 
due to the problem dimension)
"""
import pypesto.sample as sample
N_EXPERIMENT_CONVERGED = 4
N_EXPERIMENT_NON_CONVERGED = 94

STORAG_BASE_DIR = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_ConversionReaction"

LB_WRONG_TP = np.array([-1, -1, -1, -1], dtype=float)
UB_WRONG_TP = np.array([1, 1, 2, 2], dtype=float)
N_SAMPLES= 500_000


# read all 4 results
result_full_converged = store.read_result(f"{STORAG_BASE_DIR}/res_{N_EXPERIMENT_CONVERGED}/result_full.hdf5")
result_marginalized_converged = store.read_result(f"{STORAG_BASE_DIR}/res_{N_EXPERIMENT_CONVERGED}/result_marginalized.hdf5")

result_full_non_converged = store.read_result(f"{STORAG_BASE_DIR}/res_{N_EXPERIMENT_NON_CONVERGED}/result_full.hdf5")
result_marginalized_non_converged = store.read_result(f"{STORAG_BASE_DIR}/res_{N_EXPERIMENT_NON_CONVERGED}/result_marginalized.hdf5")

# get the burn-in for all methods
burn_in_full_converged = sample.geweke_test(result_full_converged)
burn_in_marginalized_converged = sample.geweke_test(result_marginalized_converged)

burn_in_full_non_converged = sample.geweke_test(result_full_non_converged)
burn_in_marginalized_non_converged = sample.geweke_test(result_marginalized_non_converged)

# extract parameter samples (only the ones _after_ burn-in...)
samples_full_converged = result_full_converged.sample_result.trace_x[0, burn_in_full_converged:, :4]
samples_marginalized_converged = result_marginalized_converged.sample_result.trace_x[0, burn_in_marginalized_converged:, :4]

samples_full_non_converged = result_full_non_converged.sample_result.trace_x[0, burn_in_full_non_converged:, :4]
samples_marginalized_non_converged = result_marginalized_non_converged.sample_result.trace_x[0, burn_in_marginalized_non_converged:, :4]

N_D_CONVERGED = result_full_converged.sample_result.trace_x.shape[-1] - 4
N_D_NON_CONVERGED = result_full_non_converged.sample_result.trace_x.shape[-1] - 4

# now plot the traces in a 2x2 subplot (1st row full, 2nd: marginal, 1st col converged, ...)
# as scatter

ALPHA = 0.12
S=10

PLOT_EVERY_NTH = 10  # easy to change
param_idx = 0        # only first kinetic parameter

fig, axs_all = plt.subplots(1, 5, figsize=(20, 5), 
                             gridspec_kw={'width_ratios': [1, 1, 0.15, 1, 1], 
                                          'wspace': 0.15})
axs_all[2].set_visible(False)  # hide the spacer
axs = [axs_all[0], axs_all[1], axs_all[3], axs_all[4]]
"""
fig.suptitle(
    "Parameter Traces",
    fontsize=FONT_SIZE_TITLE,
    y=1.02)
"""
panels = [
    (samples_full_converged,             burn_in_full_converged,             COLOR_FULL,         "Joint",        N_D_CONVERGED),
    (samples_marginalized_converged,     burn_in_marginalized_converged,     COLOR_MARGINALIZED, "Marginalized", N_D_CONVERGED),
    (samples_full_non_converged,         burn_in_full_non_converged,         COLOR_FULL,         "Joint",        N_D_NON_CONVERGED),
    (samples_marginalized_non_converged, burn_in_marginalized_non_converged, COLOR_MARGINALIZED, "Marginalized", N_D_NON_CONVERGED),
]

for ax, (samples, burn_in, color, method_label, n_t) in zip(axs, panels):
    indices_full = np.arange(burn_in, N_SAMPLES)
    indices = indices_full[::PLOT_EVERY_NTH]
    values  = samples[:len(indices_full):PLOT_EVERY_NTH, param_idx]

    ax.scatter(indices, values, color=color, s=S, alpha=0.06, rasterized=True)
    """
    ax.set_title(
        method_label + r", $n_t = " + str(n_t) + r"$",
        fontsize=FONT_SIZE_TITLE,
        pad=10)
    """
    # ax.set_xlabel("Sampling Index", fontsize=FONT_SIZE_AXES)
    ax.set_xlim((burn_in, N_SAMPLES))
    ax.set_ylim(LB_WRONG_TP[param_idx], UB_WRONG_TP[param_idx])

    # x ticks at 1e5, 3e5, 5e5, ... up to N_SAMPLES
    xticks = [t for t in [1e5, 3e5, 5e5, 1e6, 3e6, 5e6] if burn_in <= t <= N_SAMPLES]
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [r"$" + f"{t:.0e}".replace("e+0", r"\times 10^{").replace("e+", r"\times 10^{") + "}$"
         for t in xticks],
        rotation=30, ha='right', fontsize=FONT_SIZE_TICKS)

    # y ticks
    yticks = [y for y in [-1, 0, 1, 2] if LB_WRONG_TP[param_idx] <= y <= UB_WRONG_TP[param_idx]]
    ax.set_yticks(yticks)

    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
    ax.minorticks_off()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
"""
axs[0].set_ylabel(r"$\log_{10}(\theta_1)$", fontsize=FONT_SIZE_AXES)
"""
axs[1].set_yticklabels([])
axs[3].set_yticklabels([])
fig.tight_layout()

plt.savefig(f"{FIGURE_STORE_DIR}/figure_4_d.{DATAFORMAT}", bbox_inches="tight", dpi=750)  




"""
for param_idx in range(4):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    fig.suptitle(
        r"Parameter Traces for $\log(\theta_{"+str(param_idx+1)+"})$", 
        fontsize=FONT_SIZE_TITLE, 
        y=0.98)

    fig.supxlabel(
        "Sampling indices", 
        fontsize=FONT_SIZE_TITLE, 
        y=0.02)
    fig.supylabel(
        r"$\log(\theta_{"+str(param_idx+1)+r"})$", 
        fontsize=FONT_SIZE_TITLE, 
        x=0.02)


    # 1st row, 1st col: full, converged
    axs[0, 0].scatter(
        range(burn_in_full_converged -1, N_SAMPLES), 
        samples_full_converged[:, param_idx], 
        color=COLOR_FULL, s=S, alpha=ALPHA)
    # axs[0, 0].set_title(r"$J_{full}$, converged", fontsize=FONT_SIZE_TITLE)
    axs[0, 0].set_ylabel(r"$J_{full}$", fontsize=FONT_SIZE_AXES, labelpad=15)
    axs[0, 0].set_xlim((burn_in_full_converged - 1, N_SAMPLES))
    axs[0, 0].set_title(
        r"$n_{\mathcal{D}} = "+str(N_D_CONVERGED)+r"$", 
        fontsize=FONT_SIZE_TITLE, 
        pad=15)

    # 1st row, 2nd col: full, non-converged
    axs[0, 1].scatter(
        range( burn_in_full_non_converged - 1, N_SAMPLES), 
        samples_full_non_converged[:, param_idx], 
        color=COLOR_FULL, s=S, alpha=ALPHA)
    # axs[0, 1].set_title(r"$J_{full}$, non-converged", fontsize=FONT_SIZE_TITLE)
    axs[0, 1].set_xlim((burn_in_full_non_converged - 1, N_SAMPLES))
    axs[0, 1].set_title(
        r"$n_{\mathcal{D}} = "+str(N_D_NON_CONVERGED)+r"$", 
        fontsize=FONT_SIZE_TITLE, 
        pad=15)

    # 2nd row, 1st col: marginalized, converged
    axs[1, 0].scatter(
        range(burn_in_marginalized_converged - 1, N_SAMPLES), 
        samples_marginalized_converged[:, param_idx], 
        color=COLOR_MARGINALIZED, s=S, alpha=ALPHA)
    # axs[1, 0].set_title(r"$J_{marginalized}$, converged", fontsize=FONT_SIZE_TITLE)
    axs[1, 0].set_ylabel(r"$J_{marginalized}$", fontsize=FONT_SIZE_AXES, labelpad=15)
    axs[1, 0].set_xlim((burn_in_marginalized_converged - 1, N_SAMPLES))

    # 2nd row, 2nd col: marginalized, non-converged
    axs[1, 1].scatter(
        range(burn_in_marginalized_non_converged - 1, N_SAMPLES), 
        samples_marginalized_non_converged[:, param_idx], 
        color=COLOR_MARGINALIZED, s=S, alpha=ALPHA)
    # axs[1, 1].set_title(r"$J_{marginalized}$, non-converged", fontsize=FONT_SIZE_TITLE)
    axs[1, 1].set_xlim((burn_in_marginalized_non_converged - 1, N_SAMPLES))

    for ax in axs.flat:
        # --- X-axis ticks every 200,000 ---
        x_min, x_max = ax.get_xlim()
        tick_start = int(np.ceil(x_min / 100_000) * 100_000)
        tick_end = int(np.floor(x_max / 100_000) * 100_000)
        xticks = np.arange(tick_start, tick_end + 1, 200_000)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:,}" for t in xticks])

        # --- Y-axis ticks only from [-1, 0, 1, 2], if within limits ---
        ax.set_ylim(LB_WRONG_TP[param_idx], UB_WRONG_TP[param_idx])
        yticks = [y for y in [-1, 0, 1, 2] if LB_WRONG_TP[param_idx] <= y <= UB_WRONG_TP[param_idx]]
        ax.set_yticks(yticks)

        # --- Tick label font size ---
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICKS)

        # remove minor ticks
        ax.minorticks_off()
        # --- Spines ---
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(LINE_WIDTH)
        ax.spines['bottom'].set_linewidth(LINE_WIDTH)

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])
"""