import numpy as np
from math import exp
import pypesto
import matplotlib.pyplot as plt

import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.visualize as visualize
import pypesto.store as store
from copy import deepcopy

import yaml2sbml
import amici
import importlib
import sys
import os
import math
from collections.abc import Iterable

import seaborn as sns
import pandas as pd

os.chdir(os.path.dirname(__file__))

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

#% Set up the constants

# parameters
THETA_TRUE = 1.0
T_WRONG = np.array([0.5, 2])

SIGMA_X = 0.1
SIGMA_T = 0.5

# end time for integral, and time steps of the integral
T_SIM_END = 10 
H_SIM = 0.05

# computational set up 
N_SAMPLES=100000
N_STARTS=50
N_SIM = 100
N_SAMPLES_i = N_SAMPLES
N_STARTS_i = N_STARTS

LB_FULL = [0, 0, 0]
UB_FULL = [5, 4, 4]
LB_MARGINALIZED = [0]
UB_MARGINALIZED = [5]

FONT_SIZE_TICKS = 14
FONT_SIZE_AXES  = 16
FONT_SIZE_TITLE = 20

LINE_WIDTH = 3
CAP_SIZE   = 6
CAP_THICK  = LINE_WIDTH 

FIGURE_STORE_DIR = str(REPO_ROOT / "Figures Paper" / "building_blocks")
os.makedirs(FIGURE_STORE_DIR, exist_ok=True)


COLOR_FULL = [0, 78/255, 159/255] # blue
# COLOR_MARGINALIZED = [252/255, 186/255, 0] # orange
COLOR_MARGINALIZED = [243/255, 118/255, 40/255] # orange
COLOR_WRONG_TP = [144/255, 144/255, 133/255] # grey
COLOR_GROUND_TRUTH = [0, 0, 0] # black

COLORS = [COLOR_WRONG_TP, COLOR_FULL, COLOR_MARGINALIZED]

x_data = np.loadtxt(f'results/x_data.csv', delimiter=',')
t_true = np.loadtxt('results/t_true.csv', delimiter=',')

T_PLOT_MAX = 4
CI_VAL = 0.95

FONT_SIZE_TICKS = 16
FONT_SIZE_AXES  = 18
FONT_SIZE_TITLE = 22


"""
Figure 2 A: Model vs. Data
"""
NUMERICAL_EVALUATION = True

storage_dir = f'results/numerical_evaluation_{str(NUMERICAL_EVALUATION)}'
image_dir = f'results/numerical_evaluation_{str(NUMERICAL_EVALUATION)}/images'

results = [
    store.read_result(f'{storage_dir}/result_wrong_tp.hdf5'),
    store.read_result(f'{storage_dir}/result_full.hdf5'),
    store.read_result(f'{storage_dir}/result_marginalized.hdf5')
]

result_wrong_tp = results[0]
result_full = results[1]
result_marginalized = results[2]

titles = ['Standard Approach', 'Joint Approach', 'Marginalized Approach']

INFO_TO_PLOT_LIST = [
    [True, True,  False, False, True, False],  # model 0: ground truth + wrong TPs + data
    [True, False, True,  False, True, False],  # model 1: ground truth + full model + data
    [True, False, False, True,  True, False],  # model 2: ground truth + marginalized + data
]


"""
Now, let's set up the tiling
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Unit size in inches
U = 1  # 1 unit = 1 inch

panel   = 5 * U   # 5 units
gap     = 2 * U   # 2 units

fig_w = 3 * panel + 2 * gap   # = 19 inches
fig_h = panel + gap + panel + gap + panel  # row A + gap + row B(C+gap+E) = 5+2+5+2+5 = 19 inches... 
# wait: row A = 5, gap = 2, row B = 12 (=5+2+5) → total = 19

fig = plt.figure(figsize=(fig_w, fig_h))

# Convert gaps and panels to figure fractions
def to_frac_x(inches): return inches / fig_w
def to_frac_y(inches): return inches / fig_h

pw = to_frac_x(panel)
gw = to_frac_x(gap)
ph_row = to_frac_y(panel)
gh = to_frac_y(gap)

# ── Row A: three 5×5 panels ──────────────────────────────────────────────────
ax_A0 = fig.add_axes([to_frac_x(0),              to_frac_y(fig_h - panel), pw, ph_row])
ax_A1 = fig.add_axes([to_frac_x(panel + gap),    to_frac_y(fig_h - panel), pw, ph_row])
ax_A2 = fig.add_axes([to_frac_x(2*(panel+gap)),  to_frac_y(fig_h - panel), pw, ph_row])

# ── Row B left: C (5×5) and E (5×5) stacked with 2-unit gap ─────────────────
row_b_top    = to_frac_y(fig_h - panel - gap - panel)        # top of C
row_e_top    = to_frac_y(fig_h - panel - gap - panel - gap - panel)  # top of E

ax_C = fig.add_axes([to_frac_x(0), row_b_top, pw, ph_row])
ax_E = fig.add_axes([to_frac_x(0), row_e_top, pw, ph_row])

# ── Row B right: D (12×12) ───────────────────────────────────────────────────
d_size  = to_frac_x(panel + gap + panel)   # 12 units wide  (= 5+2+5)
d_left  = to_frac_x(panel + gap)           # starts after left col + gap
d_bottom = to_frac_y(fig_h - panel - gap - panel - gap - panel)  # bottom of E = bottom of D

ax_D = fig.add_axes([d_left, d_bottom, d_size, d_size])
ax_D.set_visible(False)

t_sim = np.linspace(0, T_PLOT_MAX, 100)
labels = ["Standard\nApproach", "Joint\nApproach", "Marginalized\nApproach"]

for col, ax in enumerate([ax_A0, ax_A1, ax_A2]):

    INFO_TO_PLOT = INFO_TO_PLOT_LIST[col]

    # Model fits
    for i in range(len(results)):
        if not INFO_TO_PLOT[i + 1]:
            continue
        theta = results[i].optimize_result.x[0][0]
        x_sim = np.array([exp(-theta * t) for t in t_sim])
        ax.plot(
            t_sim, x_sim,
            color=COLORS[i],
            label=labels[i],
            linewidth=LINE_WIDTH
        )

    # Ground truth
    x_gt = np.array([exp(-THETA_TRUE * t) for t in t_sim])
    ax.plot(
        t_sim, x_gt,
        label='Ground\nTruth',
        color=COLOR_GROUND_TRUTH,
        linestyle='--',
        linewidth=LINE_WIDTH
    )

    # Data with uncertainty
    ax.scatter(T_WRONG, x_data, color=COLOR_GROUND_TRUTH, s=40)
    ax.errorbar(
        T_WRONG, x_data,
        yerr=SIGMA_X,
        xerr=SIGMA_T,
        fmt='o',
        color=COLOR_WRONG_TP,
        elinewidth=LINE_WIDTH,
        capsize=CAP_SIZE,
        capthick=CAP_THICK,
        label='Measurement\nData'
    )

    ax.set_xlabel('Time', fontsize=FONT_SIZE_AXES)
    ax.set_ylabel('Observable', fontsize=FONT_SIZE_AXES)
    # ax.set_title(titles[col], fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_AXES, frameon=False, loc='upper right')
    ax.set_xlim(0, T_PLOT_MAX)
    ax.set_ylim(0, 1.1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS)

ax_A0.set_ylabel('x(t)', fontsize=FONT_SIZE_AXES)

# plt.tight_layout()
# plt.show()


"""
Figure 2 C: Marginals of theta.

"""
import seaborn as sns
from matplotlib.lines import Line2D

NUMERICAL_EVALUATION = True
storage_dir = f'results/numerical_evaluation_{str(NUMERICAL_EVALUATION)}'
image_dir = f'results/numerical_evaluation_{str(NUMERICAL_EVALUATION)}/images'

results = [
    store.read_result(f'{storage_dir}/result_wrong_tp.hdf5'),
    store.read_result(f'{storage_dir}/result_full.hdf5'),
    store.read_result(f'{storage_dir}/result_marginalized.hdf5')
]

result_wrong_tp = results[0]
result_full = results[1]
result_marginalized = results[2]


samples_wrong_tp    = result_wrong_tp.sample_result.trace_x[0, :, 0]
samples_full        = result_full.sample_result.trace_x[0, :, 0]
samples_marginalized = result_marginalized.sample_result.trace_x[0, :, 0]

df = pd.DataFrame({
    "Standard Approach$": samples_wrong_tp,
    "Joint Approach$":      samples_full,
    "Marginalized Approach$": samples_marginalized
})
df_melted = df.melt(var_name='Model', value_name=r'$\theta$')

ax=ax_C

palette={
    "Standard Approach$": COLOR_WRONG_TP,
    "Joint Approach$":     COLOR_FULL,
    "Marginalized Approach$": COLOR_MARGINALIZED,
}

# plt.figure(figsize=(5, 5))
ax = sns.kdeplot(
    data=df_melted,
    x=r'$\theta$',
    hue='Model',
    fill=False,
    legend=False,
    common_norm=False,
    palette=palette,
    linewidth=LINE_WIDTH, 
    ax=ax
)

ax.set_xlabel(r"$\theta$", fontsize=FONT_SIZE_AXES)
ax.set_ylabel(r"$p(\theta \mid \mathcal{D})$", fontsize=FONT_SIZE_AXES)
# ax.set_title(r"Posterior $p(\theta \mid \mathcal{D})$", fontsize=FONT_SIZE_TITLE)

ax.set_xlim(0, 5)
ax.set_ylim(0)

y_max = ax.get_ylim()[1]
ax.plot([THETA_TRUE, THETA_TRUE], [0, y_max],
        color=COLOR_GROUND_TRUTH, linestyle='--', linewidth=LINE_WIDTH)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(LINE_WIDTH)
ax.spines['bottom'].set_linewidth(LINE_WIDTH)
ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)

"""
lines = ax.get_lines()
custom_labels = ["Standard\nApproach", "Joint\nApproach", "Marginalized\nApproach"]
ax.legend(handles=lines[:3], labels=custom_labels, fontsize=FONT_SIZE_AXES)
"""

legend_elements = [
    Line2D([0], [0], color=COLOR_WRONG_TP,    linewidth=LINE_WIDTH, label="Standard\nApproach"),
    Line2D([0], [0], color=COLOR_FULL,         linewidth=LINE_WIDTH, label="Joint\nApproach"),
    Line2D([0], [0], color=COLOR_MARGINALIZED, linewidth=LINE_WIDTH, label="Marginalized\nApproach"),
]
ax.legend(handles=legend_elements, fontsize=FONT_SIZE_AXES, frameon=False)

# plt.savefig(f'{image_dir}/posterior_theta.png', bbox_inches='tight', dpi=600)

"""
Figure 2E: Comparative Q-Q Plots
"""
NUMERICAL_EVALUATION =False

qq_data_wrong_tp=[]
qq_data_full=[]
qq_data_marginalized=[]


for i in range(100):
    print(i)
    # read in result number i
    storage_dir = f'results/numerical_evaluation_{str(NUMERICAL_EVALUATION)}'

    result_wrong_tp = store.read_result(f'{storage_dir}/result_wrong_tp_{i}.hdf5')
    result_full = store.read_result(f'{storage_dir}/result_full_{i}.hdf5')
    result_marginalized = store.read_result(f'{storage_dir}/result_marginalized_{i}.hdf5')

    sample_theta_wrong_tp = result_wrong_tp.sample_result.trace_x[0, :, 0]
    sample_theta_full = result_full.sample_result.trace_x[0, :, 0]
    sample_theta_marginalized = result_marginalized.sample_result.trace_x[0, :, 0]

    percentile_wrong_tp = float(np.sum(sample_theta_wrong_tp < THETA_TRUE) / len(sample_theta_wrong_tp))
    percentile_full = float(np.sum(sample_theta_full < THETA_TRUE) / len(sample_theta_full))
    percentile_marginalized = float(np.sum(sample_theta_marginalized < THETA_TRUE) / len(sample_theta_marginalized))

    qq_data_wrong_tp.append(percentile_wrong_tp)
    qq_data_full.append(percentile_full)
    qq_data_marginalized.append(percentile_marginalized)

qq_data_wrong_tp = sorted([2* abs(q-0.5) for q in qq_data_wrong_tp])
qq_data_full = sorted([2* abs(q-0.5) for q in qq_data_full])
qq_data_marginalized = sorted([2* abs(q-0.5) for q in qq_data_marginalized])

n = len(qq_data_wrong_tp)
expected_quantiles = np.arange(1, n + 1) / (n + 1) 

ax = ax_E

ax.plot(qq_data_wrong_tp, expected_quantiles, color=COLOR_WRONG_TP, linestyle='-', linewidth=LINE_WIDTH, label='Standard\nApproach')
ax.plot(qq_data_full, expected_quantiles, color=COLOR_FULL, linestyle='-', linewidth=LINE_WIDTH, label='Joint\nApproach')
ax.plot(qq_data_marginalized, expected_quantiles, color=COLOR_MARGINALIZED, linestyle='-', linewidth=LINE_WIDTH, label='Marginalized\nApproach')
ax.plot([0, 1], [0, 1], color=COLOR_GROUND_TRUTH, linestyle='--', linewidth=LINE_WIDTH)

ax.set_xlabel('Expected Quantiles', fontsize=FONT_SIZE_AXES)
ax.set_ylabel('Observed Quantiles', fontsize=FONT_SIZE_AXES)
# ax.set_title('Sampling Quantiles', fontsize=FONT_SIZE_TITLE)
ax.legend(fontsize=FONT_SIZE_AXES, frameon=False)

ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(LINE_WIDTH)
ax.spines['bottom'].set_linewidth(LINE_WIDTH)
ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)

for percentage in [0.5, 0.8, 0.9, 0.95]:
    print(f"Run for {percentage*100}%")
    print(f"Wrong TP: {np.sum(np.array(qq_data_wrong_tp) < percentage)}")
    print(f"Full: {np.sum(np.array(qq_data_full) < percentage)}")
    print(f"Marginal: {np.sum(np.array(qq_data_marginalized) < percentage)}")
    print("_________________________________")


fig.savefig(f'{FIGURE_STORE_DIR}/figure_2_bce.svg', bbox_inches='tight')


"""
Figure 2D: Sampling scatter plot => integrate manually
"""
NUMERICAL_EVALUATION = True

"""
Figure 2D: Sampling scatter plot
"""
NUMERICAL_EVALUATION = True

storage_dir = f'results/numerical_evaluation_{str(NUMERICAL_EVALUATION)}'
image_dir = f'results/numerical_evaluation_{str(NUMERICAL_EVALUATION)}/images'

results = [
    store.read_result(f'{storage_dir}/result_wrong_tp.hdf5'),
    store.read_result(f'{storage_dir}/result_full.hdf5'),
    store.read_result(f'{storage_dir}/result_marginalized.hdf5')
]

result_wrong_tp = results[0]
result_full = results[1]
result_marginalized = results[2]

samples = result_full.sample_result.trace_x[0, :, :]
samples = pd.DataFrame(samples, columns=[r'$\theta$', r'$\tau_1$', r'$\tau_2$'])

scatter_kws = dict(s=6, alpha=0.02, linewidths=0, edgecolors='none', rasterized=True)
diag_kws    = dict(fill=False, linewidth=LINE_WIDTH, color=COLOR_FULL, bw_adjust=1.0)
ranges = [(0, 5), (0, 4), (0, 4)]
theta_true_full = [THETA_TRUE, t_true[0], t_true[1]]

g = sns.pairplot(
    samples,
    diag_kind='kde',
    plot_kws={**scatter_kws, 'color': COLOR_FULL},   # keep your color
    diag_kws=diag_kws
)
g.figure.set_size_inches(2 * panel + gap, 2 * panel + gap)
g.figure.subplots_adjust(hspace=0.15, wspace=0.15)

for i in range(3):
    # diagonal x-limits
    axd = g.diag_axes[i]
    axd.set_xlim(*ranges[i])

    for j in range(3):
        if i == j:
            continue
        ax = g.axes[i, j]
        ax.set_xlim(*ranges[j])
        ax.set_ylim(*ranges[i])

for i, t in enumerate(theta_true_full):
    axd = g.diag_axes[i]
    y0, y1 = axd.get_ylim()  # remember current y-scale
    axd.axvline(t, color=COLOR_GROUND_TRUTH, linestyle='--', linewidth=LINE_WIDTH)
    axd.set_ylim(y0, y1) 

for ax in g.axes.flatten():
    if ax is None: 
        continue
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)
    ax.xaxis.label.set_size(FONT_SIZE_AXES)
    ax.yaxis.label.set_size(FONT_SIZE_AXES)


ticks_theta = [0, 2.5, 5]
ticks_tau   = [0, 2, 4]

tick_map = {
    0: ticks_theta,
    1: ticks_tau,
    2: ticks_tau,
}

for i in range(3):
    for j in range(3):
        ax = g.axes[i, j]
        if ax is None or i == j:
            continue
        ax.set_xticks(tick_map[j])
        ax.set_yticks(tick_map[i])
        ax.set_xticklabels([str(v) for v in tick_map[j]])
        ax.set_yticklabels([str(v) for v in tick_map[i]])

for i, axd in enumerate(g.diag_axes):
    axd.set_xticks(tick_map[i])
    axd.set_xticklabels([str(v) for v in tick_map[i]])

g.savefig(f'{FIGURE_STORE_DIR}/figure_2_d.svg', bbox_inches='tight')

"""
Concept Figure: Figure 1
Plot the data + x/y Error bars vs. the true model:
"""
FONT_SIZE_TICKS = 18
FONT_SIZE_AXES  = 22
FONT_SIZE_TITLE = 26

LINE_WIDTH = 3          
CAP_SIZE   = 6            
CAP_THICK  = LINE_WIDTH 


plt.figure(figsize=(8, 6))
t_sim =  np.linspace(0, T_PLOT_MAX, 100)

# ground truth data
x_sim = [np.array([exp(-THETA_TRUE * t) for t in t_sim])]
plt.plot(
    t_sim, x_sim[0],
    label='Ground Truth',
    color=COLOR_GROUND_TRUTH,
    linestyle='--',
    linewidth=LINE_WIDTH
)

# plot data
plt.scatter(
    T_WRONG, x_data,
    label='Reported Data',
    color=COLOR_WRONG_TP,
    s=80  # marker size; independent of line thickness
)
plt.errorbar(
    T_WRONG, x_data,
    xerr=SIGMA_T, yerr=None,
    fmt='none',                   # no marker here (marker is the scatter above)
    ecolor=COLOR_WRONG_TP,
    elinewidth=LINE_WIDTH,
    capsize=CAP_SIZE,
    capthick=CAP_THICK,
    label='Temp. Uncertainty'
)
plt.errorbar(
    T_WRONG, x_data,
    xerr=None, yerr=SIGMA_X,
    fmt='none',
    ecolor=COLOR_WRONG_TP,
    elinewidth=LINE_WIDTH,
    capsize=CAP_SIZE,
    capthick=CAP_THICK,
    label='Msmt. Uncertainty'
)

# plt.scatter(T_WRONG, x_data, label='Data', color=COLOR_WRONG_TP)
# plt.errorbar(T_WRONG, x_data, yerr=SIGMA_X, fmt='o', color=COLOR_WRONG_TP, capsize=5)
# plt.errorbar(T_WRONG, x_data, xerr=SIGMA_T, fmt='o', color=COLOR_WRONG_TP, capsize=5)

# plot true time points
plt.plot([t_true[0], t_true[0]], [0, 1.1], label='True Time Points', color=COLOR_WRONG_TP, linestyle='--', linewidth=LINE_WIDTH)
plt.plot([t_true[1], t_true[1]], [0, 1.1], color=COLOR_WRONG_TP,  linestyle='--', linewidth=LINE_WIDTH)


plt.xlabel('Time', fontsize=FONT_SIZE_AXES)
plt.ylabel(r'$x(t)$', fontsize=FONT_SIZE_AXES)
plt.title('Ground Truth and Data', fontsize=FONT_SIZE_TITLE)
plt.legend(fontsize=FONT_SIZE_AXES)

plt.xlim(0, T_PLOT_MAX)
plt.ylim(0, 1.1)

ax = plt.gca()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_linewidth(LINE_WIDTH)
ax.spines['bottom'].set_linewidth(LINE_WIDTH)

ax.tick_params(axis='both', labelsize=FONT_SIZE_TICKS, width=LINE_WIDTH)

# now let's order the legend....
# Get current handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Desired order of groups
order = [
    "Ground Truth",
    "Reported Data",
    "Temp. Uncertainty",
    "Msmt. Uncertainty",
    "True Time Points",
]

# Reorder handles + labels
ordered_handles = [handles[labels.index(o)] for o in order]
ordered_labels  = order

# Apply to legend
plt.legend(ordered_handles, ordered_labels, fontsize=FONT_SIZE_AXES)


# plt.savefig(f'{image_dir}/ground_truth.png', dpi=600)

"""
Figure 1B: Likelihood for both data points...
"""
plt.figure(figsize=(8, 2.5))


# p(x|tau, theta) = compute the Extended Likelihood given tau
p_x_given_tau_and_theta = [list(), list()] # contains the PDFs for both data points
for tau in t_sim:

    x = np.exp(-THETA_TRUE * tau)
    pdf = 1/(np.sqrt(2*np.pi)*SIGMA_X) * np.array(
        [np.exp(-(x_data[0] - x)**2/(2*SIGMA_X**2)), 
        np.exp(-(x_data[1] - x)**2/(2*SIGMA_X**2))]
        )

    p_x_given_tau_and_theta[0].append(pdf[0])
    p_x_given_tau_and_theta[1].append(pdf[1])

# plot them in COLOR_FULL
plt.plot(
    t_sim, p_x_given_tau_and_theta[0],
    label=r'$p(\bar{y}_1|\tau, \theta)$',
    color=COLOR_FULL,
    linewidth=LINE_WIDTH
)
plt.plot(
    t_sim, p_x_given_tau_and_theta[1],
    label=r'$p(\bar{y}_2|\tau, \theta)$',
    color=COLOR_MARGINALIZED,
    linewidth=LINE_WIDTH
)

plt.xlabel(r'$\tau$', fontsize=FONT_SIZE_AXES)
plt.ylabel(r'$p(D|\tau, \theta)$', fontsize=FONT_SIZE_AXES)
plt.title('Full Likelihood', fontsize=FONT_SIZE_TITLE)
plt.legend(fontsize=FONT_SIZE_AXES, loc='lower right')

plt.xlim(0, T_PLOT_MAX)
plt.gca().set_ylim(bottom=0)

# style the axis appearance
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Thicken visible spines + ticks using the master knob
ax.spines['left'].set_linewidth(LINE_WIDTH)
ax.spines['bottom'].set_linewidth(LINE_WIDTH)
ax.tick_params(axis='both', which='both', width=LINE_WIDTH, labelsize=FONT_SIZE_TICKS)

# plt.savefig(f'{image_dir}/extended_likelihood.png', dpi=600)

"""
Figure 1C: Prior for both data points...
"""
plt.figure(figsize=(8, 2.))

def prior(tau, t_wrong):
    return 1/(np.sqrt(2*np.pi)*SIGMA_T) * np.exp(-0.5*(tau - t_wrong)**2/SIGMA_T**2)

p_tau = [
    [prior(tau, T_WRONG[0]) for tau in t_sim],
    [prior(tau, T_WRONG[1]) for tau in t_sim]
]

# plot  in COLOR_FULL
plt.plot(
    t_sim, p_tau[0],
    label=r'$p(\tau_1)$',
    color=COLOR_FULL,
    linewidth=LINE_WIDTH
)
plt.plot(
    t_sim, p_tau[1],
    label=r'$p(\tau_2)$',
    color=COLOR_MARGINALIZED,
    linewidth=LINE_WIDTH
)

# labels & co
plt.xlabel(r'$\tau$', fontsize=FONT_SIZE_AXES)
plt.ylabel(r'$p(\tau)$', fontsize=FONT_SIZE_AXES)
plt.title('Prior', fontsize=FONT_SIZE_TITLE)

plt.legend(fontsize=FONT_SIZE_AXES, loc='upper right')

plt.xlim(0, T_PLOT_MAX)
plt.ylim(0, 1.1)

# style the axis appearance
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Thicken visible spines + ticks using master knob
ax.spines['left'].set_linewidth(LINE_WIDTH)
ax.spines['bottom'].set_linewidth(LINE_WIDTH)
ax.tick_params(axis='both', which='both', width=LINE_WIDTH, labelsize=FONT_SIZE_TICKS)

# plt.savefig(f'{image_dir}/prior_tau.png', dpi=600)

"""
Figure 1D: Posterior (i.e. p(\\tau|x, \\theta) = p(x|\tau, theta) * p(theta)) 
for both data points...
"""
plt.figure(figsize=(8, 6))

extended_posterior = [
    [p_x_given_tau_and_theta[0][i] * p_tau[0][i] for i in range(len(t_sim))],
    [p_x_given_tau_and_theta[1][i] * p_tau[1][i] for i in range(len(t_sim))]
]

# plot  in COLOR_FULL
plt.fill_between(
    t_sim, extended_posterior[0],
    alpha=0.6,
    label=r'$p(\bar{y}_1, \tau|\theta)$',
    color=COLOR_FULL
)
plt.fill_between(
    t_sim, extended_posterior[1],
    alpha=0.6,
    label=r'$p(\bar{y}_2, \tau|\theta)$',
    color=COLOR_MARGINALIZED
)

# labels & co
plt.xlabel(r'$\tau$', fontsize=FONT_SIZE_AXES)
plt.ylabel(r'$\dot z(\tau)$', fontsize=FONT_SIZE_AXES)
# plt.title(r"$\dot z_i(\tau) = p(\bar{y}_i|\tau, \theta) \cdot p(\tau)$", fontsize=FONT_SIZE_TITLE)
plt.title(r"Derivative of $z_i(\tau)$", fontsize=FONT_SIZE_TITLE)

plt.legend(fontsize=FONT_SIZE_AXES)

plt.xlim(0, T_PLOT_MAX)
plt.gca().set_ylim(bottom=0)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Thicken visible spines + ticks
ax.spines['left'].set_linewidth(LINE_WIDTH)
ax.spines['bottom'].set_linewidth(LINE_WIDTH)
ax.tick_params(axis='both', which='both', width=LINE_WIDTH, labelsize=FONT_SIZE_TICKS)

# plt.savefig(f'{image_dir}/posterior_tau.png', dpi=600)

"""
Figure 1E: Cumulative integral of p(x|tau, theta) for both data points...
"""
from scipy.integrate import cumulative_trapezoid

plt.figure(figsize=(8, 6))

# -----------------------
# Compute cumulative integrals of p_x_given_tau_and_theta
# -----------------------
cum_int_0 = cumulative_trapezoid(extended_posterior[0], x=t_sim, initial=0)
cum_int_1 = cumulative_trapezoid(extended_posterior[1], x=t_sim, initial=0)

# -----------------------
# Plot as line plots (not filled)
# -----------------------
plt.plot(
    t_sim, cum_int_0,
    label=r'$z_1(\tau)$',
    color=COLOR_FULL,
    linewidth=LINE_WIDTH
)
plt.plot(
    t_sim, cum_int_1,
    label=r'$z_2(\tau)$',
    color=COLOR_MARGINALIZED,
    linewidth=LINE_WIDTH
)

# -----------------------
# Labels, limits, legend
# -----------------------
plt.xlabel(r'$\tau$', fontsize=FONT_SIZE_AXES)
plt.ylabel(r'$z_i(\tau)$', fontsize=FONT_SIZE_AXES)
# plt.title(r'$z_i(\tau)$ for $\theta=1$', fontsize=FONT_SIZE_TITLE)
plt.title(r'Trajectory of $z_i(\tau)$', fontsize=FONT_SIZE_TITLE)

plt.legend(fontsize=FONT_SIZE_AXES)
plt.xlim(0, T_PLOT_MAX)

# -----------------------
# Axes cosmetics (local only)
# -----------------------
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(LINE_WIDTH)
ax.spines['bottom'].set_linewidth(LINE_WIDTH)
ax.tick_params(axis='both', which='both', width=LINE_WIDTH, labelsize=FONT_SIZE_TICKS)

# plt.savefig(f'{image_dir}/cumulative_integral.png', dpi=600)