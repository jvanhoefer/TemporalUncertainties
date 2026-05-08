"""
Create Figure No. 6 - Bruno application - Model selection.
"""
import pypesto
import pypesto.store as store
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rows = []

RESULT_DIR = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/results"
FIGURE_STORE_DIR = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Figures Paper/building_blocks"
SIGMA_TAU = 0.0
N_DATA = 77
N_PARAMETERS_WRONG_TP = 13
N_PARAMETERS_MARGINALIZED = N_PARAMETERS_WRONG_TP + 1


DATAFORMAT = "svg"


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
Extend an existing nll_summary.csv with any missing file x folder combinations.
"""
import os
import re
import numpy as np
import pandas as pd
import pypesto.store as store

RESULT_DIR = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/results"
CSV_PATH = os.path.join(RESULT_DIR, "nll_summary.csv")

FOLDERS = [
    "marginalized_multi_tau",
    # "marginalized_shared_tau",
    "full_multi_tau",
    # "full_shared_tau",
    "wrong_tp",
]

best_params = {folder: {} for folder in FOLDERS}

def parse_filename(filename: str) -> dict:
    stem = filename.replace(".hdf5", "")
    if stem.startswith("synthetic_data"):
        data_method, is_synthetic = "synthetic_data", True
    elif stem.startswith("copied_data"):
        data_method, is_synthetic = "copied_data", False
    elif stem == "sigma_tau_0_0":
        data_method, is_synthetic = "ground_truth", False
    else:
        data_method, is_synthetic = "unknown", None

    timepoints = (
        "multi_timepoints" if "multi_timepoints" in stem
        else "shared_timepoints" if "shared_timepoints" in stem
        else None
    )
    match = re.search(r"sigma_tau_(\d+)$", stem)
    sigma_tau = int(match.group(1)) if match else None

    return {"data_method": data_method, "is_synthetic": is_synthetic,
            "timepoints": timepoints, "sigma_tau": sigma_tau}


df = pd.DataFrame()

# --- Collect all filenames currently on disk ---
all_filenames: set[str] = set()
for folder in FOLDERS:
    folder_path = os.path.join(RESULT_DIR, folder)
    if os.path.isdir(folder_path):
        all_filenames |= {f for f in os.listdir(folder_path) if f.endswith(".hdf5")}
    else:
        print(f"Warning: folder not found: {folder_path}")

all_filenames = sorted(all_filenames)

# --- Determine which rows/columns are missing ---
existing_filenames = set(df["result_name"].tolist()) if "result_name" in df.columns else set()
new_filenames = set(all_filenames) - existing_filenames

# Add any missing folder columns to the existing df
for folder in FOLDERS:
    if folder not in df.columns:
        df[folder] = np.nan
        print(f"Added missing column: {folder}")

# --- Fill missing fvals for existing rows (new folder columns) ---
for idx, row in df.iterrows():
    for folder in FOLDERS:
        if pd.isna(df.at[idx, folder]):
            filepath = os.path.join(RESULT_DIR, folder, row["result_name"])
            if os.path.isfile(filepath):
                try:
                    result = store.read_result(filepath)
                    df.at[idx, folder] = result.optimize_result.list[0].fval
                    best_params[folder][row["result_name"]] = result.optimize_result.list[0].x
                    print(f"  Filled {folder}/{row['result_name']}")
                except Exception as e:
                    print(f"  Could not read {folder}/{row['result_name']}: {e}")

# --- Append entirely new filenames ---
if new_filenames:
    print(f"\nFound {len(new_filenames)} new result files to add.")
    new_rows = []
    for filename in sorted(new_filenames):
        meta = parse_filename(filename)
        row = {"result_name": filename, **meta}
        for folder in FOLDERS:
            filepath = os.path.join(RESULT_DIR, folder, filename)
            if not os.path.isfile(filepath):
                row[folder] = np.nan
            else:
                try:
                    result = store.read_result(filepath)
                    row[folder] = result.optimize_result.list[0].fval
                    best_params[folder][filename] = result.optimize_result.list[0].x
                except Exception as e:
                    print(f"  Could not read {folder}/{filename}: {e}")
                    row[folder] = np.nan
        new_rows.append(row)
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
else:
    print("\nNo new result files found.")

# --- Compute AIC columns ---
N_PARAMS_WRONG_TP   = 13
N_PARAMS_MARGINALIZED = 14
N_PARAMS_FULL_MULTI  = 91
N_PARAMS_FULL_SHARED = 56

df["AIC_wrong_tp"]  = 2 * df["wrong_tp"] + 2 * N_PARAMS_WRONG_TP

df["AIC_marginalized_multi_tau"]  = 2 * df["marginalized_multi_tau"] + 2 * N_PARAMS_MARGINALIZED
# df["AIC_marginalized_shared_tau"] = 2 * df["marginalized_shared_tau"] + 2 * N_PARAMS_MARGINALIZED


# df["AIC_full_multi_tau"] = 2 * df["full_multi_tau"]  + (2 * N_PARAMS_FULL_MULTI)
# df["AIC_full_shared_tau"] = 2 * df["full_shared_tau"] + (2 * N_PARAMS_FULL_SHARED)



# df["BIC_wrong_tp"]  = df["wrong_tp"] + (N_PARAMS_WRONG_TP * np.log(N_DATA))

# df["BIC_marginalized_multi_tau"]  = df["marginalized_multi_tau"] +(N_PARAMS_MARGINALIZED * np.log(N_DATA))
# df["BIC_marginalized_shared_tau"] = df["marginalized_shared_tau"] +(N_PARAMS_MARGINALIZED * np.log(N_DATA))

# df["BIC_full_multi_tau"] = df["full_multi_tau"]  + (N_PARAMS_FULL_MULTI * np.log(N_DATA))
# df["BIC_full_shared_tau"] = df["full_shared_tau"] + (N_PARAMS_FULL_SHARED * np.log(N_DATA))


def parse_sigma_tau(filename: str) -> float:
    stem = filename.replace(".hdf5", "")
    
    # Special case: ground truth sigma_tau_0_0
    if stem.endswith("_0_0"):
        return 0.0
    # Decimal case: _0_5 => 0.5
    if re.search(r"_0_\d+$", stem):
        match = re.search(r"_0_(\d+)$", stem)
        return float(f"0.{match.group(1)}")
    # Integer case: _5 => 5.0
    match = re.search(r"_(\d+)$", stem)
    if match:
        return float(match.group(1))
    return np.nan

df["sigma"] = df["result_name"].apply(parse_sigma_tau)
if "sigma_tau" in df.columns:
    df.drop(columns=["sigma_tau"], inplace=True)

# drop all rows, where marginalized_multi_tau is NaN 
df = df[df["marginalized_multi_tau"].notna()]

df.to_csv(CSV_PATH, index=False)
"""
# --- Save ---
COLUMNS = ["result_name", "data_method", "is_synthetic", "timepoints", "sigma_tau"] + FOLDERS
df = df[COLUMNS]
df = df.sort_values("result_name").reset_index(drop=True)
df.to_csv(CSV_PATH, index=False)
print(f"\nSaved updated CSV to: {CSV_PATH}")
print(df.to_string(index=False))
"""

COLOR_ORIGINAL    = [144/255, 144/255, 133/255]  # grey
COLOR_MARGINALIZED = [252/255, 186/255, 0]        # orange
COLOR_FULL        = [0, 78/255, 159/255]          # blue

FONT_SIZE_TICKS = 18
FONT_SIZE_AXES  = 22
FONT_SIZE_TITLE = 26
LINE_WIDTH = 3


# copied_data + multi_timepoints
mask = (
    (df["data_method"] == "copied_data") & (df["timepoints"] == "multi_timepoints")
)
plot_df_copied_multi = df[mask].sort_values("sigma")


# copied_data + shared_timepoints
mask = (
    (df["data_method"] == "copied_data") & (df["timepoints"] == "shared_timepoints")
)
plot_df_copied_shared = df[mask].sort_values("sigma")

# synthetic_data + multi_timepoints
mask = (
    (df["data_method"] == "synthetic_data") & (df["timepoints"] == "multi_timepoints")
)
plot_df_synthetic_multi = df[mask].sort_values("sigma")

# synthetic_data + shared_timepoints
mask = (
    (df["data_method"] == "synthetic_data") & (df["timepoints"] == "shared_timepoints")
)
plot_df_synthetic_shared = df[mask].sort_values("sigma")

for plot_df in [plot_df_copied_multi, plot_df_copied_shared, plot_df_synthetic_multi, plot_df_synthetic_shared]:
    print(10*"---")

    fig, ax = plt.subplots(figsize=(8, 6))
    """
    ax.plot(
        plot_df["sigma"], 
        plot_df["AIC_wrong_tp"].to_numpy() - plot_df["AIC_marginalized_multi_tau"].to_numpy(),
            label="Marginalized Approach", color=COLOR_MARGINALIZED,
            marker="o", linewidth=LINE_WIDTH)

    ax.plot(
        plot_df["sigma"], 
        plot_df["AIC_wrong_tp"].to_numpy() - plot_df["AIC_marginalized_shared_tau"].to_numpy(),
            label=r"Marginalized Approach - shared $\tau$s", color=COLOR_MARGINALIZED,
            linestyle="--", 
            marker="o", linewidth=LINE_WIDTH)
    """
    ax.plot(
        plot_df["sigma"], 
        plot_df["BIC_marginalized_multi_tau"].to_numpy() - plot_df["BIC_marginalized_shared_tau"].to_numpy(),
            label=r"$BIC_{multi} - BIC_{shared}$", color=COLOR_MARGINALIZED,
            linestyle="--", 
            marker="o", linewidth=LINE_WIDTH)

    """
    ax.plot(
        plot_df["sigma"],
        plot_df["AIC_wrong_tp"].to_numpy() - plot_df["AIC_full_multi_tau"].to_numpy(),
            label="Joint Approach", color=COLOR_FULL,
            marker="o", linewidth=LINE_WIDTH)
    """
    ax.plot([0, 16], [0, 0], color=COLOR_GROUND_TRUTH, linestyle="--", linewidth=LINE_WIDTH)

    ax.set_xlabel(r"$\sigma_\tau$", fontsize=FONT_SIZE_AXES)
    ax.set_ylabel(r"$\Delta$ BIC", fontsize=FONT_SIZE_AXES)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(fontsize=FONT_SIZE_TICKS)
    ax.set_xlim((0, 16))
    plt.tight_layout()

"""
ax.plot(plot_df["sigma"], plot_df["AIC_marginalized_multi_tau"],
        label="Marginalized", color=COLOR_MARGINALIZED,
        marker="o", linewidth=LINE_WIDTH)

ax.plot(plot_df["sigma"], plot_df["AIC_full_multi_tau"],
        label="Full", color=COLOR_FULL,
        marker="o", linewidth=LINE_WIDTH)
"""
ax.set_xlabel(r"$\sigma_\tau$", fontsize=FONT_SIZE_AXES)
ax.set_ylabel("AIC", fontsize=FONT_SIZE_AXES)
ax.set_title("AIC — copied data, multi timepoints", fontsize=FONT_SIZE_TITLE)
ax.tick_params(labelsize=FONT_SIZE_TICKS)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.legend(fontsize=FONT_SIZE_TICKS)
# ax.set_ylim((-1, None))
ax.set_xlim((0, 16))
plt.tight_layout()
plt.show()

"""
FUTURE FIGURE 6A: Plot the l2 distance for the best parameter fit vs. the original parameters
"""
import re
ref = best_params["wrong_tp"]["sigma_tau_0_0.hdf5"][:13]

mse_distances = {}

for folder, files in best_params.items():
    mse_distances[folder] = {
        key: float(np.linalg.norm(val[:13] - ref, ord=2))**2
        for key, val in files.items()
        if key.startswith("copied_data_multi_timepoints")
    }


def extract_sigma_tau(key):
    stem = key.replace(".hdf5", "")
    match = re.search(r"sigma_tau_([\d]+(?:_[\d]+)?)$", stem)
    return float(match.group(1).replace("_", ".")) if match else None

dfs = {}
for folder, distances in mse_distances.items():
    # Filter copied_data_multi_timepoints entries
    filtered = {k: v for k, v in distances.items() 
                if k.startswith("copied_data_multi_timepoints")}
    
    # Group by sigma_tau
    grouped = {}
    for key, val in filtered.items():
        sigma_tau = extract_sigma_tau(key)
        if sigma_tau not in grouped:
            grouped[sigma_tau] = []
        grouped[sigma_tau].append(val)
    
    # Build DataFrame: rows = sigma_tau, columns = replicates
    _df = pd.DataFrame.from_dict(grouped, orient="index")
    _df.index.name = "sigma_tau"
    _df.columns = [f"replicate_{i+1}" for i in range(_df.shape[1])]
    _df = _df.sort_index()
    
    dfs[folder] = _df

# Find sigma_tau values present in all folders
common_sigma_taus = set.intersection(*[set(_df.index) for _df in dfs.values()])

# Filter each df to only keep common sigma_tau values
dfs = {folder: df.loc[sorted(common_sigma_taus)] for folder, df in dfs.items()}

"""
Figure 6 A:
"""

fig, ax = plt.subplots(figsize=(4, 4))

FOLDER_STYLES = {
    "wrong_tp":               (COLOR_WRONG_TP,     "Standard\nApproach",      0.15),
    # "full_multi_tau":         (COLOR_FULL,         "Joint",         0.0),
    "marginalized_multi_tau": (COLOR_MARGINALIZED, "Marginalized\nApproach", -0.15),
}

for folder, (color, label, offset) in FOLDER_STYLES.items():
    df = dfs[folder][dfs[folder].index > 0]
    x      = df.index.values
    median = df.median(axis=1).values

    ax.plot(x, median, color=color, linewidth=LINE_WIDTH, label=label)

    for sigma_tau, row in df.iterrows():
        bp = ax.boxplot(row.values, positions=[sigma_tau + offset], widths=0.25,
                        patch_artist=True, manage_ticks=False)
        bp['boxes'][0].set_facecolor(color)
        # bp['boxes'][0].set_alpha(0.6)
        for element in ['whiskers', 'caps', 'medians', 'fliers']:
            for item in bp[element]:
                item.set_color(color)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel(r"$\sigma_t$", fontsize=FONT_SIZE_AXES)
ax.set_ylabel("MSE of MAP", fontsize=FONT_SIZE_AXES)
ax.tick_params(labelsize=FONT_SIZE_TICKS)
ax.legend(fontsize=FONT_SIZE_TICKS, loc="upper left", frameon=False)
ax.set_yticks([0.01, 0.02, 0.03, 0.04])
ax.set_xlim((0, 16))
ax.set_ylim((0, 0.045))
# ax.set_title(r"$||\hat{\theta} - \theta_{true}||^2_2$", fontsize=FONT_SIZE_TITLE)


plt.tight_layout()
plt.savefig(f"{FIGURE_STORE_DIR}/figure_6_a.{DATAFORMAT}", bbox_inches="tight")
plt.show()

"""
Figure 6 B
"""
fig, ax = plt.subplots(figsize=(4,4))

# Collect (sigma_tau_true, sigma_tau_fitted) pairs
sigma_true_list  = []
sigma_fitted_list = []

files_marginalized = set(best_params["marginalized_multi_tau"].keys())

for fname in sorted(files_marginalized):
    if not fname.startswith("copied_data_multi_timepoints"):
        continue
    sigma_true  = extract_sigma_tau(fname)
    sigma_fitted = best_params["marginalized_multi_tau"][fname][13]  # last parameter
    sigma_true_list.append(sigma_true)
    sigma_fitted_list.append(sigma_fitted)

sigma_true_arr   = np.array(sigma_true_list)
sigma_fitted_arr = np.array(sigma_fitted_list)

# exclude sigma_tau = 0
mask = sigma_true_arr > 0
sigma_true_arr   = sigma_true_arr[mask]
sigma_fitted_arr = sigma_fitted_arr[mask]

# --- Boxplot per sigma_tau_true value ---
unique_sigmas = sorted(set(sigma_true_arr))
for s in unique_sigmas:
    mask = sigma_true_arr == s
    fitted_vals = sigma_fitted_arr[mask]
    bp = ax.boxplot(fitted_vals, positions=[s], widths=0.4,
                    patch_artist=True, manage_ticks=False)
    bp['boxes'][0].set_facecolor(COLOR_MARGINALIZED)
    # bp['boxes'][0].set_alpha(0.6)
    for element in ['whiskers', 'caps', 'medians', 'fliers']:
        for item in bp[element]:
            item.set_color(COLOR_MARGINALIZED)

# Median line
medians = [np.median(sigma_fitted_arr[sigma_true_arr == s]) for s in unique_sigmas]
ax.plot(unique_sigmas, medians, color=COLOR_MARGINALIZED,
        linewidth=LINE_WIDTH, label="Marginalized")

# --- Ground truth diagonal ---
lim = [-0.5, max(unique_sigmas) + 1]
ax.plot(lim, lim, color=COLOR_GROUND_TRUTH, linestyle="--",
        linewidth=LINE_WIDTH)

# Text on top of the line, top-right area, rotated 45°
ax.text(
    max(unique_sigmas) * 0.79,   # x position — adjust to taste
    max(unique_sigmas) * 0.82,  # y position — slightly above the line
    r"$\hat{\sigma}_t = \sigma_t$",
    fontsize=FONT_SIZE_AXES,
    color=COLOR_GROUND_TRUTH,
    rotation=45,
    rotation_mode="anchor",
    ha="left", va="bottom",
)

ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel(r"$\sigma_t (true)$", fontsize=FONT_SIZE_AXES)
ax.set_ylabel(r"$\hat{\sigma}_t (fitted)$", fontsize=FONT_SIZE_AXES)
ax.tick_params(labelsize=FONT_SIZE_TICKS)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# ax.legend(fontsize=FONT_SIZE_TICKS)
# ax.set_title(r"Fitted vs. true $\sigma_\tau$", fontsize=FONT_SIZE_TITLE)

plt.tight_layout()
plt.savefig(f"{FIGURE_STORE_DIR}/figure_6_b.{DATAFORMAT}", bbox_inches="tight")
plt.show()


"""
Figure 6C: LR Test
"""
from scipy.stats import chi2

df_lr = plot_df_copied_multi.copy()
df_lr["LR_statistic_marginalized"] = 2 * (df_lr["wrong_tp"] - df_lr["marginalized_multi_tau"])
df_lr["p_value_marginalized"] = chi2.sf(df_lr["LR_statistic_marginalized"], df=1)

unique_sigmas = sorted(df_lr["sigma"].unique())
unique_sigmas = [s for s in unique_sigmas if s > 0]

fig, ax = plt.subplots(figsize=(4, 4))

medians = []
for s in unique_sigmas:
    vals = df_lr.loc[df_lr["sigma"] == s, "p_value_marginalized"].values
    bp = ax.boxplot(vals, positions=[s], widths=0.4,
                    patch_artist=True, manage_ticks=False)
    bp['boxes'][0].set_facecolor(COLOR_MARGINALIZED)
    # bp['boxes'][0].set_alpha(0.6)
    for element in ['whiskers', 'caps', 'medians', 'fliers']:
        for item in bp[element]:
            item.set_color(COLOR_MARGINALIZED)
    medians.append(np.median(vals))

ax.plot(unique_sigmas, medians, color=COLOR_MARGINALIZED,
        linewidth=LINE_WIDTH, label="Marginalized\nApproach")

ax.axhline(0.05, color=COLOR_GROUND_TRUTH, linestyle="--",
           linewidth=LINE_WIDTH)  # no label

ax.text(
    14.5,   # x — near the right edge, adjust to taste
    0.07,   # y — slightly above the line
    r"$p = 0.05$",
    fontsize=FONT_SIZE_AXES,
    color=COLOR_GROUND_TRUTH,
    ha="right", va="bottom",
)

ax.set_xlabel(r"$\sigma_t$", fontsize=FONT_SIZE_AXES)
ax.set_ylabel("p-value", fontsize=FONT_SIZE_AXES)
ax.set_xlim((0, 16))
ax.set_ylim((-0.05, 1.02))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# ax.set_title("p-value w.r.t LR test", fontsize=FONT_SIZE_TITLE)
ax.tick_params(labelsize=FONT_SIZE_TICKS)
# ax.legend(fontsize=FONT_SIZE_TICKS)

plt.tight_layout()
plt.savefig(f"{FIGURE_STORE_DIR}/figure_6_c.{DATAFORMAT}", bbox_inches="tight")
plt.show()

"""
Figure 6D: Rejection rate with the LR test
"""

df_lr = plot_df_copied_multi.copy()
df_lr["LR_statistic_marginalized"] = 2 * (df_lr["wrong_tp"] - df_lr["marginalized_multi_tau"])
df_lr["p_value_marginalized"] = chi2.sf(df_lr["LR_statistic_marginalized"], df=1)
df_lr["rejected"] = df_lr["p_value_marginalized"] < 0.05

rejection_rate = 100 * df_lr.groupby("sigma")["rejected"].mean()  # fraction in [0, 1]
rejection_rate = rejection_rate[rejection_rate.index > 0]

fig, ax = plt.subplots(figsize=(4, 4))

ax.plot(rejection_rate.index, rejection_rate.values,
        color=COLOR_MARGINALIZED, linewidth=LINE_WIDTH, marker="o",
        label="Marginalized\nApproach")

ax.set_xlabel(r"$\sigma_t$", fontsize=FONT_SIZE_AXES)
ax.set_ylabel("Rejection rate [%]", fontsize=FONT_SIZE_AXES)
ax.set_xlim((0, 16))
ax.set_ylim((-0.5, 102))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(labelsize=FONT_SIZE_TICKS)
# ax.legend(fontsize=FONT_SIZE_TICKS)
# ax.set_title("Rejection rate of the LR test", fontsize=FONT_SIZE_TITLE)

plt.tight_layout()
plt.savefig(f"{FIGURE_STORE_DIR}/figure_6_d.{DATAFORMAT}", bbox_inches="tight")
plt.show()

