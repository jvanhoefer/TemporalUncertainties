"""
Create Figure No. 5 - Bruno application.
"""
import pypesto
import pypesto.store as store
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rows = []



RESULT_DIR = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/results"
SIGMA_TAU = 0.0
N_DATA = 77
N_PARAMETERS_WRONG_TP = 13
N_PARAMETERS_MARGINALIZED = N_PARAMETERS_WRONG_TP + 1





COLOR_FULL = [0, 78/255, 159/255] # blue
COLOR_MARGINALIZED = [252/255, 186/255, 0] # orange
COLOR_WRONG_TP = [144/255, 144/255, 133/255] # grey
COLOR_GROUND_TRUTH = [0, 0, 0] # black

COLORS = [COLOR_WRONG_TP, COLOR_FULL, COLOR_MARGINALIZED]

FONT_SIZE_TICKS = 18 # 14
FONT_SIZE_AXES  = 22 # 16
FONT_SIZE_TITLE = 26 # 20

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
    "marginalized_shared_tau",
    "full_multi_tau",
    "full_shared_tau",
    "wrong_tp",
]


def parse_filename(filename: str) -> dict:
    stem = filename.replace(".hdf5", "")
    if stem.startswith("synthetic_data"):
        data_method, is_synthetic = "synthetic_data", True
    elif stem.startswith("copied_data"):
        data_method, is_synthetic = "copied_data", False
    elif stem == "sigma_tau_0_0":
        data_method, is_synthetic = "ground_truth", None
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


# --- Load existing CSV or start fresh ---
if os.path.isfile(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded existing CSV with {len(df)} rows.")
else:
    df = pd.DataFrame()
    print("No existing CSV found, starting fresh.")

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

df["AIC_original"]  = 2 * df["wrong_tp"]          + (2 * N_PARAMS_WRONG_TP)

df["AIC_marginal"]  = 2 * df["marginalized_multi_tau"].fillna(df["marginalized_shared_tau"]) \
                         + (2 * N_PARAMS_MARGINALIZED)

df["AIC_full"] = np.where(
    df["timepoints"] == "multi_timepoints",
    2 * df["full_multi_tau"]  + (2 * N_PARAMS_FULL_MULTI),
    np.where(
        df["timepoints"] == "shared_timepoints",
        2 * df["full_shared_tau"] + (2 * N_PARAMS_FULL_SHARED),
        np.nan  # ground truth row has no timepoints
    )
)

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

df.to_csv(CSV_PATH, index=False)
print(df[["result_name", "sigma"]].to_string(index=False))

df.to_csv(CSV_PATH, index=False)
print("AIC columns added and CSV saved.")
print(df[["result_name", "AIC_original", "AIC_marginal", "AIC_full"]].to_string(index=False))

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

# --- Filter: copied_data + multi_timepoints ---
mask = (df["data_method"] == "copied_data") & (df["timepoints"] == "multi_timepoints")
plot_df = df[mask].sort_values("sigma_tau")

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(plot_df["sigma_tau"], plot_df["AIC_original"],
        label="Original (wrong TP)", color=COLOR_ORIGINAL,
        marker="o", linewidth=LINE_WIDTH)

ax.plot(plot_df["sigma_tau"], plot_df["AIC_marginal"],
        label="Marginalized", color=COLOR_MARGINALIZED,
        marker="o", linewidth=LINE_WIDTH)
"""
ax.plot(plot_df["sigma_tau"], plot_df["AIC_full"],
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

plt.tight_layout()
plt.show()

from scipy.stats import chi2

df_lr = df[
    (df["is_synthetic"] == True) & 
    (df["timepoints"] == "shared_timepoints") & 
    (df["marginalized_multi_tau"].notna())
].copy()
df_lr = df_lr.sort_values('sigma')

# df_lr["LR_statistic"] = 2 * (df_lr["wrong_tp"] - df_lr["marginalized_multi_tau"])
df_lr["LR_statistic"] = 2 * (df_lr["wrong_tp"] - df_lr["marginalized_shared_tau"])
df_lr["p_value"] = chi2.sf(df_lr["LR_statistic"], df=1)

# plt.plot(df_lr["sigma"], np.log(df_lr["LR_statistic"]))
plt.plot(df_lr["sigma"], np.log(df_lr["p_value"]))