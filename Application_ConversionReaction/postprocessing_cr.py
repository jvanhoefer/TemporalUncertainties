"""
This file does all the post-processing for Figure 4 in the paper:

In simulation_cr, we ran the simulation for different numbers of time points and
different random seeds. Those results are stored in 

        "res_<n_experiment>/measurement_setup.yaml".

In this file, we load those results, and create a common data frame with all the
results, and store them.

There are two dfs:

- per_experiment_df: 
    Every column corresponds to one experiment (i.e. "res_<n_experiment>").
    The rows correspond to entries of the .yaml file (e.g. "ESS_full", 
    "random_seed" or "run_time_optimization_full").
- per_setup_df:
    Every column corresponds to one measurement setup (i.e. one fixed number of
    time points = data points). The rows correspond to keys of "per_experiment_df"
    FOR A GIVEN RANDOM_SEED (i.e. the setup w. 5 data points was done for various
    random seeds. One Row are all of the "ESS_full", "random_seed", 
    "run_time_optimization_full"... for all of the random seeds.) The keys are 
    "<original_key>_<random_seed>" 
    (e.g. "ESS_full_3", "run_time_optimization_marginalized_4")
"""
import re
from pathlib import Path
import h5py
import numpy as np
from mcmc_diagnostics import N_THETA, chain_diagnostics

def extract_values(file_path, keys):
    """
    Extracts values for given keys from a YAML-like text file using regex.

    Args:
        file_path (str or Path): Path to the text file.
        keys (list[str]): List of keys to extract.

    Returns:
        dict: Mapping {key: value}, where value is parsed to float/int if possible, otherwise left as string.
    """
    text = Path(file_path).read_text()

    results = {}
    for key in keys:
        # regex: match "<key>: <value>" until first newline
        pattern = rf"^{re.escape(key)}:\s*(.+)$"
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            raw_val = match.group(1).strip()
            # Try to cast to number, else keep string
            if raw_val.lower() in {"none", "null"}:
                results[key] = None
            else:
                try:
                    # int if no dot, else float
                    results[key] = int(raw_val) if raw_val.isdigit() else float(raw_val)
                except ValueError:
                    results[key] = raw_val
        else:
            results[key] = None  # key not found

    return results

FILE_PATH = Path(__file__).parent
# get all the res_folders in FILE_PATH
res_folders = sorted(p for p in FILE_PATH.iterdir() if p.is_dir() and p.name.startswith("res_"))


# Example usage
keys_to_find = [
        "n_data",
        "random_seed",
        "ESS_full", 
        "ESS_marginalized", 
        "ESS_per_second_full", 
        "ESS_per_second_marginalized",
        "geweke_burn_in_full", 
        "geweke_burn_in_marginalized", 
        "run_time_optimization_full", 
        "run_time_optimization_marginalized", 
        "run_time_sampling_full", 
        "run_time_sampling_marginalized"]


# ESS, ESS/s and burn-in in measurement_setup.yaml come from pyPESTO's Geweke
# test, which is miscalibrated (see mcmc_diagnostics.py). They are kept with a
# "pypesto_" prefix for reference and recomputed from the stored chains below.
PYPESTO_KEYS = [
        "ESS_full",
        "ESS_marginalized",
        "ESS_per_second_full",
        "ESS_per_second_marginalized",
        "geweke_burn_in_full",
        "geweke_burn_in_marginalized"]


run_dicts = {}
for res_i in res_folders:
    yaml_path = FILE_PATH / res_i.name / "measurement_setup.yaml"
    if not yaml_path.exists():
        # e.g. sampling crashed for this setup before the run finished
        # (measurement_setup.yaml is only written once, at the very end of
        # a setup's run) -- skip it rather than crash on a missing file.
        print(f"Warning: {res_i.name} has no measurement_setup.yaml (run did not finish) -- skipping")
        continue

    extracted_values = extract_values(str(yaml_path), keys_to_find)
    for key in PYPESTO_KEYS:
        extracted_values[f"pypesto_{key}"] = extracted_values.pop(key)

    extracted_values["converged_full"] = None
    extracted_values["converged_marginalized"] = None

    # Geweke z-score, burn-in and ESS on theta, recomputed from the stored chain
    for model in ["full", "marginalized"]:
        hdf5_path = FILE_PATH / res_i.name / f"result_{model}.hdf5"
        diagnostics = dict.fromkeys(
            ["geweke_z_score", "geweke_window_ess", "burn_in", "ESS", "ESS_per_second"])
        if hdf5_path.exists():
            try:
                with h5py.File(hdf5_path, "r") as f:
                    chain = f["sampling/results/trace_x"][0, :, :N_THETA]
                    sampling_time = float(f["sampling/results"].attrs["time"])
                diagnostics = chain_diagnostics(chain, sampling_time)
            except Exception as e:
                print(f"Warning: could not compute diagnostics for {res_i.name}/{model}: {e}")

        extracted_values[f"geweke_z_score_{model}"] = diagnostics["geweke_z_score"]  # max |z| over theta
        extracted_values[f"geweke_window_ess_{model}"] = diagnostics["geweke_window_ess"]
        extracted_values[f"geweke_burn_in_{model}"] = diagnostics["burn_in"]  # None = not converged
        extracted_values[f"ESS_{model}"] = diagnostics["ESS"]
        extracted_values[f"ESS_per_second_{model}"] = diagnostics["ESS_per_second"]

    run_dicts[res_i.name] = extracted_values

"""
run_dicts = {}
for res_i in res_folders: 
    extracted_values = extract_values(f"{FILE_PATH}/{res_i.name}/measurement_setup.yaml", keys_to_find)
    
    extracted_values["converged_full"] = None
    extracted_values["converged_marginalized"] = None
    
    run_dicts[res_i.name] = extracted_values
"""

# add leading zeros to the keys, so that they are sorted correctly
# i.e. res_1 -> res_001, res_10 -> res_010
run_dicts = {f"res_{int(k.split('_')[1]):03d}": v for k, v in run_dicts.items()}

# for all dicts, if the key starts with "ESS_full" or "ESS_marginalized" is None, 
# replace the ESS with 0
for k, v in run_dicts.items():
    if (v["ESS_full"] is None) or (v["ESS_per_second_full"] is None):
        v["ESS_full"] = 0
        v["ESS_per_second_full"] = 0
        v["converged_full"] = False
        
    if (v["ESS_marginalized"] is None) or (v["ESS_per_second_marginalized"] is None):
        v["ESS_marginalized"] = 0
        v["ESS_per_second_marginalized"] = 0
        v["converged_marginalized"] = False

# now let's have a look at all of the convergences
# (this was done via visual inspection). If the sampling is not converged,
# then the convergence is set to false:

ADD_CONVERGENCE_INFO = True

def _set_convergence(key, converged_full, converged_marginalized):
    # res_014 (and any setup whose run crashed/is missing from disk) has no
    # entry in run_dicts -- skip rather than KeyError.
    if key not in run_dicts:
        print(f"Warning: {key} not in run_dicts -- skipping hardcoded convergence info")
        return
    run_dicts[key]["converged_full"] = converged_full
    run_dicts[key]["converged_marginalized"] = converged_marginalized

if ADD_CONVERGENCE_INFO:
    _set_convergence("res_004", True, True)

    _set_convergence("res_005", True, True)

    _set_convergence("res_006", True, True)

    _set_convergence("res_007", True, True)

    _set_convergence("res_008", True, True)

    _set_convergence("res_009", False, True)

    _set_convergence("res_010", False, True)

    _set_convergence("res_011", False, True)

    _set_convergence("res_012", True, True)

    _set_convergence("res_013", False, True)

    _set_convergence("res_014", False, False)

    _set_convergence("res_015", False, False)

    _set_convergence("res_016", False, True)

    _set_convergence("res_017", False, True)

    _set_convergence("res_018", False, False)

    _set_convergence("res_019", False, True)

    _set_convergence("res_020", False, False)

    _set_convergence("res_021", False, True)

    _set_convergence("res_022", False, True)

    _set_convergence("res_023", False, True)

    _set_convergence("res_024", False, False)

    _set_convergence("res_025", False, True)

    _set_convergence("res_026", True, True)

    _set_convergence("res_027", True, True)

    _set_convergence("res_028", True, True)

    _set_convergence("res_029", True, True)

    _set_convergence("res_030", True, True)

    _set_convergence("res_031", False, True)

    _set_convergence("res_032", False, True)

    _set_convergence("res_033", False, True)

    _set_convergence("res_034", False, True)

    _set_convergence("res_035", False, True)

    _set_convergence("res_036", False, False)

    _set_convergence("res_037", False, False)

    _set_convergence("res_038", False, True)

    _set_convergence("res_039", False, True)

    _set_convergence("res_040", False, True)

    _set_convergence("res_041", False, True)

    _set_convergence("res_042", True, False)

    _set_convergence("res_043", False, True)

    _set_convergence("res_044", True, True)

    _set_convergence("res_045", True, True)

    _set_convergence("res_046", False, True)

    _set_convergence("res_047", False, True)

    _set_convergence("res_048", False, True)

    _set_convergence("res_049", False, True)

    _set_convergence("res_050", False, False)

    _set_convergence("res_051", False, True)

    _set_convergence("res_052", False, False)

    _set_convergence("res_053", False, True)

    _set_convergence("res_054", True, True)

    _set_convergence("res_055", True, True)
    
    _set_convergence("res_056", True, True)

    _set_convergence("res_057", False, True)

    _set_convergence("res_058", False, True)

    _set_convergence("res_059", True, True)

    _set_convergence("res_060", True, True)

    _set_convergence("res_061", True, True)

    _set_convergence("res_062", False, True)

    _set_convergence("res_063", False, False)

    _set_convergence("res_064", False, True)

    _set_convergence("res_065", False, True)

    _set_convergence("res_066", False, True)

    _set_convergence("res_067", False, True)
        
    _set_convergence("res_068", True, True)

    _set_convergence("res_069", True, True)

    _set_convergence("res_070", True, False)

    _set_convergence("res_071", False, True)

    _set_convergence("res_072", True, True)

    _set_convergence("res_073", True, True)

    _set_convergence("res_074", True, True)

    _set_convergence("res_075", False, True)

    _set_convergence("res_076", True, True)

    _set_convergence("res_077", False, True)

    _set_convergence("res_078", False, False)

    _set_convergence("res_079", False, True)

    _set_convergence("res_080", False, True)

    _set_convergence("res_081", False, True)

    _set_convergence("res_082", True, True)

    _set_convergence("res_083", False, True)

    _set_convergence("res_084", False, True)

    _set_convergence("res_085", False, True)

    _set_convergence("res_086", True, True)

    _set_convergence("res_087", False, True)

    _set_convergence("res_088", True, True)

    _set_convergence("res_089", False, True)

    _set_convergence("res_090", False, False)

    _set_convergence("res_091", False, True)

    _set_convergence("res_092", False, False)

    _set_convergence("res_093", False, True)

    _set_convergence("res_094", False, True)

    _set_convergence("res_095", False, True)

    _set_convergence("res_096", True, True)

    _set_convergence("res_097", True, True)

    _set_convergence("res_098", False, True)

    _set_convergence("res_099", True, True)

    _set_convergence("res_100", True, True)

    _set_convergence("res_101", False, True)

    _set_convergence("res_102", False, True)

    _set_convergence("res_103", False, True)

    _set_convergence("res_104", False, True)

    _set_convergence("res_105", False, True)

    _set_convergence("res_106", False, True)

    _set_convergence("res_107", False, False)

    _set_convergence("res_108", False, True)

    _set_convergence("res_109", False, False)
    
    _set_convergence("res_110", True, True)

    _set_convergence("res_111", True, True)

    _set_convergence("res_112", False, True)

    _set_convergence("res_113", True, True)

    _set_convergence("res_114", True, True)

    _set_convergence("res_115", True, None)

    _set_convergence("res_116", True, True)

    _set_convergence("res_117", False, True)

    _set_convergence("res_118", False, True)

    _set_convergence("res_119", False, True)

    _set_convergence("res_120", False, True)

    _set_convergence("res_121", False, True)
    
    _set_convergence("res_122", False, False)

    _set_convergence("res_123", False, True)
    
    _set_convergence("res_124", True, True)
    
    _set_convergence("res_125", True, True)
    
    _set_convergence("res_126", True, True)
    
    _set_convergence("res_127", True, True)
    
    _set_convergence("res_128", False, False)
    
    _set_convergence("res_129", True, True)

    _set_convergence("res_130", False, True)

    _set_convergence("res_131", False, False)

    _set_convergence("res_132", False, True)
    
    _set_convergence("res_133", False, True)
    
    _set_convergence("res_134", False, True)
    
    _set_convergence("res_135", False, True)
    
    _set_convergence("res_136", False, False)
    
    _set_convergence("res_137", False, True)
    
    _set_convergence("res_138", True, True)
    
    _set_convergence("res_139", True, True)
    
    _set_convergence("res_140", True, True)
    
    _set_convergence("res_141", True, True)
    
    _set_convergence("res_142", True, True)
    
    _set_convergence("res_143", True, True)
    
    _set_convergence("res_144", False, True)
    
    _set_convergence("res_145", False, True)
    
    _set_convergence("res_146", False, False)
    
    _set_convergence("res_147", False, True)
    
    _set_convergence("res_148", False, True)
    
    _set_convergence("res_149", False, True)
    
    _set_convergence("res_150", False, True)
    
    _set_convergence("res_151", False, False)
    
    _set_convergence("res_152", False, True)
    
    _set_convergence("res_153", False, True)
    
    _set_convergence("res_154", True, True)
    
    _set_convergence("res_155", False, True)
    
    _set_convergence("res_156", False, True)
    
    _set_convergence("res_157", False, False)
    
    _set_convergence("res_158", False, True)
    
    _set_convergence("res_159", False, False)
    
    _set_convergence("res_160", False, True)



# drop all, where n_data is not in n_to_plot
n_to_plot = [5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 30, 40, 50]
run_dicts = {k: v for k, v in run_dicts.items() if v["n_data"] in n_to_plot}



# now let's get this info into both DFs.
import pandas as pd

per_experiment_df = pd.DataFrame.from_dict(run_dicts, orient='index')
per_experiment_df.index.name = 'n_experiment'

# sort the columns alphabetically by the index
per_experiment_df = per_experiment_df.sort_index(axis=0)

# now let's create the per_setup_df
# first, we need to melt per_experiment_df, so that we have one row per
# n_data and random_seed

random_seeds = sorted(per_experiment_df["random_seed"].unique())

COLUMNS_PER_EXPERIMENT_DF = list()
for random_seed in random_seeds:
    for col in per_experiment_df.columns:
        if (col == "random_seed") or (col == "n_data"):
            continue
        COLUMNS_PER_EXPERIMENT_DF.append(f"{col}_{random_seed}")

index = sorted(per_experiment_df["n_data"].unique())

per_setup_df = pd.DataFrame(columns=COLUMNS_PER_EXPERIMENT_DF, index=index)
per_setup_df.index.name = "n_data"

for random_seed in random_seeds:
    df_subset = per_experiment_df[per_experiment_df["random_seed"] == random_seed]
    for col in per_experiment_df.columns:
        if (col == "random_seed") or (col == "n_data"):
            continue
        per_setup_df[f"{col}_{random_seed}"] = df_subset.set_index("n_data")[col]

# finally, let's store both DFs
per_experiment_df.to_csv(FILE_PATH / "per_experiment_df.csv")
per_setup_df.to_csv(FILE_PATH / "per_setup_df.csv")

