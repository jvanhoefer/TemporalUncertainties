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
import numpy as np
import pypesto.store as store
from pypesto.sample.geweke_test import calculate_zscore
from scipy.stats import chi2

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


def joint_geweke_test_from_chain(chain, burn_in=None, a=0.1, b=0.5):
    chain = np.asarray(chain)  # use the full chain, no burn-in removal
    z_scores, _ = calculate_zscore(chain, a=a, b=b)
    z_scores = z_scores.flatten()
    T = np.max(np.abs(z_scores))
    # T = float(np.sum(z_scores**2))
    # p_joint = float(1 - chi2.cdf(T, df=len(z_scores)))
    p_joint = float(1 - chi2.cdf(T, df=1))
    return T, p_joint


run_dicts = {}
for res_i in res_folders:
    extracted_values = extract_values(
        f"{FILE_PATH}/{res_i.name}/measurement_setup.yaml", keys_to_find)

    extracted_values["converged_full"] = None
    extracted_values["converged_marginalized"] = None

    # Compute joint Geweke z-score and p-value from HDF5
    for model in ["full", "marginalized"]:
        hdf5_path = FILE_PATH / res_i.name / f"result_{model}.hdf5"
        T_val, p_val = None, None
        if hdf5_path.exists():
            try:
                result = store.read_result(str(hdf5_path), problem=False)
                chain = result.sample_result.trace_x[0]
                T_val, p_val = joint_geweke_test_from_chain(chain)
            except Exception as e:
                print(f"Warning: could not compute Geweke for {res_i.name}/{model}: {e}")

        extracted_values[f"geweke_z_score_{model}"] = T_val   # this is the chi2 statistic T
        extracted_values[f"geweke_p_value_{model}"] = p_val

    run_dicts[res_i.name] = extracted_values

"""
run_dicts = {}
# STORAGE_FOR_REPEATED_EXPERIMENTS = "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_ConversionReaction/"
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

if ADD_CONVERGENCE_INFO:
    run_dicts[f"res_004"]["converged_full"] = True
    run_dicts[f"res_004"]["converged_marginalized"] = True

    run_dicts[f"res_005"]["converged_full"] = True
    run_dicts[f"res_005"]["converged_marginalized"] = True

    run_dicts[f"res_006"]["converged_full"] = True
    run_dicts[f"res_006"]["converged_marginalized"] = True

    run_dicts[f"res_007"]["converged_full"] = True
    run_dicts[f"res_007"]["converged_marginalized"] = True

    run_dicts[f"res_008"]["converged_full"] = True
    run_dicts[f"res_008"]["converged_marginalized"] = True

    run_dicts[f"res_009"]["converged_full"] = False
    run_dicts[f"res_009"]["converged_marginalized"] = True

    run_dicts[f"res_010"]["converged_full"] = False
    run_dicts[f"res_010"]["converged_marginalized"] = True

    run_dicts[f"res_011"]["converged_full"] = False
    run_dicts[f"res_011"]["converged_marginalized"] = True

    run_dicts[f"res_012"]["converged_full"] = True
    run_dicts[f"res_012"]["converged_marginalized"] = True

    run_dicts[f"res_013"]["converged_full"] = False
    run_dicts[f"res_013"]["converged_marginalized"] = True

    run_dicts[f"res_014"]["converged_full"] = False
    run_dicts[f"res_014"]["converged_marginalized"] = False

    run_dicts[f"res_015"]["converged_full"] = False
    run_dicts[f"res_015"]["converged_marginalized"] = False

    run_dicts[f"res_016"]["converged_full"] = False
    run_dicts[f"res_016"]["converged_marginalized"] = True

    run_dicts[f"res_017"]["converged_full"] = False
    run_dicts[f"res_017"]["converged_marginalized"] = True

    run_dicts[f"res_018"]["converged_full"] = False
    run_dicts[f"res_018"]["converged_marginalized"] = False

    run_dicts[f"res_019"]["converged_full"] = False
    run_dicts[f"res_019"]["converged_marginalized"] = True

    run_dicts[f"res_020"]["converged_full"] = False
    run_dicts[f"res_020"]["converged_marginalized"] = False

    run_dicts[f"res_021"]["converged_full"] = False
    run_dicts[f"res_021"]["converged_marginalized"] = True

    run_dicts[f"res_022"]["converged_full"] = False
    run_dicts[f"res_022"]["converged_marginalized"] = True

    run_dicts[f"res_023"]["converged_full"] = False
    run_dicts[f"res_023"]["converged_marginalized"] = True

    run_dicts[f"res_024"]["converged_full"] = False
    run_dicts[f"res_024"]["converged_marginalized"] = False

    run_dicts[f"res_025"]["converged_full"] = False
    run_dicts[f"res_025"]["converged_marginalized"] = True

    run_dicts[f"res_026"]["converged_full"] = True
    run_dicts[f"res_026"]["converged_marginalized"] = True

    run_dicts[f"res_027"]["converged_full"] = True
    run_dicts[f"res_027"]["converged_marginalized"] = True

    run_dicts[f"res_028"]["converged_full"] = True
    run_dicts[f"res_028"]["converged_marginalized"] = True

    run_dicts[f"res_029"]["converged_full"] = True
    run_dicts[f"res_029"]["converged_marginalized"] = True

    run_dicts[f"res_030"]["converged_full"] = True
    run_dicts[f"res_030"]["converged_marginalized"] = True

    run_dicts[f"res_031"]["converged_full"] = False
    run_dicts[f"res_031"]["converged_marginalized"] = True

    run_dicts[f"res_032"]["converged_full"] = False
    run_dicts[f"res_032"]["converged_marginalized"] = True

    run_dicts[f"res_033"]["converged_full"] = False
    run_dicts[f"res_033"]["converged_marginalized"] = True

    run_dicts[f"res_034"]["converged_full"] = False
    run_dicts[f"res_034"]["converged_marginalized"] = True

    run_dicts[f"res_035"]["converged_full"] = False
    run_dicts[f"res_035"]["converged_marginalized"] = True

    run_dicts[f"res_036"]["converged_full"] = False
    run_dicts[f"res_036"]["converged_marginalized"] = False

    run_dicts[f"res_037"]["converged_full"] = False
    run_dicts[f"res_037"]["converged_marginalized"] = False

    run_dicts[f"res_038"]["converged_full"] = False
    run_dicts[f"res_038"]["converged_marginalized"] = True

    run_dicts[f"res_039"]["converged_full"] = False
    run_dicts[f"res_039"]["converged_marginalized"] = True

    run_dicts[f"res_040"]["converged_full"] = False
    run_dicts[f"res_040"]["converged_marginalized"] = True

    run_dicts[f"res_041"]["converged_full"] = False
    run_dicts[f"res_041"]["converged_marginalized"] = True

    run_dicts[f"res_042"]["converged_full"] = True
    run_dicts[f"res_042"]["converged_marginalized"] = False

    run_dicts[f"res_043"]["converged_full"] = False
    run_dicts[f"res_043"]["converged_marginalized"] = True

    run_dicts[f"res_044"]["converged_full"] = True
    run_dicts[f"res_044"]["converged_marginalized"] = True

    run_dicts[f"res_045"]["converged_full"] = True
    run_dicts[f"res_045"]["converged_marginalized"] = True

    run_dicts[f"res_046"]["converged_full"] = False
    run_dicts[f"res_046"]["converged_marginalized"] = True

    run_dicts[f"res_047"]["converged_full"] = False
    run_dicts[f"res_047"]["converged_marginalized"] = True

    run_dicts[f"res_048"]["converged_full"] = False
    run_dicts[f"res_048"]["converged_marginalized"] = True

    run_dicts[f"res_049"]["converged_full"] = False
    run_dicts[f"res_049"]["converged_marginalized"] = True

    run_dicts[f"res_050"]["converged_full"] = False
    run_dicts[f"res_050"]["converged_marginalized"] = False

    run_dicts[f"res_051"]["converged_full"] = False
    run_dicts[f"res_051"]["converged_marginalized"] = True

    run_dicts[f"res_052"]["converged_full"] = False
    run_dicts[f"res_052"]["converged_marginalized"] = False

    run_dicts[f"res_053"]["converged_full"] = False
    run_dicts[f"res_053"]["converged_marginalized"] = True

    run_dicts[f"res_054"]["converged_full"] = True
    run_dicts[f"res_054"]["converged_marginalized"] = True

    run_dicts[f"res_055"]["converged_full"] = True
    run_dicts[f"res_055"]["converged_marginalized"] = True
    
    run_dicts[f"res_056"]["converged_full"] = True
    run_dicts[f"res_056"]["converged_marginalized"] = True

    run_dicts[f"res_057"]["converged_full"] = False
    run_dicts[f"res_057"]["converged_marginalized"] = True

    run_dicts[f"res_058"]["converged_full"] = False
    run_dicts[f"res_058"]["converged_marginalized"] = True

    run_dicts[f"res_059"]["converged_full"] = True
    run_dicts[f"res_059"]["converged_marginalized"] = True

    run_dicts[f"res_060"]["converged_full"] = True
    run_dicts[f"res_060"]["converged_marginalized"] = True

    run_dicts[f"res_061"]["converged_full"] = True
    run_dicts[f"res_061"]["converged_marginalized"] = True

    run_dicts[f"res_062"]["converged_full"] = False
    run_dicts[f"res_062"]["converged_marginalized"] = True

    run_dicts[f"res_063"]["converged_full"] = False
    run_dicts[f"res_063"]["converged_marginalized"] = False

    run_dicts[f"res_064"]["converged_full"] = False
    run_dicts[f"res_064"]["converged_marginalized"] = True

    run_dicts[f"res_065"]["converged_full"] = False
    run_dicts[f"res_065"]["converged_marginalized"] = True

    run_dicts[f"res_066"]["converged_full"] = False
    run_dicts[f"res_066"]["converged_marginalized"] = True

    run_dicts[f"res_067"]["converged_full"] = False
    run_dicts[f"res_067"]["converged_marginalized"] = True
        
    run_dicts[f"res_068"]["converged_full"] = True
    run_dicts[f"res_068"]["converged_marginalized"] = True

    run_dicts[f"res_069"]["converged_full"] = True
    run_dicts[f"res_069"]["converged_marginalized"] = True

    run_dicts[f"res_070"]["converged_full"] = True
    run_dicts[f"res_070"]["converged_marginalized"] = False

    run_dicts[f"res_071"]["converged_full"] = False
    run_dicts[f"res_071"]["converged_marginalized"] = True

    run_dicts[f"res_072"]["converged_full"] = True
    run_dicts[f"res_072"]["converged_marginalized"] = True

    run_dicts[f"res_073"]["converged_full"] = True
    run_dicts[f"res_073"]["converged_marginalized"] = True

    run_dicts[f"res_074"]["converged_full"] = True
    run_dicts[f"res_074"]["converged_marginalized"] = True

    run_dicts[f"res_075"]["converged_full"] = False
    run_dicts[f"res_075"]["converged_marginalized"] = True

    run_dicts[f"res_076"]["converged_full"] = True
    run_dicts[f"res_076"]["converged_marginalized"] = True

    run_dicts[f"res_077"]["converged_full"] = False
    run_dicts[f"res_077"]["converged_marginalized"] = True

    run_dicts[f"res_078"]["converged_full"] = False
    run_dicts[f"res_078"]["converged_marginalized"] = False

    run_dicts[f"res_079"]["converged_full"] = False
    run_dicts[f"res_079"]["converged_marginalized"] = True

    run_dicts[f"res_080"]["converged_full"] = False
    run_dicts[f"res_080"]["converged_marginalized"] = True

    run_dicts[f"res_081"]["converged_full"] = False
    run_dicts[f"res_081"]["converged_marginalized"] = True

    run_dicts[f"res_082"]["converged_full"] = True
    run_dicts[f"res_082"]["converged_marginalized"] = True

    run_dicts[f"res_083"]["converged_full"] = False
    run_dicts[f"res_083"]["converged_marginalized"] = True

    run_dicts[f"res_084"]["converged_full"] = False
    run_dicts[f"res_084"]["converged_marginalized"] = True

    run_dicts[f"res_085"]["converged_full"] = False
    run_dicts[f"res_085"]["converged_marginalized"] = True

    run_dicts[f"res_086"]["converged_full"] = True
    run_dicts[f"res_086"]["converged_marginalized"] = True

    run_dicts[f"res_087"]["converged_full"] = False
    run_dicts[f"res_087"]["converged_marginalized"] = True

    run_dicts[f"res_088"]["converged_full"] = True
    run_dicts[f"res_088"]["converged_marginalized"] = True

    run_dicts[f"res_089"]["converged_full"] = False
    run_dicts[f"res_089"]["converged_marginalized"] = True

    run_dicts[f"res_090"]["converged_full"] = False
    run_dicts[f"res_090"]["converged_marginalized"] = False

    run_dicts[f"res_091"]["converged_full"] = False
    run_dicts[f"res_091"]["converged_marginalized"] = True

    run_dicts[f"res_092"]["converged_full"] = False
    run_dicts[f"res_092"]["converged_marginalized"] = False

    run_dicts[f"res_093"]["converged_full"] = False
    run_dicts[f"res_093"]["converged_marginalized"] = True

    run_dicts[f"res_094"]["converged_full"] = False
    run_dicts[f"res_094"]["converged_marginalized"] = True

    run_dicts[f"res_095"]["converged_full"] = False
    run_dicts[f"res_095"]["converged_marginalized"] = True

    run_dicts[f"res_096"]["converged_full"] = True
    run_dicts[f"res_096"]["converged_marginalized"] = True

    run_dicts[f"res_097"]["converged_full"] = True
    run_dicts[f"res_097"]["converged_marginalized"] = True

    run_dicts[f"res_098"]["converged_full"] = False
    run_dicts[f"res_098"]["converged_marginalized"] = True

    run_dicts[f"res_099"]["converged_full"] = True
    run_dicts[f"res_099"]["converged_marginalized"] = True

    run_dicts[f"res_100"]["converged_full"] = True
    run_dicts[f"res_100"]["converged_marginalized"] = True

    run_dicts[f"res_101"]["converged_full"] = False
    run_dicts[f"res_101"]["converged_marginalized"] = True

    run_dicts[f"res_102"]["converged_full"] = False
    run_dicts[f"res_102"]["converged_marginalized"] = True

    run_dicts[f"res_103"]["converged_full"] = False
    run_dicts[f"res_103"]["converged_marginalized"] = True

    run_dicts[f"res_104"]["converged_full"] = False
    run_dicts[f"res_104"]["converged_marginalized"] = True

    run_dicts[f"res_105"]["converged_full"] = False
    run_dicts[f"res_105"]["converged_marginalized"] = True

    run_dicts[f"res_106"]["converged_full"] = False
    run_dicts[f"res_106"]["converged_marginalized"] = True

    run_dicts[f"res_107"]["converged_full"] = False
    run_dicts[f"res_107"]["converged_marginalized"] = False

    run_dicts[f"res_108"]["converged_full"] = False
    run_dicts[f"res_108"]["converged_marginalized"] = True

    run_dicts[f"res_109"]["converged_full"] = False
    run_dicts[f"res_109"]["converged_marginalized"] = False
    
    run_dicts[f"res_110"]["converged_full"] = True
    run_dicts[f"res_110"]["converged_marginalized"] = True

    run_dicts[f"res_111"]["converged_full"] = True
    run_dicts[f"res_111"]["converged_marginalized"] = True

    run_dicts[f"res_112"]["converged_full"] = False
    run_dicts[f"res_112"]["converged_marginalized"] = True

    run_dicts[f"res_113"]["converged_full"] = True
    run_dicts[f"res_113"]["converged_marginalized"] = True

    run_dicts[f"res_114"]["converged_full"] = True
    run_dicts[f"res_114"]["converged_marginalized"] = True

    run_dicts[f"res_115"]["converged_full"] = True
    run_dicts[f"res_115"]["converged_marginalized"] = None

    run_dicts[f"res_116"]["converged_full"] = True
    run_dicts[f"res_116"]["converged_marginalized"] = True

    run_dicts[f"res_117"]["converged_full"] = False
    run_dicts[f"res_117"]["converged_marginalized"] = True

    run_dicts[f"res_118"]["converged_full"] = False
    run_dicts[f"res_118"]["converged_marginalized"] = True

    run_dicts[f"res_119"]["converged_full"] = False
    run_dicts[f"res_119"]["converged_marginalized"] = True

    run_dicts[f"res_120"]["converged_full"] = False
    run_dicts[f"res_120"]["converged_marginalized"] = True

    run_dicts[f"res_121"]["converged_full"] = False
    run_dicts[f"res_121"]["converged_marginalized"] = True
    
    run_dicts[f"res_122"]["converged_full"] = False
    run_dicts[f"res_122"]["converged_marginalized"] = False

    run_dicts[f"res_123"]["converged_full"] = False
    run_dicts[f"res_123"]["converged_marginalized"] = True
    
    run_dicts[f"res_124"]["converged_full"] = True
    run_dicts[f"res_124"]["converged_marginalized"] = True
    
    run_dicts[f"res_125"]["converged_full"] = True
    run_dicts[f"res_125"]["converged_marginalized"] = True
    
    run_dicts[f"res_126"]["converged_full"] = True
    run_dicts[f"res_126"]["converged_marginalized"] = True
    
    run_dicts[f"res_127"]["converged_full"] = True
    run_dicts[f"res_127"]["converged_marginalized"] = True
    
    run_dicts[f"res_128"]["converged_full"] = False
    run_dicts[f"res_128"]["converged_marginalized"] = False
    
    run_dicts[f"res_129"]["converged_full"] = True
    run_dicts[f"res_129"]["converged_marginalized"] = True

    run_dicts[f"res_130"]["converged_full"] = False
    run_dicts[f"res_130"]["converged_marginalized"] = True

    run_dicts[f"res_131"]["converged_full"] = False
    run_dicts[f"res_131"]["converged_marginalized"] = False

    run_dicts[f"res_132"]["converged_full"] = False
    run_dicts[f"res_132"]["converged_marginalized"] = True
    
    run_dicts[f"res_133"]["converged_full"] = False
    run_dicts[f"res_133"]["converged_marginalized"] = True
    
    run_dicts[f"res_134"]["converged_full"] = False
    run_dicts[f"res_134"]["converged_marginalized"] = True
    
    run_dicts[f"res_135"]["converged_full"] = False
    run_dicts[f"res_135"]["converged_marginalized"] = True
    
    run_dicts[f"res_136"]["converged_full"] = False
    run_dicts[f"res_136"]["converged_marginalized"] = False
    
    run_dicts[f"res_137"]["converged_full"] = False
    run_dicts[f"res_137"]["converged_marginalized"] = True
    
    run_dicts[f"res_138"]["converged_full"] = True
    run_dicts[f"res_138"]["converged_marginalized"] = True
    
    run_dicts[f"res_139"]["converged_full"] = True
    run_dicts[f"res_139"]["converged_marginalized"] = True
    
    run_dicts[f"res_140"]["converged_full"] = True
    run_dicts[f"res_140"]["converged_marginalized"] = True
    
    run_dicts[f"res_141"]["converged_full"] = True
    run_dicts[f"res_141"]["converged_marginalized"] = True
    
    run_dicts[f"res_142"]["converged_full"] = True
    run_dicts[f"res_142"]["converged_marginalized"] = True
    
    run_dicts[f"res_143"]["converged_full"] = True
    run_dicts[f"res_143"]["converged_marginalized"] = True
    
    run_dicts[f"res_144"]["converged_full"] = False
    run_dicts[f"res_144"]["converged_marginalized"] = True
    
    run_dicts[f"res_145"]["converged_full"] = False
    run_dicts[f"res_145"]["converged_marginalized"] = True
    
    run_dicts[f"res_146"]["converged_full"] = False
    run_dicts[f"res_146"]["converged_marginalized"] = False
    
    run_dicts[f"res_147"]["converged_full"] = False
    run_dicts[f"res_147"]["converged_marginalized"] = True
    
    run_dicts[f"res_148"]["converged_full"] = False
    run_dicts[f"res_148"]["converged_marginalized"] = True
    
    run_dicts[f"res_149"]["converged_full"] = False
    run_dicts[f"res_149"]["converged_marginalized"] = True
    
    run_dicts[f"res_150"]["converged_full"] = False
    run_dicts[f"res_150"]["converged_marginalized"] = True
    
    run_dicts[f"res_151"]["converged_full"] = False
    run_dicts[f"res_151"]["converged_marginalized"] = False
    
    run_dicts[f"res_152"]["converged_full"] = False
    run_dicts[f"res_152"]["converged_marginalized"] = True
    
    run_dicts[f"res_153"]["converged_full"] = False
    run_dicts[f"res_153"]["converged_marginalized"] = True
    
    run_dicts[f"res_154"]["converged_full"] = True
    run_dicts[f"res_154"]["converged_marginalized"] = True
    
    run_dicts[f"res_155"]["converged_full"] = False
    run_dicts[f"res_155"]["converged_marginalized"] = True
    
    run_dicts[f"res_156"]["converged_full"] = False
    run_dicts[f"res_156"]["converged_marginalized"] = True
    
    run_dicts[f"res_157"]["converged_full"] = False
    run_dicts[f"res_157"]["converged_marginalized"] = False
    
    run_dicts[f"res_158"]["converged_full"] = False
    run_dicts[f"res_158"]["converged_marginalized"] = True
    
    run_dicts[f"res_159"]["converged_full"] = False 
    run_dicts[f"res_159"]["converged_marginalized"] = False
    
    run_dicts[f"res_160"]["converged_full"] = False
    run_dicts[f"res_160"]["converged_marginalized"] = True



import numpy as np
from scipy.stats import chi2
from pypesto.sample.geweke_test import calculate_zscore



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

