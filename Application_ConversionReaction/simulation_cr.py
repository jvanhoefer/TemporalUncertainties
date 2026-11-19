"""
Implementation of the conversion reaction example with 
temporal uncertainty.

Reaction Network:
 -> A <=>B ->

This file is responsible for all simulations:


- Define (and store) the synthetic measurement setup
- Generate (and store) the synthetic data
- Implement all three objective function:
    - wrong_tp
    - full
    - marginalized
- fit and sample the objective functions
- Bonus: Refit for different values of simga_tau
"""
import yaml2sbml
import yaml
import pypesto 
import amici
import numpy as np
import os
import sys

import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.store as store

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "Application_ConversionReaction"

# Full historical sweep (used only when PETAB_INDEX/SLURM_ARRAY_TASK_ID
# selects into it by position -- see below). Generated from this file's
# own previously-commented `setups` entries; do not hand-edit, regenerate
# via the extraction command in the plan if entries are ever added/removed.
ALL_SETUPS = [
    {"n_experiment": 4, "n_time": 5, "random_seed": 42},
    {"n_experiment": 5, "n_time": 6, "random_seed": 42},
    {"n_experiment": 6, "n_time": 7, "random_seed": 42},
    {"n_experiment": 7, "n_time": 8, "random_seed": 42},
    {"n_experiment": 8, "n_time": 9, "random_seed": 42},
    {"n_experiment": 9, "n_time": 10, "random_seed": 42},
    {"n_experiment": 10, "n_time": 11, "random_seed": 42},
    {"n_experiment": 11, "n_time": 12, "random_seed": 42},
    {"n_experiment": 12, "n_time": 13, "random_seed": 42},
    {"n_experiment": 13, "n_time": 14, "random_seed": 42},
    {"n_experiment": 15, "n_time": 16, "random_seed": 42},
    {"n_experiment": 16, "n_time": 17, "random_seed": 42},
    {"n_experiment": 17, "n_time": 18, "random_seed": 42},
    {"n_experiment": 18, "n_time": 19, "random_seed": 42},
    {"n_experiment": 19, "n_time": 20, "random_seed": 42},
    {"n_experiment": 20, "n_time": 25, "random_seed": 42},
    {"n_experiment": 21, "n_time": 30, "random_seed": 42},
    {"n_experiment": 22, "n_time": 35, "random_seed": 42},
    {"n_experiment": 23, "n_time": 40, "random_seed": 42},
    {"n_experiment": 24, "n_time": 45, "random_seed": 42},
    {"n_experiment": 25, "n_time": 50, "random_seed": 42},
    {"n_experiment": 26, "n_time": 5, "random_seed": 1},
    {"n_experiment": 27, "n_time": 6, "random_seed": 1},
    {"n_experiment": 28, "n_time": 7, "random_seed": 1},
    {"n_experiment": 29, "n_time": 8, "random_seed": 1},
    {"n_experiment": 30, "n_time": 9, "random_seed": 1},
    {"n_experiment": 31, "n_time": 10, "random_seed": 1},
    {"n_experiment": 32, "n_time": 12, "random_seed": 1},
    {"n_experiment": 33, "n_time": 14, "random_seed": 1},
    {"n_experiment": 34, "n_time": 16, "random_seed": 1},
    {"n_experiment": 35, "n_time": 18, "random_seed": 1},
    {"n_experiment": 36, "n_time": 20, "random_seed": 1},
    {"n_experiment": 37, "n_time": 30, "random_seed": 1},
    {"n_experiment": 38, "n_time": 40, "random_seed": 1},
    {"n_experiment": 39, "n_time": 50, "random_seed": 1},
    {"n_experiment": 40, "n_time": 5, "random_seed": 7},
    {"n_experiment": 41, "n_time": 6, "random_seed": 7},
    {"n_experiment": 42, "n_time": 7, "random_seed": 7},
    {"n_experiment": 43, "n_time": 8, "random_seed": 7},
    {"n_experiment": 44, "n_time": 9, "random_seed": 7},
    {"n_experiment": 45, "n_time": 10, "random_seed": 7},
    {"n_experiment": 46, "n_time": 12, "random_seed": 7},
    {"n_experiment": 47, "n_time": 14, "random_seed": 7},
    {"n_experiment": 48, "n_time": 16, "random_seed": 7},
    {"n_experiment": 49, "n_time": 18, "random_seed": 7},
    {"n_experiment": 50, "n_time": 20, "random_seed": 7},
    {"n_experiment": 51, "n_time": 30, "random_seed": 7},
    {"n_experiment": 52, "n_time": 40, "random_seed": 7},
    {"n_experiment": 53, "n_time": 50, "random_seed": 7},
    {"n_experiment": 54, "n_time": 5, "random_seed": 2},
    {"n_experiment": 55, "n_time": 6, "random_seed": 2},
    {"n_experiment": 56, "n_time": 7, "random_seed": 2},
    {"n_experiment": 57, "n_time": 8, "random_seed": 2},
    {"n_experiment": 58, "n_time": 9, "random_seed": 2},
    {"n_experiment": 59, "n_time": 10, "random_seed": 2},
    {"n_experiment": 60, "n_time": 12, "random_seed": 2},
    {"n_experiment": 61, "n_time": 14, "random_seed": 2},
    {"n_experiment": 62, "n_time": 16, "random_seed": 2},
    {"n_experiment": 63, "n_time": 18, "random_seed": 2},
    {"n_experiment": 64, "n_time": 20, "random_seed": 2},
    {"n_experiment": 65, "n_time": 30, "random_seed": 2},
    {"n_experiment": 66, "n_time": 40, "random_seed": 2},
    {"n_experiment": 67, "n_time": 50, "random_seed": 2},
    {"n_experiment": 68, "n_time": 5, "random_seed": 3},
    {"n_experiment": 69, "n_time": 6, "random_seed": 3},
    {"n_experiment": 70, "n_time": 7, "random_seed": 3},
    {"n_experiment": 71, "n_time": 8, "random_seed": 3},
    {"n_experiment": 72, "n_time": 9, "random_seed": 3},
    {"n_experiment": 73, "n_time": 10, "random_seed": 3},
    {"n_experiment": 74, "n_time": 12, "random_seed": 3},
    {"n_experiment": 75, "n_time": 14, "random_seed": 3},
    {"n_experiment": 76, "n_time": 16, "random_seed": 3},
    {"n_experiment": 77, "n_time": 18, "random_seed": 3},
    {"n_experiment": 78, "n_time": 20, "random_seed": 3},
    {"n_experiment": 79, "n_time": 30, "random_seed": 3},
    {"n_experiment": 80, "n_time": 40, "random_seed": 3},
    {"n_experiment": 81, "n_time": 50, "random_seed": 3},
    {"n_experiment": 82, "n_time": 5, "random_seed": 4},
    {"n_experiment": 83, "n_time": 6, "random_seed": 4},
    {"n_experiment": 84, "n_time": 7, "random_seed": 4},
    {"n_experiment": 85, "n_time": 8, "random_seed": 4},
    {"n_experiment": 86, "n_time": 9, "random_seed": 4},
    {"n_experiment": 87, "n_time": 10, "random_seed": 4},
    {"n_experiment": 88, "n_time": 12, "random_seed": 4},
    {"n_experiment": 89, "n_time": 14, "random_seed": 4},
    {"n_experiment": 90, "n_time": 16, "random_seed": 4},
    {"n_experiment": 91, "n_time": 18, "random_seed": 4},
    {"n_experiment": 92, "n_time": 20, "random_seed": 4},
    {"n_experiment": 93, "n_time": 30, "random_seed": 4},
    {"n_experiment": 94, "n_time": 40, "random_seed": 4},
    {"n_experiment": 95, "n_time": 50, "random_seed": 4},
    {"n_experiment": 96, "n_time": 5, "random_seed": 5},
    {"n_experiment": 97, "n_time": 6, "random_seed": 5},
    {"n_experiment": 98, "n_time": 7, "random_seed": 5},
    {"n_experiment": 99, "n_time": 8, "random_seed": 5},
    {"n_experiment": 100, "n_time": 9, "random_seed": 5},
    {"n_experiment": 101, "n_time": 10, "random_seed": 5},
    {"n_experiment": 102, "n_time": 12, "random_seed": 5},
    {"n_experiment": 103, "n_time": 14, "random_seed": 5},
    {"n_experiment": 104, "n_time": 16, "random_seed": 5},
    {"n_experiment": 105, "n_time": 18, "random_seed": 5},
    {"n_experiment": 106, "n_time": 20, "random_seed": 5},
    {"n_experiment": 107, "n_time": 30, "random_seed": 5},
    {"n_experiment": 108, "n_time": 40, "random_seed": 5},
    {"n_experiment": 109, "n_time": 50, "random_seed": 5},
    {"n_experiment": 110, "n_time": 5, "random_seed": 6},
    {"n_experiment": 111, "n_time": 6, "random_seed": 6},
    {"n_experiment": 112, "n_time": 7, "random_seed": 6},
    {"n_experiment": 113, "n_time": 8, "random_seed": 6},
    {"n_experiment": 114, "n_time": 9, "random_seed": 6},
    {"n_experiment": 115, "n_time": 10, "random_seed": 6},
    {"n_experiment": 116, "n_time": 12, "random_seed": 6},
    {"n_experiment": 117, "n_time": 14, "random_seed": 6},
    {"n_experiment": 118, "n_time": 16, "random_seed": 6},
    {"n_experiment": 119, "n_time": 18, "random_seed": 6},
    {"n_experiment": 120, "n_time": 20, "random_seed": 6},
    {"n_experiment": 121, "n_time": 30, "random_seed": 6},
    {"n_experiment": 122, "n_time": 40, "random_seed": 6},
    {"n_experiment": 123, "n_time": 50, "random_seed": 6},
    {"n_experiment": 124, "n_time": 5, "random_seed": 8},
    {"n_experiment": 125, "n_time": 6, "random_seed": 8},
    {"n_experiment": 126, "n_time": 7, "random_seed": 8},
    {"n_experiment": 127, "n_time": 8, "random_seed": 8},
    {"n_experiment": 128, "n_time": 9, "random_seed": 8},
    {"n_experiment": 129, "n_time": 10, "random_seed": 8},
    {"n_experiment": 130, "n_time": 12, "random_seed": 8},
    {"n_experiment": 131, "n_time": 14, "random_seed": 8},
    {"n_experiment": 132, "n_time": 16, "random_seed": 8},
    {"n_experiment": 133, "n_time": 18, "random_seed": 8},
    {"n_experiment": 134, "n_time": 20, "random_seed": 8},
    {"n_experiment": 135, "n_time": 30, "random_seed": 8},
    {"n_experiment": 136, "n_time": 40, "random_seed": 8},
    {"n_experiment": 137, "n_time": 50, "random_seed": 8},
    {"n_experiment": 138, "n_time": 5, "random_seed": 9},
    {"n_experiment": 139, "n_time": 6, "random_seed": 9},
    {"n_experiment": 140, "n_time": 7, "random_seed": 9},
    {"n_experiment": 141, "n_time": 8, "random_seed": 9},
    {"n_experiment": 142, "n_time": 9, "random_seed": 9},
    {"n_experiment": 143, "n_time": 10, "random_seed": 9},
    {"n_experiment": 144, "n_time": 12, "random_seed": 9},
    {"n_experiment": 145, "n_time": 14, "random_seed": 9},
    {"n_experiment": 146, "n_time": 16, "random_seed": 9},
    {"n_experiment": 147, "n_time": 18, "random_seed": 9},
    {"n_experiment": 148, "n_time": 20, "random_seed": 9},
    {"n_experiment": 149, "n_time": 30, "random_seed": 9},
    {"n_experiment": 150, "n_time": 40, "random_seed": 9},
    {"n_experiment": 151, "n_time": 50, "random_seed": 9},
    {"n_experiment": 152, "n_time": 15, "random_seed": 1},
    {"n_experiment": 153, "n_time": 15, "random_seed": 2},
    {"n_experiment": 154, "n_time": 15, "random_seed": 3},
    {"n_experiment": 155, "n_time": 15, "random_seed": 4},
    {"n_experiment": 156, "n_time": 15, "random_seed": 5},
    {"n_experiment": 157, "n_time": 15, "random_seed": 6},
    {"n_experiment": 158, "n_time": 15, "random_seed": 7},
    {"n_experiment": 159, "n_time": 15, "random_seed": 8},
    {"n_experiment": 160, "n_time": 15, "random_seed": 9},
]

setups = [
    # START WITH RANDOM_SEED = 42
    # {"n_experiment": 4, "n_time": 5, "random_seed": 42}, 
    # {"n_experiment": 5, "n_time": 6, "random_seed": 42},
    # {"n_experiment": 6, "n_time": 7, "random_seed": 42}, 
    # {"n_experiment": 7, "n_time": 8, "random_seed": 42}, 
    # {"n_experiment": 8, "n_time": 9, "random_seed": 42},
    # {"n_experiment": 9, "n_time": 10, "random_seed": 42}, 
    # {"n_experiment": 10, "n_time": 11, "random_seed": 42},
    # {"n_experiment": 11, "n_time": 12, "random_seed": 42},
    # {"n_experiment": 12, "n_time": 13, "random_seed": 42},
    # {"n_experiment": 13, "n_time": 14, "random_seed": 42},
    # {"n_experiment": 15, "n_time": 16, "random_seed": 42},
    # {"n_experiment": 16, "n_time": 17, "random_seed": 42},
    # {"n_experiment": 17, "n_time": 18, "random_seed": 42},
    # {"n_experiment": 18, "n_time": 19, "random_seed": 42}, 
    # {"n_experiment": 19, "n_time": 20, "random_seed": 42},
    # {"n_experiment": 20, "n_time": 25, "random_seed": 42},
    # {"n_experiment": 21, "n_time": 30, "random_seed": 42}, 
    # {"n_experiment": 22, "n_time": 35, "random_seed": 42}, 
    # {"n_experiment": 23, "n_time": 40, "random_seed": 42},
    # {"n_experiment": 24, "n_time": 45, "random_seed": 42},
    # {"n_experiment": 25, "n_time": 50, "random_seed": 42},
    # START WITH RANDOM_SEED = 1
    # {"n_experiment": 26, "n_time": 5, "random_seed": 1},
    # {"n_experiment": 27, "n_time": 6, "random_seed": 1},
    # {"n_experiment": 28, "n_time": 7, "random_seed": 1}, 
    # {"n_experiment": 29, "n_time": 8, "random_seed": 1},
    # {"n_experiment": 30, "n_time": 9, "random_seed": 1},
    # {"n_experiment": 31, "n_time": 10, "random_seed": 1},
    # {"n_experiment": 32, "n_time": 12, "random_seed": 1},
    # {"n_experiment": 33, "n_time": 14, "random_seed": 1},
    # {"n_experiment": 34, "n_time": 16, "random_seed": 1},
    # {"n_experiment": 35, "n_time": 18, "random_seed": 1},
    # {"n_experiment": 36, "n_time": 20, "random_seed": 1},
    # {"n_experiment": 37, "n_time": 30, "random_seed": 1},
    # {"n_experiment": 38, "n_time": 40, "random_seed": 1},
    # {"n_experiment": 39, "n_time": 50, "random_seed": 1},
    # START WITH RANDOM_SEED = 7
    # {"n_experiment": 40, "n_time": 5, "random_seed": 7},
    # {"n_experiment": 41, "n_time": 6, "random_seed": 7},
    # {"n_experiment": 42, "n_time": 7, "random_seed": 7},
    # {"n_experiment": 43, "n_time": 8, "random_seed": 7},
    # {"n_experiment": 44, "n_time": 9, "random_seed": 7},
    # {"n_experiment": 45, "n_time": 10, "random_seed": 7},
    # {"n_experiment": 46, "n_time": 12, "random_seed": 7},
    # {"n_experiment": 47, "n_time": 14, "random_seed": 7},
    # {"n_experiment": 48, "n_time": 16, "random_seed": 7},
    # {"n_experiment": 49, "n_time": 18, "random_seed": 7},
    # {"n_experiment": 50, "n_time": 20, "random_seed": 7},
    # {"n_experiment": 51, "n_time": 30, "random_seed": 7},
    # {"n_experiment": 52, "n_time": 40, "random_seed": 7},
    # {"n_experiment": 53, "n_time": 50, "random_seed": 7},
    # START WITH RANDOM_SEED = 2
    # {"n_experiment": 54, "n_time": 5, "random_seed": 2},
    # {"n_experiment": 55, "n_time": 6, "random_seed": 2},
    # {"n_experiment": 56, "n_time": 7, "random_seed": 2},
    # {"n_experiment": 57, "n_time": 8, "random_seed": 2},
    # {"n_experiment": 58, "n_time": 9, "random_seed": 2},
    # {"n_experiment": 59, "n_time": 10, "random_seed": 2},
    # {"n_experiment": 60, "n_time": 12, "random_seed": 2},
    # {"n_experiment": 61, "n_time": 14, "random_seed": 2},
    # {"n_experiment": 62, "n_time": 16, "random_seed": 2},
    # {"n_experiment": 63, "n_time": 18, "random_seed": 2},
    # {"n_experiment": 64, "n_time": 20, "random_seed": 2},
    # {"n_experiment": 65, "n_time": 30, "random_seed": 2},
    # {"n_experiment": 66, "n_time": 40, "random_seed": 2},
    # {"n_experiment": 67, "n_time": 50, "random_seed": 2},  NOT DONE YET -> SAMPLING FAILS FOR FULL MODEL
    # START WITH RANDOM_SEED = 3
    # {"n_experiment": 68, "n_time": 5, "random_seed": 3},
    # {"n_experiment": 69, "n_time": 6, "random_seed": 3},
    # {"n_experiment": 70, "n_time": 7, "random_seed": 3},
    # {"n_experiment": 71, "n_time": 8, "random_seed": 3},
    # {"n_experiment": 72, "n_time": 9, "random_seed": 3},
    # {"n_experiment": 73, "n_time": 10, "random_seed": 3},
    # {"n_experiment": 74, "n_time": 12, "random_seed": 3},
    # {"n_experiment": 75, "n_time": 14, "random_seed": 3},
    # {"n_experiment": 76, "n_time": 16, "random_seed": 3},
    # {"n_experiment": 77, "n_time": 18, "random_seed": 3},
    # {"n_experiment": 78, "n_time": 20, "random_seed": 3},
    # {"n_experiment": 79, "n_time": 30, "random_seed": 3},
    # {"n_experiment": 80, "n_time": 40, "random_seed": 3},
    # {"n_experiment": 81, "n_time": 50, "random_seed": 3},
    # START WITH RANDOM SEED = 4
    # {"n_experiment": 82, "n_time": 5, "random_seed": 4},
    # {"n_experiment": 83, "n_time": 6, "random_seed": 4},
    # {"n_experiment": 84, "n_time": 7, "random_seed": 4},
    # {"n_experiment": 85, "n_time": 8, "random_seed": 4},
    # {"n_experiment": 86, "n_time": 9, "random_seed": 4},
    # {"n_experiment": 87, "n_time": 10, "random_seed": 4},
    # {"n_experiment": 88, "n_time": 12, "random_seed": 4},
    # {"n_experiment": 89, "n_time": 14, "random_seed": 4},
    # {"n_experiment": 90, "n_time": 16, "random_seed": 4},
    # {"n_experiment": 91, "n_time": 18, "random_seed": 4},
    # {"n_experiment": 92, "n_time": 20, "random_seed": 4},
    # {"n_experiment": 93, "n_time": 30, "random_seed": 4},
    # {"n_experiment": 94, "n_time": 40, "random_seed": 4},
    # {"n_experiment": 95, "n_time": 50, "random_seed": 4},
    # START WITH RANDOM SEED = 5
    # {"n_experiment": 96, "n_time": 5, "random_seed": 5},
    # {"n_experiment": 97, "n_time": 6, "random_seed": 5},
    # {"n_experiment": 98, "n_time": 7, "random_seed": 5},
    # {"n_experiment": 99, "n_time": 8, "random_seed": 5},
    # {"n_experiment": 100, "n_time": 9, "random_seed": 5},
    # {"n_experiment": 101, "n_time": 10, "random_seed": 5},
    # {"n_experiment": 102, "n_time": 12, "random_seed": 5},
    # {"n_experiment": 103, "n_time": 14, "random_seed": 5},
    # {"n_experiment": 104, "n_time": 16, "random_seed": 5},
    # {"n_experiment": 105, "n_time": 18, "random_seed": 5},
    # {"n_experiment": 106, "n_time": 20, "random_seed": 5},
    # {"n_experiment": 107, "n_time": 30, "random_seed": 5},
    # {"n_experiment": 108, "n_time": 40, "random_seed": 5},
    # {"n_experiment": 109, "n_time": 50, "random_seed": 5},
    # START WITH RANDOM SEED = 6
    # {"n_experiment": 110, "n_time": 5, "random_seed": 6},
    # {"n_experiment": 111, "n_time": 6, "random_seed": 6},
    # {"n_experiment": 112, "n_time": 7, "random_seed": 6},
    # {"n_experiment": 113, "n_time": 8, "random_seed": 6},
    # {"n_experiment": 114, "n_time": 9, "random_seed": 6},
    # {"n_experiment": 115, "n_time": 10, "random_seed": 6},
    # {"n_experiment": 116, "n_time": 12, "random_seed": 6},
    # {"n_experiment": 117, "n_time": 14, "random_seed": 6},
    # {"n_experiment": 118, "n_time": 16, "random_seed": 6},
    # {"n_experiment": 119, "n_time": 18, "random_seed": 6},
    # {"n_experiment": 120, "n_time": 20, "random_seed": 6},
    # {"n_experiment": 121, "n_time": 30, "random_seed": 6}, NOT DONE YET -> SAMPLING FAILS FOR FULL MODEL
    # {"n_experiment": 122, "n_time": 40, "random_seed": 6},
    # {"n_experiment": 123, "n_time": 50, "random_seed": 6} NOT DONE YET -> SAMPLING FAILS FOR FULL MODEL
    # START WITH RANDOM SEED = 8
    # {"n_experiment": 124, "n_time": 5, "random_seed": 8},
    # {"n_experiment": 125, "n_time": 6, "random_seed": 8},
    # {"n_experiment": 126, "n_time": 7, "random_seed": 8},
    # {"n_experiment": 127, "n_time": 8, "random_seed": 8},
    # {"n_experiment": 128, "n_time": 9, "random_seed": 8},
    # {"n_experiment": 129, "n_time": 10, "random_seed": 8},
    # {"n_experiment": 130, "n_time": 12, "random_seed": 8},
    # {"n_experiment": 131, "n_time": 14, "random_seed": 8},
    # {"n_experiment": 132, "n_time": 16, "random_seed": 8},
    # {"n_experiment": 133, "n_time": 18, "random_seed": 8},
    # {"n_experiment": 134, "n_time": 20, "random_seed": 8},
    # {"n_experiment": 135, "n_time": 30, "random_seed": 8},
    # {"n_experiment": 136, "n_time": 40, "random_seed": 8},
    # {"n_experiment": 137, "n_time": 50, "random_seed": 8},
    # START WITH RANDOM SEED = 9
    # {"n_experiment": 138, "n_time": 5, "random_seed": 9},
    # {"n_experiment": 139, "n_time": 6, "random_seed": 9},
    # {"n_experiment": 140, "n_time": 7, "random_seed": 9},
    # {"n_experiment": 141, "n_time": 8, "random_seed": 9},
    # {"n_experiment": 142, "n_time": 9, "random_seed": 9},
    # {"n_experiment": 143, "n_time": 10, "random_seed": 9},
    # {"n_experiment": 144, "n_time": 12, "random_seed": 9},
    # {"n_experiment": 145, "n_time": 14, "random_seed": 9},
    # {"n_experiment": 146, "n_time": 16, "random_seed": 9},
    # {"n_experiment": 147, "n_time": 18, "random_seed": 9},
    # {"n_experiment": 148, "n_time": 20, "random_seed": 9},
    # {"n_experiment": 149, "n_time": 30, "random_seed": 9},  NOT DONE YET -> SAMPLING FAILS FOR FULL MODEL
    # {"n_experiment": 150, "n_time": 40, "random_seed": 9},
    # {"n_experiment": 151, "n_time": 50, "random_seed": 9},
    # NOW LET'S REDO n_time = 15, in order to have nicer plotting
    # {"n_experiment": 152, "n_time": 15, "random_seed": 1}
    # {"n_experiment": 153, "n_time": 15, "random_seed": 2},
    # {"n_experiment": 154, "n_time": 15, "random_seed": 3},
    # {"n_experiment": 155, "n_time": 15, "random_seed": 4},
    # {"n_experiment": 156, "n_time": 15, "random_seed": 5},
    # {"n_experiment": 157, "n_time": 15, "random_seed": 6},
    # {"n_experiment": 158, "n_time": 15, "random_seed": 7},
    # {"n_experiment": 159, "n_time": 15, "random_seed": 8},
    {"n_experiment": 160, "n_time": 15, "random_seed": 9}
]
GENERATE_FIGURE_3B = False

"""
util function:
    - Generate sbml from yaml
    - generate AMICI model from SBML
    - do a function, that sets all the parameters from theta
    - do a function, that does the amici simulations for given ts

Hoisted out of the per-setup loop below: the base model's SBML/ODEs are
identical across setups, and AMICI's staleness guard is mtime-based, not
content-based -- recompiling the SAME SBML under the SAME model name on a
second loop iteration still raises "the file was modified on disk" (see
tests/verify_amici_model_naming.py::check_shared_name_crashes_even_with_identical_sbml).
Compiling once here and reusing the resulting `amici_model_base` instance
across setups is safe because `simulate_amici_base` re-sets all parameters
and timepoints on every call via `_set_parameter_base`, so no per-setup
state leaks between iterations.
"""
# generate the SBML for the base model (i.e. without temporal uncertainty) --
# only if it isn't already compiled. The base model's content never depends
# on `setup` (only parameter VALUES like THETA_TRUE/N_TIME/N_EXPERIMENT are
# set per setup below, after this compile), so every concurrent array task
# can safely compile once (first-caller-wins, see submit_slurm_array.sh's
# prep step) and reuse the same compiled module read-only, instead of every
# task racing to (re)write the same SBML file and compiled model directory.
yaml_dir = str(APP_DIR / "models" / "conversion_reaction_base.yaml")
sbml_dir = str(APP_DIR / "models" / "conversion_reaction_base.xml")

model_name_base = "Conversion_Reaction_Base"
amici_model_dir = str(APP_DIR / "ode_sbml" / "amici_files")

# AMICI's compiled package ends up at amici_model_dir/model_name_base/__init__.py
# (see amici.import_model_module below), not directly in amici_model_dir -- the
# previous check here (Path(amici_model_dir) / "__init__.py") looked in the wrong
# place and was therefore always False, so every process (prep job AND every
# concurrent array task) unconditionally recompiled the base model, racing on
# amici's DEExporter._prepare_model_folder(), which blindly os.remove()s every
# file in amici_model_dir -- symptom: "FileNotFoundError: ... dJydsigma.cpp"
# when a sibling process's compile already deleted it.
if not (Path(amici_model_dir) / model_name_base / "__init__.py").exists():
    yaml2sbml.yaml2sbml(yaml_dir, sbml_dir)
    amici_importer = amici.SbmlImporter(sbml_dir)
    amici_importer.sbml2amici(model_name_base, amici_model_dir)

# COMPILE_ONLY lets a short prep job compile the base model and exit,
# without running any (multi-hour) setup -- see submit_slurm_array.sh's
# prep stage, which depends the array job on this exiting 0 quickly rather
# than on a full setup finishing within the prep job's time limit.
if os.environ.get("COMPILE_ONLY"):
    print("COMPILE_ONLY set: base model compiled, exiting without running any setup.")
    sys.exit(0)

amici_module = amici.import_model_module(model_name_base, amici_model_dir)
amici_model_base = amici_module.getModel()

"""
Define the Measurement Setup:
"""
"""
Select which setup this process runs.

PETAB_INDEX < 0 (the default): reproduce today's exact behavior -- iterate
`setups` as manually uncommented above (today, just the one active
n_experiment=160 entry), sequentially, in this one process.

PETAB_INDEX >= 0 (set via SLURM_ARRAY_TASK_ID or the PETAB_INDEX env var):
claim ALL_SETUPS[PETAB_INDEX] by position, and exit immediately if that
setup's result already exists -- this is what makes concurrent SLURM array
tasks safe.
"""
PETAB_INDEX = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("PETAB_INDEX", "-1")))

if PETAB_INDEX >= 0:
    setup = ALL_SETUPS[PETAB_INDEX]
    _result_check_path = APP_DIR / f"res_{setup['n_experiment']}" / "result_marginalized.hdf5"
    if _result_check_path.exists():
        print(f"Result for n_experiment={setup['n_experiment']} (index {PETAB_INDEX}) already exists, skipping.")
        sys.exit(0)
    setups = [setup]

for setup in setups:
    # theta lin is defined in Log10 space
    THETA_TRUE = np.array([0.5, 0, 1, 1], dtype=float)
    SIGMA_X = 0.1
    SIGMA_TAU = 1

    # 
    # N_TIME = 5
    N_TIME = setup["n_time"]
    T_MAX = 8
    T_WRONG = np.linspace(0, T_MAX, N_TIME+1, dtype=float)[1:] 

    # OBS = ['A', 'B']
    OBS = ['B']

    T_SIM_END = T_WRONG[-1] + 4 * SIGMA_TAU # the end of the integration for the marginal approach

    Z_MIN = 1e-300

    N_EXPERIMENT = setup["n_experiment"]
    # RANDOM_SEED = 42
    # RANDOM_SEED = 1
    # RANDOM_SEED = 7
    # RANDOM_SEED = 2
    RANDOM_SEED = setup.get("random_seed")
    STORAGE_DIR_BASE = str(APP_DIR / f"res_{N_EXPERIMENT}")

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


    """
    GENERATE SYNTHETIC DATA
    """
    np.random.seed(RANDOM_SEED)

    delta_t = np.random.normal(loc=0.0, scale=SIGMA_TAU, size=T_WRONG.shape)
    T_TRUE = T_WRONG + delta_t
    while np.any(T_TRUE < 0):
        delta_t = np.random.normal(loc=0.0, scale=SIGMA_TAU, size=T_WRONG.shape)
        T_TRUE = T_WRONG + delta_t
        print("Resample T_TRUE")

    X_DATA, _, _ = simulate_amici_base(amici_model_base, THETA_TRUE, T_TRUE)

    X_DATA = X_DATA + np.random.normal(
            loc=0.0, scale=SIGMA_X, size=X_DATA.shape)

    # make directory (may already exist from a previous run of this setup)
    os.makedirs(STORAGE_DIR_BASE, exist_ok=True)

    # save t_true and x_data as .csv
    np.savetxt(f'{STORAGE_DIR_BASE}/t_true.csv', T_TRUE, delimiter=',')
    np.savetxt(f'{STORAGE_DIR_BASE}/x_data.csv', X_DATA, delimiter=',')

    del delta_t


    def objective_wrong_tp(theta_log: np.array):
        """
        Objective function for the wrong time points.
        """
        # Simulate the model with the true parameters
        x_sim, sx_sim, _ = simulate_amici_base(amici_model_base, theta_log, T_WRONG)

        # Compute the objective value (e.g., negative log-likelihood)
        obj = X_DATA.size/2 * np.log(2*np.pi*SIGMA_X**2) + 1/2 * np.sum((X_DATA - x_sim) ** 2) / SIGMA_X ** 2

        # gradients    
        if X_DATA.ndim > 1:
            raise NotImplementedError("Gradient computation for more then just one observable is not supported yet.")
        
        grad = np.zeros_like(theta_log)
        for i in range(len(X_DATA)):
            grad += (x_sim[i] - X_DATA[i]) / SIGMA_X ** 2 * sx_sim[i, :]

        # chain rule for log trafo...
        grad *= np.log(10) * 10 ** theta_log
        return obj, grad


    def objective_full(theta_full: np.array):
        """
        Objective function for the full model. .e. with time points as model parameters
        """
        theta_log = np.array(theta_full[:4], dtype=float) 
        tau = np.array(theta_full[4:], dtype=float)

        if np.any(tau < 0):
            raise NotImplementedError("Negative time points are not supported.")

        # Simulate the model with the true parameters
        x_sim, sx_sim, x_full = simulate_amici_base(amici_model_base, theta_log, tau)

        # Compute the objective value (e.g., negative log-likelihood)
        nll = X_DATA.size/2 * np.log(2*np.pi*SIGMA_X**2) + 1/2 * np.sum((X_DATA - x_sim) ** 2) / SIGMA_X ** 2
        nl_prior = T_WRONG.size/2 * np.log(2*np.pi*SIGMA_TAU**2) + 1/2 * np.sum((tau - T_WRONG) ** 2) / SIGMA_TAU ** 2

        # compute the gradient
        if X_DATA.ndim > 1:
            raise NotImplementedError("Gradient computation for more then just one observable is not supported yet.")

        def f(x, theta_lin):
            """Compute the right hand side of the ODE for the gradient w.r.t tau"""
            # 'k_1 - k_2 * A + k_3 * B'
            # 'k_2 * A - (k_3 + k_4) * B'

            f_x = np.array([
                theta_lin[0] - theta_lin[1] * x[0] + theta_lin[2] * x[1],
                theta_lin[1] * x[0] - (theta_lin[2] + theta_lin[3]) * x[1]
            ])

            if len(OBS) == 2:
                raise NotImplementedError("Gradient computation for more than one observable is not supported yet.")
            if "A" in OBS:
                return f_x[0]
            elif "B" in OBS:
                return f_x[1]
            return None

        grad = np.zeros_like(theta_full)
        for i in range(len(X_DATA)):
            grad[:4] += (x_sim[i] - X_DATA[i]) / SIGMA_X ** 2 * sx_sim[i, :]
            # nll-part
            grad[4+i] += (x_sim[i] - X_DATA[i]) / SIGMA_X ** 2 * f(x_full[i], 10**theta_log) 
            # prior-Part
            grad[4+i] += (tau[i] - T_WRONG[i]) / SIGMA_TAU ** 2

        # chain rule for log trafo...
        grad[:4] *= np.log(10) * 10 ** theta_log[:4]

        return nll + nl_prior, grad


    """
    Let's set up the yaml model, that contains the z-states as parameters
    """
    yaml_model_marginalized = yaml2sbml.YamlModel.load_from_yaml(yaml_dir)
    sbml_dir_marginnalized = f"{STORAGE_DIR_BASE}/conversion_reaction_marginalized.xml"

    yaml_model_marginalized.set_time("time")


    # add ODEs
    if len(OBS) > 1:
        raise NotImplementedError("TBD")
    else:
        if "A" in OBS:
            obs_id = "A"
        else:
            obs_id = "B"

        for i, (t_i, data_i) in enumerate(zip(T_WRONG, X_DATA)):

            lsq_data = f"pow({obs_id} - {data_i}, 2) / {SIGMA_X**2}"
            lsq_time = f"pow(time - {t_i}, 2) / {SIGMA_TAU**2}"
            z_dot = f"exp(-(1/2) * ( ({lsq_data}) + ({lsq_time}) ))"



            yaml_model_marginalized.add_ode(
                state_id=f"z_{i}",
                right_hand_side=z_dot,
                initial_value=0.0
            )

    yaml_model_marginalized.write_to_yaml(f"{STORAGE_DIR_BASE}/extended_model.yaml", overwrite=True)
    yaml_model_marginalized.write_to_sbml(sbml_dir_marginnalized, overwrite=True)


    # import into amici:
    # N_EXPERIMENT is this file's existing unique-per-setup key (also used
    # for STORAGE_DIR_BASE) -- each setup's marginal model has different
    # ODEs (T_WRONG/X_DATA baked into z_dot above), so it needs its own
    # compiled AMICI module name or the next setup's import crashes with
    # "the file was modified on disk" (see tests/verify_amici_model_naming.py).
    # It also needs its own build DIRECTORY, not just its own name: AMICI's
    # sbml2amici() treats amici_model_dir as a shared project root (its
    # CMakeLists.txt/setup.py and the cmake/ninja build_model_ext staging
    # dir are not namespaced per model), so concurrent SLURM array tasks
    # compiling different setups into the same directory race on those
    # shared files -- symptom: "ninja: error: failed recompaction: No such
    # file or directory".
    model_name_marginal = f"Conversion_Reaction_marginal_{N_EXPERIMENT}"
    amici_model_dir = str(APP_DIR / "ode_sbml" / "amici_files" / f"marginal_{N_EXPERIMENT}")
    os.makedirs(amici_model_dir, exist_ok=True)

    amici_importer_marginal = amici.SbmlImporter(sbml_dir_marginnalized)
    amici_importer_marginal.sbml2amici(model_name_marginal, amici_model_dir)

    amici_module_marginal = amici.import_model_module(model_name_marginal, amici_model_dir)
    amici_model_marginal = amici_module_marginal.getModel()

    def _set_parameter_marginal(amici_model_marginal, theta_log: np.array):
        """Set the parameters for the marginal Model"""
        amici_model_marginal = _set_parameter_base(amici_model_marginal, theta_log)
        
        return amici_model_marginal


    def simulate_amici_marginal(
        amici_model_marginal, 
        theta_log: np.array, 
        t_sim: np.array):
        """
        Simulate the AMICI model for the base case.
        """
        amici_model_marginal = _set_parameter_marginal(amici_model_marginal, theta_log)
        amici_model_marginal.setTimepoints(t_sim)

        # Run the simulation
        solver = amici_model_marginal.getSolver()
        solver.setSensitivityOrder(1)

        rdata = amici.runAmiciSimulation(amici_model_marginal, solver)

        return rdata.x[:, 2:], rdata.sx[:, :, 2:]


    def objective_marginalized(theta_log: np.array):
        """
        Objective function for the marginalized model.
        """
        # Simulate the model with the true parameters
        z_sim, szim_sim = simulate_amici_marginal(
            amici_model_marginal,
            theta_log, 
            np.array([0, T_SIM_END], dtype=float))
        
        res = 0
        grad = np.zeros_like(theta_log)
        for i in range(len(z_sim[-1])):
            
            if z_sim[-1, i] > Z_MIN:

                z_i = z_sim[-1, i]
                res -= np.log(z_i)

                grad -= szim_sim[-1, :, i] / z_i

                if len(OBS) == 1:
                    res += np.log(2*np.pi*SIGMA_X * SIGMA_TAU)
                else:
                    raise NotImplementedError("Here there is some more work")

            else:
                res -= np.log(Z_MIN)

        # chain rule for log trafo:
        grad *= np.log(10) * 10 ** theta_log
        return res, grad


    """
    Now finally the moment of truth: We define all the pypesto problems, 
    in order to optimize & sample.
    """
    # X_FIXED_INDICES = None
    # X_FIXED_VALS = None

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

    np.random.seed(RANDOM_SEED)
    
    pypesto_objective_wrong_tp = pypesto.Objective(fun=objective_wrong_tp, grad=True)
    problem_wrong_tp = pypesto.Problem(
        objective=pypesto_objective_wrong_tp,
        lb=LB_WRONG_TP,
        ub=UB_WRONG_TP)

    # full objective
    pypesto_objective_full = pypesto.Objective(fun=objective_full, grad=True)
    problem_full = pypesto.Problem(
        objective=pypesto_objective_full,
        lb=LB_FULL,
        ub=UB_FULL)

    # marginalized objective
    pypesto_objective_marginalized = pypesto.Objective(fun=objective_marginalized, grad=True)
    problem_marginalized = pypesto.Problem(
        objective=pypesto_objective_marginalized,
        lb=LB_MARGINAL,
        ub=UB_MARGINAL)


    result_wrong_tp = optimize.minimize(
        problem_wrong_tp, 
        n_starts=N_STARTS, 
        optimizer=OPTIMIZER, 
        startpoint_method=STARTPOINT_METHOD)
    sample.sample(problem_wrong_tp, n_samples=N_SAMPLES, sampler=SAMPLER, result=result_wrong_tp)
    store.write_result(result_wrong_tp, f'{STORAGE_DIR_BASE}/result_wrong_tp.hdf5', overwrite=True)


    # full result
    result_full = optimize.minimize(
        problem_full, 
        n_starts=N_STARTS, 
        optimizer=OPTIMIZER, 
        startpoint_method=STARTPOINT_METHOD)
    sample.sample(problem_full, n_samples=N_SAMPLES, sampler=SAMPLER, result=result_full)
    store.write_result(result_full, f'{STORAGE_DIR_BASE}/result_full.hdf5', overwrite=True)

    # marginalized result
    result_marginalized = optimize.minimize(
        problem_marginalized, 
        n_starts=N_STARTS, 
        optimizer=OPTIMIZER, 
        startpoint_method=STARTPOINT_METHOD)
    sample.sample(problem_marginalized, n_samples=N_SAMPLES, sampler=SAMPLER, result=result_marginalized)
    store.write_result(result_marginalized, f'{STORAGE_DIR_BASE}/result_marginalized.hdf5', overwrite=True)


    """
    Compute all the ESS & Sampling statistics:

    This needs to be this complicated for the case, that ess or burn in is None (happens if the sampler did not converge...)
    """
    ess_wrong_tp = sample.effective_sample_size(result_wrong_tp)
    ess_per_second_wrong_tp = None
    ess_full = sample.effective_sample_size(result_full)
    ess_per_second_full = None
    ess_marginalized = sample.effective_sample_size(result_marginalized)
    ess_per_second_marginalized = None

    if ess_wrong_tp:
        ess_wrong_tp = float(ess_wrong_tp)
        ess_per_second_wrong_tp = float(ess_wrong_tp / result_wrong_tp.sample_result.time)

    if ess_full:
        ess_full = float(ess_full)
        ess_per_second_full = float(ess_full / result_full.sample_result.time)

    if ess_marginalized:
        ess_marginalized = float(ess_marginalized)
        ess_per_second_marginalized = float(ess_marginalized / result_marginalized.sample_result.time)

    geweke_burn_in_wrong_tp = sample.geweke_test(result_wrong_tp)
    geweke_burn_in_full = sample.geweke_test(result_full)
    geweke_burn_in_marginalized = sample.geweke_test(result_marginalized)

    if geweke_burn_in_wrong_tp is not None:
        geweke_burn_in_wrong_tp = float(geweke_burn_in_wrong_tp)

    if geweke_burn_in_full is not None:
        geweke_burn_in_full = float(geweke_burn_in_full)

    if geweke_burn_in_marginalized is not None:
        geweke_burn_in_marginalized = float(geweke_burn_in_marginalized)


    # save setup
    os.makedirs(STORAGE_DIR_BASE, exist_ok=True)
    with open(f'{STORAGE_DIR_BASE}/measurement_setup.yaml', 'w') as f:
        yaml.dump({
            'true_parameters': str(THETA_TRUE),
            'sigma_x': float(SIGMA_X),
            'sigma_tau': float(SIGMA_TAU),
            't_wrong': str(T_WRONG),
            'observables': OBS,
            'n_experiment': int(N_EXPERIMENT),
            'random_seed': int(RANDOM_SEED),
            't_sim_end': float(T_SIM_END),
            't_true': str(T_TRUE),
            'x_data': str(X_DATA),
            'n_data': int(X_DATA.size),
            'lb': str(LB_WRONG_TP),
            'ub': str(UB_WRONG_TP),
            'lb_full': str(LB_FULL),
            'ub_full': str(UB_FULL),
            'n_starts': N_STARTS,
            'n_samples': N_SAMPLES,

            'run_time_optimization_wrong_tp': float(np.sum(result_wrong_tp.optimize_result.time)),
            'run_time_optimization_full': float(np.sum(result_full.optimize_result.time)),
            'run_time_optimization_marginalized': float(np.sum(result_marginalized.optimize_result.time)),

            'run_time_sampling_wrong_tp': float(result_wrong_tp.sample_result.time),
            'run_time_sampling_full': float(result_full.sample_result.time),
            'run_time_sampling_marginalized': float(result_marginalized.sample_result.time),

            'ESS_wrong_tp': ess_wrong_tp,
            'ESS_full': ess_full,
            'ESS_marginalized': ess_marginalized,

            # 'ESS_per_second_wrong_tp': float(sample.effective_sample_size(result_wrong_tp) / result_wrong_tp.sample_result.time),
            'ESS_per_second_full': ess_per_second_full,
            'ESS_per_second_marginalized': ess_per_second_marginalized,

            'geweke_burn_in_wrong_tp': geweke_burn_in_wrong_tp,
            'geweke_burn_in_full': geweke_burn_in_full,
            'geweke_burn_in_marginalized': geweke_burn_in_marginalized
        }, f)


    """
    Save all the visualizations, in order to have everything in one go:
    """
    import pypesto.visualize as visualize
    import matplotlib.pyplot as plt
    """
    reference_wrong_tp = visualize.ReferencePoint(
        x=THETA_TRUE,
        fval=objective_wrong_tp(np.array(THETA_TRUE, dtype=float))[0]
        )


    reference_full = visualize.ReferencePoint(
        x=np.concatenate([THETA_TRUE, T_TRUE]), 
        fval=objective_full(np.concatenate([THETA_TRUE, T_TRUE], dtype=float))[0]
        )

    reference_marginalized = visualize.ReferencePoint(
        x=THETA_TRUE, 
        fval=objective_marginalized(np.array(THETA_TRUE, dtype=float))[0]
    )
    """
    os.makedirs(f"{STORAGE_DIR_BASE}/img", exist_ok=True)


    # Sampling fval traces
    visualize.sampling_fval_traces(result_wrong_tp)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/wrong_tp_sampling_fval_traces.png')

    visualize.sampling_fval_traces(result_full)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/full_sampling_fval_traces.png')

    visualize.sampling_fval_traces(result_marginalized)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/marginalized_sampling_fval_traces.png')

    # Parameter traces
    visualize.sampling_parameter_traces(result_wrong_tp)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/wrong_tp_sampling_parameter_traces.png')

    visualize.sampling_parameter_traces(result_full)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/full_sampling_parameter_traces.png')

    visualize.sampling_parameter_traces(result_marginalized)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/marginalized_sampling_parameter_traces.png')

    # 1D Marginals
    visualize.sampling_1d_marginals(result_wrong_tp)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/wrong_tp_sampling_1d_marginals.png')

    visualize.sampling_1d_marginals(result_full)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/full_sampling_1d_marginals.png')

    visualize.sampling_1d_marginals(result_marginalized)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/marginalized_sampling_1d_marginals.png')

    # waterfall plots
    visualize.waterfall(result_wrong_tp)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/wrong_tp_sampling_waterfall.png')

    visualize.waterfall(result_full)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/full_sampling_waterfall.png')

    visualize.waterfall(result_marginalized)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/marginalized_sampling_waterfall.png')

    # parameter plot
    visualize.parameters(result_wrong_tp)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/wrong_tp_sampling_parameters.png')

    visualize.parameters(result_full)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/full_sampling_parameters.png')

    visualize.parameters(result_marginalized)
    plt.savefig(f'{STORAGE_DIR_BASE}/img/marginalized_sampling_parameters.png')

    plt.close('all')
    """
    Visualization for Paper:
    """

    """
    Fig 3A: Plot Model + data + model fitts
    """
    import matplotlib.pyplot as plt

    H = 0.05 # step size in time for simulations
    T_VIS = np.array([H * i for i in range(int((T_WRONG[-1]+2)/H))])

    COLOR_FULL = [0, 78/255, 159/255] # blue
    COLOR_MARGINALIZED = [252/255, 186/255, 0] # orange
    COLOR_WRONG_TP = [144/255, 144/255, 133/255] # grey
    COLOR_GROUND_TRUTH = [0, 0, 0] # black

    parameter_dict = {
        "$\\theta_{true}$": 
            {"theta": THETA_TRUE, 
            "color": COLOR_GROUND_TRUTH},
        "$J_{wrong tp}$": 
            {"theta": result_wrong_tp.optimize_result.x[0],
            "color": COLOR_WRONG_TP},
        "$J_{full}$": 
            {"theta": result_full.optimize_result.x[0], 
            "color": COLOR_FULL},
        "$J_{marginalized}$": 
            {"theta": result_marginalized.optimize_result.x[0], 
            "color": COLOR_MARGINALIZED}
    }

    plt.figure(figsize=(8, 6))


    # plot the data with error bars
    plt.errorbar(
        T_WRONG, X_DATA,
        yerr=SIGMA_X, xerr=SIGMA_TAU,
        fmt='o',          # 'o' = points
        capsize=3,        # little lines at ends of errorbars
        color=COLOR_WRONG_TP,
        label="Data"
    )



    # plot the simulations....
    for label, val in parameter_dict.items():
        theta = val["theta"]
        color = val["color"]

        x, _, _ = simulate_amici_base(
            amici_model_base,
            theta_log=theta[:4],
            t_sim=T_VIS)

        plt.plot(T_VIS, x, label=label, color=color)

    plt.xlim((0, T_VIS[-1]))
    plt.ylim((0, np.max(X_DATA) + SIGMA_X + 0.1))
    plt.legend(loc='lower right')

    plt.xlabel('Time')
    plt.ylabel('$x_B$')
    plt.title("Model with Data")

    # remove spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # save figure
    plt.savefig(f"{STORAGE_DIR_BASE}/img/figure_3A.png", bbox_inches='tight')

    plt.close('all')
    """
    Figure 3B: Model with data + uncertainty bands
    """

    parameter_dict = {
        "$J_{wrong tp}$": 
            {"result": result_wrong_tp,
            "theta_max": result_wrong_tp.optimize_result.x[0],
            "color": COLOR_WRONG_TP},
        "$J_{full}$": 
            {"result": result_full, 
            "color": COLOR_FULL},
        "$J_{marginalized}$": 
            {"result": result_marginalized,
            "color": COLOR_MARGINALIZED}
    }

    if GENERATE_FIGURE_3B:

        for label, val in parameter_dict.items():
            plt.figure(figsize=(8, 6))

            # plot the ground truth
            x, _, _ = simulate_amici_base(
                amici_model_base,
                theta_log=THETA_TRUE,
                t_sim=T_VIS)

            plt.plot(T_VIS, x, label="Ground Truth", color=COLOR_GROUND_TRUTH, linestyle='--')

            result = val["result"]
            color = val["color"]

            x_sims = []
            theta_old = None
            for theta in result.sample_result.trace_x[0]:
                if theta_old is not None and (theta != theta_old).any():
                    x, _, _ = simulate_amici_base(
                        amici_model_base,
                        theta_log=theta[:4],
                        t_sim=T_VIS)
                    x_sims.append(x)
                
                theta_old = theta
                x_sims.append(x)

            # Plot the mean
            plt.plot(T_VIS, np.mean(x_sims, axis=0), label=f"{label} Median", color=color)

            # Plot the uncertainty bands
            plt.fill_between(
                T_VIS,
                np.percentile(x_sims, 5, axis=0),
                np.percentile(x_sims, 95, axis=0),
                color=color,
                alpha=0.2,
                label=f"{label} 90% CI"
            )


            plt.xlim((0, T_VIS[-1]))
            plt.ylim((0, np.max(X_DATA) + SIGMA_X + 0.1))
            plt.legend(loc='lower right')

            plt.xlabel('Time')
            plt.ylabel('$x_B$')
            plt.title("Predictions with uncertainty bands")

            # remove spines
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)

            # save figure
            plt.savefig(f"{STORAGE_DIR_BASE}/img/figure_3B_{label}.png", bbox_inches='tight')


    plt.close('all')
    """
    Figure 3C: Kernel Density estimates of the samples:
    """
    parameter_dict = {
        "$J_{wrong tp}$": 
            {"result": result_wrong_tp,
            "theta_max": result_wrong_tp.optimize_result.x[0],
            "color": COLOR_WRONG_TP},
        "$J_{full}$": 
            {"result": result_full, 
            "color": COLOR_FULL},
        "$J_{marginalized}$": 
            {"result": result_marginalized,
            "color": COLOR_MARGINALIZED}
    }




    for n_parameter in [0, 1, 2, 3]:
        plt.figure(figsize=(8, 6))

        runs = [
            result_wrong_tp.sample_result.trace_x[0][:, n_parameter], 
            result_full.sample_result.trace_x[0][:, n_parameter], 
            result_marginalized.sample_result.trace_x[0][:, n_parameter] 
        ]

        colors = [COLOR_WRONG_TP, COLOR_FULL, COLOR_MARGINALIZED]
        labels = ["$J_{wrong tp}$", "$J_{full}$", "$J_{marginalized}$"]

        # Define common bin edges
        bins = np.linspace(min(map(np.min, runs)), max(map(np.max, runs)), 30)
        bin_width = bins[1] - bins[0]
        bar_width = bin_width / len(runs)


        for run, label, color in zip(runs, labels, colors):
            # histogram counts
            counts, _ = np.histogram(run, bins=bins)
            # shift each run horizontally so bars are side by side
            plt.hist(
                run, 
                bins=bins, 
                density=True, 
                histtype="step", 
                linewidth=3, 
                label=label,
                color=color)
            #plt.plot(bins[:-1], counts, color=color, linewidth=2)

        plt.legend()
        plt.xlabel(f"$\\theta_{n_parameter + 1}$")
        plt.ylabel("density")
        plt.title(f"Histogram for $\\theta_{n_parameter + 1}$")
        plt.ylim((0, None))
        plt.xlim((np.min(bins), np.max(bins)))
        plt.legend()

        # remove spines
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        # save figure
        plt.savefig(f"{STORAGE_DIR_BASE}/img/figure_3C_theta_{n_parameter + 1}.png", bbox_inches='tight')

    plt.close('all')
