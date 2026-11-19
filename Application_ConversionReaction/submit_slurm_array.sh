#!/bin/bash
# Submit simulation_cr.py across every historical setup in ALL_SETUPS as a
# SLURM job array, followed by a dependent aggregation step.
#
# Usage: ./submit_slurm_array.sh
#
# This does not replace running simulation_cr.py directly for a single
# local setup -- that remains the sequential, single-machine fallback
# (PETAB_INDEX unset). This script is the cluster path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../" && pwd)"
PYTHON="${PYTHON:-python3}"
MAIL_USER="${MAIL_USER:-}"
# Optional e-mail notification: set MAIL_USER=you@example.org to enable.
MAIL_ARGS=()
if [ -n "$MAIL_USER" ]; then
    MAIL_ARGS=(--mail-type=END,FAIL --mail-user="$MAIL_USER")
fi
LOG_DIR="$SCRIPT_DIR/log"
mkdir -p "$LOG_DIR"

# Optional per-app environment setup (module loads, venv activation, ...).
# Not required -- only sourced if you create it; kept out of this repo
# since it's cluster-specific.
if [ -f "$REPO_ROOT/prerequisites.sh" ]; then
    source "$REPO_ROOT/prerequisites.sh"
    echo "sourcing preprequisites"
fi

N_SETUPS=$("$PYTHON" -c "
import re
text = open('$SCRIPT_DIR/simulation_cr.py').read()
block = text.split('ALL_SETUPS = [')[1].split(']')[0]
print(len(re.findall(r'\"n_experiment\"', block)))
")
if [ "$N_SETUPS" -eq 0 ]; then
    echo "No entries found in ALL_SETUPS in simulation_cr.py -- run Task 6 first." >&2
    exit 1
fi
LAST_INDEX=$((N_SETUPS - 1))

echo "Found $N_SETUPS setups (array indices 0-$LAST_INDEX)" >&2

# Stage 1: prep -- compile the base AMICI model once, before any array task
# starts, so concurrent array tasks never race on the first-time compile
# (or on regenerating the shared SBML file). COMPILE_ONLY makes this exit
# right after compiling instead of running a (multi-hour) setup, so this
# job finishes well within its time limit and --dependency=afterok below
# actually triggers.
PREP_JOB=$(sbatch --parsable \
    --job-name="conversion_reaction_prep" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --time=00:30:00 \
    ${MAIL_ARGS[@]+"${MAIL_ARGS[@]}"} \
    --output="$LOG_DIR/conversion_reaction_prep_%A.out" \
    --wrap="COMPILE_ONLY=1 $PYTHON $SCRIPT_DIR/simulation_cr.py")

echo "Submitted prep job $PREP_JOB" >&2

# Stage 2: array -- one task per setup, each claiming its own entry via
# SLURM_ARRAY_TASK_ID (see Task 6 of the plan this script implements).
# %20 caps this array at 20 concurrently running tasks so it doesn't flood
# the cluster, regardless of how many setups are in ALL_SETUPS. Each setup
# runs a 100-start optimization + 500,000-sample MCMC per objective
# (wrong_tp/full/marginalized) -- --time is a generous starting estimate,
# not a measured value; verify against a real timed run and adjust.
# --cpus-per-task stays at 1: simulation_cr.py uses
# pyPESTO's default sequential engine, not MultiThreadEngine, so extra
# cores here wouldn't be used.
ARRAY_JOB=$(sbatch --parsable \
    --dependency=afterok:"$PREP_JOB" \
    --job-name="conversion_reaction_array" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --array=0-"$LAST_INDEX"%20 \
    --time=1-00:00:00 \
    ${MAIL_ARGS[@]+"${MAIL_ARGS[@]}"} \
    --output="$LOG_DIR/conversion_reaction_array_%A_%a.out" \
    --wrap="$PYTHON $SCRIPT_DIR/simulation_cr.py")

echo "Submitted array job $ARRAY_JOB (depends on $PREP_JOB)" >&2

# Stage 3: aggregate -- only meaningful once every setup has a result;
# postprocessing_cr.py builds per_experiment_df.csv/per_setup_df.csv,
# visualization_cr.py builds Figure 4 from them.
AGG_JOB=$(sbatch --parsable \
    --dependency=afterok:"$ARRAY_JOB" \
    --job-name="conversion_reaction_aggregate" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --time=01:00:00 \
    ${MAIL_ARGS[@]+"${MAIL_ARGS[@]}"} \
    --output="$LOG_DIR/conversion_reaction_aggregate_%A.out" \
    --wrap="$PYTHON $SCRIPT_DIR/postprocessing_cr.py && $PYTHON $SCRIPT_DIR/visualization_cr.py")

echo "Submitted aggregation job $AGG_JOB (depends on $ARRAY_JOB)" >&2
echo "$AGG_JOB"
