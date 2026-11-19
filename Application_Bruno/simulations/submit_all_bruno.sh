#!/bin/bash
# Submit Bruno objective types' chains (prep -> array each), one after
# another. Each objective type's prep job depends (--dependency=afterany)
# on the previous objective type's array job, so no two objective types'
# arrays run at the same time. Node list, CPUs per task and concurrency are
# set in submit_slurm_array.sh (ALLOWED_NODES, CPUS_PER_TASK, MAX_CONCURRENT).
#
# Usage: ./submit_all_bruno.sh [script-name ...]
#   Without arguments: the default rerun set below.
#   Example: ALLOWED_NODES=node01,node02 ./submit_all_bruno.sh full_shared_tau
#
# This is a convenience wrapper. You can still submit any single objective
# type on its own with ./submit_slurm_array.sh <script-name> -- this script
# just chains all 5 calls together so they queue sequentially instead of
# all starting at once.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The three objectives used in the paper (Figures 5 and 6).
DEFAULT_SCRIPTS=(wrong_tp full_multi_tau marginalized_multi_tau)
if [ "$#" -gt 0 ]; then
    SCRIPTS=("$@")
else
    SCRIPTS=("${DEFAULT_SCRIPTS[@]}")
fi
for name in "${SCRIPTS[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$name.py" ]; then
        echo "ERROR: unknown script '$name' (no $SCRIPT_DIR/$name.py)." >&2
        exit 1
    fi
done

PREV_JOB=""
for name in "${SCRIPTS[@]}"; do
    echo "=== Submitting $name (after: ${PREV_JOB:-nothing}) ===" >&2
    PREV_JOB=$("$SCRIPT_DIR/submit_slurm_array.sh" "$name" "$PREV_JOB" | tail -1)
    if ! [[ "$PREV_JOB" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $name chain did not return a job id (got: '$PREV_JOB'), stopping." >&2
        exit 1
    fi
    echo "=== $name array job: $PREV_JOB ===" >&2
done

echo "Submitted ${#SCRIPTS[@]} Bruno chains sequentially (${SCRIPTS[*]}). Final job: $PREV_JOB"
