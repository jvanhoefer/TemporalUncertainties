#!/bin/bash
# Submit one Bruno fitting script across every PEtab variant as a SLURM
# job array, preceded by a one-time compile step.
#
# Usage: ./submit_slurm_array.sh <script-name> [after-job-id]
#   <script-name>   one of: wrong_tp, full_multi_tau, full_shared_tau,
#                   marginalized_multi_tau, marginalized_shared_tau
#   [after-job-id]  optional -- if given, this whole chain's prep job waits
#                   (--dependency=afterany) on that job id first, so it
#                   never runs concurrently with whatever that job id
#                   belongs to. Used by submit_all_bruno.sh to run the 5
#                   objective types one after another instead of all at
#                   once. Leave unset to start immediately.
#
# Settings (environment variables, defaults in brackets):
#   ALLOWED_NODES   comma-separated nodes the jobs may run on
#                   [empty = any node]. Implemented as --exclude of every
#                   other node: with --nodes=1, --nodelist of several nodes
#                   would require ALL of them, and --nodelist can't be
#                   changed on a pending array (ExcNodeList can).
#   CPUS_PER_TASK   CPUs (= MultiThreadEngine threads) per array task [2].
#                   The objectives are GIL-bound Python: 2 threads keep
#                   ~84% of the CPUs busy, 8 only ~34%.
#   MAX_CONCURRENT  max. running tasks of this array [48]
#   ARRAY_TIME      wall-time limit per array task [5-00:00:00]
# Example: ALLOWED_NODES=node01,node02 MAX_CONCURRENT=24 ./submit_slurm_array.sh wrong_tp
#
# This does not replace run_all.sh -- run_all.sh remains the local,
# single-machine, sequential fallback. This script is the cluster path.
#
# All informational progress lines go to stderr; the LAST line on stdout
# is always this chain's array job id, so callers can do:
#   LAST_JOB=$(./submit_slurm_array.sh wrong_tp | tail -1)
#   ./submit_slurm_array.sh full_multi_tau "$LAST_JOB"

set -euo pipefail

SCRIPT_NAME="${1:?Usage: $0 <script-name> [after-job-id]}"
AFTER_JOB="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PETAB_DIR="$SCRIPT_DIR/PEtab_versions"
PYTHON="${PYTHON:-python3}"
MAIL_USER="${MAIL_USER:-}"
# Optional e-mail notification: set MAIL_USER=you@example.org to enable.
MAIL_ARGS=()
if [ -n "$MAIL_USER" ]; then
    MAIL_ARGS=(--mail-type=END,FAIL --mail-user="$MAIL_USER")
fi
LOG_DIR="$SCRIPT_DIR/log"
mkdir -p "$LOG_DIR"

ALLOWED_NODES="${ALLOWED_NODES:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
MAX_CONCURRENT="${MAX_CONCURRENT:-48}"
EXCLUDE_NODES=""
if [ -n "$ALLOWED_NODES" ]; then
    ALL_NODES=$(sinfo -h -N -o %N | sort -u)
    for node in ${ALLOWED_NODES//,/ }; do
        if ! grep -qxF "$node" <<< "$ALL_NODES"; then
            echo "ERROR: node '$node' from ALLOWED_NODES not found by sinfo (known: $(paste -sd, - <<< "$ALL_NODES"))." >&2
            exit 1
        fi
    done
    EXCLUDE_NODES=$(grep -vxF -f <(tr ',' '\n' <<< "$ALLOWED_NODES") <<< "$ALL_NODES" | paste -sd, - || true)
fi
echo "Allowed nodes: ${ALLOWED_NODES:-any} (excluding: ${EXCLUDE_NODES:-none}); $CPUS_PER_TASK CPUs/task, max $MAX_CONCURRENT tasks at once" >&2

# Optional shared (repo-root) environment setup (module loads, venv
# activation, ...), used by every submit_slurm_array.sh in this repo. Not
# required -- only sourced if you create it; kept out of this repo since
# it's cluster-specific.
if [ -f "$REPO_ROOT/prerequisites.sh" ]; then
    source "$REPO_ROOT/prerequisites.sh" >&2
fi

N_VARIANTS=$(find "$PETAB_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
if [ "$N_VARIANTS" -eq 0 ]; then
    echo "No PEtab variants found under $PETAB_DIR -- run generate_petabs.py first." >&2
    exit 1
fi
LAST_INDEX=$((N_VARIANTS - 1))

echo "Found $N_VARIANTS PEtab variants (array indices 0-$LAST_INDEX)" >&2

# marginalized_*_tau additionally sample (200,000 draws) on top of the
# 1,000-start optimization every objective runs -- give them more wall
# time. These are still estimates: verify against a real timed run on your
# cluster and adjust once you have data.
ARRAY_TIME="${ARRAY_TIME:-5-00:00:00}"

# Stage 1: prep -- compile the AMICI model(s) once, before any array task
# starts, so concurrent array tasks never race on a first-time compile.
# COMPILE_ONLY makes every script exit right after compiling instead of
# running a (multi-hour) fit, so this job finishes well within its time
# limit and --dependency=afterok below actually triggers (a naive
# "PETAB_INDEX=0" prep job would instead run a full variant and get killed
# by the time limit before completing, so the array would never start).
PREP_DEPENDENCY=()
if [ -n "$AFTER_JOB" ]; then
    # afterany, not afterok: this chain's own correctness doesn't depend on
    # the PREVIOUS objective type's chain having succeeded -- we only want
    # to avoid running at the same time as it.
    PREP_DEPENDENCY=(--dependency="afterany:$AFTER_JOB")
fi
PREP_JOB=$(sbatch --parsable \
    ${PREP_DEPENDENCY[@]+"${PREP_DEPENDENCY[@]}"} \
    --job-name="${SCRIPT_NAME}_prep" \
    --nodes=1 \
    --ntasks=1 \
    ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
    --cpus-per-task=1 \
    --time=00:30:00 \
    ${MAIL_ARGS[@]+"${MAIL_ARGS[@]}"} \
    --output="$LOG_DIR/${SCRIPT_NAME}_prep_%A.out" \
    --wrap="COMPILE_ONLY=1 $PYTHON $SCRIPT_DIR/${SCRIPT_NAME}.py")

echo "Submitted prep job $PREP_JOB${AFTER_JOB:+ (after $AFTER_JOB)}" >&2

# Stage 2: array -- one task per PEtab variant, each claiming its own
# variant via SLURM_ARRAY_TASK_ID (see Task 1 of the plan this script
# implements). %MAX_CONCURRENT caps the number of concurrently running
# tasks so the array doesn't flood the cluster. SLURM_CPUS_PER_TASK sets the
# number of MultiThreadEngine threads in every fitting script.
ARRAY_JOB=$(sbatch --parsable \
    --dependency=afterok:"$PREP_JOB" \
    --job-name="${SCRIPT_NAME}_array" \
    --nodes=1 \
    --ntasks=1 \
    ${EXCLUDE_NODES:+--exclude="$EXCLUDE_NODES"} \
    --cpus-per-task="$CPUS_PER_TASK" \
    --array=0-"$LAST_INDEX"%"$MAX_CONCURRENT" \
    --time="$ARRAY_TIME" \
    ${MAIL_ARGS[@]+"${MAIL_ARGS[@]}"} \
    --output="$LOG_DIR/${SCRIPT_NAME}_array_%A_%a.out" \
    --wrap="$PYTHON $SCRIPT_DIR/${SCRIPT_NAME}.py")

echo "Submitted array job $ARRAY_JOB (depends on $PREP_JOB)" >&2

echo "$ARRAY_JOB"
