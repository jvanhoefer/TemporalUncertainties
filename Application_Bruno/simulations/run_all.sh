#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

caffeinate -i &
CAFFEINATE_PID=$!

trap "kill $CAFFEINATE_PID" EXIT


while true; do
    echo "Starting run at $(date)"

    # echo "Running marginalized_multi_tau.py..."
    # "$PYTHON" "$SCRIPT_DIR/marginalized_multi_tau.py"
    # echo "Running marginalized_shared_tau.py..."
    # "$PYTHON" "$SCRIPT_DIR/marginalized_shared_tau.py"

    echo "Running wrong_tp.py..."
    "$PYTHON" "$SCRIPT_DIR/wrong_tp.py"

    # echo "Running full_multi_tau.py..."
    # "$PYTHON" "$SCRIPT_DIR/full_multi_tau.py"
    # echo "Running full_shared_tau.py..."
    # "$PYTHON" "$SCRIPT_DIR/full_shared_tau.py"

    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Script failed with exit code $EXIT_CODE, stopping."
        break
    fi
    
    echo "Run finished at $(date)"
done

echo "All runs completed."