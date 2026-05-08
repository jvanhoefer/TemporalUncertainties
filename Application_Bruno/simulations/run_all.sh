#!/bin/bash

caffeinate -i &
CAFFEINATE_PID=$!

trap "kill $CAFFEINATE_PID" EXIT


while true; do
    echo "Starting run at $(date)"
    
    # echo "Running marginalized_multi_tau.py..."
    # "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/.venv/bin/python" "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/marginalized_multi_tau.py"
    # echo "Running marginalized_shared_tau.py..."
    # "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/.venv/bin/python" "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/marginalized_shared_tau.py"

    echo "Running wrong_tp.py..."
    # "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/.venv/bin/python" "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/wrong_tp.py"

    # echo "Running full_multi_tau.py..."
    # "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/.venv/bin/python" "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/full_multi_tau.py"    
    # echo "Running full_shared_tau.py..."
    # "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/.venv/bin/python" "/Users/jakobvanhoefer/Documents/PhD/Projects/Temporal Uncertainties/Application_Bruno/simulations/full_shared_tau.py"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Script failed with exit code $EXIT_CODE, stopping."
        break
    fi
    
    echo "Run finished at $(date)"
done

echo "All runs completed."