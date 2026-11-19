# Applied Conversion Reaction: README:

This code contains all the implementations for the example:

```
x_1' =   k_{-2}*x_2 - k_2*x_1 + k_1
x_2' = - k_{-2}*x_2 + k_2*x_1 - k_3 * x_2

x_1(0) = 1
x_2(0) = 0
```

The model is defined in `models/conversion_reaction_base.yaml` (converted to SBML with yaml2sbml).

- `simulation_cr.py` runs one measurement setup (selected by `PETAB_INDEX`) and stores the results in `res_<n_experiment>/`.
- `postprocessing_cr.py` collects all `res_*` folders into `per_experiment_df.csv` and `per_setup_df.csv`, including MCMC convergence diagnostics from `mcmc_diagnostics.py`.
- `visualization_cr.py` loads these results and stores Figures 3 and 4.

`mcmc_diagnostics.py` computes the Geweke test, burn-in and effective sample size on the mechanistic parameters. It replaces the corresponding pyPESTO functions, whose spectral-density estimate does not centre the chain windows.