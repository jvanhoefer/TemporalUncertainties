# Bruno Example: Readme:

This folder implements two methods to extend the Bruno et al. model [1](https://academic.oup.com/jxb/article/67/21/5993/2738937):

Experimental Setup: 

- The `multi_tau` version(s) have an individual value for tau for every data point.
- The `shared_tau` version(s) have a shared value for tau for every data point within an experimental condition, that was taken at the same time point (i.e. when condition and time point is the same, tau is shared)

Note that in the paper, only the `multi_tau` methods are presented.

Further, there are two algorithmic approaches, that are tested:

- `marginalized` means, that tau is integrated out via the ODE solver.
- `full` means, that tau remains a model parameter for fitting/sampling (in the paper "Joint Approach").

Hence there are in the end four `pypesto` problems, which we try out: `problem_marginalized_multi_tau`, `problem_marginalized_shared_tau`, `problem_full_multi_tau` and `problem_full_shared_tau`. Further there is the original implementation, neglecting temporal uncertainties, given in `problem_wrong_tp`.

The implementations of the different objectives are in the corresponding files in the `./simulations` folder. Further you can find the code for the Figures in the paper in `visualizations_Fig05.py` and `visualizations_Fig06.py`

The PEtab files for the different temporal perturbations (including different replicates)
are generated from `petab_bruno_original/` by the `generate_petabs.py` script. Replicate `r` (1-19) uses random seed `r`; see the top-level README for the exact commands.
`simulations/full_parametrization_df.tsv` maps every data point to its time parameter tau; `simulations/shared_parametrization_df.tsv` does the same for the shared-tau version.

Further scripts not needed for the paper figures: `debug_and_plot.py` (trajectory plots per condition, written to `data/plots/`), `figures.py` (objective-value summary across all results), `utils.py` (truncated-Gaussian helper) and `run_all.sh` (local sequential loop over the fitting scripts).