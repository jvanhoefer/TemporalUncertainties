# Applied didactic: README:

This code contains all the implementations for the minimal example:

```
x' = -theta *  x
x(0) = 1
```

With measurements at `t=T_TRUE = [0.5, 2]` + N(0, 0.5) noise on t and N(0, 0.1) noise on x.  `THETA_TRUE = 1`.

For the implementation details & visualizations, see `simulations.py` & `visualizations.py`. The simulation file does all the simulation and stores the results, while `visualizations.py` loads these results for visualizations and stores the corresponding figures.