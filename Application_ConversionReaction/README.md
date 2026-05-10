# Applied Conversion Reaction: README:

This code contains all the implementations for the example:

```
x_1' =   k_{-2}*x_2 - k_2*x_1 + k_1
x_2' = - k_{-2}*x_2 + k_2*x_1 - k_3 * x_2

x_1(0) = 1
x_2(0) = 0
```

For the implementation details & visualizations, see `simulation_cr.py` & `visualization_cr.py`. The simulation file does all the simulation and stores the results, while `visualization_cr.py` loads these results for visualizations and stores the corresponding figures.