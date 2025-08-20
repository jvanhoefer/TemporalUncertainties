# Getting started with the Bruno model:


## TLDR

We are trying to implement two methods to extend the Bruno et al. model [1](https://academic.oup.com/jxb/article/67/21/5993/2738937):

- The `multi_tau` version(s) have an individual value for tau for every data point.
- The `shared_tau` version(s) have a shared value for tau for every data point within an experimental condition, that was taken at the same time point (i.e. when condition and time point is the same, tau is shared)


Further, there are two algorithmic approaches, that are tested:

- `marginalized` means, that tau is integrated out via the ODE solver.
- `full` means, that tau remains a model parameter for fitting/sampling.


Hence there are in the end four `pypesto` problems, which we try out: `problem_marginalized_multi_tau`, `problem_marginalized_shared_tau`, `problem_full_multi_tau` and `problem_full_shared_tau`. Further there is the original implementation from the paper, given in `problem_wrong_tp`.



The implementations are in the corresponding files in the `./simulations` folder. Every problem can be 

## Biological background for Bruno:

This paper is about "How does Arabidopsis (model plant) do clevage on carotenoid (the orange color)" i.e. how get carotenoid chopped down by enzymes. This meachanism is currently unknown. This study researches, how some of the intermediary elenents of ths cleavage are working.

ChatGPT: "Carotenoid-cleaving enzymes in plants make important hormones and possibly other unknown signals that influence plant development. One enzyme, AtCCD4, has been suspected of making such signals from certain carotenoid intermediates, but this hasn’t been proven. The authors are testing whether AtCCD4 can actually produce these signals — or whether another enzyme, AtCCD7, is the real source."