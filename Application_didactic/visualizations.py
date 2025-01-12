#%% imports and load the results
import pypesto.optimize as optimize 
import pypesto.sample as sample
import pypesto.store as store
import pypesto.visualize as visualize

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from example import THETA_TRUE, T_WRONG, SIGMA_X, SIGMA_T, N_SIM,  N_STARTS_i, N_SAMPLES_i

#%% do the setup & recompute the data (using a random seed...)
np.random.seed(42)

# simulate the data
x_data, t_true = simulate_data()

result_wrong_tp_ = store.read_result(f'results/result_wrong_tp.hdf5')
result_full = store.read_result(f'results/result_full.hdf5')
result_marginalized = store.read_result(f'results/result_marginalized.hdf5')


# Define Colors for purposes

COLOR_ORANGE_marginalized = '#fcba00' # [252,186,0]
COLOR_BLUE_full = '#004e9f' # [0, 78, 159]
COLOR_GREY_wrong_tp = '#909085'  # [144, 144, 133]

COLOR_BLACK_ground_truth = [0, 0, 0]

"""Color schema: dashed line for true trajectory, solid line for fitted trajectory, shaded area for samples"""

# Visualisation 1: Plotting the data generation and the best fit when ignoring the uncertainty
def data_generation_plot():
    """
    Set up the trajectory plots. This function:
    - plots the data (using the true measurement time points)
    - plots the true trajectory
    - plots the assumed time points
    - sets the axes limits
    """
    # axes limits
    plt.xlim((0, 4))
    plt.ylim((0, 1.1))

    plt.xlabel('t')
    plt.ylabel('x(t)')


    # get rid of the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Plot data
    plt.plot(t_true, x_data, 'x', color=COLOR_BLACK_ground_truth, label='data (true time points)')
    plt.plot(T_WRONG, x_data, 'o', color=COLOR_GREY_wrong_tp, label='data (assumed time points)')
    
    # Plot assumed (i.e. wrong) time points
    for t in T_WRONG:
        plt.plot([t, t], [0, 1], color=COLOR_BLACK_ground_truth, linestyle='--', alpha=0.5)
    
    # plot the true trajectory
    x_true = [np.exp(-THETA_TRUE * t) for t in np.arange(0, 4.1, 0.1)]
    plt.plot(np.arange(0, 4.1, 0.1), x_true, color=COLOR_BLACK_ground_truth, linestyle='--', label='true trajectory')

    plt.title('Data generation process')
    plt.legend()

def plot_parameter_histograms_1d(
    result_wrong_tp, 
    result_full, 
    result_wrong_tp, color, title):
    """
    Plot the 1d histograms of the samples for the parameter theta.
    """
    # full objective
    sns.kdeplot(
        result_full.sample_result.trace_x[0, :, 0],
        color=COLOR_BLUE_full, 
        fill=False, 
        label='Full objective')
    # marginalized objective
    sns.kdeplot(
        result_marginalized.sample_result.trace_x[0, :, 0], 
        color=COLOR_ORANGE_marginalized,
        fill=False, 
        label='Marginalized objective')
    # wrong time points
    sns.kdeplot(
        result_wrong_tp.sample_result.trace_x[0, :, 0], 
        color=COLOR_GREY_wrong_tp, 
        fill=False, 
        label='Wrong time points')

    plt.plot([THETA_TRUE, THETA_TRUE], plt.gca().get_ylim(), 'k--')
    plt.xlim((0, 4))    

    # get rid of the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlabel('$\theta$')
    plt.ylabel('Density')

    plt.title(title)






def plot_for_parameter(result,
                       result_wrong_tp,
                       x_data,
                       t_true):

    _set_up_trajectory_plot()

    x_fit = [np.exp(-result.optimize_result.x[0][0] * t) for t in np.arange(0, 4.1, 0.1)]
    plt.plot(np.arange(0, 4.1, 0.1), x_fit, 'b', label='fitted trajectory')

    x_fit_wrong_tp = [np.exp(-result_wrong_tp.optimize_result.x[0][0] * t) for t in np.arange(0, 4.1, 0.1)]
    plt.plot(np.arange(0, 4.1, 0.1), x_fit_wrong_tp, 'k', label='fitted trajectory for wrong tps')

    plt.legend()

    # plot the samples of the marginalized objective
    _set_up_figure()
    x_fit = []
    for x in result.sample_result.trace_x[0, :, 0]:
        x_fit.append([np.exp(-x * t) for t in np.arange(0, 4.1, 0.1)])

    plt.fill_between(np.arange(0, 4.1, 0.1),
                     np.percentile(np.array(x_fit), q=2.5, axis=0),
                     np.percentile(np.array(x_fit), q=97.5, axis=0))
    plt.title('fits')

    # plt samples
    _set_up_figure()

    x_fit_wrong_tp = []
    for x in result_wrong_tp.sample_result.trace_x[0, :, 0]:
        x_fit_wrong_tp.append([np.exp(-x * t) for t in np.arange(0, 4.1, 0.1)])

    plt.fill_between(np.arange(0, 4.1, 0.1),
                     np.percentile(np.array(x_fit_wrong_tp), q=2.5, axis=0),
                     np.percentile(np.array(x_fit_wrong_tp), q=97.5, axis=0))
    plt.title('Wrong timepoints')

    plt.legend()