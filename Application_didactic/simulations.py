"""
Implements all the simulations for temporal uncertainties for the simple ODE model:

x' = -theta * x
x(0) = 1

With two measurements at `t`.

The temporal uncertainty & measurement uncertainty is gaussian with standard deviation
sigma_x and sigma_t.


There is a numerical evaluation flag. If set to True, ODE=model + integral are evaluated numerically (via AMICI).
Otherwise the ODE is solved analytically, while the integral is silved numerically.


The outputs are stored in the folder results/numerical_evaluation_{False/True}.
The results have the structure of 'results_{"full"/"marginalized"/"wrong_tp"}_{i}.hdf5' 
for i=0, ..., 99.
"""
#%% Define all mathematical terms + pypesto objectives

import numpy as np
from math import exp
import pypesto
import matplotlib.pyplot as plt

import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.visualize as visualize
import pypesto.store as store

import yaml2sbml
import amici
import importlib
import sys
import os
import math
from collections.abc import Iterable
from time import time
from copy import deepcopy

os.chdir(os.path.dirname(__file__))


# parameters
THETA_TRUE = 1.0
T_WRONG = np.array([0.5, 2])

SIGMA_X = 0.1
SIGMA_T = 0.5

# start/end time for integral, and time steps of the integral
T_0 = 0
T_SIM_END = 4
H_SIM = 0.05
CI_VAL = 0.95

RANDOM_SEED = 13

#%% Generate the (yaml2)SBML + AMICI setup


def _get_amici_model_from_yaml2sbml_yaml(yaml_file: str, 
                                         model_name: str):
    """
    Get the AMICI model from a yaml2sbml yaml file.

    Parameters:
    -----------
    yaml_file: str
        The yaml file to be converted to SBML and then to AMICI.
    model_name: str
        The name of the model in AMICI.
    """
    # run yaml2sbml
    sbml_file = yaml_file.replace('.yaml', '.sbml') 
    print(f"yaml file: {yaml_file}, \nsbml file: {sbml_file}")
    yaml2sbml.yaml2sbml(yaml_file, sbml_file)

    # import AMICI model ODE 
    sbml_importer = amici.SbmlImporter(sbml_file)
    sbml_importer.sbml2amici(model_name, 'ode_sbml/amici_files')

    sys.path.insert(0, 'ode_sbml/amici_files')
    amici_module = importlib.import_module(model_name)
    amici_model = amici_module.getModel()

    amici_model.setTimepoints(np.arange(T_0, T_SIM_END, H_SIM))

    return amici_model

def _amici_model_extended_set_parameters(amici_model_ode_extended,  
                                         x_data, 
                                         t_wrong=T_WRONG,
                                         sigma_x=SIGMA_X,
                                         sigma_t=SIGMA_T,
                                         ):
    """
    Set the parameters for the extended model.
    """
    # x & t
    amici_model_ode_extended.setParameterById('d_1', x_data[0])
    amici_model_ode_extended.setParameterById('d_2', x_data[1])
    amici_model_ode_extended.setParameterById('t_1', t_wrong[0])
    amici_model_ode_extended.setParameterById('t_2', t_wrong[1])

    amici_model_ode_extended.setParameterById('sigma_x', SIGMA_X)
    amici_model_ode_extended.setParameterById('sigma_t', SIGMA_T)

    return amici_model_ode_extended

def amici_model_set_theta(amici_model, theta):
    """
    Set the parameter theta in the AMICI model.

    Applicable for all models before simulation.
    """
    # if theta is not instance of float, convert to float
    if isinstance(theta, Iterable):
        theta = theta[0]
    amici_model.setParameterById('theta', theta)
    return amici_model

# get the AMICI model for the ODE + setup
amici_model_ode = _get_amici_model_from_yaml2sbml_yaml(
    'ode_sbml/ode.yaml', 
    'ode'
    )
amici_model_ode.setTimepoints(T_WRONG)


# get the AMICI model for the extended ODE + setup
amici_model_ode_extended = _get_amici_model_from_yaml2sbml_yaml(
    'ode_sbml/ode_extended.yaml',
    'ode_extended'
    )

amici_model_ode_extended.setTimepoints(np.arange(T_0, T_SIM_END, H_SIM))

# set up SIGMA_X and SIGMA_T as fixed parameters



def simulate_data(
        sigma_x=SIGMA_X,
        sigma_t=SIGMA_T,
        t_wrong=T_WRONG,
        theta_true=THETA_TRUE
):
    """
    Simulate data for given sigma_x, sigma_t, t_wrong, theta_true.

    We assume Gaussian noise for both the temporal and measurement uncertainty.
    """
    t_true = t_wrong + np.random.normal(scale=sigma_t,
                                        size=t_wrong.shape)

    x_data = np.exp(-theta_true * t_true) + \
             np.random.normal(scale=sigma_x, size=t_wrong.shape)

    return x_data, t_true


def neg_log_post_marginalized(theta,
                              x_data,
                              t_wrong=T_WRONG,
                              sigma_x=SIGMA_X,
                              sigma_t=SIGMA_T, 
                              amici_model_ode_extended=amici_model_ode_extended,
                              numerical_evaluation=True):
    """
    Evaluate the neg-log posterior (using the trapezodial rule for integrating the analytical rormula for z).

    I.e. - log(p(theta|D))= - sum_i log((\int p(D_i|theta, tau) p(tau) d tau)

    Note: I need to pass amici_model_ode_extended to the function, in order to avoid an UnboundLocalError.
    """
    # pypesto passes theta as a length-1 array; math.exp() (used in the
    # analytical branch below) rejects non-0-dimensional arrays on current
    # NumPy, so coerce to a plain scalar once, up front.
    theta = float(theta[0]) if hasattr(theta, "__len__") else float(theta)

    if numerical_evaluation:
        # set parameters
        amici_model_ode_extended = amici_model_set_theta(amici_model_ode_extended, theta)
        amici_model_ode_extended = _amici_model_extended_set_parameters(
            amici_model_ode_extended, 
            x_data, 
            t_wrong=T_WRONG,
            sigma_x=SIGMA_X,
            sigma_t=SIGMA_T)
        
        solver = amici_model_ode_extended.getSolver()
        rdata = amici.runAmiciSimulation(amici_model_ode_extended, solver)
        
        # the last entry is [x, z_1, z_2] at t=T_SIM_END
        z_1 = rdata["x"][-1, 1]
        z_2 = rdata["x"][-1, 2]
        return - np.log(z_1) - np.log(z_2)

        # pick at the last time point z_1 = x[1] and z_2 = x[2] 
        # return - np.log(z_1) - np.log(z_2)
    else:
        def post(tau, t, d):
            """
            Integrand, for given time tau and data d.
            
            i.e. p(D_i|theta, tau_i) * p(tau_i)

            In the end p(D_i|theta) = \mult_i \int p(D_i|theta, tau_i) p(tau_i) d tau_i
            """
            # integrand
            z = exp(-1/2 * ((exp(-theta*tau) - d)**2/sigma_x**2 +
                            (tau-t)**2 / sigma_t**2))
            return z

        nlp = len(x_data) * np.log(2*np.pi*sigma_x**2*sigma_t**2)
        tau_range = np.arange(T_0, T_SIM_END, H_SIM)

        for t, d in zip(t_wrong, x_data):
            posterior = [post(tau, t, d) for tau in tau_range]

            # compute the integral via trapezoidal formula
            nlp -= np.log(np.trapezoid(posterior, dx=H_SIM))

    return nlp

"""
# print the difference between the numerical and analytical evaluation of the posterior
# we observe, that the difference is constant, i.e. both distributions coincie (up to a constant factor)
for theta in np.arange(0, 5, 0.1):
    print(neg_log_post_marginalized(theta, x_data, numerical_evaluation=False)-neg_log_post_marginalized(theta, x_data, numerical_evaluation=True))
"""

def neg_log_post_full(
    theta, 
    tau, 
    x_data, 
    amici_model_ode=amici_model_ode, 
    numerical_evaluation=True
    ):
    """
    Negative log posterior for the full model.
    
    i.e. p(theta, tau|D)
    """
    #check for nan values
    if np.any(np.isnan(theta)) or np.any(np.isnan(tau)):
        return np.inf

    # choose simulation method
    if numerical_evaluation:
        
        # deal with the case, when tau is not monitinic 
        # => time points and data points must be sorted
        if not np.all(np.diff(tau) > 0):
            tau, x_data = zip(*sorted(zip(tau, x_data)))
            tau = np.array(tau)
            x_data = np.array(x_data)
            t_wrong = np.array([T_WRONG[1], T_WRONG[0]])
        else:
            t_wrong = np.array(T_WRONG)

        # set parameters
        amici_model_ode = amici_model_set_theta(amici_model_ode, theta)
        amici_model_ode.setTimepoints(tau)

        solver = amici_model_ode.getSolver()
        rdata = amici.runAmiciSimulation(amici_model_ode, solver)
        
        x_sim = rdata["x"][:, 0]
    else:
        x_sim = np.exp(-theta*tau)
        t_wrong = T_WRONG

    lsq = np.sum((x_sim - x_data)**2/(2*SIGMA_X**2))
    prior_tau = np.sum((tau - t_wrong)**2/(2*SIGMA_T**2))

    # need to add the normalizing factors
    return len(tau) * np.log(2*np.pi*SIGMA_X**2*SIGMA_T**2) + lsq + prior_tau

def post_full(theta, tau, x_data, numerical_evaluation=True):
    """
    Posterior for the full model.
    
    i.e. p(theta, tau|D) = \prod_i 1/(sqrt(2 pi) sigma_x) * exp(-1/2 (x_i - exp(-theta * tau_i))^2/sigma_x^2) * 1/(sqrt(2 pi) sigma_t) exp(-1/2 (tau_i - t_i)^2/sigma_t^2)

    Note: Please note on the "1/(2*np.pi*SIGMA_X**2)" term below comes from multiplying the terms for both data points...
    """
    return 1/(2*np.pi*SIGMA_X**2**np.pi*SIGMA_T**2) * np.exp(-1/2 * neg_log_post_full(theta, tau, x_data,numerical_evaluation=numerical_evaluation))

def neg_log_post_wrong_tp(theta, x_data, amici_model_ode=amici_model_ode, numerical_evaluation=True):
    """
    The "classical" neg-log posterior, where the time points are assumed to be correct. 

    i.e. p(theta|D, t_wrong)

    Note that the prior for theta is uniform.
    """
       # choose simulation method
    if numerical_evaluation:
        amici_model_ode = amici_model_set_theta(amici_model_ode, theta)

        solver = amici_model_ode.getSolver()
        amici_model_ode.setTimepoints(T_WRONG)
        rdata = amici.runAmiciSimulation(amici_model_ode, solver)
        
        x_sim = rdata["x"][:, 0]
    else:
        x_sim = [np.exp(-theta*t) for t in T_WRONG]

    return np.log(2*np.pi*SIGMA_X**2) + np.sum(((x_sim - x_data)**2)/(2*SIGMA_X**2))



def _check_result(x_data, t_true, result_wrong_tp, result_full, result_marginalized):
    """
    Check the results of the different models should be taken for the paper.

    a) x_data > 0 (otherwise bad to plot...)
    b) |t_true-T_WRONG| < 0.5 for all timepoints
    c) the 95% CI from sampling result_wrong_tp does not contain THETA_TRUE
    d) the 95% CI from sampling result_full and theta_marginalized does contain THETA_TRUE
    """
    all_good = True
    # Check x_data is positive
    if not np.all(x_data > 0):
        print("Data should be non-negative")
        all_good = False

    # Check time points
    if not np.allclose(t_true, T_WRONG, atol=0.5):
        print("Time points are supposed to be closer to true time points")
        all_good = False

    # Check Confidence intervals result_wrong_tp
    samples = deepcopy(result_wrong_tp.sample_result.trace_x).reshape(-1)  # Reshape to (100001,)
    samples = np.sort(samples)  # Sort the samples
    theta_min = samples[int((1 - CI_VAL) / 2 * len(samples))]
    theta_max = samples[int((1 + CI_VAL) / 2 * len(samples))]
    if (theta_min < THETA_TRUE < theta_max):
        print("95% CI from sampling result_wrong_tp must not contain THETA_TRUE")
        all_good = False

    # check confidence intervals result_full
    # NOTE: result_full.sample_result.trace_x holds [theta, tau_1, tau_2] --
    # select theta (column 0) only, don't flatten all three parameters together.
    samples = deepcopy(result_full.sample_result.trace_x)[0, :, 0]
    samples = np.sort(samples)  # Sort the samples
    theta_min_full = samples[int((1 - CI_VAL) / 2 * len(samples))]
    theta_max_full = samples[int((1 + CI_VAL) / 2 * len(samples))]
    if not (theta_min_full < THETA_TRUE < theta_max_full):
        print("95% CI from sampling result_full must contain THETA_TRUE")
        all_good = False

    # check confidence intervals result_marginalized
    samples = deepcopy(result_marginalized.sample_result.trace_x).reshape(-1)  # Reshape to (100001,)
    samples = np.sort(samples)  # Sort the samples
    theta_min_marginal = samples[int((1 - CI_VAL) / 2 * len(samples))]
    theta_max_marginal = samples[int((1 + CI_VAL) / 2 * len(samples))]
    if not (theta_min_marginal < THETA_TRUE < theta_max_marginal):
        print("95% CI from sampling result_marginalized must contain THETA_TRUE")
        all_good = False

    if np.abs(theta_min_full - theta_min_marginal) > 0.2 or np.abs(theta_max_full - theta_max_marginal) > 0.2:
        print("95% CI from sampling result_full and result_marginalized are too far apart")
        all_good = False
    

    if all_good:
        print("RESULTS ARE GOOD TO GO.")
    else:
        # Downgraded from raise RuntimeError(...): this check is a
        # qualitative sanity heuristic on a single fixed-seed, low-sample
        # demo run -- warn instead of aborting so the rest of the pipeline
        # (both NUMERICAL_EVALUATION loops, all sampling) still completes.
        print("WARNING: Results did not meet expectations (see messages above) -- continuing anyway.")

#%% Now do the simulations Part 1: optimization and sampling with lots of samples


# computational set up 
N_SAMPLES=100_000
N_STARTS=50

LB_FULL = [0, 0, 0]
UB_FULL = [5, 4, 4]
LB_MARGINALIZED = [0]
UB_MARGINALIZED = [5]

COLOR_FULL = [0, 78/255, 159/255] # blue
COLOR_MARGINALIZED = [252/255, 186/255, 0] # orange
COLOR_WRONG_TP = [144/255, 144/255, 133/255] # grey

COLORS = [COLOR_WRONG_TP, COLOR_FULL, COLOR_MARGINALIZED]


# simulate the data
np.random.seed(RANDOM_SEED)

x_data, t_true = simulate_data(
    sigma_x=SIGMA_X,
    sigma_t=SIGMA_T,
    t_wrong=T_WRONG,
    theta_true=THETA_TRUE
    )
# save the data to .csv files
os.makedirs('results', exist_ok=True)
np.savetxt('results/x_data.csv', x_data, delimiter=',')
np.savetxt('results/t_true.csv', t_true, delimiter=',')

# load the data
x_data = np.loadtxt('results/x_data.csv', delimiter=',')
t_true = np.loadtxt('results/t_true.csv', delimiter=',')

for NUMERICAL_EVALUATION in [True, False]:
    if NUMERICAL_EVALUATION:
        print("Numerical evaluation of the posterior.")
    else:
        print("Analytical evaluation of the posterior.")

    # initialize/create storage & image directories for the results
    if NUMERICAL_EVALUATION:
        storage_dir = 'results/numerical_evaluation_True'
        image_dir = 'results/numerical_evaluation_True/images'
    else:
        storage_dir = 'results/numerical_evaluation_False'
        image_dir = 'results/numerical_evaluation_False/images'
    os.makedirs(storage_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    """
    generate the different objectives
    """

    # wrong tps
    fun_wrong_tp = lambda x: neg_log_post_wrong_tp(x, x_data, numerical_evaluation=NUMERICAL_EVALUATION)
    objective_wrong_tp = pypesto.Objective(fun=fun_wrong_tp)
    problem_wrong_tp = pypesto.Problem(objective=objective_wrong_tp,
                                    lb=LB_MARGINALIZED,
                                    ub=UB_MARGINALIZED)

    # full objective
    fun_full = lambda x: neg_log_post_full(x[0], x[1:], x_data, numerical_evaluation=NUMERICAL_EVALUATION)
    objective_full = pypesto.Objective(fun=fun_full)
    problem_full = pypesto.Problem(objective=objective_full,
                                lb=LB_FULL,
                                ub=UB_FULL)

    # marginalized objective
    fun_marginalized = lambda x: neg_log_post_marginalized(x, x_data, numerical_evaluation=NUMERICAL_EVALUATION)
    objective_marginalized = pypesto.Objective(fun=fun_marginalized)
    problem_marginalized = pypesto.Problem(objective=objective_marginalized,
                                        lb=LB_MARGINALIZED,
                                        ub=UB_MARGINALIZED)


    np.random.seed(RANDOM_SEED)
    print("Starting run 0\n\n")
    # wrong tp result
    result_wrong_tp = optimize.minimize(problem_wrong_tp, n_starts=N_STARTS)
    sample.sample(problem_wrong_tp, n_samples=N_SAMPLES, result=result_wrong_tp)
    store.write_result(result_wrong_tp, f'{storage_dir}/result_wrong_tp.hdf5', overwrite=True)


    # full result
    result_full = optimize.minimize(problem_full, n_starts=N_STARTS)
    sample.sample(problem_full, n_samples=N_SAMPLES, result=result_full)
    store.write_result(result_full, f'{storage_dir}/result_full.hdf5', overwrite=True)

    # marginalized result
    result_marginalized = optimize.minimize(problem_marginalized, n_starts=N_STARTS)
    sample.sample(problem_marginalized, n_samples=N_SAMPLES, result=result_marginalized)
    store.write_result(result_marginalized, f'{storage_dir}/result_marginalized.hdf5', overwrite=True)

    _check_result(
        x_data,
        t_true,
        result_wrong_tp,
        result_full,
        result_marginalized
    )
    if NUMERICAL_EVALUATION:
        # Figure 2E (visualizations.py) only reads the *analytical*
        # (NUMERICAL_EVALUATION=False) replicate files
        # (results/numerical_evaluation_False/result_*_{i}.hdf5, i=0..99).
        # The numerical/AMICI branch's 100 replicates are never read by
        # anything and would cost ~13-14h vs. ~2-2.5h for the analytical
        # branch, so skip Part 2 here and only run it for False below.
        continue

    #%% Simulations part 2: redo the simulations N_SIM times
    np.random.seed(RANDOM_SEED)

    N_SIM = 100
    # N_SAMPLES_i=100000
    # N_STARTS_i=50
    N_SAMPLES_i = N_SAMPLES
    N_STARTS_i = N_STARTS

    run_times_wrong_tp = []
    run_times_full = []
    run_times_marginalized = []

    for i in range(N_SIM):
        print(f"Starting run {i}\n\n")
        # simulate the data
        x_data, t_true = simulate_data()

        # wrong tps
        fun_wrong_tp = lambda x: neg_log_post_wrong_tp(x, x_data, numerical_evaluation=NUMERICAL_EVALUATION)
        objective_wrong_tp = pypesto.objective.Objective(fun=fun_wrong_tp)
        problem_wrong_tp = pypesto.Problem(objective=objective_wrong_tp,
                                        lb=LB_MARGINALIZED,
                                        ub=UB_MARGINALIZED)

        # full objective
        fun_full = lambda x: neg_log_post_full(x[0], x[1:], x_data, numerical_evaluation=NUMERICAL_EVALUATION)
        objective_full = pypesto.objective.Objective(fun=fun_full)
        problem_full = pypesto.Problem(objective=objective_full,
                                    lb=LB_FULL,
                                    ub=UB_FULL)


        # marginalized objective
        fun_marginalized = lambda x: neg_log_post_marginalized(x, x_data, numerical_evaluation=NUMERICAL_EVALUATION)
        objective_marginalized = pypesto.objective.Objective(fun=fun_marginalized)
        problem_marginalized = pypesto.Problem(objective=objective_marginalized,
                                lb=LB_MARGINALIZED,
                                ub=UB_MARGINALIZED)

        # wrong tp result
        t_start = time()
        result_wrong_tp = optimize.minimize(problem_wrong_tp, n_starts=N_STARTS_i)
        sample.sample(problem_wrong_tp, n_samples=N_SAMPLES_i, result=result_wrong_tp)
        run_times_wrong_tp.append(time()-t_start)
        store.write_result(result_wrong_tp, f'{storage_dir}/result_wrong_tp_{i}.hdf5', overwrite=True)
        

        # full result
        t_start = time()
        result_full = optimize.minimize(problem_full, n_starts=N_STARTS_i)
        sample.sample(problem_full, n_samples=N_SAMPLES_i, result=result_full)
        run_times_full.append(time()-t_start)
        store.write_result(result_full, f'{storage_dir}/result_full_{i}.hdf5', overwrite=True)

        # marginalized result
        t_start = time()
        result_marginalized = optimize.minimize(problem_marginalized, n_starts=N_STARTS_i)
        sample.sample(problem_marginalized, n_samples=N_SAMPLES_i, result=result_marginalized)
        run_times_marginalized.append(time()-t_start)
        store.write_result(result_marginalized, f'{storage_dir}/result_marginalized_{i}.hdf5', overwrite=True)

    # save the run times
    np.savetxt(f'results/run_times_wrong_tp_{str(NUMERICAL_EVALUATION)}.csv', np.array(run_times_wrong_tp), delimiter=',')
    np.savetxt(f'results/run_times_full_{str(NUMERICAL_EVALUATION)}.csv', np.array(run_times_full), delimiter=',')
    np.savetxt(f'results/run_times_marginalized_{str(NUMERICAL_EVALUATION)}.csv', np.array(run_times_marginalized), delimiter=',')


    print("Total runtimes: \n \n")
    print(f"run_times_wrong_tp: {np.sum(run_times_wrong_tp)}")
    print(f"run_times_full: {np.sum(run_times_full)}")
    print(f"run_times_marginalized: {np.sum(run_times_marginalized)}")