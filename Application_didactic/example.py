"""
Implements the method for temporal uncertainties for the simple ODE model:

x' = -theta * x
x(0) = 1

With two measurements at `t`.

The temporal uncertainty & measurement uncertainty is gaussian with standard deviation
sigma_x and sigma_t.

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



# parameters
THETA_TRUE = 1.0
T_WRONG = np.array([0.5, 2])

SIGMA_X = 0.1
SIGMA_T = 0.5

# end time for integral, and time steps of the integral
T_SIM_END = 10 
H_SIM = 0.05

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

    amici_model.setTimepoints(np.arange(0, T_SIM_END, H_SIM))

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
    amici_model.setParameterById('d_1', x_data[0])
    amici_model.setParameterById('d_2', x_data[1])
    amici_model.setParameterById('t_1', t_wrong[0])
    amici_model.setParameterById('t_2', t_wrong[1])

    amici_model.setParameterById('sigma_x', SIGMA_X)
    amici_model.setParameterById('sigma_t', SIGMA_T)

    return amici_model

def amici_model_set_theta(amici_model, theta):
    """
    Set the parameter theta in the AMICI model.

    Applicable for all models before simulation.
    """
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

amici_model_ode_extended.setTimepoints(np.arange(0, T_SIM_END, H_SIM))

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

    x_data = np.exp(-theta_true * t_wrong) + \
             np.random.normal(scale=sigma_x, size=t_wrong.shape)

    return x_data, t_true


def neg_log_post_marginalized(theta,
                              x_data,
                              t_wrong=T_WRONG,
                              sigma_x=SIGMA_X,
                              sigma_t=SIGMA_T):
    """
    Evaluate the neg-log posterior (using the trapezodial rule for integrating the analytical rormula for z).

    I.e. - log(p(theta|D))= - sum_i log((\int p(D_i|theta, tau) p(tau) d tau)
    """

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
    tau_range = np.arange(0, T_SIM_END, H_SIM)

    for t, d in zip(t_wrong, x_data):
        posterior = [post(tau, t, d) for tau in tau_range]

        # compute the integral via trapezoidal formula
        nlp -= np.log(np.trapezoid(posterior, dx=h))

    return nlp

def neg_log_post_full(theta, tau, x_data):
    """
    Negative log posterior for the full model.
    
    i.e. p(theta, tau|D)
    """

    lsq = np.sum((np.exp(-theta*tau) - x_data)**2/(2*SIGMA_X**2))
    prior_tau = np.sum((tau - T_WRONG)**2/(2*SIGMA_T**2))

    # need to add the normalizing factors
    return len(tau) * np.log(2*np.pi*SIGMA_X**2*SIGMA_T**2) + lsq + prior_tau

def post_full(theta, tau, x_data):
    """
    Posterior for the full model.
    
    i.e. p(theta, tau|D) = \prod_i 1/(sqrt(2 pi) sigma_x) * exp(-1/2 (x_i - exp(-theta * tau_i))^2/sigma_x^2) * 1/(sqrt(2 pi) sigma_t) exp(-1/2 (tau_i - t_i)^2/sigma_t^2)

    Note: Please note on the "1/(2*np.pi*SIGMA_X**2)" term below comes from muliitplying the terms for both data points...
    """
    return 1/(2*np.pi*SIGMA_X**2**np.pi*SIGMA_T**2) * np.exp(-1/2 * neg_log_post_full(theta, tau, x_data))

def neg_log_post_wrong_tp(theta, x_data):
    """
    The "classical" neg-log posterior, where the time points are assumed to be correct. 

    i.e. p(theta|D, t_wrong)

    Note that the prior for theta is uniform.
    """

    x_sim = [np.exp(-theta*t) for t in T_WRONG]

    return np.sum((x_sim - x_data)**2/SIGMA_X**2)



#%% Now do the simulations Part 1: optimization and sampling with lots of samples

np.random.seed(42)

# simulate the data
x_data, t_true = simulate_data(
    sigma_x=SIGMA_X,
    sigma_t=SIGMA_T,
    t_wrong=T_WRONG,
    theta_true=THETA_TRUE
    )

# computational set up 
N_SAMPLES=100000
N_STARTS=50

# wrong tps
fun_wrong_tp = lambda x: neg_log_post_wrong_tp(x, x_data)
objective_wrong_tp = pypesto.objective.Objective(fun=fun_wrong_tp)
problem_wrong_tp = pypesto.Problem(objective=objective_wrong_tp,
                                   lb=[0],
                                   ub=[5])

# full objective
fun_full = lambda x: neg_log_post_full(x[0], x[1:], x_data)
objective_full = pypesto.objective.Objective(fun=fun_full)
problem_full = pypesto.Problem(objective=objective_full,
                               lb=[0, 0, 0],
                               ub=[5, 4, 4])


# marginalized objective
fun_marginalized = lambda x: neg_log_post_marginalized(x, x_data)
objective_marginalized = pypesto.objective.Objective(fun=fun_marginalized)
problem_marginalized = pypesto.Problem(objective=objective_marginalized,
                          lb=[0],
                          ub=[5])

# wrong tp result
result_wrong_tp = optimize.minimize(problem_wrong_tp, n_starts=N_STARTS)
sample.sample(problem_wrong_tp, n_samples=N_SAMPLES, result=result_wrong_tp)
store.write_result(result_marginalized, f'results/result_wrong_tp.hdf5', overwrite=True)


# full result
result_full = optimize.minimize(problem_full, n_starts=N_STARTS)
sample.sample(problem_full, n_samples=N_SAMPLES, result=result_full)
store.write_result(result_marginalized, f'results/result_full.hdf5', overwrite=True)

# marginalized result
result_marginalized = optimize.minimize(problem_marginalized, n_starts=N_STARTS)
sample.sample(problem_marginalized, n_samples=N_SAMPLES, result=result_marginalized)
store.write_result(result_marginalized, f'results/result_marginalized.hdf5', overwrite=True)



# plot marginals
for r in [result_wrong_tp, result_full, result_marginalized]:
    fig = plt.figure()
    ax = visualize.sampling_1d_marginals(r, par_indices=[0])
    plt.xlim((0, 4))
    ax = plt.gca()
    plt.plot([THETA_TRUE, THETA_TRUE], ax.get_ylim(), 'k--')
    plt.title('Marginal for theta')

# compute correlation matrix within the parameters for the sample results for result_full

corr = np.corrcoef(result_full.sample_result.trace_x[0, :, :].T)
visualize.sampling_scatter(result_full)

# seaborn pairplot with kde

import seaborn as sns
import pandas as pd

df = pd.DataFrame(result_full.sample_result.trace_x[0, :, :])
df.columns = ['theta', 'tau1', 'tau2']
sns.pairplot(df, kind='kde')



#%% Simulations part 2: redo the simulations N_SIM times
np.random.seed(42)

N_SIM = 100
N_SAMPLES_i=100000
N_STARTS_i=50

for i in range(N_SIM):
    print(f"Starting run {i}\n\n")
    # simulate the data
    x_data, t_true = simulate_data()

    # wrong tps
    fun_wrong_tp = lambda x: neg_log_post_wrong_tp(x, x_data)
    objective_wrong_tp = pypesto.objective.Objective(fun=fun_wrong_tp)
    problem_wrong_tp = pypesto.Problem(objective=objective_wrong_tp,
                                    lb=[0],
                                    ub=[5])

    # full objective
    fun_full = lambda x: neg_log_post_full(x[0], x[1:], x_data)
    objective_full = pypesto.objective.Objective(fun=fun_full)
    problem_full = pypesto.Problem(objective=objective_full,
                                lb=[0, 0, 0],
                                ub=[5, 4, 4])


    # marginalized objective
    fun_marginalized = lambda x: neg_log_post_marginalized(x, x_data)
    objective_marginalized = pypesto.objective.Objective(fun=fun_marginalized)
    problem_marginalized = pypesto.Problem(objective=objective_marginalized,
                            lb=[0],
                            ub=[5])

    # wrong tp result
    result_wrong_tp = optimize.minimize(problem_wrong_tp, n_starts=N_STARTS_i)
    sample.sample(problem_wrong_tp, n_samples=N_SAMPLES_i, result=result_wrong_tp)
    store.write_result(result_wrong_tp, f'results/result_wrong_tp_{i}.hdf5', overwrite=True)
    

    # full result
    result_full = optimize.minimize(problem_full, n_starts=N_STARTS_i)
    sample.sample(problem_full, n_samples=N_SAMPLES_i, result=result_full)
    store.write_result(result_full, f'results/result_full_{i}.hdf5', overwrite=True)

    # marginalized result
    result_marginalized = optimize.minimize(problem_marginalized, n_starts=N_STARTS_i)
    sample.sample(problem_marginalized, n_samples=N_SAMPLES_i, result=result_marginalized)
    store.write_result(result_marginalized, f'results/result_marginalized_{i}.hdf5', overwrite=True)


# read in all the results and compute the position of THETA_TRUE in the marginals in the full model (i.e. parameter nr. 0)
quantile_wrong_tp = []
quantile_full = []
quantile_marginalized = []

for j in range(N_SIM):
    # read for wrong tp
    result = store.read_result(f'results/result_wrong_tp_{j}.hdf5')
    quantile_wrong_tp.append(int(np.searchsorted(sorted(result.sample_result.trace_x[0, :, 0]), v=THETA_TRUE)))

    # read for full objective
    result = store.read_result(f'results/result_full_{j}.hdf5')
    quantile_full.append(int(np.searchsorted(sorted(result.sample_result.trace_x[0, :, 0]), v=THETA_TRUE)))

    # read for marginalized objective
    result = store.read_result(f'results/result_marginalized_{j}.hdf5')
    quantile_marginalized.append(int(np.searchsorted(sorted(result.sample_result.trace_x[0, :, 0]), v=THETA_TRUE)))

for i in range(N_SIM):
    print(f"Wrong TP: {quantile_wrong_tp[i]}, Full: {quantile_full[i]}, Marginalized: {quantile_marginalized[i]}")

plt.hist(quantile_wrong_tp, range=(0, 100000),bins=20, alpha=0.5, label='wrong tp')

#get fraction in the 90% quantile
print(f"Wrong timepoint: {np.sum(2*np.abs(np.array(quantile_wrong_tp)/N_SAMPLES_i-1/2)<0.90)}")
print(f"Full: {np.sum(2*np.abs(np.array(quantile_full)/N_SAMPLES_i-1/2)<0.90)}")
print(f"Marginalized: {np.sum(2*np.abs(np.array(quantile_marginalized)/N_SAMPLES_i-1/2)<0.90)}")


# plot all the quantiles

plt.plot([np.sum(2*np.abs(np.array(quantile_wrong_tp)/N_SAMPLES_i-1/2)<p/100) for p in range(101)])
plt.plot([np.sum(2*np.abs(np.array(quantile_full)/N_SAMPLES_i-1/2)<p/100) for p in range(101)])
plt.plot([np.sum(2*np.abs(np.array(quantile_marginalized)/N_SAMPLES_i-1/2)<p/100) for p in range(101)])
plt.plot([i for i in range(101)])

plt.legend(['wrong tp', 'full', 'marginalized', 'expected'])