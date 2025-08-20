"""
The shared multi-tau model means, is the same as full_multi_tau, but with shared 
tau parameters for data points, that are at the same time point for the same condition.


theta_extended = [theta_estimate, taus, sigma_tau]

i.e. all taus are additional model parameters.


Further No taus are shared, i.e. every data point has its own tau.
"""
import numpy as np
import pandas as pd
# All functions for simulations etc. can be reused, after mapping the parameters
from .full_multi_tau import _simulate_single_data_point_dicts_for_condition

tau_shared_df = pd.read_csv(
    "Application_Bruno/simulations/shared_parametrization_df.tsv", sep="\t")

DELTA_TIME_BOUND = 5 # the time variables are bound by +/- delta time

LB_FULL_SHARED_TAU = np.concatenate(
    [np.log10(importer.petab_problem.parameter_df['lowerBound'].values),
    tau_shared_df.drop_duplicates("tau_number").set_index("tau_number")["time"].reindex(range(42)).to_numpy()-DELTA_TIME_BOUND,
    [0]
    ])

UB_FULL_SHARED_TAU = np.concatenate(
    [np.log10(importer.petab_problem.parameter_df['upperBound'].values),
    tau_shared_df.drop_duplicates("tau_number").set_index("tau_number")["time"].reindex(range(42)).to_numpy()+DELTA_TIME_BOUND,
    [5]
    ])

theta_shared_true = np.concatenate(
    [np.log10(importer.petab_problem.parameter_df['nominalValue'].values), 
    tau_shared_df.drop_duplicates("tau_number").set_index("tau_number")["time"].reindex(range(42)).to_numpy(),
    [2]
    ]
    )    


def theta_shared_to_theta_multi(theta_extended: np.array):
    """
    Maps a theta_extended with shared taus to a theta_extended with multi taus.
    """
    # Let`s chop the vector into pieces, but do no log trafo
    theta_extended, taus_shared, sigma_tau = _theta_extended_to_thetas(theta_extended, log_trafo=False)

    taus_multi = np.zeros((77, ))
    tau_shared_index_to_tau_multi_index = tau_shared_df['tau_number'].values
    # i.e. tau_shared_index_to_tau_multi_index[i] is the index of the tau in taus_multi
    
    for i, tau_idx in enumerate(tau_shared_index_to_tau_multi_index):
        taus_multi[i] = taus_shared[tau_idx]

    return np.concatenate([theta_extended, taus_multi, [sigma_tau]])



def neg_log_post_shared_with_grad(
    theta_extended: np.ndarray, 
    amici_model
    ) -> Tuple[float, np.ndarray]:
    """
    Negative log-posterior:
        nlp = -log p(data | theta, taus) - log p(taus | times, sigma_tau)
    where
        data likelihood: y_i ~ N(sim_i(theta, tau_i), sigma_y_i^2)
        tau prior:       tau_i ~ N(t_i (from PEtab), sigma_tau^2)

    theta_extended layout is defined in _theta_extended_to_thetas().

    Returns a posterior + gradient.
    """
    theta_extended = theta_shared_to_theta_multi(theta_extended)

    theta_estimate, _, sigma_tau = _theta_extended_to_thetas(theta_extended)

    _simulated_datapoints_list = []
    for condition in ["model1_data1", "model1_data2", "model1_data3", "model1_data4", "model1_data5", "model1_data6"]:
        _simulated_datapoints_for_condition = _simulate_single_data_point_dicts_for_condition(condition, theta_extended, amici_model)
        for e in _simulated_datapoints_for_condition:
            _simulated_datapoints_list.append(e) 

    # correct measurement_df_index and theta_extended_index
    tau_shared_index_to_tau_multi_index = tau_shared_df['tau_number'].values
    for i, sim_dict in enumerate(_simulated_datapoints_list):
        sim_dict["measurement_df_index"] = tau_shared_index_to_tau_multi_index[i]
        sim_dict["theta_extended_index"] = tau_shared_index_to_tau_multi_index[i] + 13

    # From here: unpack them by looping over the list
    # update/add to log_post + gradient.
    neg_log_post = 0.0
    grad = np.zeros((56,))

    # list of tau indices, that already are in the posterior
    evaluated_tau_indices = []
    
    for sim_dict in _simulated_datapoints_list:
        sim = float(sim_dict["sim"])
        data = float(sim_dict["data"])
        tau = float(sim_dict["tau"])
        t = float(sim_dict["t"])
        sigma_y = float(sim_dict["sigma_y"])
        theta_extended_index = int(sim_dict["theta_extended_index"])

        # check if sim is NaN
        if np.isnan(sim):
            return np.inf

        # calculate the obj function
        resid = sim - data

        neg_log_likelihood = 0.5 * (np.log(2*np.pi*sigma_y**2) + (resid**2) / (sigma_y**2))
    
        # adapt the way in which the posterior is evaluated to 
        # only count for every time point index once
        if theta_extended_index not in evaluated_tau_indices:
            neg_log_prior_tau = 0.5 * (np.log(2*np.pi*sigma_tau**2) + ((tau - t)**2) / (sigma_tau**2))        
            d_tau = (tau - t) / (sigma_tau**2)
            evaluated_tau_indices.append(theta_extended_index)
        else:
            # The posterior is zero, the posterior part of the gradient w.r.t tau is zero.
            neg_log_prior_tau = 0.0
            d_tau = 0.0

        neg_log_post += neg_log_likelihood + neg_log_prior_tau
        # calculate the gradient
        # w.r.t theta_estimate => resid/sigma^2 * "grad_theta_estimate of sim"
        # w.r.t tau => resid/sigma^2 * "grad_tau of sim" for the corresponding tau index
        # w.r.t sigma_tau => derivative of neg log prior w.r.t sigma_tau
        grad[:len(theta_estimate)] += (resid / sigma_y**2) * sim_dict["dsim_dtheta"]
        grad[theta_extended_index] += (resid / sigma_y**2) * sim_dict["dsim_dtau"]  + d_tau
        grad[-1] += 1 / sigma_tau - ((tau - t)**2) / (sigma_tau**3)

    # do the log-trafo chain rule for theta_estimate
    grad[:len(theta_estimate)] *= 10 ** theta_extended[:13] * np.log(10)

    return float(neg_log_post), grad


"""
pypesto Optimization FTW :D
"""
import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.visualize as visualize
import pypesto.store as store
from pypesto.petab import PetabImporter


def obj_full_shared_tau(theta_shared):
    return neg_log_post_shared_with_grad(theta_shared, amici_model)

objective_full_shared_tau = pypesto.Objective(
    fun=obj_full_shared_tau,
    grad=True,
)

problem_full_shared_tau = pypesto.Problem(
    objective=objective_full_shared_tau,
    lb=LB_FULL_SHARED_TAU,
    ub=UB_FULL_SHARED_TAU
)

fides_optimizer = optimize.FidesOptimizer(verbose=0)
# engine = pypesto.engine.MultiThreadEngine(n_threads=8)

# your optimizer stays the same
result_full_shared_tau = optimize.minimize(
    problem_full_shared_tau,
    optimizer=fides_optimizer,
    # engine=engine,
    n_starts=10,
)