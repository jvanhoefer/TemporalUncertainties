"""
Here some playground code for debugging
"""

condition = "model1_data2"
res1 = _simulate_single_data_point_dicts_for_condition(condition, theta_extended, amici_model)
res1_aggregated = {key: [d[key] for d in res1] for key in res1[0]}


res2 = simulate_for_condition(condition, theta_extended, amici_model)

print(f"Diff in sim {np.max(np.abs(res1_aggregated["sim"] - res2["sim"]))}")
print(f"Diff in data {np.max(np.abs(res1_aggregated["data"] - res2["data"]))}")
print(f"Diff in {key} {np.max(np.abs(res1_aggregated["t"] - res2["ts"]))}")
print(f"Diff in {key} {np.max(np.abs(res1_aggregated["tau"] - res2["taus"]))}")
print(f"Diff in {key} {np.max(np.abs(res1_aggregated["sigma_y"] - res2["sigma_ys"]))}")



"""
Look if the gradient is correct by comparing to a FD gradient:
"""

g_fd = objective_fd(theta_true, sensi_orders=(0, 1))[1]
g_sim = neg_log_post_full_with_grad(theta_true, amici_model)[1]


err1, err2, err3 = _theta_extended_to_thetas(np.abs(g_fd - g_sim))
# g11, g12, g13 = _theta_extended_to_thetas(g_fd)
# g21, g22, g23 = _theta_extended_to_thetas(g_sim)


"""
Next debugging thing: Let's calculate FDs for _simulate_single_data_point_dicts_for_condition
On a "per Datapoint" basis and compare it with dsim_dtau and dsim_dtheta
"""
from scipy.optimize import approx_derivative

theta_true = np.concatenate([
    np.log10(importer.petab_problem.parameter_df['nominalValue'].values), 
    importer.petab_problem.measurement_df['time'].values, 
    [2]
])
# create a random theta in [0, 10]
theta_test = np.random.uniform(0, 10, size=theta_true.shape)

for condition, n_datapoints in [
    ("model1_data1", 7),
    ("model1_data2", 14),
    ("model1_data3", 14),
    ("model1_data4", 21),
    ("model1_data5", 7),
    ("model1_data6", 14) 
    ]:

    for n_datapoint in range(n_datapoints):
        # theta_test = theta_true
        theta_test = np.asarray(theta_true, dtype=float)
        # condition = "model1_data6"

        def _fd_test_fun_jakob(theta):
            """
            My Implementation of the derivative
            """
            """
            dp_dicts = _simulate_single_data_point_dicts_for_condition(condition, theta, amici_model)
            theta_grad = np.zeros_like(theta)
            # chain rule for log trafo
            theta_grad[:13] = dp_dicts[0]["dsim_dtheta"] * 10 ** theta[:13] * np.log(10)
            # theta_grad[:13] = dp_dicts[0]["dsim_dtheta"]
            theta_grad[dp_dicts[n_datapoint]["theta_extended_index"]] = dp_dicts[n_datapoint]["dsim_dtau"]
            """
            theta_grad = neg_log_post_full_with_grad(theta, amici_model)[1]
            return  theta_grad

        def _fd_test_fun_for_scipy(theta):
            """
            The function f(theta), that we want to differentiate with scipy
            """
            # dp_dicts = _simulate_single_data_point_dicts_for_condition(condition, theta, amici_model)
            # return np.atleast_1d([dp_dicts[n_datapoint]["sim"]])
            return neg_log_post_full_with_grad(theta, amici_model)[0]
            

        grad_num = approx_derivative(
            _fd_test_fun_for_scipy,
            theta_test,
            method="3-point",     # better accuracy than 2-point
            rel_step=None,        # or set e.g.  rel_step=1e-6
            abs_step=None         # or abs_step=1e-8 for small thetas
        )

        grad_jakob = _fd_test_fun_jakob(theta_test)

        """
        tol = 1e-3
        if np.max(np.abs(grad_jakob - grad_num)) > tol and np.max(np.abs(grad_jakob - grad_num)/grad_jakob) > tol:
            # print(f"Gradient check failed for condition {condition}, datapoint {n_datapoint}")
            # print(np.max(grad_jakob[14:] - grad_num[14:]))
            # print(condition)
            # print(n_datapoint)
            # print(np.max(np.abs(grad_jakob[13:] - grad_num[13:])))
            for i in np.flatnonzero(np.abs(grad_jakob-grad_num)>tol):
                print(f"         {i}, {float(grad_jakob[i])}, {float(grad_num[i])}, {float(abs(grad_jakob[i]-grad_num[i]))}")
        """
        tol = 1e-3
        eps = 1e-12

        # If you want only the first 15, change to grad_jakob[:15], grad_num[:15]
        grad_jakob, grad_num

        abs_err = np.abs(grad_jakob - grad_num)
        rel_err = abs_err / np.maximum(np.abs(grad_num), eps)  # relative to numeric grad

        fail_mask = (abs_err > tol) & (rel_err > tol)
        bad_idx = np.flatnonzero(fail_mask)

        if bad_idx.size:
            # optional context
            # print(f"max abs={abs_err.max():.3e}, max rel={rel_err.max():.3e}")
            for i in bad_idx:
                print(f"        {i}, {float(grad_jakob[i])}, {float(grad_num[i])}, abs={float(abs_err[i])}, rel={float(rel_err[i])}")




# Plot the results
import matplotlib.pyplot as plt

tau_df = pd.read_csv(
        "Application_Bruno/simulations/full_parametrization_df.tsv", sep="\t"
        ).reset_index()

def visualize_for_condition(
    condition: str, 
    amici_model: amici.Model,
    theta: np.ndarray,
    plot_errorbars: bool = False,
    save_in_dir: str = None
):
    """
    Visualizes the simulation results for a given condition.

    Args:
        condition: str
            The condition to visualize.
        amici_model: amici.Model
            The AMICI model to use for the visualization.
        theta: np.ndarray
            The parameter vector to use for the visualization.
        plot_errorbars: bool
            Whether to plot error bars.
        save_in_dir: str
            The directory to save the plots. If None, the plots are not saved
    """
    theta_estimate, taus, sigma_tau = _theta_extended_to_thetas(theta)

    # dp_dicts contains a) all predictions and b) all data.
    dp_dicts = _simulate_single_data_point_dicts_for_condition(condition, theta, amici_model)
    amici_model = set_parameters_for_condition(condition, theta_estimate, amici_model)
    
    # set Time Points and simulate
    amici_model.setTimepoints([t for t in range(185)])
    solver = amici_model.getSolver()
    rdata = amici.runAmiciSimulation(amici_model, solver)

    # extract the number of observables for the current condition
    df_for_condition = tau_df[
        tau_df["simulationConditionId"] == condition]
    observables_for_condition = set(df_for_condition["observableId"])
    n_observables_for_condition = len(observables_for_condition)

    # Keep track of all plotted observables to avoid overlapping labels
    plotted_observable=[]
    for observable in observables_for_condition:
        plt.figure(figsize=(10, 6))

        if observable == "ybcar":
            x_trajectors = rdata.x[:, 0]
        elif observable == "ybcry":
            x_trajectors = rdata.x[:, 1]
        elif observable == "yb10":
            x_trajectors = rdata.x[:, 2]
        elif observable == "yohb10":
            x_trajectors = rdata.x[:, 4]
        elif observable == "yzea":
            x_trajectors = rdata.x[:, 6]

        t_trajectors = rdata.t

        plt.plot(t_trajectors, x_trajectors, label=observable)

        if plot_errorbars:
            for dp in dp_dicts:
                if dp["observable_name"] == observable:
                    plt.errorbar(
                        dp["t"], dp["data"], 
                        yerr=dp["sigma_y"], # in observable direction
                        xerr=theta[-1],  # in time direction
                        fmt='o',
                        color="red",
                        ecolor="gray",
                        elinewidth=1,
                        capsize=3,
                        label="Data" if observable in plotted_observable else None)
        
        plotted_observable.append(observable)

        plt.xlabel("Time")
        plt.ylabel(observable)
        plt.title(f"Simulation Results for {condition}")
        plt.legend()
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        plt.xlim((0, 181))
        plt.ylim((0, np.max(x_trajectors) * 1.2))

        if save_in_dir is not None:
            plt.savefig(f"{save_in_dir}/{condition}_{observable}.png")
        plt.show()
        plt.close()

save_in_dir = "Application_Bruno/data/plots"
for condition in tau_df["simulationConditionId"].unique():
    visualize_for_condition(
        condition=condition,
        amici_model=amici_model,
        theta=theta,
        plot_errorbars=True,
        save_in_dir=save_in_dir
    )

"""

"""

grad_theta = [np.sum(np.abs(x[:13]))/13 for x in result_full_multi_tau.optimize_result.grad]
grad_tau = [np.sum(np.abs(x[13:-1]))/77 for x in result_full_multi_tau.optimize_result.grad]
grad_sigma_tau = [np.sum(np.abs(x[-1])) for x in result_full_multi_tau.optimize_result.grad]

#. with yscale: log

plt.plot(grad_theta[:50], label=r"${||\nabla_\theta||_1}/n_{\theta}$")
plt.plot(grad_tau[:50], label=r"${||\nabla_\tau||_1}/n_{\tau}$")
plt.plot(grad_sigma_tau[:50], label=r"${||\nabla_{\sigma_{\tau}}||_1}/n_{\sigma_{\tau}}$")
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")
plt.yscale("log")
plt.xlim((0, 50))
plt.title("L1 Norm of Gradients")

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.legend(bbox_to_anchor=(0.98, 0.85))
