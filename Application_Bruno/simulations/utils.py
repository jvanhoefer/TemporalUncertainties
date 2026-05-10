def truncated_gaussian_log_correction(sigma_tau, lower_bound=0):
    """
    Computes the log correction term for a truncated Gaussian distribution.

    Parameters:
    - sigma_tau: Standard deviation of the Gaussian distribution.
    - lower_bound: The lower bound of the truncation (default is 0).

    Returns:
    - log_correction: The log correction term to be added to the log-likelihood.
    """
    # Compute the cumulative distribution function (CDF) at the lower bound
    cdf_lower = 0.5 * (1 + np.erf(lower_bound / (sigma_tau * np.sqrt(2))))
    
    # Compute the log correction term
    log_correction = -np.log(1 - cdf_lower)
    
    return log_correction
    
