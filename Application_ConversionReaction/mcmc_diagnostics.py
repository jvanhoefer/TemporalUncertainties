"""
Convergence and efficiency diagnostics for single MCMC chains (Figure 4 B/C).

Replaces pyPESTO's `geweke_test` / `effective_sample_size` (pyPESTO <= 0.7.0,
unchanged on `main` as of 2026-10): pyPESTO estimates the spectral density at
frequency 0 without subtracting the window mean, so the Geweke z-score shrinks
towards 0 for every parameter whose posterior mean is not close to 0. Since
pyPESTO also sets the burn-in (and with it the ESS) from that z-score, both
were too optimistic. See `.scratch/fixes/pypesto-geweke-bug.md`.

Here:
- the variance of a window mean is var * tau_int / n, with the integrated
  autocorrelation time tau_int from an FFT autocorrelation and Sokal's
  automatic window (c = 5), so every window is centered;
- all diagnostics are restricted to the mechanistic parameters theta (the
  first N_THETA entries of the chain), which both approaches share;
- the threshold is Bonferroni-corrected over theta (2.50 for 4 parameters);
- the burn-in follows Ballnus et al. (2017): candidates 0 %, 5 %, ... are
  tested in order and the first that passes is taken (a fixed-sequence
  procedure, so no correction across candidates is needed);
- a chain whose burn-in would exceed half its length counts as not converged.
  Without this cap, a late burn-in leaves a short remainder in which slow
  drifts are invisible, and slowly mixing chains pass;
- the reported z-score (Figure 4B) is the one after the burn-in, or after the
  last candidate for chains that do not converge. So a chain is converged
  iff its z-score is below Z_THRESHOLD (and its window ESS is sufficient),
  and Figure 4B/C/D use one and the same rule.
"""
import numpy as np
from scipy.stats import norm

N_THETA = 4
ALPHA = 0.05
Z_THRESHOLD = float(norm.ppf(1 - ALPHA / (2 * N_THETA)))  # 2.498 for 4 parameters
MIN_WINDOW_ESS = 50  # below this, no Geweke variant is calibrated
N_FRAGMENTS = 20  # candidate burn-ins in steps of 5 % (as in pyPESTO)
MAX_BURN_IN_FRACTION = 0.5  # candidates 0 %, 5 %, ..., 50 %


def tau_int(x: np.ndarray, c: float = 5.0) -> np.ndarray:
    """Integrated autocorrelation time per column (FFT + Sokal window), >= 1."""
    n = x.shape[0]
    x = x - x.mean(axis=0)
    f = np.fft.rfft(x, n=2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=2 * n, axis=0)[:n]
    taus = np.full(x.shape[1], np.inf)  # constant column: no information
    for j in range(x.shape[1]):
        if acf[0, j] <= 0:
            continue
        cs = 2 * np.cumsum(acf[:, j] / acf[0, j]) - 1
        inside = np.arange(n) < c * cs
        w = np.argmin(inside) if not inside.all() else n - 1
        taus[j] = max(cs[w], 1.0)
    return taus


def ess_per_parameter(x: np.ndarray) -> np.ndarray:
    """Effective sample size per column."""
    return x.shape[0] / tau_int(x)


def geweke_zscores(chain: np.ndarray, a: float = 0.1, b: float = 0.5):
    """
    Centered Geweke z per column: first `a` fraction vs. last `b` fraction.

    Returns (z, ess_first_window). A column without variation in either
    window gets z = nan and ESS = 0.
    """
    n = chain.shape[0]
    xa, xb = chain[: int(a * n)], chain[n - int(b * n):]
    ta, tb = tau_int(xa), tau_int(xb)
    var_a = xa.var(axis=0, ddof=1) * ta / len(xa)
    var_b = xb.var(axis=0, ddof=1) * tb / len(xb)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (xa.mean(axis=0) - xb.mean(axis=0)) / np.sqrt(var_a + var_b)
    return z, len(xa) / ta


def passes_geweke(chain: np.ndarray) -> bool:
    """Stationarity test on theta: Bonferroni threshold and enough samples."""
    z, ess_window = geweke_zscores(chain[:, :N_THETA])
    return bool(
        np.all(np.isfinite(z))
        and np.max(np.abs(z)) < Z_THRESHOLD
        and np.min(ess_window) >= MIN_WINDOW_ESS
    )


def burn_in_candidates(n: int) -> list:
    """Candidate burn-ins k * n / N_FRAGMENTS up to MAX_BURN_IN_FRACTION * n."""
    step = n // N_FRAGMENTS
    return [k * step for k in range(int(MAX_BURN_IN_FRACTION * N_FRAGMENTS) + 1)]


def burn_in(chain: np.ndarray):
    """
    First candidate burn-in (see `burn_in_candidates`) after which the
    remaining chain passes `passes_geweke`, or None if no candidate passes
    (not converged).
    """
    for b in burn_in_candidates(chain.shape[0]):
        if passes_geweke(chain[b:]):
            return b
    return None


def chain_diagnostics(chain: np.ndarray, sampling_time: float) -> dict:
    """
    All Figure 4 quantities for one chain (theta = first N_THETA columns).

    - geweke_z_score: max |z| over theta after the burn-in, or after the last
      candidate if not converged (panel B)
    - geweke_window_ess: min over theta of the ESS in the first Geweke window
      of that same test
    - burn_in: see `burn_in`; None = not converged
    - ESS / ESS_per_second: min over theta after burn-in, None if not converged
    """
    theta = np.asarray(chain)[:, :N_THETA]
    b = burn_in(theta)
    start = burn_in_candidates(len(theta))[-1] if b is None else b
    z, ess_window = geweke_zscores(theta[start:])
    ess = None if b is None else float(np.min(ess_per_parameter(theta[b:])))
    return {
        "geweke_z_score": float(np.max(np.abs(z))),
        "geweke_window_ess": float(np.min(ess_window)),
        "burn_in": b,
        "ESS": ess,
        "ESS_per_second": None if ess is None else ess / sampling_time,
    }
