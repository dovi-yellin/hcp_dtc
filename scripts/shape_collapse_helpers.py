"""
Compact analysis outline for testing vertex-based PSD shape collapse.

Assumptions
-----------
- psd_arr: array of shape (n_regions, n_freqs)
- freqs:   array of shape (n_freqs,)
- dtc:     array of shape (n_regions,)   # or control parameter G, etc.
- PSDs are already positive and defined on the same frequency grid

Goal
----
1. Estimate vertex location and power for each PSD
2. Apply empirical-style collapse:
      - reduce vertex power
      - apply model-based multiplicative correction from DTC
3. Quantify collapse quality
4. Compare empirical / approximated / simulated PSD sets
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.stats import pearsonr


# ============================================================
# 1. Helpers
# ============================================================

def safe_log10(x, eps=1e-12):
    return np.log10(np.maximum(x, eps))


def smooth_log_psd(psd, sigma=2.0):
    """Smooth PSD in log-space to stabilize derivative-based vertex detection."""
    return gaussian_filter1d(safe_log10(psd), sigma=sigma)


def compute_vertex(freqs, psd, sigma=2.0, valid_band=None):
    """
    Estimate vertex as the point where the smoothed log-PSD curvature peaks
    or slope changes most strongly.
    """
    x = safe_log10(freqs)
    y = smooth_log_psd(psd, sigma=sigma)

    if valid_band is not None:
        mask = (freqs >= valid_band[0]) & (freqs <= valid_band[1])
    else:
        mask = np.ones_like(freqs, dtype=bool)

    # First and second derivatives in log-log space
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)

    # A simple practical choice: vertex = strongest curvature within band
    idx_candidates = np.where(mask)[0]
    idx_vertex = idx_candidates[np.argmax(np.abs(d2y[idx_candidates]))]

    return freqs[idx_vertex], psd[idx_vertex], idx_vertex, y


def estimate_curvature_at_vertex(freqs, psd, idx_vertex, sigma=2.0):
    """Second derivative of smoothed log-PSD at the vertex."""
    x = safe_log10(freqs)
    y = smooth_log_psd(psd, sigma=sigma)
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    return d2y[idx_vertex]


# ============================================================
# 2. Model-based multiplicative correction
# ============================================================

def model_multiplier(freqs, dtc_i, dtc_ref, beta=1.0, f0=0.1):

    ff = freqs / f0
    return ((dtc_ref + ff) / (dtc_i + ff)) ** beta


# ============================================================
# 3. Collapse quality metrics
# ============================================================

def compute_template(psd_arr):
    """Mean template in log-space."""
    return np.nanmean(safe_log10(psd_arr), axis=0)


def collapse_error_per_region(psd_arr):
    """
    Region-wise squared deviation from the common collapsed template
    in log-space.
    """
    template = compute_template(psd_arr)
    log_psd = safe_log10(psd_arr)
    err = np.nanmean((log_psd - template[None, :]) ** 2, axis=1)
    return err, template


def mean_pairwise_similarity(psd_arr):
    """
    Mean pairwise Pearson correlation among log-PSDs.
    """
    X = safe_log10(psd_arr)
    n = X.shape[0]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = pearsonr(X[i], X[j])
            vals.append(r)
    return np.mean(vals), np.array(vals)


def variance_reduction_score(original_psd_arr, collapsed_psd_arr):
    """
    Quantify how much between-region shape variance is reduced by collapse.
    """
    err_before, _ = collapse_error_per_region(original_psd_arr)
    err_after, _ = collapse_error_per_region(collapsed_psd_arr)
    return 1.0 - (np.mean(err_after) / np.mean(err_before))


# ============================================================
# 4. Main analysis wrapper
# ============================================================

# study shape collapse analytically
def generate_analytic_psds(
    freqs,
    dtc_values,
    tau_fast,
    tau_slow_scale=1.0,
    slow_weight_func=None,
    amplitude_func=None,
    normalize=False,
):
    """
    Generate synthetic PSDs from a two-mode analytical formula across a grid of DTC values.

    Model
    -----
    For each DTC value d:
        P_d(f) = A(d) * [ w(d) / (f^2 + f_s(d)^2) + (1 - w(d)) / (f^2 + f_f^2) ]

    where:
        f_f = 1 / (2*pi*tau_fast)
        f_s(d) = d / (2*pi*tau_slow_scale*tau_fast)
    """

    freqs = np.asarray(freqs, dtype=float)
    dtc_values = np.asarray(dtc_values, dtype=float)

    if np.any(freqs <= 0):
        raise ValueError("All frequencies must be positive.")
    if tau_fast <= 0:
        raise ValueError("tau_fast must be positive.")
    if tau_slow_scale <= 0:
        raise ValueError("tau_slow_scale must be positive.")

    if slow_weight_func is None:
        slow_weight_func = lambda d: 1.0 / (1.0 + d)

    if amplitude_func is None:
        amplitude_func = lambda d: 1.0

    f_fast = 1.0 / (2.0 * np.pi * tau_fast)

    psd_list = []
    f_slow_list = []
    slow_weights = []
    amplitudes = []

    for d in dtc_values:
        if d <= 0:
            raise ValueError("All dtc_values must be positive.")

        f_slow = d / (2.0 * np.pi * tau_slow_scale * tau_fast)
        w = float(slow_weight_func(d))
        A = float(amplitude_func(d))

        w = np.clip(w, 0.0, 1.0)

        psd = A * (
            w / (freqs**2 + f_slow**2) +
            (1.0 - w) / (freqs**2 + f_fast**2)
        )

        if normalize:
            psd = psd / np.max(psd)

        psd_list.append(psd)
        f_slow_list.append(f_slow)
        slow_weights.append(w)
        amplitudes.append(A)

    psd_arr = np.vstack(psd_list)

    param_dict = {
        "dtc_values": dtc_values,
        "f_fast": f_fast,
        "f_slow": np.asarray(f_slow_list),
        "slow_weights": np.asarray(slow_weights),
        "amplitudes": np.asarray(amplitudes),
    }

    return psd_arr, param_dict

def two_mode_psd(freqs, A, w, f_slow, f_fast):
    """
    Two-mode PSD:
        P(f) = A * [ w/(f^2 + f_slow^2) + (1-w)/(f^2 + f_fast^2) ]
    """
    freqs = np.asarray(freqs, dtype=float)
    return A * (
        w / (freqs**2 + f_slow**2) +
        (1.0 - w) / (freqs**2 + f_fast**2)
    )


def fit_two_mode_psd(
    freqs,
    psd_emp,
    fit_band=None,
    p0=None,
    bounds=None,
    robust_loss="soft_l1",
    f_scale=0.1,
):
    """
    Fit empirical PSD to the analytical two-mode formula using bounded
    nonlinear least squares in log10-PSD space.

    """
    freqs = np.asarray(freqs, dtype=float)
    psd_emp = np.asarray(psd_emp, dtype=float)

    eps = 1e-12

    if fit_band is not None:
        mask = (freqs >= fit_band[0]) & (freqs <= fit_band[1])
        f = freqs[mask]
        y = psd_emp[mask]
    else:
        f = freqs
        y = psd_emp

    if np.any(f <= 0):
        raise ValueError("Frequencies must be positive.")
    if np.any(y <= 0):
        raise ValueError("PSD values must be positive.")

    # Default initial guess
    if p0 is None:
        A0 = np.median(y)
        w0 = 0.5
        f_slow0 = max(np.min(f) * 2, 1e-3)
        f_fast0 = min(np.median(f), np.max(f))
        p0 = [A0, w0, f_slow0, f_fast0]

    # Default bounds
    if bounds is None:
        lb = [1e-12, 0.0, 1e-4, 1e-4]
        ub = [1e12,   1.0, np.max(f), np.max(f) * 10]
        bounds = (lb, ub)

    logy = np.log10(np.maximum(y, eps))

    def residuals(params):
        A, w, f_slow, f_fast = params

        # Optional ordering penalty: enforce f_slow < f_fast by swapping
        if f_slow > f_fast:
            f_slow, f_fast = f_fast, f_slow
            w = 1.0 - w

        yhat = two_mode_psd(f, A, w, f_slow, f_fast)
        logyhat = np.log10(np.maximum(yhat, eps))
        return logyhat - logy

    res = least_squares(
        residuals,
        x0=p0,
        bounds=bounds,
        loss=robust_loss,
        f_scale=f_scale,
    )

    A_fit, w_fit, f_slow_fit, f_fast_fit = res.x

    # reorder if needed so slow < fast
    if f_slow_fit > f_fast_fit:
        f_slow_fit, f_fast_fit = f_fast_fit, f_slow_fit
        w_fit = 1.0 - w_fit

    psd_fit = two_mode_psd(freqs, A_fit, w_fit, f_slow_fit, f_fast_fit)

    ss_res = np.sum((np.log10(np.maximum(psd_emp, eps)) -
                     np.log10(np.maximum(psd_fit, eps))) ** 2)
    ss_tot = np.sum((np.log10(np.maximum(psd_emp, eps)) -
                     np.mean(np.log10(np.maximum(psd_emp, eps)))) ** 2)
    r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "A": A_fit,
        "w": w_fit,
        "f_slow": f_slow_fit,
        "f_fast": f_fast_fit,
        "tau_slow": 1.0 / (2.0 * np.pi * f_slow_fit),
        "tau_fast": 1.0 / (2.0 * np.pi * f_fast_fit),
        "psd_fit": psd_fit,
        "success": res.success,
        "cost": res.cost,
        "r2_log": r2_log,
        "optimizer_result": res,
    }

def fit_two_mode_psd_array(freqs, psd_arr, fit_band=(0.01, 0.5)):
    fits = []
    for i in range(psd_arr.shape[0]):
        fit = fit_two_mode_psd(freqs, psd_arr[i], fit_band=fit_band)
        fits.append(fit)
    return fits

def plot_all_empirical_vs_fitted_psds(freqs, psd_arr, fits, fig_num=None, alpha=0.2):
    """
    Compact overlay plot of all empirical PSDs and all fitted PSDs.

    Parameters
    ----------
    freqs : array, shape (n_freqs,)
    psd_arr : array, shape (n_regions, n_freqs)
        Empirical PSDs.
    fits : list of dict
        Output of fit_two_mode_psd_array(...), each containing "psd_fit".
    fig_num : int or None
    alpha : float
        Transparency for individual curves.
    """
    import matplotlib.pyplot as plt

    if fig_num is not None:
        plt.figure(fig_num, figsize=(7, 5))
    else:
        plt.figure(figsize=(7, 5))

    for i in range(psd_arr.shape[0]):
        plt.loglog(freqs, psd_arr[:,i], alpha=alpha)
        plt.loglog(freqs, fits[i]["psd_fit"], '--') #, alpha=alpha)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title("All empirical and fitted PSD profiles")
    plt.tight_layout()

#==============================

def two_mode_psd_fixed(freqs, f_slow, A, w, f_fast):
    """
    Two-mode PSD with only f_slow free:
        P(f) = A * [ w/(f^2 + f_slow^2) + (1-w)/(f^2 + f_fast^2) ]
    """
    freqs = np.asarray(freqs, dtype=float)
    return A * (
        w / (freqs**2 + f_slow**2) +
        (1.0 - w) / (freqs**2 + f_fast**2)
    )

def fit_two_mode_psd_fslow_only(
    freqs,
    psd_emp,
    A,
    w,
    f_fast,
    fit_band=None,
    f_slow0=None,
    bounds=None,
    robust_loss="soft_l1",
    f_scale=0.1,
):
    """
    Fit one empirical PSD with only f_slow free, while A, w, and f_fast are fixed.
    Fitting is done in log10-PSD space.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd_emp = np.asarray(psd_emp, dtype=float)
    eps = 1e-12

    if fit_band is not None:
        mask = (freqs >= fit_band[0]) & (freqs <= fit_band[1])
        f = freqs[mask]
        y = psd_emp[mask]
    else:
        f = freqs
        y = psd_emp

    if f_slow0 is None:
        f_slow0 = max(np.min(f) * 2, 1e-3)

    if bounds is None:
        bounds = ([1e-5], [np.max(f)])

    logy = np.log10(np.maximum(y, eps))

    def residuals(params):
        f_slow = params[0]
        yhat = two_mode_psd_fixed(f, f_slow, A, w, f_fast)
        return np.log10(np.maximum(yhat, eps)) - logy

    res = least_squares(
        residuals,
        x0=[f_slow0],
        bounds=bounds,
        loss=robust_loss,
        f_scale=f_scale,
    )

    f_slow_fit = res.x[0]
    psd_fit = two_mode_psd_fixed(freqs, f_slow_fit, A, w, f_fast)

    ss_res = np.sum((np.log10(np.maximum(psd_emp, eps)) -
                     np.log10(np.maximum(psd_fit, eps))) ** 2)
    ss_tot = np.sum((np.log10(np.maximum(psd_emp, eps)) -
                     np.mean(np.log10(np.maximum(psd_emp, eps)))) ** 2)
    r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "A": A,
        "w": w,
        "f_fast": f_fast,
        "f_slow": f_slow_fit,
        "tau_slow": 1.0 / (2.0 * np.pi * f_slow_fit),
        "tau_fast": 1.0 / (2.0 * np.pi * f_fast),
        "psd_fit": psd_fit,
        "success": res.success,
        "cost": res.cost,
        "r2_log": r2_log,
        "optimizer_result": res,
    }


def fit_two_mode_psd_array_fslow_only(
    freqs,
    psd_arr,
    A,
    w,
    f_fast,
    fit_band=(0.01, 0.5),
    f_slow0=None,
    bounds=None,
    robust_loss="soft_l1",
    f_scale=0.1,
):
    """
    Fit all PSD profiles assuming shared A, w, and f_fast across regions,
    with only f_slow varying from region to region.
    """
    fits = []
    for i in range(psd_arr.shape[0]):
        fit = fit_two_mode_psd_fslow_only(freqs=freqs, psd_emp=psd_arr[i], A=A, w=w, f_fast=f_fast,
            fit_band=fit_band, f_slow0=f_slow0, bounds=bounds, robust_loss=robust_loss, f_scale=f_scale,
        )
        fits.append(fit)
    return fits

#===========================================================================

import numpy as np
from scipy.optimize import least_squares


def two_mode_psd(freqs, A, w, f_slow, f_fast):
    freqs = np.asarray(freqs, dtype=float)
    return A * (
        w / (freqs**2 + f_slow**2) +
        (1.0 - w) / (freqs**2 + f_fast**2)
    )


def fit_two_mode_psd_array_joint(
    freqs,
    psd_arr,
    fit_band=(0.01, 0.5),
    p0_global=None,
    fslow0=None,
    bounds_global=None,
    bounds_fslow=None,
    robust_loss="soft_l1",
    f_scale=0.1,
):
    """
    Joint fit across all PSDs:
      shared across all regions: A, w, f_fast
      region-specific: f_slow[i]

    """
    freqs = np.asarray(freqs, dtype=float)
    psd_arr = np.asarray(psd_arr, dtype=float)
    eps = 1e-12

    mask = np.ones_like(freqs, dtype=bool)
    if fit_band is not None:
        mask = (freqs >= fit_band[0]) & (freqs <= fit_band[1])

    f = freqs[mask]
    Y = psd_arr[:, mask]
    n_regions = Y.shape[0]

    if np.any(f <= 0):
        raise ValueError("Frequencies must be positive.")
    if np.any(Y <= 0):
        raise ValueError("PSD values must be positive.")

    if p0_global is None:
        A0 = float(np.median(Y))
        w0 = 0.5
        ffast0 = float(np.median(f))
        p0_global = [A0, w0, ffast0]

    if fslow0 is None:
        fslow0 = np.full(n_regions, max(np.min(f) * 2, 1e-3), dtype=float)
    elif np.isscalar(fslow0):
        fslow0 = np.full(n_regions, float(fslow0), dtype=float)
    else:
        fslow0 = np.asarray(fslow0, dtype=float)
        if len(fslow0) != n_regions:
            raise ValueError("fslow0 must be scalar or length n_regions.")

    if bounds_global is None:
        bounds_global = ([1e-12, 0.0, 1e-4], [1e12, 1.0, np.max(f) * 10])

    if bounds_fslow is None:
        bounds_fslow = (1e-5, np.max(f))

    lb = np.array(list(bounds_global[0]) + [bounds_fslow[0]] * n_regions, dtype=float)
    ub = np.array(list(bounds_global[1]) + [bounds_fslow[1]] * n_regions, dtype=float)

    x0 = np.array(list(p0_global) + list(fslow0), dtype=float)

    logY = np.log10(np.maximum(Y, eps))

    def residuals(params):
        A, w, f_fast = params[:3]
        f_slow_vec = params[3:]

        pred = np.zeros_like(Y)
        for i in range(n_regions):
            pred[i] = two_mode_psd(f, A, w, f_slow_vec[i], f_fast)

        logPred = np.log10(np.maximum(pred, eps))
        return (logPred - logY).ravel()

    res = least_squares(
        residuals,
        x0=x0,
        bounds=(lb, ub),
        loss=robust_loss,
        f_scale=f_scale,
    )

    A_fit, w_fit, f_fast_fit = res.x[:3]
    f_slow_fit = res.x[3:]

    psd_fit_arr = np.zeros_like(psd_arr, dtype=float)
    for i in range(n_regions):
        psd_fit_arr[i] = two_mode_psd(freqs, A_fit, w_fit, f_slow_fit[i], f_fast_fit)

    log_emp = np.log10(np.maximum(psd_arr, eps))
    log_fit = np.log10(np.maximum(psd_fit_arr, eps))

    ss_res_total = np.sum((log_emp - log_fit) ** 2)
    ss_tot_total = np.sum((log_emp - np.mean(log_emp)) ** 2)
    r2_log_global = 1.0 - ss_res_total / ss_tot_total if ss_tot_total > 0 else np.nan

    r2_log_per_region = np.full(n_regions, np.nan)
    for i in range(n_regions):
        ss_res_i = np.sum((log_emp[i] - log_fit[i]) ** 2)
        ss_tot_i = np.sum((log_emp[i] - np.mean(log_emp[i])) ** 2)
        r2_log_per_region[i] = 1.0 - ss_res_i / ss_tot_i if ss_tot_i > 0 else np.nan

    return {
        "A": A_fit,
        "w": w_fit,
        "f_fast": f_fast_fit,
        "tau_fast": 1.0 / (2.0 * np.pi * f_fast_fit),
        "f_slow": f_slow_fit,
        "tau_slow": 1.0 / (2.0 * np.pi * f_slow_fit),
        "psd_fit_arr": psd_fit_arr,
        "r2_log_global": r2_log_global,
        "r2_log_per_region": r2_log_per_region,
        "mean_r2_log_per_region": np.nanmean(r2_log_per_region),
        "median_r2_log_per_region": np.nanmedian(r2_log_per_region),
        "success": res.success,
        "cost": res.cost,
        "optimizer_result": res,
    }

#=================================================================================

def solve_A_from_vertex(f_vertex, p_vertex, f_slow, w, f_fast, eps=1e-12):
    """
    Solve analytically for A so the fitted curve passes exactly through
    the target vertex point (f_vertex, p_vertex).
    """
    denom = (
        w / (f_vertex**2 + f_slow**2) +
        (1.0 - w) / (f_vertex**2 + f_fast**2)
    )
    return p_vertex / max(denom, eps)


def fit_two_mode_psd_vertex(
    freqs,
    psd_emp,
    w,
    f_fast,
    f_vertex,
    p_vertex,
    fit_band=(0.01, 0.5),
    mode="joint",                  # "joint" or "iterative"
    enforce_vertex_exact=False,    # if True, A is derived analytically from vertex
    vertex_weight=10.0,            # penalty weight if not exact
    f_slow0=None,
    A0=None,
    bounds_fslow=None,
    bounds_A=None,
    n_iter=6,
    robust_loss="soft_l1",
    f_scale=0.1,
):
    """
    Fit one empirical PSD with fixed w and f_fast, free f_slow and A,
    while targeting a shared empirical vertex point.

    Returns
    -------
    dict with fitted parameters and fitted PSD
    """
    freqs = np.asarray(freqs, dtype=float)
    psd_emp = np.asarray(psd_emp, dtype=float)
    eps = 1e-12

    mask = np.ones_like(freqs, dtype=bool)
    if fit_band is not None:
        mask = (freqs >= fit_band[0]) & (freqs <= fit_band[1])

    f = freqs[mask]
    y = psd_emp[mask]
    logy = np.log10(np.maximum(y, eps))

    if f_slow0 is None:
        f_slow0 = max(np.min(f) * 2, 1e-3)
    if A0 is None:
        A0 = np.median(y)

    if bounds_fslow is None:
        bounds_fslow = (1e-5, np.max(f))
    if bounds_A is None:
        bounds_A = (1e-12, 1e12)

    def residuals_joint(params):
        f_slow, A = params
        yhat = two_mode_psd_fixed(f, f_slow, A, w, f_fast)
        res = np.log10(np.maximum(yhat, eps)) - logy

        if not enforce_vertex_exact:
            p_v_hat = two_mode_psd_fixed(np.array([f_vertex]), f_slow, A, w, f_fast)[0]
            vertex_res = np.sqrt(vertex_weight) * (
                np.log10(max(p_v_hat, eps)) - np.log10(max(p_vertex, eps))
            )
            res = np.concatenate([res, [vertex_res]])
        return res

    def residuals_fslow_only(params, A_fixed):
        f_slow = params[0]
        if enforce_vertex_exact:
            A_use = solve_A_from_vertex(f_vertex, p_vertex, f_slow, w, f_fast)
        else:
            A_use = A_fixed
        yhat = two_mode_psd_fixed(f, f_slow, A_use, w, f_fast)
        res = np.log10(np.maximum(yhat, eps)) - logy

        if not enforce_vertex_exact:
            p_v_hat = two_mode_psd_fixed(np.array([f_vertex]), f_slow, A_use, w, f_fast)[0]
            vertex_res = np.sqrt(vertex_weight) * (
                np.log10(max(p_v_hat, eps)) - np.log10(max(p_vertex, eps))
            )
            res = np.concatenate([res, [vertex_res]])
        return res

    def residuals_A_only(params, f_slow_fixed):
        A = params[0]
        yhat = two_mode_psd_fixed(f, f_slow_fixed, A, w, f_fast)
        res = np.log10(np.maximum(yhat, eps)) - logy

        p_v_hat = two_mode_psd_fixed(np.array([f_vertex]), f_slow_fixed, A, w, f_fast)[0]
        vertex_res = np.sqrt(vertex_weight) * (
            np.log10(max(p_v_hat, eps)) - np.log10(max(p_vertex, eps))
        )
        res = np.concatenate([res, [vertex_res]])
        return res

    # --------------------------------------------------
    # Joint mode
    # --------------------------------------------------
    if mode == "joint":
        if enforce_vertex_exact:
            def residuals_fslow_exact(params):
                f_slow = params[0]
                A = solve_A_from_vertex(f_vertex, p_vertex, f_slow, w, f_fast)
                yhat = two_mode_psd_fixed(f, f_slow, A, w, f_fast)
                return np.log10(np.maximum(yhat, eps)) - logy

            res = least_squares(
                residuals_fslow_exact,
                x0=[f_slow0],
                bounds=([bounds_fslow[0]], [bounds_fslow[1]]),
                loss=robust_loss,
                f_scale=f_scale,
            )
            f_slow_fit = res.x[0]
            A_fit = solve_A_from_vertex(f_vertex, p_vertex, f_slow_fit, w, f_fast)
        else:
            res = least_squares(
                residuals_joint,
                x0=[f_slow0, A0],
                bounds=([bounds_fslow[0], bounds_A[0]],
                        [bounds_fslow[1], bounds_A[1]]),
                loss=robust_loss,
                f_scale=f_scale,
            )
            f_slow_fit, A_fit = res.x

    # --------------------------------------------------
    # Iterative mode
    # --------------------------------------------------
    elif mode == "iterative":
        f_slow_fit = f_slow0
        A_fit = solve_A_from_vertex(f_vertex, p_vertex, f_slow_fit, w, f_fast) if enforce_vertex_exact else A0

        for _ in range(n_iter):
            # step 1: optimize f_slow
            res1 = least_squares(
                residuals_fslow_only,
                x0=[f_slow_fit],
                bounds=([bounds_fslow[0]], [bounds_fslow[1]]),
                args=(A_fit,),
                loss=robust_loss,
                f_scale=f_scale,
            )
            f_slow_fit = res1.x[0]

            # step 2: update A
            if enforce_vertex_exact:
                A_fit = solve_A_from_vertex(f_vertex, p_vertex, f_slow_fit, w, f_fast)
            else:
                res2 = least_squares(
                    residuals_A_only,
                    x0=[A_fit],
                    bounds=([bounds_A[0]], [bounds_A[1]]),
                    args=(f_slow_fit,),
                    loss=robust_loss,
                    f_scale=f_scale,
                )
                A_fit = res2.x[0]
    else:
        raise ValueError("mode must be 'joint' or 'iterative'")

    psd_fit = two_mode_psd_fixed(freqs, f_slow_fit, A_fit, w, f_fast)

    ss_res = np.sum((np.log10(np.maximum(psd_emp, eps)) -
                     np.log10(np.maximum(psd_fit, eps))) ** 2)
    ss_tot = np.sum((np.log10(np.maximum(psd_emp, eps)) -
                     np.mean(np.log10(np.maximum(psd_emp, eps)))) ** 2)
    r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    p_vertex_fit = two_mode_psd_fixed(np.array([f_vertex]), f_slow_fit, A_fit, w, f_fast)[0]

    return {
        "A": A_fit,
        "w": w,
        "f_fast": f_fast,
        "f_slow": f_slow_fit,
        "tau_fast": 1.0 / (2.0 * np.pi * f_fast),
        "tau_slow": 1.0 / (2.0 * np.pi * f_slow_fit),
        "psd_fit": psd_fit,
        "p_vertex_fit": p_vertex_fit,
        "vertex_log_error": np.log10(max(p_vertex_fit, eps)) - np.log10(max(p_vertex, eps)),
        "r2_log": r2_log,
    }


def fit_two_mode_psd_array_vertex(
    freqs,
    psd_arr,
    w,
    f_fast,
    f_vertex,
    p_vertex,
    fit_band=(0.01, 0.5),
    mode="joint",
    enforce_vertex_exact=False,
    vertex_weight=10.0,
    f_slow0=None,
    A0=None,
    bounds_fslow=None,
    bounds_A=None,
    n_iter=6,
):
    """
    Apply the vertex-targeted fit to an array of PSDs.
    """
    fits = []
    for i in range(psd_arr.shape[0]):
        fit = fit_two_mode_psd_vertex(
            freqs=freqs,
            psd_emp=psd_arr[i],
            w=w,
            f_fast=f_fast,
            f_vertex=f_vertex,
            p_vertex=p_vertex,
            fit_band=fit_band,
            mode=mode,
            enforce_vertex_exact=enforce_vertex_exact,
            vertex_weight=vertex_weight,
            f_slow0=f_slow0,
            A0=A0,
            bounds_fslow=bounds_fslow,
            bounds_A=bounds_A,
            n_iter=n_iter,
        )
        fits.append(fit)
    return fits


