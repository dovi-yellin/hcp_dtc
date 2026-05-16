from os.path import join
from pathlib import Path
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, least_squares
from scipy.stats import pearsonr

num_subjects = 40
num_rois = 180
_REPO_ROOT = Path(__file__).resolve().parents[1]
path_str = _REPO_ROOT / 'data'
path_result_str = _REPO_ROOT / 'results'

path_result_str = 'results'
subject_filename = ''
lettersize = 22

hierarchy_filename = 'Glasser_2016_Table_my_Hierarchy.csv'
df_glasser = pd.read_csv(join(path_str, hierarchy_filename))
hierarchy = df_glasser['Hierarchy'].values


str_all_subjects_all_rois_mean_psd = 'all_subjects_all_rois_mean_psd_v2'
with open(join(path_str, str_all_subjects_all_rois_mean_psd + '.pickle'), 'rb') as handle:
    data_all_subs_all_rois_mean_psd = pickle.load(handle)

empirical_psd = np.array(data_all_subs_all_rois_mean_psd).T

df = 1 / 300  # 0.003333... Hz
fr = np.arange(0, 0.5 + df / 2, df)
fr_tag = fr[(fr >= 0.01) & (fr <= 0.25)]

lettersize = 20
fig1 = plt.figure(1)
plt.loglog(fr_tag, empirical_psd)
plt.xlabel('Freq (Hz)')
plt.ylabel('Power (Index)')
plt.ylim(0.8, 11)
plt.tight_layout()


def approximate_intersection(x, y, kind='cubic'):
    """    Approximate the intersection point of multiple curves.
    Params:
        x (numpy.ndarray): 1D array of x-values common to all curves.
        y (numpy.ndarray): 2D array (n x m) of y-values where each column represents a curve.
        kind (str): Kind of interpolation to use for each curve.

    Returns: dispersion minimizing difference between maximum and minimum y values at x_intersect
    """
    # Create an interpolation function for each curve. Using the same x vector for all curves.
    interpolators = [interp1d(x, y[:, i], kind=kind, fill_value="extrapolate") for i in range(y.shape[1])]

    # Given x value, return the diff between max and min y of the curves. At the intersection, this should be minimal.
    def objective(x_val):
        # Interpolate each curve at x_val.
        y_vals = np.array([interp(x_val) for interp in interpolators])
        return np.max(y_vals) - np.min(y_vals)

    # Use a minimization algorithm to find x within the original range that minimizes the objective.
    res = minimize_scalar(objective, bounds=(x[0], x[-1]), method='bounded')

    # The x location of minimal dispersion
    x_intersect = res.x
    # Calculate the y-values from each curve at x_intersect and then average.
    y_vals_at_intersect = np.array([interp(x_intersect) for interp in interpolators])
    y_intersect = np.mean(y_vals_at_intersect)

    return x_intersect, y_intersect, res.fun


corrcoef_matrix = np.corrcoef(empirical_psd.T)
plt.figure(0, figsize=(8, 4))
im = plt.imshow(corrcoef_matrix, cmap='jet')
plt.yticks([0, 50, 100, 150])  # fontsize=font_size)
plt.ylabel('ROI Index', fontsize=20)
cbar = plt.colorbar(im)

x = fr_tag
y = empirical_psd

y_mean = np.mean(y, axis=1)

# Compute the approximate intersection (vertex point).
xi, yi, dispersion = approximate_intersection(x, y)

print(f"Approximate intersection at x = {xi:.4f}, y = {yi:.4f}")
print(f"Dispersion (max - min of y-values) at this point: {dispersion:.4e}")

plt.figure(2, figsize=(8, 6))
plt.loglog(x, y)  # plt.plot(x, y[:, i], label=f"Curve {i + 1}")
plt.loglog(x, y_mean, linewidth=6, color='red')  # plt.plot(x,y_mean, linewidth=6, color='red')
plt.loglog(xi, yi, 'ko', markersize=8, label="Intersection Approx.")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

x_tag = x - xi
y_tag = y - yi

plt.figure(3, figsize=(8, 6))
plt.plot(x_tag, y_tag)
plt.plot(0.0, 0.0, 'ko', markersize=8, label="Intersection Approx.")

psd = y_tag
psd_mean = np.mean(y_tag, axis=1)

# retrieve mean fALFF per ROI
mean_fALFF_arr = []
for sub_ind in range(num_subjects):
    with open(join(path_str, 'criticality_markers_subject_n' + str(sub_ind + 1) + '.pickle'), 'rb') as handle:
        critical_markers = pickle.load(handle)
    mean_fALFF_arr.append(critical_markers['mean_falff'])
mean_fALFF_arr = np.array(mean_fALFF_arr)
falff = mean_fALFF_arr[:, 0:180].mean(axis=0)

offset = 0.2  # 0.2 # without HRF - 0.06 # the offset with HRF - 0.2
exponent = 0.24  # 0.24 # without HRF 0.5 # with HRF - 0.24
control_param = 1 - (offset / falff) ** (1 / exponent)


# Example parametric model for g(falff)
def g_model(control_param, params):
    a, b = params
    # return a + b * falff
    return b * (control_param ** a)


def residuals(params, psd, control_param, threshold=1.0):
    if threshold < 1.0:
        control_param = control_param[control_param > threshold]
        psd = psd.T[control_param > threshold]

    # Compute the model g for each value in falff.
    g_values = g_model(control_param, params)
    # Calculate the mean of psd over the frequency range.
    mean_psd = np.mean(psd, axis=1)
    # Compute the residual per each ROI PSD: note that we divide by mean_psd to normalize.
    r = np.mean(((psd / g_values).T - mean_psd) / mean_psd, axis=1)
    return r


# Initial guess for the parameters [a, b]
initial_guess = [1.0, 0.1]

# Use the least_squares function from SciPy to minimize the residuals.
result = least_squares(residuals, initial_guess, args=(psd, control_param))

# Extract the optimal parameters.
optimal_params = result.x

print("Optimal parameters for fit:", optimal_params)

# Compute the fitted g values using the optimal parameters.
g_fit = g_model(control_param, optimal_params)

# Compute the fitted scaled psd: psd_fit = psd / g_fit.
psd_collapsed = psd / g_fit
psd_mean_scaled = psd_collapsed.mean(axis=1)

# Visualize the Results by plotting the original psd, the fitted g function, and the scaled psd vs. the mean psd.
plt.figure(4, figsize=(10, 6))
plt.plot(fr_tag,
         psd_collapsed)  # plt.loglog(fr_tag, psd_collapsed[:,126]*2.3, linewidth=3, color='yellow') # plt.loglog(fr_tag, psd_collapsed[:,12]*0.6, linewidth=3, color='green')
plt.plot(fr_tag, psd_mean_scaled, linewidth=6, color='red')
plt.xlabel('fr')
plt.ylabel('Scaled psd')
# plt.legend()
plt.title('Scaled psd Relative to Mean(psd)')
plt.tight_layout()

from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

norm = Normalize(vmin=np.min(control_param), vmax=np.max(control_param))
cmap = get_cmap('jet')
colors = [cmap(norm(cp)) for cp in control_param]

plt.figure(5, figsize=(8, 6))
lines = plt.loglog(fr_tag, empirical_psd)  # , color=np.squeeze(control_param), cmap='jet')
# assign colors (a tiny loop; still O(K))
for ln, c in zip(lines, colors):
    ln.set_color(c)
plt.xlabel('Freq (Hz)', fontsize=lettersize)
plt.ylabel('Power (Index)', fontsize=lettersize)
plt.xticks(fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.ylim(0.8, 11)
plt.tight_layout()

# correct back to original y values
psd_collapsed_org = psd_collapsed + yi
psd_mean_scaled_org = psd_mean_scaled + yi

plt.figure(6, figsize=(8, 6))
lines = plt.loglog(fr_tag, psd_collapsed_org)
for ln, c in zip(lines, colors):
    ln.set_color(c)
plt.xlabel('Freq (Hz)', fontsize=lettersize)
plt.ylabel('Power (Index)', fontsize=lettersize)
plt.ylim(0.8, 11)
plt.xticks(fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.tight_layout()
plt.show()
# plt.savefig('xyz')

from matplotlib import cm, colors as mcolors


def plot_empirical_vs_analytical_psds(fr_tag, empirical_psd, analytical_psd, control_param, ylabel="PSD",
                                      xlabel="Freq (Hz)", cmap="jet", empirical_alpha=0.18, analytical_alpha=0.95,
                                      empirical_lw=1.0,
                                      analytical_lw=2.2, analytical_ls="--", ylim=None, lettersize=12, figsize=(7, 5),
                                      add_colorbar=True):
    fr_tag = np.asarray(fr_tag)
    empirical_psd = np.asarray(empirical_psd)
    analytical_psd = np.asarray(analytical_psd)
    control_param = np.asarray(control_param)

    if empirical_psd.shape != analytical_psd.shape:
        raise ValueError("empirical_psd and analytical_psd must have the same shape.")
    if empirical_psd.shape[1] != control_param.shape[0]:
        raise ValueError("Number of PSD profiles must match length of control_param.")

    norm = mcolors.Normalize(vmin=np.min(control_param), vmax=np.max(control_param))
    cmap_obj = cm.get_cmap(cmap)
    curve_colors = [cmap_obj(norm(c)) for c in control_param]

    fig, ax = plt.subplots(figsize=figsize)

    for i, col in enumerate(curve_colors):
        ax.loglog(
            fr_tag,
            empirical_psd[:, i],
            color=col,
            alpha=empirical_alpha,
            linewidth=empirical_lw,
            zorder=1,
        )
        ax.loglog(
            fr_tag,
            analytical_psd[:, i],
            analytical_ls,
            color=col,
            alpha=analytical_alpha,
            linewidth=analytical_lw,
            zorder=2,
        )

    ax.set_xlabel(xlabel, fontsize=lettersize)
    ax.set_ylabel(ylabel, fontsize=lettersize)
    ax.tick_params(axis="both", labelsize=lettersize)

    if ylim is not None:
        ax.set_ylim(ylim)

    if add_colorbar:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Control parameter", fontsize=lettersize)
        cbar.ax.tick_params(labelsize=lettersize)

    plt.tight_layout()
    return fig, ax


# analytical PSD to empirical shape fitting
from shape_collapse_helpers import fit_two_mode_psd_array, fit_two_mode_psd_array_fslow_only, \
    fit_two_mode_psd_array_joint, fit_two_mode_psd_array_vertex

joint_fit = fit_two_mode_psd_array_joint(freqs=fr_tag, psd_arr=empirical_psd.T, fit_band=(0.01, 0.5))

print("A =", joint_fit["A"], "; w =", joint_fit["w"], "; f_fast =", joint_fit["f_fast"])
print("tau_fast =", joint_fit["tau_fast"])
print("Global log-R2 =", joint_fit["r2_log_global"])
print("Mean per-region log-R2 =", joint_fit["mean_r2_log_per_region"])

fitted_psd_arr = np.vstack([joint_fit["psd_fit_arr"]])
analytical_collapsed = ((fitted_psd_arr.T - yi) / g_fit[None, :]) + yi

fig, ax = plot_empirical_vs_analytical_psds(fr_tag, psd_collapsed_org, analytical_collapsed,
                                            control_param=control_param, ylabel="Power (Index)", ylim=(0.8, 11),
                                            lettersize=12)

plt.figure(figsize=(7, 5))
for i in range(psd_collapsed_org.shape[0]):
    plt.loglog(fr_tag, psd_collapsed_org.T[i], alpha=0.2)
    plt.loglog(fr_tag, analytical_collapsed.T[i], "--", alpha=0.9)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Collapsed PSD")
plt.tight_layout()

fits = fit_two_mode_psd_array(fr_tag, empirical_psd.T, fit_band=(0.01, 0.5))
fitted_psd_arr = np.vstack([fit["psd_fit"] for fit in fits])

fig, ax = plot_empirical_vs_analytical_psds(fr_tag, empirical_psd, fitted_psd_arr.T,
                                            control_param=control_param, ylabel="Power (Index)", ylim=(0.8, 11),
                                            lettersize=14)

# Collapse both empirical and analytical PSDs with the same scaling
empirical_collapsed = empirical_psd / g_fit[None, :]
fitted_psd_arr = np.vstack([fit["psd_fit"] for fit in fits])
analytical_collapsed = ((fitted_psd_arr.T - yi) / g_fit[None, :]) + yi

fig, ax = plot_empirical_vs_analytical_psds(fr_tag, psd_collapsed_org, analytical_collapsed,
                                            control_param=control_param, ylabel="Power (Index)", ylim=(0.8, 11),
                                            lettersize=14)

fits = fit_two_mode_psd_array_fslow_only(freqs=fr_tag, psd_arr=empirical_psd.T, fit_band=(0.01, 0.5),
                                         A=2.5, w=0.005, f_fast=1.4)
fitted_psd_arr = np.vstack([fit["psd_fit"] for fit in fits])

fig, ax = plot_empirical_vs_analytical_psds(fr_tag, empirical_psd, fitted_psd_arr.T,
                                            control_param=control_param, ylabel="Power (Index)", ylim=(0.8, 11),
                                            lettersize=14)

analytical_collapsed = ((fitted_psd_arr.T - yi) / g_fit[None, :]) + yi

fig, ax = plot_empirical_vs_analytical_psds(fr_tag, psd_collapsed_org, analytical_collapsed,
                                            control_param=control_param, ylabel="Power (Index)", ylim=(0.8, 11),
                                            lettersize=14)

fits = fit_two_mode_psd_array_vertex(freqs=fr_tag, psd_arr=empirical_psd.T, fit_band=(0.01, 0.5), w=0.005, f_fast=1.4,
                                     f_vertex=xi, p_vertex=yi, mode="joint", enforce_vertex_exact=True)
fitted_psd_arr = np.vstack([fit["psd_fit"] for fit in fits])

fig, ax = plot_empirical_vs_analytical_psds(fr_tag, empirical_psd, fitted_psd_arr.T,
                                            control_param=control_param, ylabel="Power (Index)", ylim=(0.8, 11),
                                            lettersize=14)

# plot_all_empirical_vs_fitted_psds(fr_tag, empirical_psd, fits)

analytical_collapsed = ((fitted_psd_arr.T - yi) / g_fit[None, :]) + yi

fig, ax = plot_empirical_vs_analytical_psds(fr_tag, psd_collapsed_org, analytical_collapsed,
                                            control_param=control_param, ylabel="Power (Index)", ylim=(0.8, 11),
                                            lettersize=14)

# Load results from simulations on top of the empirical and shape collapse analysis
sim_G = np.array([0.096, 0.097, 0.098, 0.0985, 0.099, 0.0995]) * 0.2 * 49.88
simulated_mean_s_0_096 = pd.read_csv(
    join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_096_multi_realizations.csv')).values
simulated_mean_s_0_097 = pd.read_csv(
    join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_097_multi_realizations.csv')).values
simulated_mean_s_0_098 = pd.read_csv(
    join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_098_multi_realizations.csv')).values
simulated_mean_s_0_0985 = pd.read_csv(
    join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_0985_multi_realizations.csv')).values
simulated_mean_s_0_099 = pd.read_csv(
    join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_099_multi_realizations.csv')).values
simulated_mean_s_0_0995 = pd.read_csv(
    join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_0995_multi_realizations.csv')).values

simulated_mean_psd_arr = np.array([np.mean(np.array(simulated_mean_s_0_096), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_097), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_098), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_0985), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_099), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_0995), axis=0),
                                   ])

######      Temporally overriding the previous simulation results    #######################
# sim_G = np.array([0.95, 0.96, 0.97, 0.98, 0.99])
# simulated_mean_s_0_95 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_tau_60_G_0.95_multi_realizations.csv')).values
# simulated_mean_s_0_96 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_tau_60_G_0.96_multi_realizations.csv')).values
# simulated_mean_s_0_97 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_tau_60_G_0.97_multi_realizations.csv')).values
# simulated_mean_s_0_98 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_tau_60_G_0.98_multi_realizations.csv')).values
# simulated_mean_s_0_99 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_tau_60_G_0.99_multi_realizations.csv')).values
#
# simulated_mean_psd_arr = np.array([np.mean(np.array(simulated_mean_s_0_95), axis=0),
#                                 np.mean(np.array(simulated_mean_s_0_96), axis=0),
#                                 np.mean(np.array(simulated_mean_s_0_97), axis=0),
#                                 np.mean(np.array(simulated_mean_s_0_98), axis=0),
#                                 np.mean(np.array(simulated_mean_s_0_99), axis=0),
#                                 ])

sim_G = np.array([0.96, 0.97, 0.98, 0.99])
simulated_mean_s_0_96 = pd.read_csv(
    join(path_result_str, 'simulated_N400_near_criticality_psd_tau_60_G_0.96_w_HRF_multi_realizations.csv')).values
simulated_mean_s_0_97 = pd.read_csv(
    join(path_result_str, 'simulated_N400_near_criticality_psd_tau_60_G_0.97_w_HRF_multi_realizations.csv')).values
simulated_mean_s_0_98 = pd.read_csv(
    join(path_result_str, 'simulated_N400_near_criticality_psd_tau_60_G_0.98_w_HRF_multi_realizations.csv')).values
simulated_mean_s_0_99 = pd.read_csv(
    join(path_result_str, 'simulated_N400_near_criticality_psd_tau_60_G_0.99_w_HRF_multi_realizations.csv')).values

# simulated_mean_s_0_96 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_G_0_96_w_HRF_multi_realizations.csv')).values
# simulated_mean_s_0_97 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_G_0_97_w_HRF_multi_realizations.csv')).values
# simulated_mean_s_0_98 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_G_0_98_w_HRF_multi_realizations.csv')).values
# simulated_mean_s_0_99 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_G_0_99_w_HRF_multi_realizations.csv')).values


simulated_mean_psd_arr = np.array([np.mean(np.array(simulated_mean_s_0_96), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_98), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_97), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_99), axis=0),
                                   ])

sim_mean_psd = simulated_mean_psd_arr
plt.figure()
plt.loglog(fr_tag, sim_mean_psd.T)


##########################################################################
def affine_normalize_sim_psds_to_empirical(
        freqs,
        empirical_psd_arr,
        sim_psd_arr,
        center=True,
        fit_to="mean",  # "mean" or "median"
        eps=1e-12,
):
    empirical_psd_arr = np.asarray(empirical_psd_arr, dtype=float)
    sim_psd_arr = np.asarray(sim_psd_arr, dtype=float)

    if empirical_psd_arr.ndim != 2 or sim_psd_arr.ndim != 2:
        raise ValueError("empirical_psd_arr and sim_psd_arr must both be 2D arrays.")

    if empirical_psd_arr.shape[1] != sim_psd_arr.shape[1]:
        raise ValueError("Empirical and simulated PSD arrays must share the same number of frequencies.")

    log_emp = np.log10(np.maximum(empirical_psd_arr, eps))
    log_sim = np.log10(np.maximum(sim_psd_arr, eps))

    if fit_to == "mean":
        emp_template = np.mean(log_emp, axis=0)
        sim_template = np.mean(log_sim, axis=0)
    elif fit_to == "median":
        emp_template = np.median(log_emp, axis=0)
        sim_template = np.median(log_sim, axis=0)
    else:
        raise ValueError('fit_to must be "mean" or "median".')

    x = sim_template.copy()
    y = emp_template.copy()

    if center:
        x = x - np.mean(x)
        y = y - np.mean(y)

    # Fit y ≈ a + b*x
    b, a = np.polyfit(x, y, 1)

    # If centered, recover a in original coordinates
    if center:
        a = np.mean(emp_template) - b * np.mean(sim_template)

    log_sim_norm = a + b * log_sim
    sim_psd_norm = 10 ** log_sim_norm

    return {
        "sim_psd_norm": sim_psd_norm,
        "a": float(a),
        "b": float(b),
        "empirical_template_log": emp_template,
        "sim_template_log": sim_template,
    }


def plot_empirical_and_affine_normalized_sim(freqs, empirical_psd_arr, sim_psd_norm, alpha_emp=0.12, alpha_sim=0.8):
    plt.figure(figsize=(7, 5))

    for i in range(empirical_psd_arr.shape[0]):
        plt.loglog(freqs, empirical_psd_arr[i], alpha=alpha_emp)

    # lines_sim = np.array(sim_psd_norm.shape[0])
    # for i in range(sim_psd_norm.shape[0]):
    #    lines_sim[i] = plt.loglog(freqs, sim_psd_norm[i], '--', alpha=alpha_sim, linewidth=8)
    lines_sim = plt.loglog(freqs, sim_psd_norm.T, linewidth=8)

    plt.xlabel("Frequency (Hz)", fontsize=lettersize)
    plt.ylabel("PSD (Index)", fontsize=lettersize)
    new_labels = ['0.96', '0.97', '0.98', '0.99']
    plt.legend(lines_sim, new_labels)
    # plt.title("Empirical PSDs and affine-normalized simulated PSDs")
    plt.xticks(fontsize=lettersize)
    plt.yticks(fontsize=lettersize)
    plt.tight_layout()


out = affine_normalize_sim_psds_to_empirical(
    freqs=fr_tag,
    empirical_psd_arr=empirical_psd.T,  # shape (180, 73)
    sim_psd_arr=simulated_mean_psd_arr,  # shape (5, 73) or (6, 73)
    center=True,
    fit_to="mean",
)

sim_psd_norm = out["sim_psd_norm"]
print("a =", out["a"])
print("b =", out["b"])

plot_empirical_and_affine_normalized_sim(fr_tag, empirical_psd.T, sim_psd_norm)
plt.show()

##########################################################################

# normalize scales by positive (only) area under the curve of the highest empirical and simulated spectra
min_ind = np.argmin(np.abs(fr_tag - xi))  # index in vertex

sim_mean_psd_tag = np.transpose(np.array([sim_mean_psd[i] - (sim_mean_psd[i][min_ind]) for i in range(len(sim_G))]))
plt.figure()
plt.plot(fr_tag, sim_mean_psd_tag)

auc_sim_psd_max = np.trapz(sim_mean_psd_tag[:min_ind, -1])

psd_max_ind = np.argmax(control_param)
auc_psd_max = np.trapz(psd[:, psd_max_ind])

rescaling_factor = auc_psd_max / auc_sim_psd_max

simulated_mean_psd_arr_rescaled = np.array(
    [(sim_mean_psd_tag[:, i] * rescaling_factor + np.mean(sim_mean_psd[i][min_ind]))
     for i in range(len(sim_G))])

plt.figure(figsize=(7, 5))
for i in range(empirical_psd.shape[0]):
    plt.loglog(fr_tag, psd_collapsed_org.T[i], alpha=0.2)
plt.loglog(fr_tag, simulated_mean_psd_arr_rescaled.T, linewidth=8)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Collapsed PSD")
plt.tight_layout()

sim_result = least_squares(residuals, initial_guess, args=(simulated_mean_psd_arr.T, sim_G))
sim_optimal_params = sim_result.x
sim_g_fit = g_model(sim_G, sim_optimal_params)

# Try no HRF setting for the simulation fitting
offset = 0.06  # the offset with HRF - 0.2
exponent = 0.5  # with HRF - 0.24
control_param = 1 - (offset / falff) ** (1 / exponent)
result = least_squares(residuals, initial_guess, args=(psd, control_param))
no_hrf_optimal_params = result.x
no_hrf_g_fit = g_model(control_param, no_hrf_optimal_params)
# end no HRF

simulated_mean_psd_arr_tag = np.array([simulated_mean_psd_arr[i] - (simulated_mean_psd_arr[i][min_ind])
                                       for i in range(len(sim_G))])
additional_exponent = 2.5  # for best fit use 2.5, to show base result (i.e. no fitting exponent) use 1
scale_fitter = np.array([(1 / val) ** additional_exponent for val in sim_g_fit])
simulated_mean_psd_arr_tag_scaled = np.array(
    [simulated_mean_psd_arr_tag[i] * scale_fitter[i] for i in range(len(sim_G))])

# no_hrf_scale_fitter = np.array([(1 / val) for val in no_hrf_g_fit])
# simulated_mean_psd_arr_tag_scaled = np.array([simulated_mean_psd_arr_tag[i]*no_hrf_scale_fitter[i] for i in range(len(sim_G))])

# scale_fitter = np.array([(1 / val)**1.5 for val in sim_g_fit])
# simulated_mean_psd_arr_tag_scaled = np.array([simulated_mean_psd_arr_tag[i]*scale_fitter[i] / sim_g_fit[i]
#                                               for i in range(len(sim_G))])
plt.figure(12, figsize=(8, 6))
plt.plot(fr_tag, simulated_mean_psd_arr_tag_scaled.T, linewidth=8)  # , color='darkred')

scale_factor = rescaling_factor  # 4000000 # currently an arbitrary factor, but should later normalize so that all signals would be divided by area under largest value
simulated_mean_psd_arr_rescaled = np.array(
    [(simulated_mean_psd_arr_tag_scaled[i] + np.mean(simulated_mean_psd_arr[:, min_ind])) * scale_factor
     for i in range(len(sim_G))])
plt.figure(13, figsize=(8, 6))
lines = plt.loglog(fr_tag, psd_collapsed_org, alpha=0.2)
for ln, c in zip(lines, colors):
    ln.set_color(c)
    ln.set_alpha(0.2)
lines_sim = plt.loglog(fr_tag, simulated_mean_psd_arr_rescaled[1:5, :].T * 5, linewidth=8)
plt.xlabel('Freq', fontsize=lettersize)
plt.ylabel('Power (Index)', fontsize=lettersize)
plt.ylim(0.8, 11)
# plt.legend(lines_sim, np.around(sim_G, 2))
new_labels = ['0.96', '0.97', '0.98', '0.99']
plt.legend(lines_sim, new_labels)
plt.xticks(fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.tight_layout()

# Back to scaled. But need to check yi again here... It seems off by a bit
plt.figure(14, figsize=(8, 6))
lines = plt.loglog(fr_tag, empirical_psd, alpha=0.2)
for ln, c in zip(lines, colors):
    ln.set_color(c)
    ln.set_alpha(0.2)
lines_sim = plt.loglog(fr_tag, (((simulated_mean_psd_arr_rescaled[1:5, :].T * 5 - yi) * sim_g_fit[1:5]) + yi),
                       linewidth=8)
plt.xlabel('Freq', fontsize=lettersize)
plt.ylabel('Power (Index)', fontsize=lettersize)
plt.ylim(0.8, 11)
plt.legend(lines_sim, np.round(sim_G, 2))
plt.xticks(fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.tight_layout()

# normalize the empirical data for later comparison with simulated data
max_norm_empirical_psd = empirical_psd / empirical_psd.max(axis=1, keepdims=True)

fits = fit_two_mode_psd_array_vertex(freqs=fr_tag, psd_arr=empirical_psd.T, fit_band=(0.01, 0.5), w=0.005, f_fast=1.4,
                                     f_vertex=xi, p_vertex=yi, mode="joint", enforce_vertex_exact=True)

###########################################################################

from scipy.optimize import least_squares


def vertex_lorentzian_psd(freqs, G, f_vertex=0.12, tau=0.020, A_fast=0.2, scale=1.0):
    """
    Simple vertex-calibrated analytical PSD.

    epsilon(G) = 1 - G

    P(f;G) =
        scale * [
            (f_vertex^2 + epsilon^2) / (f^2 + epsilon^2)
            + A_fast / (f^2 + f_fast^2)
        ]

    where

        f_fast = 1 / (2*pi*tau)
    """
    freqs = np.asarray(freqs, dtype=float).ravel()
    G = np.asarray(G, dtype=float).ravel()

    epsilon = 1.0 - G
    f_fast = 1.0 / (2.0 * np.pi * tau)

    slow = (f_vertex ** 2 + epsilon[None, :] ** 2) / (
            freqs[:, None] ** 2 + epsilon[None, :] ** 2
    )

    fast = A_fast / (freqs[:, None] ** 2 + f_fast ** 2)

    return scale * (slow + fast)


def vertex_lorentzian_psd_fixed_geometry(
        freqs,
        G,
        A_slow=1.0,
        A_fast=0.2,
        f_vertex=0.12,
        tau=0.020,
):
    """
    Vertex-calibrated analytical PSD with fixed vertex and fixed tau.

    P(f;G) =
        A_slow * (f_vertex^2 + epsilon^2) / (f^2 + epsilon^2)
        + A_fast / (f^2 + f_fast^2)

    where:
        epsilon = 1 - G
        f_fast = 1 / (2*pi*tau)
    """

    freqs = np.asarray(freqs, dtype=float).ravel()
    G = np.asarray(G, dtype=float).ravel()

    epsilon = 1.0 - G
    f_fast = 1.0 / (2.0 * np.pi * tau)

    slow = (f_vertex ** 2 + epsilon[None, :] ** 2) / (
            freqs[:, None] ** 2 + epsilon[None, :] ** 2
    )

    fast = 1.0 / (freqs[:, None] ** 2 + f_fast ** 2)

    return A_slow * slow + A_fast * fast


def fit_vertex_lorentzian_amplitudes_only(
        freqs,
        empirical_psd,
        G,
        f_vertex=0.12,
        tau=0.020,
        fit_band=(0.01, 0.5),
        p0=(1.0, 0.2),
        bounds=((1e-6, 0.0), (1e3, 1e3)),
        robust_loss="soft_l1",
        f_scale=0.1,
):
    """
    Fit only A_slow and A_fast, while keeping the vertex frequency,
    tau, and epsilon(G)=1-G fixed.
    """

    freqs = np.asarray(freqs, dtype=float).ravel()
    emp = np.asarray(empirical_psd, dtype=float)
    G = np.asarray(G, dtype=float).ravel()
    eps = 1e-12

    # Ensure empirical PSD is (n_freq, n_roi)
    if emp.shape[0] != freqs.size and emp.shape[1] == freqs.size:
        emp = emp.T

    if emp.shape[0] != freqs.size:
        raise ValueError("empirical_psd must have one dimension equal to len(freqs).")

    if emp.shape[1] != G.size:
        raise ValueError("Number of empirical PSD curves must match len(G).")

    if fit_band is not None:
        mask = (freqs >= fit_band[0]) & (freqs <= fit_band[1])
    else:
        mask = np.ones_like(freqs, dtype=bool)

    log_emp = np.log10(np.maximum(emp[mask, :], eps))

    def residuals(params):
        A_slow, A_fast = params

        pred = vertex_lorentzian_psd_fixed_geometry(
            freqs=freqs[mask],
            G=G,
            A_slow=A_slow,
            A_fast=A_fast,
            f_vertex=f_vertex,
            tau=tau,
        )

        return (
                np.log10(np.maximum(pred, eps)) - log_emp
        ).ravel()

    res = least_squares(
        residuals,
        x0=np.asarray(p0, dtype=float),
        bounds=bounds,
        loss=robust_loss,
        f_scale=f_scale,
    )

    A_slow, A_fast = res.x

    pred = vertex_lorentzian_psd_fixed_geometry(
        freqs=freqs,
        G=G,
        A_slow=A_slow,
        A_fast=A_fast,
        f_vertex=f_vertex,
        tau=tau,
    )

    return {
        "A_slow": A_slow,
        "A_fast": A_fast,
        "f_vertex": f_vertex,
        "tau": tau,
        "f_fast": 1.0 / (2.0 * np.pi * tau),
        "analytical_psd": pred,
        "success": res.success,
        "cost": res.cost,
        "optimizer_result": res,
    }


def plot_empirical_vs_fixed_vertex_lorentzian_psds(
        freqs,
        empirical_psd,
        G,
        f_vertex=0.12,
        tau=0.020,
        fit_band=(0.01, 0.5),
        p0=(1.0, 0.2),
        ylim=None,
        lettersize=14,
        empirical_alpha=0.30,
        analytical_alpha=0.85,
        figsize=(8.5, 6.5),
):
    """
    Fit only A_slow and A_fast, then plot all empirical PSDs and all
    analytical PSDs.

    Solid lines  = empirical PSDs
    Dashed lines = analytical PSDs
    """

    freqs = np.asarray(freqs, dtype=float).ravel()
    emp = np.asarray(empirical_psd, dtype=float)
    G = np.asarray(G, dtype=float).ravel()

    # Ensure empirical PSD is (n_freq, n_roi)
    if emp.shape[0] != freqs.size and emp.shape[1] == freqs.size:
        emp = emp.T

    fit = fit_vertex_lorentzian_amplitudes_only(
        freqs=freqs,
        empirical_psd=emp,
        G=G,
        f_vertex=f_vertex,
        tau=tau,
        fit_band=fit_band,
        p0=p0,
    )

    pred = fit["analytical_psd"]

    # Sort by control parameter for smooth color ordering
    order = np.argsort(G)
    Gs = G[order]
    emp_s = emp[:, order]
    pred_s = pred[:, order]

    colors = plt.cm.viridis(np.linspace(0, 1, emp_s.shape[1]))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot all empirical curves
    for i in range(emp_s.shape[1]):
        ax.loglog(
            freqs,
            emp_s[:, i],
            color=colors[i],
            alpha=empirical_alpha,
            linewidth=1.1,
        )

    # Plot all analytical curves
    for i in range(pred_s.shape[1]):
        ax.loglog(
            freqs,
            pred_s[:, i],
            "--",
            color=colors[i],
            alpha=analytical_alpha,
            linewidth=1.4,
        )

    ax.axvline(f_vertex, linestyle=":", linewidth=1.4, color="k", alpha=0.7)

    ax.set_xlabel("Frequency (Hz)", fontsize=lettersize)
    ax.set_ylabel("PSD", fontsize=lettersize)
    ax.tick_params(labelsize=lettersize)

    ax.set_title(
        "Empirical vs fixed-vertex analytical PSDs\n"
        "solid = empirical, dashed = analytical",
        fontsize=lettersize,
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.show()

    print(
        "Fitted parameters:\n"
        f"  A_slow   = {fit['A_slow']:.4g}\n"
        f"  A_fast   = {fit['A_fast']:.4g}\n"
        f"  f_vertex = {fit['f_vertex']:.4f} Hz\n"
        f"  tau      = {fit['tau']:.4f} s\n"
        f"  f_fast   = {fit['f_fast']:.4f} Hz"
    )

    return fig, ax, fit


fig, ax, fit = plot_empirical_vs_fixed_vertex_lorentzian_psds(
    freqs=fr_tag,
    empirical_psd=empirical_psd,
    G=control_param,
    f_vertex=0.12,
    tau=0.020,  # 20 ms, in seconds
    fit_band=(0.01, 0.5),
    p0=(1.0, 0.2),
    ylim=(0.8, 11),
    lettersize=14,
)


##########################################################


def plot_empirical_vs_vertex_lorentzian_psds(
        freqs,
        empirical_psd,
        G,
        roi_names=None,
        n_show=10,
        f_vertex=0.12,
        tau=0.020,
        A_fast=0.2,
        fit_global_scale=True,
        normalize_at_vertex=False,
        figsize=(8.5, 6.5),
):
    """
    Overlay empirical ROI PSDs with a simplified vertex-calibrated analytical Lorentzian model.

    Analytical model
    ----------------
    epsilon(G) = 1 - G

    P(f;G) =
        (f_vertex^2 + epsilon^2) / (f^2 + epsilon^2)
        + A_fast / (f^2 + f_fast^2)

    where

        f_fast = 1 / (2*pi*tau)
    """

    freqs = np.asarray(freqs).ravel()
    emp = np.asarray(empirical_psd, dtype=float)
    G = np.asarray(G, dtype=float).ravel()

    # Ensure empirical PSD has shape (n_freq, n_roi)
    if emp.shape[0] != freqs.size and emp.shape[1] == freqs.size:
        emp = emp.T

    if emp.shape[0] != freqs.size:
        raise ValueError("empirical_psd must have one dimension equal to len(freqs).")

    if emp.shape[1] != G.size:
        raise ValueError("Number of empirical PSD curves must match len(G).")

    # Sort ROIs by G, from subcritical to near-critical
    order = np.argsort(G)
    Gs = G[order]
    emp = emp[:, order]

    if roi_names is not None:
        roi_names = np.asarray(roi_names)[order]

    # Representative ROIs across the G range
    n_show = min(n_show, G.size)
    show_idx = np.linspace(0, G.size - 1, n_show, dtype=int)

    # Analytical spectra
    f_fast = 1.0 / (2.0 * np.pi * tau)

    pred = np.zeros_like(emp)

    for j, Gj in enumerate(Gs):
        epsilon = 1.0 - Gj

        slow_mode = (f_vertex ** 2 + epsilon ** 2) / (freqs ** 2 + epsilon ** 2)
        fast_mode = A_fast / (freqs ** 2 + f_fast ** 2)

        pred[:, j] = slow_mode + fast_mode

    # Optional normalization of both empirical and analytical spectra at vertex
    if normalize_at_vertex:
        iv = np.argmin(np.abs(freqs - f_vertex))

        emp = emp / emp[iv, :][None, :]
        pred = pred / pred[iv, :][None, :]

    # Otherwise fit one global scale from analytical to empirical
    elif fit_global_scale:
        tiny = 1e-12
        log_ratio = np.log(emp + tiny) - np.log(pred + tiny)
        global_scale = np.exp(np.nanmean(log_ratio))
        pred *= global_scale
    else:
        global_scale = 1.0

    # Plot
    plt.figure(figsize=figsize)

    for j in show_idx:
        if roi_names is None:
            label = f"G={Gs[j]:.3f}"
        else:
            label = f"{roi_names[j]} (G={Gs[j]:.3f})"

        line, = plt.loglog(freqs, emp[:, j], linewidth=2.0, alpha=0.85, label=label)

        plt.loglog(freqs, pred[:, j], "--", linewidth=2.0, alpha=0.95, color=line.get_color())

    plt.axvline(f_vertex, linestyle=":", linewidth=1.4)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(
        "Empirical vs vertex-calibrated analytical PSDs\n"
        "solid = empirical, dashed = analytical"
    )

    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.show()

    return {
        "order": order,
        "G_sorted": Gs,
        "empirical_psd_sorted": emp,
        "analytical_psd_sorted": pred,
        "show_idx": show_idx,
        "f_vertex": f_vertex,
        "f_fast": f_fast,
        "global_scale": None if normalize_at_vertex else global_scale,
    }


out = plot_empirical_vs_vertex_lorentzian_psds(
    freqs=fr_tag,
    empirical_psd=empirical_psd.T,
    G=control_param,
    n_show=10,
    f_vertex=0.12,
    tau=0.020,
    A_fast=0.2,
    fit_global_scale=True,
    normalize_at_vertex=False,
)


def power_spectrum(f, G, f_vertex=0.1, tau=0.020, A_fast=0.2):
    """
    Two-mode analytical spectrum.

    G        : proximity to criticality, with G = 1 at criticality
    epsilon  : distance from criticality = 1 - G
    f_vertex : frequency where the slow-mode spectra are calibrated to intersect
    tau      : fast timescale in seconds
    A_fast   : amplitude of the fast mode
    """
    epsilon = 1 - G
    f_fast = 1 / (2 * np.pi * tau)

    sigma_shared_sq = f_vertex ** 2 + epsilon ** 2

    slow_mode = sigma_shared_sq / (f ** 2 + epsilon ** 2)
    fast_mode = A_fast / (f ** 2 + f_fast ** 2)

    return slow_mode + fast_mode


# Frequencies
f = np.logspace(np.log10(0.01), np.log10(0.5), 1000)

# Parameters
G_values = [0.70, 0.80, 0.90, 0.95, 0.99]
f_vertex = 0.1  # Hz
tau = 0.020  # 20 ms
A_fast = 0.2

# Plot
plt.figure(figsize=(7, 5))

for G in G_values:
    P = power_spectrum(
        f,
        G,
        f_vertex=f_vertex,
        tau=tau,
        A_fast=A_fast
    )
    plt.loglog(f, P, linewidth=2, label=f"G = {G:.2f}")

# Mark common vertex
f_fast = 1 / (2 * np.pi * tau)
vertex_power = 1 + A_fast / (f_vertex ** 2 + f_fast ** 2)

plt.axvline(f_vertex, linestyle="--", linewidth=1)
plt.scatter(f_vertex, vertex_power, s=50, zorder=5)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (a.u.)")
plt.title("Analytical power spectra with input-variance calibration")
plt.legend(title="Proximity to criticality")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlim(0.01, 0.5)

plt.tight_layout()
plt.show()