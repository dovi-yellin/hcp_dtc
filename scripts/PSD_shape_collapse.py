from os.path import join
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
path_str = 'data'
path_result_str = 'results'
subject_filename = ''
lettersize = 22

hierarchy_filename = 'Glasser_2016_Table_my_Hierarchy.csv'
df_glasser = pd.read_csv(join(path_str, hierarchy_filename))
hierarchy = df_glasser['Hierarchy'].values

#hierarchy_filename = 'Glasser_2016_Table_my_Hierarchy_2.csv'
#df_glasser = pd.read_csv(join(path_str, hierarchy_filename))
#regions = df_glasser['region'][0:180].values # areas = df_glasser['cortex'][0:180].values

str_all_subjects_all_rois_mean_psd = 'all_subjects_all_rois_mean_psd_v2'
with open(join(path_str, str_all_subjects_all_rois_mean_psd + '.pickle'), 'rb') as handle:
    data_all_subs_all_rois_mean_psd = pickle.load(handle)

empirical_psd = np.array(data_all_subs_all_rois_mean_psd).T

# define frequency list in lieu to 300 s spectral analysis methods in use
fr = np.array([0.0, 0.0033333333333333335, 0.006666666666666667, 0.01, 0.013333333333333334, 0.016666666666666666, 0.02, 0.023333333333333334, 0.02666666666666667, 0.030000000000000002, 0.03333333333333333, 0.03666666666666667, 0.04, 0.043333333333333335, 0.04666666666666667, 0.05, 0.05333333333333334, 0.05666666666666667, 0.060000000000000005, 0.06333333333333334, 0.06666666666666667, 0.07, 0.07333333333333333, 0.07666666666666667, 0.08, 0.08333333333333334, 0.08666666666666667, 0.09000000000000001, 0.09333333333333334, 0.09666666666666668, 0.1, 0.10333333333333335, 0.10666666666666667, 0.11, 0.11333333333333334, 0.11666666666666667, 0.12000000000000001, 0.12333333333333334, 0.12666666666666668, 0.13, 0.13333333333333333, 0.1366666666666667, 0.14, 0.14333333333333334, 0.14666666666666667, 0.15000000000000002, 0.15333333333333335, 0.15666666666666668, 0.16, 0.16333333333333333, 0.16666666666666669, 0.17, 0.17333333333333334, 0.17666666666666667, 0.18000000000000002, 0.18333333333333335, 0.18666666666666668, 0.19, 0.19333333333333336, 0.19666666666666668, 0.2, 0.20333333333333334, 0.2066666666666667, 0.21000000000000002, 0.21333333333333335, 0.21666666666666667, 0.22, 0.22333333333333336, 0.22666666666666668, 0.23, 0.23333333333333334, 0.2366666666666667, 0.24000000000000002, 0.24333333333333335, 0.24666666666666667, 0.25, 0.25333333333333335, 0.2566666666666667, 0.26, 0.26333333333333336, 0.26666666666666666, 0.27, 0.2733333333333334, 0.27666666666666667, 0.28, 0.2833333333333333, 0.2866666666666667, 0.29000000000000004, 0.29333333333333333, 0.2966666666666667, 0.30000000000000004, 0.30333333333333334, 0.3066666666666667, 0.31, 0.31333333333333335, 0.3166666666666667, 0.32, 0.32333333333333336, 0.32666666666666666, 0.33, 0.33333333333333337, 0.33666666666666667, 0.34, 0.3433333333333334, 0.3466666666666667, 0.35000000000000003, 0.35333333333333333, 0.3566666666666667, 0.36000000000000004, 0.36333333333333334, 0.3666666666666667, 0.37000000000000005, 0.37333333333333335, 0.3766666666666667, 0.38, 0.38333333333333336, 0.3866666666666667, 0.39, 0.39333333333333337, 0.39666666666666667, 0.4, 0.4033333333333334, 0.4066666666666667, 0.41000000000000003, 0.4133333333333334, 0.4166666666666667, 0.42000000000000004, 0.42333333333333334, 0.4266666666666667, 0.43000000000000005, 0.43333333333333335, 0.4366666666666667, 0.44, 0.44333333333333336, 0.4466666666666667, 0.45, 0.45333333333333337, 0.4566666666666667, 0.46, 0.4633333333333334, 0.4666666666666667, 0.47000000000000003, 0.4733333333333334, 0.4766666666666667, 0.48000000000000004, 0.48333333333333334, 0.4866666666666667, 0.49000000000000005, 0.49333333333333335, 0.4966666666666667, 0.5])
fr_tag = fr[(fr >= 0.01) & (fr <= 0.25)]

lettersize = 20
fig1 = plt.figure(1)
plt.loglog(fr_tag,empirical_psd)
plt.xlabel('Freq')
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
im = plt.imshow(corrcoef_matrix,cmap='jet')
plt.yticks([0, 50, 100, 150]) #fontsize=font_size)
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
plt.loglog(x,y)   #    plt.plot(x, y[:, i], label=f"Curve {i + 1}")
plt.loglog(x,y_mean, linewidth=6, color='red')  #plt.plot(x,y_mean, linewidth=6, color='red')
plt.loglog(xi, yi, 'ko', markersize=8, label="Intersection Approx.")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

x_tag = x - xi
y_tag = y - yi

plt.figure(3, figsize=(8, 6))
plt.plot(x_tag,y_tag)
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
falff = mean_fALFF_arr[:,0:180].mean(axis=0)

offset = 0.2 # 0.2 # without HRF - 0.06 # the offset with HRF - 0.2
exponent = 0.24 # 0.24 # without HRF 0.5 # with HRF - 0.24
control_param = 1 - (offset/falff)**(1/exponent)

# Example parametric model for g(falff)
def g_model(control_param, params):
    a, b = params
    #return a + b * falff
    return b*(control_param**a)


def residuals(params, psd, control_param, threshold=1.0):
    if threshold < 1.0:
        control_param = control_param[control_param > threshold]
        psd = psd.T[control_param > threshold]
        
    # Compute the model g for each value in falff.
    g_values = g_model(control_param, params)
    # Calculate the mean of psd over the frequency range.
    mean_psd = np.mean(psd, axis=1)
    # Compute the residual per each ROI PSD: note that we divide by mean_psd to normalize.
    r = np.mean(((psd / g_values).T - mean_psd)/mean_psd, axis=1)
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
plt.plot(fr_tag, psd_collapsed) # plt.loglog(fr_tag, psd_collapsed[:,126]*2.3, linewidth=3, color='yellow') # plt.loglog(fr_tag, psd_collapsed[:,12]*0.6, linewidth=3, color='green')
plt.plot(fr_tag, psd_mean_scaled, linewidth=6, color='red')
plt.xlabel('fr')
plt.ylabel('Scaled psd')
#plt.legend()
plt.title('Scaled psd Relative to Mean(psd)')
plt.tight_layout()
plt.show()

from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

norm = Normalize(vmin=np.min(control_param), vmax=np.max(control_param))
cmap = get_cmap('jet')
colors = [cmap(norm(cp)) for cp in control_param]

plt.figure(5, figsize=(8, 6))
lines = plt.loglog(fr_tag,empirical_psd) #, color=np.squeeze(control_param), cmap='jet')
# assign colors (a tiny loop; still O(K))
for ln, c in zip(lines, colors):
    ln.set_color(c)
plt.xlabel('Freq', fontsize=lettersize)
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
plt.xlabel('Freq', fontsize=lettersize)
plt.ylabel('Power (Index)', fontsize=lettersize)
plt.ylim(0.8, 11)
plt.xticks(fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.tight_layout()
plt.show()
#plt.savefig('xyz')

# Load results from simulations on top of the empirical and shape collapse analysis
sim_G = np.array([0.096, 0.097, 0.098, 0.0985, 0.099, 0.0995]) * 0.2 * 49.88
simulated_mean_s_0_096 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_096_multi_realizations.csv')).values
simulated_mean_s_0_097 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_097_multi_realizations.csv')).values
simulated_mean_s_0_098 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_098_multi_realizations.csv')).values
simulated_mean_s_0_0985 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_0985_multi_realizations.csv')).values
simulated_mean_s_0_099 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_099_multi_realizations.csv')).values
simulated_mean_s_0_0995 = pd.read_csv(join(path_result_str, 'simulated_N400_near_criticality_psd_gamma_0_0995_multi_realizations.csv')).values

simulated_mean_psd_arr = np.array([np.mean(np.array(simulated_mean_s_0_096), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_097), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_098), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_0985), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_099), axis=0),
                                   np.mean(np.array(simulated_mean_s_0_0995), axis=0),
                                   ])

sim_mean_psd = simulated_mean_psd_arr
#plt.figure()
#plt.loglog(fr_tag, sim_mean_psd.T)

# compute positive (only) area under the curve of the highest empirical and simulated spectra
min_ind = np.argmin(np.abs(fr_tag - xi)) # index in vertex

sim_mean_psd_tag = np.transpose(np.array([sim_mean_psd[i] - (sim_mean_psd[i][min_ind]) for i in range(len(sim_G))]))
#plt.figure()
#plt.plot(fr_tag, sim_mean_psd_tag)

auc_sim_psd_max = np.trapz(sim_mean_psd_tag[:min_ind,-1])

psd_max_ind = np.argmax(control_param)
auc_psd_max = np.trapz(psd[:,psd_max_ind])

rescaling_factor = auc_psd_max/auc_sim_psd_max

sim_result = least_squares(residuals, initial_guess, args=(simulated_mean_psd_arr.T, sim_G))
sim_optimal_params = sim_result.x
sim_g_fit = g_model(sim_G, sim_optimal_params)

simulated_mean_psd_arr_tag = np.array([simulated_mean_psd_arr[i] - (simulated_mean_psd_arr[i][min_ind])
                                       for i in range(len(sim_G))])

scale_fitter = np.array([(1 / val)**1.5 for val in sim_g_fit])
simulated_mean_psd_arr_tag_scaled = np.array([simulated_mean_psd_arr_tag[i]*scale_fitter[i] / sim_g_fit[i]
                                              for i in range(len(sim_G))])
plt.figure(12, figsize=(8, 6))
plt.plot(fr_tag,simulated_mean_psd_arr_tag_scaled.T, linewidth=8)  #, color='darkred')

scale_factor = rescaling_factor # 4000000 # currently an arbitrary factor, but should later normalize so that all signals would be divided by area under largest value
simulated_mean_psd_arr_rescaled = np.array([(simulated_mean_psd_arr_tag_scaled[i]+np.mean(simulated_mean_psd_arr[:,min_ind]))*scale_factor
                                              for i in range(len(sim_G))])
plt.figure(13, figsize=(8, 6))
lines = plt.loglog(fr_tag, psd_collapsed_org, alpha=0.2)
for ln, c in zip(lines, colors):
    ln.set_color(c)
    ln.set_alpha(0.2)
lines_sim = plt.loglog(fr_tag,simulated_mean_psd_arr_rescaled[1:5,:].T*5, linewidth=8)
plt.xlabel('Freq', fontsize=lettersize)
plt.ylabel('Power (Index)', fontsize=lettersize)
plt.ylim(0.8, 11)
plt.legend(lines_sim, np.around(sim_G, 2))
plt.xticks(fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.tight_layout()

# Back to scaled. But need to check yi again here... It seems off by a bit
plt.figure(14, figsize=(8, 6))
lines = plt.loglog(fr_tag, empirical_psd, alpha=0.2)
for ln, c in zip(lines, colors):
    ln.set_color(c)
    ln.set_alpha(0.2)
lines_sim = plt.loglog(fr_tag,(((simulated_mean_psd_arr_rescaled[1:5,:].T*5-yi)*sim_g_fit[1:5])+yi), linewidth=8)
plt.xlabel('Freq', fontsize=lettersize)
plt.ylabel('Power (Index)', fontsize=lettersize)
plt.ylim(0.8, 11)
plt.legend(lines_sim, np.round(sim_G, 2))
plt.xticks(fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.tight_layout()

# normalize the empirical data for later comparison with simulated data
max_norm_empirical_psd = empirical_psd / empirical_psd.max(axis=1, keepdims=True)
