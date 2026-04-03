from os.path import join
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import pearsonr
import pandas as pd
from itertools import combinations

###### Move the following plotting method to separate script method ######
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
#from matplotlib.colors import ListedColormap

from nilearn import plotting
import hcp_utils as hcp

import numpy.ma as ma
std_fontsize = 14

def add_thresholded_colorbar(fig, cmap_obj, full_vmin, full_vmax, vmin, vmax, cax_position=[0.88, 0.20, 0.03, 0.60]):

    # Ensure figure is wide enough
    w, h = fig.get_size_inches()
    if w < 8:
        fig.set_size_inches(10, h, forward=True)

    # Map revised limits into original colormap space
    full_norm = mpl.colors.Normalize(vmin=full_vmin, vmax=full_vmax)
    fmin = full_norm(vmin)
    fmax = full_norm(vmax)

    truncated = cmap_obj(np.linspace(fmin, fmax, 256))
    truncated_cmap = mcolors.ListedColormap(truncated)

    norm_cb = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = mpl.cm.ScalarMappable(norm=norm_cb, cmap=truncated_cmap)
    mappable.set_array([])

    cax = fig.add_axes(cax_position)
    cbar = fig.colorbar(mappable, cax=cax)

    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.ax.tick_params(labelsize=14)

    cbar.set_label("fALFF", fontsize=16)

    return cbar

#def plot_left_parcellation(meshL, parcellation, analysis_results, vmin=-0.01, view_mode='lateral', cmap='jet', b_save=False,
#                           tsnr_arr=None, tsnr_cutoff=None, gray_color=(0.7, 0.7, 0.7, 1.0), add_colorbar=False):
def plot_parcellation(mesh, parcellation, analysis_results, hemi='L', vmin=-0.01, view_mode='lateral', cmap='jet',
        b_save=False, tsnr_arr=None, tsnr_cutoff=None, gray_color=(0.7, 0.7, 0.7, 1.0), add_colorbar=False, vmax=None, save_path=None):
    """
    View the given parcellation on an the left hemisphere brain surface mesh.
    """

    # By default, assume left hemisphere unless specifically addressing right hemisphere
    hemi_key = 'L'
    cortex_map = hcp.left_cortex_data(parcellation.map_all)
    hemi_name = 'Left'

    if hemi in ('r', 'right', 'rh'):
        hemi_key = 'R'
        cortex_map = hcp.right_cortex_data(parcellation.map_all)
        hemi_name = 'Right'

    # for some parcellations the numerical ids need not be consecutive
    # cortex_map = hcp.left_cortex_data(parcellation.map_all)
    cortex_map = np.asarray(cortex_map)
    ids = np.unique(cortex_map)
    ids = ids[ids != 0]  # Exclude background

    normalized_cortex_map = np.zeros_like(cortex_map)

    # Check if analysis_results matches the number of ROIs
    if len(analysis_results) != len(ids): # correct for the fact that ids include unassigned vertices as 0
        raise ValueError("The length of analysis_results must match the number of unique ROI IDs.")

    # --- NEW: validate threshold vector if provided ---
    use_threshold = (tsnr_arr is not None) and (tsnr_cutoff is not None)
    if use_threshold:
        threshold_values = np.asarray(tsnr_arr, dtype=float).ravel()
        if threshold_values.size != ids.size:
            raise ValueError(f"The length of threshold_values ({threshold_values.size}) must match the number of unique ROI IDs excluding 0 ({ids.size}).")
        roi_ok = threshold_values >= float(tsnr_cutoff)
    else:
        roi_ok = np.ones(ids.size, dtype=bool)

    valid = normalized_cortex_map[np.isfinite(normalized_cortex_map)]
    if valid.size == 0:
        raise ValueError("All ROIs were masked by the threshold; nothing left to color.")

    vmin_eff = np.min(analysis_results[roi_ok]) if vmin is None else vmin
    vmax_eff = np.max(analysis_results[roi_ok])  # or allow a user-provided vmax similarly

    # Build vertex-wise stat map
    normalized_cortex_map = np.full_like(cortex_map, np.nan, dtype=float)  # NaNs -> "bad" color
    for i, roi_id in enumerate(ids):
        ind = cortex_map == roi_id
        if roi_ok[i]:
            normalized_cortex_map[ind] = analysis_results[i]
        # else: leave as NaN -> will render as gray

    # Make a cmap copy and set NaNs ("bad") to gray
    base_cmap = cm.get_cmap(cmap)
    cmap_obj = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, 256)))  # Create an independent copy manually
    cmap_obj.set_bad(gray_color)  # Set NaNs ("bad" values) to gray

    # Use nilearn's plot_surf for a static plot
    if vmin is None:
        vmin = 0.01
    display = plotting.plot_surf_stat_map(
        mesh,
        stat_map=normalized_cortex_map,
        cmap=cmap_obj,
        bg_map=None,
        view=view_mode,
        colorbar=False,
        vmin=vmin,
        vmax=vmax_eff,
        title=f'{hemi_name} Hemisphere Parcellation',
    )

    if add_colorbar:
        fig = plt.gcf()
        add_thresholded_colorbar(fig, cmap_obj, full_vmin=vmin, full_vmax=vmax_eff, vmin=vmin_eff, vmax=vmax_eff)
        plt.show()

    if b_save:
        if save_path is None:
            # Keep your original naming but include hemisphere + threshold if present
            tsnr_tag = f"_tsnr_{int(tsnr_cutoff)}" if use_threshold else ""
            save_path = f'fALFF_cortical_surface_{hemi_key}_{view_mode}{tsnr_tag}.eps'
        plt.savefig(save_path, format='eps', bbox_inches='tight')

path_str = 'data'
filenamestr = 'criticality_markers_subject_n'
#filenamestr = 'tsnr_clean_criticality_markers_subject_n'
#filenamestr = 'HS_cov_corrected_criticality_markers_subject_n'

tsnr_filenamestr = 'tsnr_metrics_subject_n'

hierarchy_filename = 'Glasser_2016_Table_my_Hierarchy_2.csv'
df_glasser = pd.read_csv(join(path_str, hierarchy_filename))
hierarchy = df_glasser['Hierarchy'].values

def get_subject_criticality_markers(sub_ind, roi_ind = -1):
    with open(join(path_str, filenamestr + str(sub_ind) + '.pickle'), 'rb') as handle:
        critical_markers = pickle.load(handle)
    return critical_markers

def get_subject_tsnr_markers(sub_ind, roi_ind = -1):
    with open(join(path_str, tsnr_filenamestr + str(sub_ind) + '.pickle'), 'rb') as handle:
        tsnr_markers = pickle.load(handle)
    return tsnr_markers

num_subjects = 40
estimated_g_arr = []
mean_fALFF_arr = []
fcd_peaks_arr = []
tsnr_mean_arr = []
for i in range(num_subjects):
    critical_markers = get_subject_criticality_markers(i+1)
    estimated_g_arr.append(critical_markers['estimated_g'])
    mean_fALFF_arr.append(critical_markers['mean_falff'])
    fcd_peaks_arr.append(critical_markers['fcd_peaks'])

    tsnr_markers = get_subject_tsnr_markers(i + 1)
    tsnr_mean_fALFF_arr = tsnr_markers['mean_falff']
    tsnr_mean_arr.append(tsnr_markers['tsnr_mean'])

estimated_g_arr = np.array(estimated_g_arr)

mean_fALFF_arr = np.array(mean_fALFF_arr)

#mean_fALFF_arr_nan = ma.masked_invalid(mean_fALFF_arr)
#corr_mean_fALFF = np.triu(np.corrcoef(mean_fALFF_arr_nan[~mean_fALFF_arr_nan.mask]), k=1) # exxtract the upper diagonal of the correlation matrix
corr_mean_fALFF = np.triu(np.corrcoef(mean_fALFF_arr), k=1)
corr_mean_fALFF = corr_mean_fALFF[corr_mean_fALFF > 0]

mean_subjects_fALFF_arr = mean_fALFF_arr[:,0:180].mean(axis=0)
idxs = np.flip(np.argsort(mean_subjects_fALFF_arr))
x = np.arange(len(mean_subjects_fALFF_arr)) # /180
mean_subjects_fALFF_arr_sorted = mean_subjects_fALFF_arr[idxs]
p = np.polyfit(x, mean_subjects_fALFF_arr_sorted, 1)
y_fit = p[0] * x + p[1]
plt.figure(55)
plt.plot(x, mean_subjects_fALFF_arr[idxs], 'bo', label='Empirical fALFF')
plt.plot(x, y_fit, 'g-', linewidth=4, label='Fit')
plt.xlabel('Sorted ROI index', fontsize=18) # ('Normalized ROI index', fontsize=18)
plt.ylabel('Mean fALFF (rank-ordered)', fontsize=18)
plt.legend()

res = stats.linregress(x, mean_subjects_fALFF_arr[idxs])
print("Slope:", res.slope)
print("Intercept:", res.intercept)
print("R²:", res.rvalue**2)
print("p-value:", res.pvalue)
print("Std error:", res.stderr)

fcd_peaks_arr = np.array(fcd_peaks_arr)
corr_fcd_peaks = np.triu(np.corrcoef(fcd_peaks_arr), k=1) # exxtract the upper diagonal of the correlation matrix
corr_fcd_peaks = corr_fcd_peaks[corr_fcd_peaks > 0]

mean_fcd_peak_arr = fcd_peaks_arr.mean(axis=0)[0:180]
mean_fcd_peak_arr[mean_fcd_peak_arr <= 0] = 0.01

def nanmean_threshold_axis(x, axis=0, max_nan_frac=0.05):
    nan_frac = np.isnan(x).mean(axis=axis)
    mean_vals = np.nanmean(x, axis=axis)

    mean_vals[nan_frac > max_nan_frac] = np.nan
    return mean_vals

nanmean_fALFF_arr_per_region = nanmean_threshold_axis(mean_fALFF_arr, axis=0, max_nan_frac=0.05)
tsnr_mean_arr_per_region = np.mean(tsnr_mean_arr, axis=0)

def plot_all_map_perspectives(metric_arr, tsnr_arr, tsnr_cutoff, hemisphere='left', vmin=None):
    hemisphere_mesh = hcp.mesh.inflated_left if hemisphere == 'left' else hcp.mesh.inflated_right
    hemi_ids = np.arange(0, 180) if hemisphere == 'left' else np.arange(180, 360)
    plot_parcellation(hemisphere_mesh,hcp.mmp, metric_arr[hemi_ids], vmin=vmin, tsnr_arr=tsnr_arr[hemi_ids],
                           hemi=hemisphere,tsnr_cutoff=tsnr_cutoff)
    plot_parcellation(hemisphere_mesh,hcp.mmp, metric_arr[hemi_ids], vmin=vmin, tsnr_arr=tsnr_arr[hemi_ids],
                           view_mode='medial', tsnr_cutoff=tsnr_cutoff)
    hemisphere_mesh = hcp.mesh.flat_left if hemisphere == 'left' else hcp.mesh.flat_right
    plot_parcellation(hemisphere_mesh,hcp.mmp, metric_arr[hemi_ids], vmin=vmin,
                           tsnr_arr=tsnr_arr[hemi_ids], tsnr_cutoff=tsnr_cutoff, add_colorbar=True)

plot_all_map_perspectives(nanmean_fALFF_arr_per_region, tsnr_mean_arr_per_region, 80)
#plot_all_map_perspectives(nanmean_fALFF_arr_per_region, tsnr_mean_arr_per_region, 80, hemisphere='right')
#plot_all_map_perspectives(mean_fcd_peak_arr, tsnr_mean_arr_per_region, 80, vmin=-0.5)

#plot_parcellation(hcp.mesh.inflated_left,hcp.mmp, mean_fcd_peak_arr, vmin=-0.5)
#plot_parcellation(hcp.mesh.inflated_left,hcp.mmp, mean_fcd_peak_arr, vmin=-0.5, view_mode='medial')
#plot_parcellation(hcp.mesh.flat_left,hcp.mmp,mean_fcd_peak_arr[0:180])


# compute and plot surface map of FCD width at half height (full width at half maximum - FWHM)
str_all_subjects_all_rois_mean_fcd = 'all_subjects_all_rois_mean_fcd_concat'
with open(join(path_str, str_all_subjects_all_rois_mean_fcd + '.pickle'), 'rb') as handle:
    data_all_subs_all_rois_mean_fcd = pickle.load(handle)

b_mean_across_subjects = True
if b_mean_across_subjects:
    temp_data_all_subs_all_rois_mean_fcd = []
    for roi_fcd in data_all_subs_all_rois_mean_fcd:
        temp_roi_fcd = np.reshape(roi_fcd,(-1,num_subjects))
        temp_data_all_subs_all_rois_mean_fcd.append(np.mean(temp_roi_fcd, axis=1))
    data_all_subs_all_rois_mean_fcd = temp_data_all_subs_all_rois_mean_fcd

from  scipy.stats import gaussian_kde
fcd_fwhm_per_roi_arr = []
for i, d in enumerate(data_all_subs_all_rois_mean_fcd):
    count, division = np.histogram(d, bins=100)
    max_ind = np.argmax(count)
    peak_fcd = division[max_ind]
    x = np.linspace(d.min(), d.max())
    y = gaussian_kde(d)(x)
    peak_height = np.max(y)
    half_max = peak_height / 2.0
    indices_above_half = np.where(y >= half_max)[0]
    fwhm = x[indices_above_half[-1]] - x[indices_above_half[0]]
    fcd_fwhm_per_roi_arr.append(fwhm)

# plot_parcellation(hcp.mesh.inflated_left,hcp.mmp,fcd_fwhm_per_roi_arr) #, vmin=0.85)

# Estimating G from fALFF simulation with or without HRF convolution
offset = 0.2 # without HRF - 0.06 # the offset with HRF - 0.2
exponent = 0.24 # without HRF 0.5 # with HRF - 0.24
simulation_estimated_G = 1 - (offset/(mean_fALFF_arr.mean(axis=0)))**(1/exponent)

#plot_all_map_perspectives(simulation_estimated_G, tsnr_mean_arr_per_region, 80, vmin=0.85)
#plot_all_map_perspectives(simulation_estimated_G, tsnr_mean_arr_per_region, 79, hemisphere='right', vmin=0.85)

lettersize = 22
hemisphere = 'LH'

def plot_criticality_measure(x_array, hemisphere, title_str, fig_index=1):
    fig = plt.figure(fig_index, figsize=(20,20))
    ax = plt.axes()
    if hemisphere == 'LH':
        roi_range = slice(0,180)
    elif hemisphere == 'RH':
        roi_range = slice(180, 360)
    im = plt.imshow(x_array[:,roi_range],cmap='jet')
    plt.title(hemisphere + ' ' + title_str, fontsize=lettersize)
    plt.xlabel("ROI index", fontsize=lettersize)
    plt.ylabel("Subject", fontsize=lettersize)
    plt.xticks(fontsize=lettersize)
    plt.yticks(fontsize=lettersize)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=lettersize)
    #plt.savefig(r'C:\Users\doviy\Downloads\fig_40_subjects_' + title_str + hemisphere + '_v.png', bbox_inches='tight')

def plot_criticality_measure_tsnr_corr(x_array,
                             hemisphere,
                             title_str,
                             fig_index=1,
                             tsnr_vec=None,
                             tsnr_threshold=None):

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax = plt.axes()

    # Hemisphere ROI slicing
    if hemisphere == 'LH':
        roi_range = slice(0, 180)
    elif hemisphere == 'RH':
        roi_range = slice(180, 360)
    else:
        raise ValueError("hemisphere must be 'LH' or 'RH'")

    data = x_array[:, roi_range]

    # ---- Apply tSNR masking if provided ----
    if tsnr_vec is not None and tsnr_threshold is not None:

        tsnr_hemi = tsnr_vec[roi_range]

        # Identify noisy ROIs (below threshold)
        noisy_mask = np.array(tsnr_hemi) < tsnr_threshold

        # Expand mask to match subjects x ROI shape
        full_mask = np.tile(noisy_mask, (data.shape[0], 1))

        data = np.ma.array(data, mask=full_mask)

        # Copy colormap and set masked values to white
        base_cmap = cm.get_cmap('jet', 256)
        cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, 256)))
        cmap.set_bad('white')

    else:
        cmap = 'jet'

    # ---- Plot ----
    im = plt.imshow(data, cmap=cmap)

    plt.title(f"{hemisphere} {title_str}", fontsize=lettersize)
    plt.xlabel("ROI index", fontsize=lettersize)
    plt.ylabel("Subject", fontsize=lettersize)
    plt.xticks(fontsize=lettersize)
    plt.yticks(fontsize=lettersize)

    # Colorbar
    cax = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.02,
                        ax.get_position().height])

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=lettersize)

    return fig

plot_criticality_measure(estimated_g_arr, 'LH', 'Estimated G', 1)
plot_criticality_measure(estimated_g_arr, 'RH', 'Estimated G', 2)

plot_criticality_measure(mean_fALFF_arr, 'LH', 'Mean fALFF', 3)
plot_criticality_measure(mean_fALFF_arr, 'RH', 'Mean fALFF', 4)
plot_criticality_measure_tsnr_corr(mean_fALFF_arr, 'LH', 'Mean fALFF', 13, tsnr_vec=tsnr_markers['tsnr_mean'], tsnr_threshold=80)


plot_criticality_measure(fcd_peaks_arr, 'LH', 'FCD Peak', 5)
plot_criticality_measure(fcd_peaks_arr, 'RH', 'FCD Peak', 6)
plot_criticality_measure_tsnr_corr(fcd_peaks_arr, 'LH', 'FCD Peak', 15, tsnr_vec=tsnr_markers['tsnr_mean'], tsnr_threshold=80)

def compute_and_plot_correlation_matrix(x_array, fig_ind):
    corr_x_array_LH = np.triu(np.corrcoef(x_array[:,0:180]), k=1) # extract the upper diagonal of the correlation matrix
    mean_corr_x_array_LH = corr_x_array_LH[corr_x_array_LH > 0].mean()

    corr_x_array_RH = np.tril(np.corrcoef(x_array[:,180:360]), k=-1) # extract the lower diagonal of the correlation matrix
    mean_corr_x_array_RH = corr_x_array_RH[corr_x_array_RH > 0].mean()

    corr_x_array_all = corr_x_array_LH + corr_x_array_RH

    fig = plt.figure(fig_ind, figsize=(10,8))
    im = plt.imshow(corr_x_array_all,cmap='jet',vmin=0.4)
    plt.title('LH, Mean= ' + str("%.3f" % mean_corr_x_array_LH), fontsize=lettersize, horizontalalignment='left')
    plt.xlabel('RH, Mean= ' + str("%.3f" % mean_corr_x_array_RH), fontsize=lettersize, position=(0., 1e6), horizontalalignment='left')
    plt.ylabel("Subject Ind", fontsize=lettersize)
    plt.xticks(fontsize=std_fontsize)
    plt.yticks(fontsize=std_fontsize)
    cbar = plt.colorbar() # (shrink=.5)
    cbar.ax.tick_params(labelsize=std_fontsize)
    #cbar.ax.set_ylim(0.5, 1)

#compute_and_plot_correlation_matrix(estimated_g_arr,11)
compute_and_plot_correlation_matrix(mean_fALFF_arr, 12)
compute_and_plot_correlation_matrix(fcd_peaks_arr, 13)

z_mean_fALFF_arr = (mean_fALFF_arr - mean_fALFF_arr.mean(axis=1, keepdims=True))/mean_fALFF_arr.std(axis=1, keepdims=True)
plot_criticality_measure(z_mean_fALFF_arr, 'LH', 'Standardized fALFF', 14)
compute_and_plot_correlation_matrix(z_mean_fALFF_arr, 15)

def generate_subject_property_pairwise_analysis(matrix: np.ndarray, fig_ind):
    """
    Given a matrix with N rows (subjects) and M columns (property values),
    generate all pair permutations of two-subjects' relevant property values.

    Returns:
        - x_values contains the properties of the first subject in each pair
        - y_values contains the properties of the second subject in each pair
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy ndarray.")

    n_subjects, n_properties = matrix.shape
    x_list = []
    y_list = []

    for i, j in combinations(range(n_subjects), 2):
        x_list.append(matrix[i])
        y_list.append(matrix[j])

    # Stack all vectors into one big array (for plotting)
    x_values = np.concatenate(x_list)
    y_values = np.concatenate(y_list)
    r_pearson = pearsonr(x_values, y_values)

    fig_x = plt.figure(fig_ind, figsize=(8, 8))
    plt.scatter(x_values, y_values)
    plt.xlabel("Subject x fALFF Values")
    plt.ylabel("Subject y fALFF Values")
    # plt.title("Scatter of Subject Property Pairs")
    plt.figtext(0.25, 0.77, f'r =  {str("%.2f" % r_pearson.correlation)}', fontsize=20)
    plt.grid(True)
    plt.show()
    # plt.xlabel('Odd subjects mean ' + measure_str, fontsize=lettersize)
    # plt.ylabel('Even subjects mean ' + measure_str_2nd, fontsize=lettersize)
    # plt.xticks(fontsize=lettersize)
    # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
    # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    # plt.yticks(fontsize=lettersize)
    # plt.figtext(0.25, 0.77, f'r =  {str("%.2f" % r_pearson.correlation)}', fontsize=28)
    return r_pearson.correlation

labels_for_pairwise, r_vals = [], []

labels_for_pairwise.append('All')
r_vals.append(generate_subject_property_pairwise_analysis(z_mean_fALFF_arr, 16))

auditory_indxs = [23,173,172,102,123,106,174,124,127,128,129,175,149] # [23,173,172,102,123,106,174,124,122,127,128,129,175,149]
auditory_mean_falff_sorted = mean_fALFF_arr[:,auditory_indxs]
z_auditory_mean_falff_sorted = z_mean_fALFF_arr[:,auditory_indxs]

generate_subject_property_pairwise_analysis(auditory_mean_falff_sorted, 17)
labels_for_pairwise.append('Auditory')
r_vals.append(generate_subject_property_pairwise_analysis(z_mean_fALFF_arr[:,auditory_indxs], 18))
#generate_subject_property_pairwise_analysis(z_auditory_mean_falff_sorted, 18)

default_indxs = [29,30,31,32,33,34,60,61,63,64,65,66,67,68,69,70,71,75,86,87,89,92,93,117,118,119,125,129,130,131,133,149,150,154,160,161,163,164,175,176]
r_vals.append(generate_subject_property_pairwise_analysis(z_mean_fALFF_arr[:,default_indxs], 19))
labels_for_pairwise.append('DMN')

dorsal_attention = [10, 45, 49, 94, 115, 116, 126, 135, 136, 142, 145]
r_vals.append(generate_subject_property_pairwise_analysis(z_mean_fALFF_arr[:,dorsal_attention], 20))
labels_for_pairwise.append('Dorsal Attention')

frontoparietal = [13, 14, 28, 62, 72, 76, 79, 81, 82, 84, 88, 90, 91, 96, 97, 110, 132, 143, 144, 148, 169, 170]
r_vals.append(generate_subject_property_pairwise_analysis(z_mean_fALFF_arr[:,frontoparietal], 21))
labels_for_pairwise.append('Frontoparietal')

#z_estimated_g_arr = (estimated_g_arr - estimated_g_arr.mean(axis=1, keepdims=True))/estimated_g_arr.std(axis=1, keepdims=True)
#labels_for_pairwise.append('Auditory')
#generate_subject_property_pairwise_analysis(z_estimated_g_arr[:,default_indxs], 20)

r_sq_values, error_values = [], []
for r in r_vals:
    r_sq_values.append(r**2)
    error_values.append(np.sqrt(1 - r**2))

plt.figure(figsize=(8, 5))
plt.bar(labels_for_pairwise, r_sq_values, capsize=8, color='skyblue', edgecolor='black', alpha=0.7)
plt.ylabel('Variance Explained ($R^2$)')
#plt.title('Comparison of Correlation and Noise Across Conditions')
plt.ylim(0, 1.3) # Extra room for error bars
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

lettersize = 20
# def compute_and_plot_half_half_measure_analysis(x_array, measure_str, fig_ind, second_x_array=None, measure_str_2nd=''):
#     mean_odd_half_array = np.nanmean(x_array[::2],axis=0)
#     mean_even_half_array = np.nanmean(x_array[1::2],axis=0)
#
#     if second_x_array is None:
#         measure_str_2nd = measure_str
#     else:
#         mean_even_half_array = np.nanmean(second_x_array[1::2],axis=0)
#
#     nas = np.logical_or(np.isnan(mean_odd_half_array), np.isnan(mean_even_half_array))
#     r_pearson = pearsonr(mean_odd_half_array[~nas], mean_even_half_array[~nas])
#
#     fig = plt.figure(fig_ind, figsize=(6, 6))
#     plt.scatter(mean_odd_half_array, mean_even_half_array, c=hierarchy, cmap='jet', s=50)  # df_glasser['Cortex_ID']
#     plt.xlabel('Odd subjects mean '+ measure_str, fontsize=lettersize)
#     plt.ylabel('Even subjects mean '+ measure_str_2nd, fontsize=lettersize)
#     plt.xticks(fontsize=lettersize)
#     plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
#     plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
#     plt.yticks(fontsize=lettersize)
#     plt.figtext(0.25, 0.77, f'r =  {str("%.2f" % r_pearson.correlation)}', fontsize=28)
#     plt.colorbar()
#     plt.tight_layout()

def compute_and_plot_half_half_measure_analysis(x_array,
                                                measure_str,
                                                fig_ind,
                                                second_x_array=None,
                                                measure_str_2nd='',
                                                tsnr_vec=None,
                                                tsnr_threshold=None):

    mean_odd_half_array  = np.nanmean(x_array[::2], axis=0)
    mean_even_half_array = np.nanmean(x_array[1::2], axis=0)

    if second_x_array is None:
        measure_str_2nd = measure_str
    else:
        mean_even_half_array = np.nanmean(second_x_array[1::2], axis=0)

    # ---- Apply tSNR mask (exclude noisy ROIs from r, show them white) ----
    if tsnr_vec is not None and tsnr_threshold is not None:
        noisy = (tsnr_vec < tsnr_threshold)

        # force noisy ROIs to be excluded from correlation
        mean_odd_half_array  = mean_odd_half_array.astype(float).copy()
        mean_even_half_array = mean_even_half_array.astype(float).copy()
        mean_odd_half_array[noisy]  = np.nan
        mean_even_half_array[noisy] = np.nan

        # also mask their color (so they appear white in scatter)
        hierarchy_for_c = np.ma.array(hierarchy, mask=noisy)

        base_cmap = cm.get_cmap('jet', 256)
        cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, 256)))
        cmap.set_bad('white')
    else:
        hierarchy_for_c = hierarchy
        cmap = 'jet'

    # ---- Correlation (after masking) ----
    nas = np.logical_or(np.isnan(mean_odd_half_array), np.isnan(mean_even_half_array))
    r_pearson = pearsonr(mean_odd_half_array[~nas], mean_even_half_array[~nas])

    # ---- Plot ----
    fig = plt.figure(fig_ind, figsize=(6, 6))
    sc = plt.scatter(mean_odd_half_array, mean_even_half_array,
                     c=hierarchy_for_c, cmap=cmap, s=50)  # df_glasser['Cortex_ID']

    plt.xlabel('Odd subjects mean ' + measure_str, fontsize=lettersize)
    plt.ylabel('Even subjects mean ' + measure_str_2nd, fontsize=lettersize)
    plt.xticks(fontsize=lettersize)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.yticks(fontsize=lettersize)
    plt.figtext(0.25, 0.77, f'r =  {str("%.2f" % r_pearson.correlation)}', fontsize=28)
    plt.colorbar(sc)
    plt.tight_layout()

    return r_pearson, fig

compute_and_plot_half_half_measure_analysis(estimated_g_arr,'estimated G', 31)
compute_and_plot_half_half_measure_analysis(mean_fALFF_arr, 'mean fALFF', 32)
compute_and_plot_half_half_measure_analysis(fcd_peaks_arr, 'FCD peak', 33)
compute_and_plot_half_half_measure_analysis(mean_fALFF_arr, 'mean fALFF', 34, tsnr_vec=np.array(tsnr_markers['tsnr_mean']), tsnr_threshold=80)
compute_and_plot_half_half_measure_analysis(fcd_peaks_arr, 'FCD peak', 35, tsnr_vec=np.array(tsnr_markers['tsnr_mean']), tsnr_threshold=80)

#compute_and_plot_half_half_measure_analysis(mean_fALFF_arr, 'mean fALFF', 34, second_x_array=fcd_peaks_arr, measure_str_2nd='FCD peak')

areas = df_glasser['cortex'][0:180].values # alternatively, can use 'cortex' or 'region'
values_mean_estimated_g = estimated_g_arr[:,0:180].mean(axis=0)
values_mean_mean_falff = mean_fALFF_arr[:,0:180].mean(axis=0)
values_mean_fcd_peak = fcd_peaks_arr[:,0:180].mean(axis=0)

def sort_and_plot_mean_measures(x_array_all, roi_str_arr, measure_str, fig_ind):
    x_array = x_array_all.mean(axis=0)
    x_array_err = x_array_all.std(axis=0)
    idxs = np.flip(np.argsort(x_array))
    x = np.arange(len(x_array))  # 100 x-axis values
    plt.figure(fig_ind,figsize=(20, 5)) # values_estimated_g
    plt.bar(x, x_array[idxs]) # values_mean_falff
    plt.xticks(ticks=x, labels=roi_str_arr[idxs], rotation=45, ha='right', fontsize=6)  # Rotate for better visibility
    plt.ylim(np.min(x_array)-0.02,np.max(x_array)+0.02)
    plt.xlabel('ROIs',fontsize=20)
    plt.ylabel(measure_str,fontsize=20)
    plt.errorbar(x, x_array[idxs], x_array_err[idxs], fmt='.', color='Black', elinewidth=2, capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2)
    plt.tight_layout()  # Adjust layout to fit everything neatly
    print(measure_str,areas[idxs])

sort_and_plot_mean_measures(estimated_g_arr[:,0:180], areas, 'estimated G', 51)  # values_mean_estimated_g
sort_and_plot_mean_measures(mean_fALFF_arr[:,0:180], areas, 'mean fALFF', 52) # values_mean_mean_falff
#sort_and_plot_mean_measures(values_mean_fcd_peak, areas, 'FCD peak', 53)

# plot fALFF pf the Audotory hierarchy
auditory_indxs = [23,173,172,102,123,106,174,124,127,128,129,175,149] # [23,173,172,102,123,106,174,124,122,127,128,129,175,149]
auditory_areas = areas[auditory_indxs]
auditory_mean_falff_sorted = mean_fALFF_arr[:,0:180].mean(axis=0)[auditory_indxs]
#auditory_mean_falff_err = mean_fALFF_arr[:,0:180].std(axis=0)[auditory_indxs]
auditory_mean_falff_err = stats.sem(mean_fALFF_arr[:,0:180][:,auditory_indxs],axis=0)
regions = df_glasser['region'][0:180].values
x = np.arange(len(auditory_indxs))

# Estimating G from fALFF simulation with or without HRF convolution
offset = 0.2 # without HRF - 0.06 # the offset with HRF - 0.2
exponent = 0.24 # without HRF 0.5 # with HRF - 0.24
auditory_simulated_est_G_sorted = 1 - (offset/auditory_mean_falff_sorted)**(1/exponent)

auditory_metric_sorted = auditory_mean_falff_sorted # auditory_simulated_est_G_sorted
plt.figure(45, figsize=(20, 5))
plt.bar(x, auditory_metric_sorted)  # values_mean_falff
plt.xticks(ticks=x, labels=regions[auditory_indxs], rotation=45, ha='right', fontsize=20)  # Rotate for better visibility
plt.yticks(fontsize=20)
plt.ylim(np.min(auditory_metric_sorted) - 0.02, np.max(auditory_metric_sorted) + 0.02)
plt.xlabel('Auditory ROIs', fontsize=20)
plt.ylabel('Mean fALFF', fontsize=20)  # 'Mean fALFF' 'Simulation Estimated G'
plt.errorbar(x, auditory_mean_falff_sorted, auditory_mean_falff_err, fmt='.', color='Black', elinewidth=2, capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2)

plt.figure(46, figsize=(8, 8))
plt.barh(x, auditory_metric_sorted, xerr=auditory_mean_falff_err, align='center', height=0.9)
plt.xlim(np.min(auditory_metric_sorted) - 0.02, np.max(auditory_metric_sorted) + 0.02)
plt.yticks(ticks=x, labels=regions[auditory_indxs], rotation=45, ha='right', fontsize=20)  # Rotate for better visibility
plt.xticks(fontsize=20)
plt.xlabel('Mean fALFF', fontsize=20)
plt.ylabel('Auditory ROIs', fontsize=20)
#plt.errorbar(x, auditory_mean_falff_sorted, xerr=auditory_mean_falff_err, fmt='.', color='Black', elinewidth=2, capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2)

r_pearson = pearsonr(values_mean_estimated_g, values_mean_mean_falff)

fig = plt.figure(61, figsize=(6, 6))
plt.scatter(values_mean_estimated_g, values_mean_mean_falff, c=hierarchy[:len(values_mean_estimated_g)], cmap='jet', s=50)  # df_glasser['Cortex_ID']
plt.xlabel('All subjects mean estimated G', fontsize=lettersize)
plt.ylabel('All subjects mean fALFF', fontsize=lettersize)
plt.xticks(fontsize=lettersize)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.yticks(fontsize=lettersize)
plt.figtext(0.25, 0.77, f'r =  {str("%.2f" % r_pearson.correlation)}', fontsize=28)
plt.colorbar()
plt.tight_layout()

rows = 5
num_subjects = estimated_g_arr.shape[0]
cols = int(num_subjects/rows)
fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
for i in range(num_subjects):
    sub_ind = i + 1
    r_pearson = pearsonr(estimated_g_arr[i,0:180], mean_fALFF_arr[i,0:180])
    axes[int(i / cols), i % cols].set_title(f's_{str(sub_ind)}: r={str("%.2f" % r_pearson.correlation)}')
    axes[int(i / cols), i % cols].scatter(estimated_g_arr[i,0:180], mean_fALFF_arr[i,0:180], s=5)

def compute_mean_across_simulations(filename_base, filename_exts, measure):
    vals_list = []
    for i, file_ext in enumerate(filename_exts):
        df_results = pd.read_csv(str(filename_base + file_ext + '.csv'))
        vals_list.append(df_results[measure].values)
    mean_measure_vals = np.array(vals_list).mean(axis=0)
    return mean_measure_vals

path_str = 'results'
filename_base_0 = join(path_str, 'simulation_N400_stats_per_G_') # first simulation in G range of ~0.95 to 0.99 spread across 180 ROIs without HRF conv
filename_base_1 = join(path_str, 'simulation_N400_stats_per_G_full_range_') # second simulation in G range of ~0.8 to 0.99 divided to 40 points without HRF conv
filename_base_2 = join(path_str, 'simulation_N400_stats_with_hrf_per_G_full_range_') # third simulation in G range of ~0.8 to 0.99 divided to 40 points with HRF conv
filename_base_3 = join(path_str, 'simulation_N400_stats_with_hrf_per_180_ROI_')
filename_base_4 = join(path_str, 'simulation_N400_stats_with_hrf_per_180_ROI_power_law_spacing_')
filename_base_5 = join(path_str, 'simulation_N400_stats_with_hrf_after_local_mean_per_180_ROI_') # average on local neurons and then apply HRF - Rafi's way
filename_base_6 = join(path_str, 'simulation_N400_stats_with_hrf_logspace10_per_180_ROI_') # based on use of logspace with decreasing spacing as G grows
filename_base_7 = join(path_str, 'simulation_N400_stats_no_hrf_logspace10_per_180_ROI_')
filename_base_8 = join(path_str, 'simulation_N400_stats_with_hrf_logspace10_per_40_ROI_')
filename_base_9 = join(path_str, 'simulation_N400_stats_no_hrf_logspace_Oren_A_0_06_Alpha_0_5_180_ROI_')
filename_base_10 = join(path_str, 'simulation_N400_stats_no_hrf_logspace_Oren_A_0_06_Alpha_0_5_180_ROI_Ex2_')
filename_base = filename_base_10
filename_extensions_0 = ['20241204_s1', '20241205_s2', '20241207_s3']
filename_extensions_1 = ['s0','s1', 's2', 's3','s4']
filename_extensions_2 = ['s0','s1']
filename_extensions_3 = ['s0']
filename_extensions_4 = ['s0']
filename_extensions_5 = ['s0','s1']
filename_extensions_6 = ['s0']
filename_extensions_7 = ['s0']
filename_extensions_8 = ['s0']
filename_extensions_9 = ['s0','s1','s2','s3','s4']
filename_extensions_10 = ['s0','s1','s2','s3','s4']
filename_extensions = filename_extensions_10
simulated_mean_fALFF = compute_mean_across_simulations(filename_base, filename_extensions, 'mean_fALFF')
simulated_FCD_peak = compute_mean_across_simulations(filename_base, filename_extensions, 'peak_FCD')
simulation_estimated_G = compute_mean_across_simulations(filename_base, filename_extensions, 'estimated_G')
G = compute_mean_across_simulations(filename_base, filename_extensions, 'G')

LH_mean_fALFF_arr = mean_fALFF_arr[:,0:180].mean(axis=0)
idxs = np.argsort(LH_mean_fALFF_arr)

indxs_slice = slice(0,180) # all is slice(0,180)
higher_g_simulated_mean_fALFF = simulated_mean_fALFF[indxs_slice]
higher_g_LH_mean_fALFF_arr_sorted = LH_mean_fALFF_arr[idxs][indxs_slice]
higher_g_idxs = idxs[indxs_slice]
G_idxs = G[indxs_slice] # only until the following method is enhanced to receive this as a suitable argument

def plot_measures_scatter(fig_ind, x_measure_vec, y_measure_vec, x_str, y_str):
    r_pearson = pearsonr(x_measure_vec, y_measure_vec)
    fig = plt.figure(fig_ind, figsize=(6, 6))
    plt.scatter(x_measure_vec, y_measure_vec, s=50, c=G_idxs, cmap='jet')  # df_glasser['Cortex_ID']
    plt.xlabel(x_str, fontsize=lettersize)
    plt.ylabel(y_str, fontsize=lettersize)
    plt.xticks(fontsize=lettersize)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.yticks(fontsize=lettersize)
    plt.figtext(0.25, 0.82, f'r =  {str("%.2f" % r_pearson.correlation)}', fontsize=28)
    plt.xlim(0.29,0.61)
    plt.ylim(0.29, 0.61)
    plt.colorbar()
    plt.tight_layout()

plot_measures_scatter(71, higher_g_simulated_mean_fALFF,higher_g_LH_mean_fALFF_arr_sorted, 'Simulated mean fALFF', 'All subjects mean fALFF')
plot_measures_scatter(69, values_mean_estimated_g[idxs],higher_g_LH_mean_fALFF_arr_sorted, 'All subjects estimated G', 'All subjects mean fALFF')


idxs_FCD = np.argsort(values_mean_fcd_peak)
LH_FCD_peak_arr_sorted = values_mean_fcd_peak[idxs_FCD]
plot_measures_scatter(72, simulated_FCD_peak,LH_FCD_peak_arr_sorted, 'Simulated FCD peak', 'All subjects LH FCD peak')

plot_measures_scatter(74, simulated_FCD_peak,higher_g_LH_mean_fALFF_arr_sorted, 'Simulated FCD peak', 'All subjects LH mean fALFF')
plot_measures_scatter(75, LH_FCD_peak_arr_sorted,higher_g_LH_mean_fALFF_arr_sorted, 'All subjects LH FCD peak', 'All subjects LH mean fALFF')

plot_measures_scatter(76, G,simulated_mean_fALFF, 'Simulation configured G', 'Simulated mean fALFF')
plot_measures_scatter(77, simulated_mean_fALFF, G, 'Simulated mean fALFF', 'Simulation configured G')

plot_measures_scatter(79, np.multiply(simulated_mean_fALFF,np.power(G,6)), higher_g_LH_mean_fALFF_arr_sorted, 'G-normalized simulated mean fALFF', 'All subjects mean fALFF')

# save increase trend of fALFF values for corrected G spacing simulation
# simulated_mean_fALFF_max_smooth = np.maximum.accumulate(simulated_mean_fALFF)
# print(repr(simulated_mean_fALFF_max_smooth))
# np.save(join(path_str, 'simulated_fALFF_spacing.npy'), simulated_mean_fALFF_max_smooth)
# end save

r_pearson = pearsonr(G,simulated_mean_fALFF) #(simulated_mean_fALFF, LH_mean_fALFF_arr[idxs])
fig = plt.figure(100, figsize=(6, 6))
plt.scatter(G,simulated_mean_fALFF, s=50)  # df_glasser['Cortex_ID']
plt.xlabel('Simulation control parameter (G)', fontsize=lettersize)
plt.ylabel('Simulated mean fALFF', fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.xticks(fontsize=lettersize)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.05))
#plt.xlim(0.95,0.99)
plt.figtext(0.25, 0.77, f'r =  {str("%.2f" % r_pearson.correlation)}', fontsize=28)
#plt.colorbar()
plt.tight_layout()

# try looking up scale free behavior of fALFF across ROIs
G_tag = 1 - G
r_pearson = pearsonr(G_tag,simulated_mean_fALFF) #(simulated_mean_fALFF, LH_mean_fALFF_arr[idxs])
fig = plt.figure(100, figsize=(6, 6))
plt.scatter(G_tag,simulated_mean_fALFF, s=50)  # df_glasser['Cortex_ID']
plt.xlabel('Simulation control parameter (1 - G)', fontsize=lettersize)
plt.ylabel('Simulated mean fALFF', fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.xticks(fontsize=lettersize)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.05))
#plt.xlim(0.95,0.99)
plt.figtext(0.25, 0.77, f'r =  {str("%.2f" % r_pearson.correlation)}', fontsize=28)
#plt.colorbar()
plt.tight_layout()

r_pearson = pearsonr(G_tag,simulated_mean_fALFF) #(simulated_mean_fALFF, LH_mean_fALFF_arr[idxs])
fig = plt.figure(101, figsize=(6, 6))
ax = plt.gca()
ax.scatter(G_tag,simulated_mean_fALFF, s=50)  # df_glasser['Cortex_ID']
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('Simulation control parameter (G)', fontsize=lettersize)
plt.ylabel('Simulated mean fALFF', fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.xticks(fontsize=lettersize)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.01))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.05))
#plt.xlim(0.95,0.99)
plt.figtext(0.25, 0.77, f'r =  {str("%.2f" % r_pearson.correlation)}', fontsize=28)
#plt.colorbar()
plt.tight_layout()

G_ex = G # np.power(G,6)
r_pearson = pearsonr(G_ex,simulated_mean_fALFF) #(simulated_mean_fALFF, LH_mean_fALFF_arr[idxs])
fig = plt.figure(101, figsize=(6, 6))
plt.loglog(G_ex,simulated_mean_fALFF, basex=10, basey=10)  # df_glasser['Cortex_ID']
plt.xlabel('Simulation control parameter (G)', fontsize=lettersize)
plt.ylabel('Simulated fALFF', fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.xticks(fontsize=lettersize)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.05))
#plt.xlim(0.95,0.99)
plt.figtext(0.25, 0.77, f'r =  {str("%.2f" % r_pearson.correlation)}', fontsize=28)
#plt.colorbar()
plt.tight_layout()