from os.path import join
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.stats import pearsonr, gamma, cramervonmises_2samp, ks_2samp, anderson_ksamp, chisquare
from scipy import signal
import scipy.io
from utils import butter_bandpass_filter

import HCP_fcd_utils as utils # get_subject_rois_tc, run_model_iteration, get_subject_fc_dist, HRF_convolve_signal, compute_mean_psd

def gamma_hrf(times):  # Return values for HRF at given times
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

hrf_times = np.arange(0, 20, 1)
# hrf_signal = gamma_hrf(hrf_times)

# roi_index below determines if full connectome is to be analyzed (== -1) or
roi_index = 329 # 283 # -1 # when == -1  => full connectome is analyzed, otherwise, for specific ROI k, set (... = k)

# get auditory ROIs
list_auditory_rois = [23, 102, 103, 106, 123, 172, 174, 203, 282, 286, 303, 352]

subject_filename = 'hcp_subject_n'
def transform_subjects_mat_to_pickles(path_str, filename):
    import mat73
    data_dict = mat73.loadmat(join(path_str, filename)) # overcoming the version 7.3 issue
    b_print_regions = False
    if b_print_regions:
        for i, e in enumerate(data_dict['ParcelData']['Regions_Names'], start=0):
            print(str(i) + '.', e)
    sub_ind = 1  #subject_n_data = data_dict['ParcelData']['Data'][n]
    for sub_data in data_dict['ParcelData']['Data']:
        subject_file = join(path_str,(subject_filename + str(sub_ind) + '.pickle'))
        with open(subject_file, 'wb') as handle:
            pickle.dump(sub_data, handle)  #, protocol=pickle.HIGHEST_PROTOCOL)
        sub_ind += 1

path_str = 'data'
mat_filename = 'data_Rest1_1_TR_300.mat'
#transform_subjects_mat_to_pickles(path_str, mat_filename)

# lettersize = 24
# fig0 = plt.figure(0)
# plt.plot(get_subject_rois_tc(1))
# plt.xlabel('Time (sec)', fontsize=lettersize)
# plt.gcf().subplots_adjust(bottom=0.15)
# plt.ylabel('Activity (Z Score)', fontsize=lettersize)

#N = utils.get_subject_rois_tc(1,roi_index).shape[1] # Dovi - note ROI indication
selected_subject = 30
roi_tc_arr = utils.get_subject_rois_tc(selected_subject, roi_index, z_score=False) # Dovi - note ROI indication
estimated_g = utils.compute_hu_sompolinsky_g(roi_tc_arr)
subject_fcd = utils.get_subject_fc_dist(roi_tc_arr)
mean_s, fr = utils.compute_mean_psd(roi_tc_arr)

num_subjects = 40
num_rois = 360

# save all 40 subject FCD for all ROIs
b_save_all_FCDs = False
if b_save_all_FCDs:
    for i in range(num_subjects):
        subject_all_fcds = []
        for j in range(num_rois):
            roi_tc_arr = utils.get_subject_rois_tc(i+1, j)
            mean_s, fr = utils.compute_mean_psd(roi_tc_arr)
            # plt.loglog(fr, mean_s)
            current_sub_roi_dict = {'subject':i+1, 'roi':j, 'fcd':utils.get_subject_fc_dist(roi_tc_arr), 'psd_s':mean_s, 'psd_fr':fr}
            print(f'adding info for subject {i} and roi {j}')
            subject_all_fcds.append(current_sub_roi_dict)

        with open(join(path_str, 'subject_' + str(i+1) + '_all_roi_fcds.pickle'), 'wb') as handle:
            pickle.dump(subject_all_fcds, handle)
# END - # save 40 subject FCD for all ROIs

n_bins = 50
rows = 5
cols = int(num_subjects/rows)
fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
for i in range(num_subjects):
    roi_tc_arr = utils.get_subject_rois_tc(i+1, roi_index)
    axes[int(i / cols), i % cols].set_title("subject " + str(i + 1))

    b_plot_roi_tc = False
    if b_plot_roi_tc:
        axes[int(i/cols), i%cols].plot(roi_tc_arr)

    b_plot_fcd = False
    if b_plot_fcd:
        subject_fcd = utils.get_subject_fc_dist(roi_tc_arr)
        axes[int(i / cols), i % cols].hist(subject_fcd, bins=n_bins)
        axes[int(i / cols), i % cols].set_title("subject " + str(i + 1))

    b_plot_psd = False
    if b_plot_psd:
        for j in range(0, roi_tc_arr.shape[1]):
            s, fr = mlab.psd(roi_tc_arr[:, j], NFFT=int(60), Fs=1)
            axes[int(i/cols), i%cols].loglog(fr, s)
        #plt.grid()
        #plt.xlabel('Freq. (Hz)', fontsize=lettersize)
        #plt.gcf().subplots_adjust(bottom=0.15)
        #plt.ylabel('Power (index)', fontsize=lettersize)
#plt.tight_layout()

# set up a comparable rate model simulation and explore match of functional connectivity correlation distribution
from initParams import initParams
from models import RateModel

N=roi_tc_arr.shape[1]
np.random.seed(42)
params = initParams(N=N, total_time=450)
params.tau = 20.0
params.gamma = 0.09
prb_divisor = 15 # 30
params.prb /= prb_divisor
params.mu *= prb_divisor
params.sigma *= prb_divisor

model = RateModel(params)
activity_downsampled = utils.run_model_iteration(model, params)

lettersize = 18
fig1 = plt.figure(1)
plt.plot(activity_downsampled)
plt.xlabel('Time (sec)', fontsize=lettersize)
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylabel('Activity (Z Score)', fontsize=lettersize)

activity_downsampled_demean = activity_downsampled - activity_downsampled.mean(axis=0, keepdims=True)
activity_downsampled_std = (activity_downsampled - activity_downsampled.mean(axis=0, keepdims=True)) / activity_downsampled.std(axis=0, keepdims=True)

# bold_signal = np.convolve(activity_downsampled[:,0], hrf_signal)

hrf_conv_activity = utils.HRF_convolve_signal(activity_downsampled_demean, z_score=False)
hrf_activity_demean = hrf_conv_activity - hrf_conv_activity.mean(axis=0, keepdims=True)

fig2 = plt.figure(2)
plt.plot(hrf_activity_demean)
plt.xlabel('Time (sec)', fontsize=lettersize)
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylabel('Activity (Z Score)', fontsize=lettersize)

mean_s_sim, fr_sim = utils.compute_mean_psd(activity_downsampled)
mean_s_sim_hrf, fr_sim_hrf = utils.compute_mean_psd(hrf_activity_demean)
k_scale = 1000000000
fig3 = plt.figure(3)
plt.loglog(fr, mean_s, label='Empirical PSD')
plt.loglog(fr, mean_s_sim * k_scale, label='Simulation PSD')
plt.loglog(fr, mean_s_sim_hrf * k_scale, label='Simulation with HRF PSD')
plt.legend()
plt.xlabel('Freq', fontsize=lettersize)
plt.ylabel('Power', fontsize=lettersize)

simulation_fcd = utils.get_subject_fc_dist(activity_downsampled_demean)
res_ks = ks_2samp(subject_fcd, simulation_fcd)
res_cvm = cramervonmises_2samp(subject_fcd, simulation_fcd)

fig4 = plt.figure(4)
_, bins, _ = plt.hist(subject_fcd, bins=n_bins, label='Empirical HCP results')
_ = plt.hist(simulation_fcd, bins=bins, alpha=0.5, label='Random recurrent model simulation')
plt.legend()
plt.xlabel('Correlation', fontsize=lettersize)
plt.ylabel('Count', fontsize=lettersize)

import levy
params_levy_sim = levy.fit_levy(simulation_fcd)
params_levy_emp = levy.fit_levy(subject_fcd)

from scipy.stats import levy_stable
data_levy_sim = levy_stable.rvs(alpha=params_levy_sim[0].x[0],beta=params_levy_sim[0].x[1], loc=params_levy_sim[0].x[2],scale=params_levy_sim[0].x[3], size=len(simulation_fcd))
data_levy_emp = levy_stable.rvs(alpha=params_levy_emp[0].x[0],beta=params_levy_emp[0].x[1], loc=params_levy_emp[0].x[2],scale=params_levy_emp[0].x[3], size=len(simulation_fcd))
fig5 = plt.figure(5)
_, bins, _ = plt.hist(np.array(subject_fcd), bins=50, label='Empirical HCP results')
plt.hist(np.array(simulation_fcd), bins=bins, label='Random recurrent model simulation')
plt.hist(np.array(data_levy_emp), bins=bins, label='Levy stable empirical HCP fit')
plt.hist(np.array(data_levy_sim), bins=bins, label='Levy stable simulation fit')
plt.legend()
plt.xlabel('Correlation', fontsize=lettersize)
plt.ylabel('Count', fontsize=lettersize)
#x = np.linspace(0, 1, 100)
#plt.plot(x, levy_stable.pdf(x, *params_levy)*50, lw=3)

from scipy.stats import beta
params_beta_sim = beta.fit(simulation_fcd)
params_beta_emp = beta.fit(subject_fcd)

prbs = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.3, 0.4]
gammas = [0.02, 0.04, 0.08, 0.085, 0.09, 0.092, 0.094, 0.095, 0.096, 0.098]
rows = 2
cols = int(len(prbs)/rows)
mu = params.mu
gamma = params.gamma
prb = 0.2
fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
ind = 0
#for prb in prbs:
for gamma in gammas:
    #params.gamma = gamma * (0.2/prb)
    params.gamma = gamma
    params.prb = prb
    params.mu = mu * (0.095/gamma) # (0.2/prb)
    print("Model params are: ", params)
    model = RateModel(params)

    node_activities_downsampled = utils.run_model_iteration(model, params)

    #fig1 = plt.figure(1)
    #plt.plot(node_activities_downsampled)

    corr_arr = []
    for i in range(0, N-1):
        for j in range(i+1, N):
            corr, _ = pearsonr(node_activities_downsampled[:, i], node_activities_downsampled[:, j])
            corr_arr.append(corr) # as in Turner, define positive values only for the connectivity level
            if corr > 0.5:
                print(f"pair of neurons are at indexes {i}, {j}, corr = {corr}")
                #break
    dist2 = corr_arr

    _, bins, _ = axes[int(ind/cols), ind%cols].hist(subject_fcd, bins=n_bins, label='Empirical HCP results')
    axes[int(ind/cols), ind%cols].hist(dist2, bins=bins, alpha=0.5, label='Random recurrent model simulation')
    axes[int(ind/cols), ind%cols].set_title("G = " + str(params.gamma * params.prb * params.mu))
    # axes[int(ind/cols), ind%cols].set_title("probability " + str(prb))

    ind += 1

count, division = np.histogram(subject_fcd, bins=50)
ind = np.argmax(count)
peak = division[ind]

plotfile = 'xyz'
plt.savefig('figure/' + plotfile, transparent=True)
plt.close()