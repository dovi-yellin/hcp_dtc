from os.path import join
from pathlib import Path
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, gamma, cramervonmises_2samp, ks_2samp, anderson_ksamp, chisquare
from scipy import signal
from utils import butter_bandpass_filter
import nitime.algorithms.spectral as spectrum
import nilearn
from numpy.linalg import eigvals, eigvalsh
from cov_spectrum_random_netw import fit_cdf_g

#"A hemodynamic response function"
def hrf(t):
    return t ** 8.6 * np.exp(-t / 0.547)

def gamma_hrf(times):  # Return values for HRF at given times
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 2) # 6
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 4) # 12
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

hrf_times = np.arange(0, 20, 1)
#hrf_signal = hrf(hrf_times)
hrf_signal = gamma_hrf(hrf_times)



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

_REPO_ROOT = Path(__file__).resolve().parents[1]
path_str = _REPO_ROOT / 'data'
mat_filename = 'data_Rest1_1_TR_300.mat'
#transform_subjects_mat_to_pickles(path_str, mat_filename)

def calculate_tsnr(roi_tcs):
    # Calculate temporal SNR for each voxel in fMRI data.
    mean_signal = np.mean(roi_tcs, axis=0)
    std_signal = np.std(roi_tcs, axis=0)
    tsnr_map = np.divide(mean_signal, std_signal, out=np.zeros_like(mean_signal), where=std_signal != 0)
    return tsnr_map

def get_subject_rois_tc(sub_ind, roi_ind=-1, z_score=True, demean=False, tsnr_threshold=-1):
    subject_file = join(path_str,(subject_filename + str(sub_ind) + '.pickle'))
    with open(subject_file, 'rb') as handle:
        data_hcp_sub = pickle.load(handle)

    roi_tc_arr = []
    if roi_ind == -1:
        for roi in data_hcp_sub:
            if tsnr_threshold > 0:
                tsnr_indx = calculate_tsnr(roi)
                roi = roi[tsnr_indx > tsnr_threshold]
            if z_score:
                roi = (roi - roi.mean(axis=1, keepdims=True)) / roi.std(axis=1, keepdims=True)
            elif demean:
                roi = (roi - roi.mean(axis=1, keepdims=True))
            roi_tc_arr.append(roi.mean(axis=0))
    else:
        for voxel_tc in data_hcp_sub[roi_ind]:
            if tsnr_threshold > 0:
                if np.divide(np.mean(voxel_tc),np.std(voxel_tc)) < tsnr_threshold:
                    continue
            if z_score:
                voxel_tc = (voxel_tc - np.mean(voxel_tc)) / np.std(voxel_tc)
            elif demean:
                voxel_tc = (voxel_tc - np.mean(voxel_tc))
            roi_tc_arr.append(voxel_tc)

    if len(roi_tc_arr) > 0:
        roi_tc_arr = np.stack(roi_tc_arr, axis=0).T
    return np.array(roi_tc_arr)

def get_subject_fc_dist(roi_tc_arr, to_console=False):
    corr_arr = []
    for i in range(0, roi_tc_arr.shape[1]-1):
        for j in range(i+1, roi_tc_arr.shape[1]):
            corr, _ = pearsonr(roi_tc_arr[:, i], roi_tc_arr[:, j])
            corr_arr.append(corr) # as in Turner, define positive values only for the connectivity level # note - adding twice due to symmetry in correlation matrix
            if to_console and corr > 0.5:
                print(f"pair of neurons are at indexes {i}, {j}, corr = {corr}")
    return corr_arr

def run_model_iteration(model, params, z_score=True, hrf_convole=False):
    control_param = params.gamma * params.mu * params.prb

    sys.stdout.flush()  # ensures progress bar starts after print
    print(f"Beginning network simulation with {params.gamma=:.6f}, {params.mu=}, {params.prb=}, {control_param=:.6f}")
    # print(f"{N=}")
    sys.stdout.flush()  # ensures progress bar starts after print

    # run the model simulation
    results_dict = model.run_local_circuit()

    r_store = results_dict['r_store']
    activity_per_unit = np.transpose(r_store)
    node_activities = activity_per_unit[:, (int(params.start_sample) - 1):-1]
    node_activities = np.transpose(node_activities)
    # resampling the signal to 1 Hz
    node_activities_downsampled = signal.resample(node_activities, int(node_activities.shape[0] / params.fs))
    if hrf_convole:
        hrf_conv_activity = []  # np.array([])
        for i in range(0, node_activities.shape[1]):
            norm_activity_downsampled = node_activities_downsampled[:, i] - np.mean(node_activities_downsampled[:, i])
            bold_activity = np.convolve(norm_activity_downsampled, hrf_signal)
            bold_activity = bold_activity[int(len(hrf_signal) / 2):-int(len(hrf_signal) / 2 - 1)]  # trim convolution ends
            hrf_conv_activity.append(bold_activity)
        node_activities_downsampled = np.transpose(np.array(hrf_conv_activity))
    if z_score:
        node_activities_downsampled = (node_activities_downsampled - node_activities_downsampled.mean(axis=0,
                                    keepdims=True)) / node_activities_downsampled.std(axis=0, keepdims=True)
    return node_activities_downsampled

def HRF_convolve_signal(network_activity, z_score=True):
    hrf_conv_activity = []  # np.array([])
    for i in range(0, network_activity.shape[1]):
        demean_activity = network_activity[:, i] - np.mean(network_activity[:, i])
        bold_activity = np.convolve(demean_activity, hrf_signal)
        bold_activity = bold_activity[int(len(hrf_signal) / 2):-int(len(hrf_signal) / 2 - 1)]  # trim convolution ends
        hrf_conv_activity.append(bold_activity)
    hrf_conv_activity = np.transpose(np.array(hrf_conv_activity))
    if z_score:
        hrf_conv_activity = (hrf_conv_activity - hrf_conv_activity.mean(axis=0, keepdims=True)) / hrf_conv_activity.std(axis=0, keepdims=True)
    return hrf_conv_activity

def compute_mean_psd(roi_tc_arr, Fs=1, alpha=1):
    s_arr = []
    for k in range(int(roi_tc_arr.shape[1]*alpha)):
        mean_activity = roi_tc_arr[:, k] - np.mean(roi_tc_arr[:, k])
        #s, fr = mlab.psd(mean_activity, NFFT=int(50), Fs=Fs)
        fr, s, _ = spectrum.multi_taper_psd(mean_activity, Fs=Fs, NW=5)
        s_arr.append(s)
    mean_s = np.mean(s_arr, axis=0)
    return mean_s, fr

def compute_mean_fALFF(roi_tc_arr, average_first=False, Fs=1):
    # time_series_cleaned = nilearn.signal.clean(roi_tc_arr, sessions=None, detrend=True,
    #                                            standardize=None, confounds=None,
    #                                            low_pass=0.25, high_pass=0.01, t_r=2.0,
    #                                            ensure_finite=True)

    #Compute power spectra
    # freqs_cleaned, psd_cleaned = signal.welch(roi_tc_arr.T, fs=0.5, window='boxcar',
    #                                           nperseg=None, noverlap=None, nfft=60,
    #                                           detrend='constant', return_onesided=False,
    #                                           scaling='spectrum', axis=-1, average='mean')
    roi_tc_arr_demean = roi_tc_arr - roi_tc_arr.mean(axis=0, keepdims=True)
    roi_tc_arr_demean_detrend = signal.detrend(roi_tc_arr_demean, axis=0)

    if average_first: # and HRF convolve
        mean_signal = np.mean(roi_tc_arr_demean_detrend, axis=1)
        demean_activity = mean_signal - np.mean(mean_signal)
        mean_signal_conv = np.convolve(demean_activity, hrf_signal)
        row = butter_bandpass_filter(mean_signal_conv, 0.01, 0.25, Fs)
        freqs_cleaned, psd_cleaned, _ = spectrum.multi_taper_psd(row, Fs=Fs, NW=5)
        low_freq = psd_cleaned[(freqs_cleaned >= 0.01) & (freqs_cleaned <= 0.08)]
        sum_low_freq = np.sum(low_freq)
        all_freq = psd_cleaned[(freqs_cleaned >= 0.01) & (freqs_cleaned <= 0.25)]
        sum_all_freq = np.sum(all_freq)
        fALFF = sum_low_freq / sum_all_freq
        dummy_std = 0
        return fALFF, dummy_std

    fALFF_container = []
    for row in roi_tc_arr_demean_detrend.T:
        row = row - np.mean(row)
        #fig111 = plt.figure(111)
        #plt.plot(row)
        row = butter_bandpass_filter(row, 0.01, 0.25, Fs)
        #plt.plot(row)
        freqs_cleaned, psd_cleaned, _ = spectrum.multi_taper_psd(row, Fs=Fs, NW=5)
        low_freq = psd_cleaned[(freqs_cleaned >= 0.01) & (freqs_cleaned <= 0.08)]
        sum_low_freq = np.sum(low_freq)
        all_freq = psd_cleaned[(freqs_cleaned >= 0.01) & (freqs_cleaned <= 0.25)]
        sum_all_freq = np.sum(all_freq)
        fALFF = sum_low_freq / sum_all_freq

        fALFF_container.append(fALFF)
    mean_fALFF = np.mean(fALFF_container)
    std_fALFF = np.std(fALFF_container)
    return mean_fALFF, std_fALFF

def compute_hu_sompolinsky_g(roi_tc_arr, Fs=1):
    roi_tc_arr_demean = roi_tc_arr - roi_tc_arr.mean(axis=0, keepdims=True)
    roi_tc_arr_demean_detrend = signal.detrend(roi_tc_arr_demean, axis=0)

    Mg = roi_tc_arr.T  # M_cell[id_re_sort[i]]
    n, T = Mg.shape  # [num units * num time points]
    a0 = n / T
    Cg = np.corrcoef(Mg)  # (Mg)
    eig_Cg = -np.sort(-eigvalsh(Cg))
    if a0 > 1:
        eig_Cg = eig_Cg[eig_Cg > 0]
    gh, s2h, cost = fit_cdf_g(eig_Cg, cost='CvM')
    return gh

def plot_activity(activity_downsampled, fig=None, lettersize=20):
    activity_downsampled_demean = activity_downsampled - activity_downsampled.mean(axis=0, keepdims=True)
    if fig == None:
        fig = plt.figure(101)
    plt.plot(activity_downsampled_demean)
    plt.xlabel('Time (s)', fontsize=lettersize)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylabel('Activity (mean deviation)', fontsize=lettersize)

    return  # return fig

def plot_FCD(activity_downsampled, fig=None, label=None, lettersize=20):
    activity_downsampled_demean = activity_downsampled - activity_downsampled.mean(axis=0, keepdims=True)
    simulation_fcd = get_subject_fc_dist(activity_downsampled_demean)
    if fig == None:
        fig = plt.figure(101)
    n_bins = 50
    _, bins, _ = plt.hist(simulation_fcd, bins=n_bins, label=label)
    #_ = plt.hist(simulation_fcd, bins=bins, alpha=0.5, label='Random recurrent model simulation')
    if label is not None:
        plt.legend()
    plt.xlabel('Correlation', fontsize=lettersize)
    plt.ylabel('Count', fontsize=lettersize)
