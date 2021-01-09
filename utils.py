# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 01:43:54 2021

@author: Raffaela
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import psd_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap
from mne.viz import plot_topomap as plt_topomap
import pandas as pd
import os

def read_eeginfo(eeg_filepath, marc_filepath, verbose = False):
    # read eeg signals file
    raw = mne.io.read_raw_brainvision(eeg_filepath) # EEG + other sensors channels
    raw.drop_channels(['GSR_MR_100_xx','74','75','76','ECG'])
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    # read events time file
    mat_content = scipy.io.loadmat(marc_filepath)
    time_events = mat_content['marcadores']
    if verbose:
        print(raw.info)
    return raw, time_events


def eeg_preprocess(raw, channels_list = None, plot_flag = False):
    filt_raw = raw.copy()
    channels_list = raw.info['ch_names']
    if channels_list is not None:
        filt_raw.pick(channels_list)
    #num_channels = filt_raw.info['nchan']
    filt_raw.load_data()
    filt_raw.filter(0.1,45, phase="zero-double", method = "iir", iir_params= dict(order=4, ftype='cheby1', rp=0.5))
    filt_raw.notch_filter(np.arange(60, 241, 60))
    filt_csd = mne.preprocessing.compute_current_source_density(filt_raw)
    if plot_flag ==True:
        filt_csd.plot(duration=2, n_channels=20, remove_dc=False);
        filt_csd.plot_psd(fmax=100);
    return filt_csd

def get_events_sequence(seq_events_sect, num_sect, time_events ):    
    seq_events = np.repeat(seq_events_sect,num_sect, axis=0)
    seq_events = seq_events.flatten()
    seq_events = np.reshape(seq_events,[len(seq_events),1])
    zero_column = np.zeros([len(seq_events),1],dtype='int32')
    array_events = np.concatenate([time_events[1:],zero_column], axis=1)
    array_events = np.concatenate([array_events,seq_events],axis=1)

    return array_events

def get_sensor_positions(info):
    dig= info['dig']
    array_pos=None
    for k in dig:
        values = list(k.values())
        if array_pos is not None:
            array_pos = np.concatenate([array_pos,values[1][0:2].reshape([1,values[1].shape[0]-1])], axis=0)
        else:
            array_pos = values[1][0:2].reshape([1,values[1].shape[0]-1])
    return array_pos

def smooth_signal(erds, fs, new_freq, time_vec, plot=False):
    window_len = np.int((1/new_freq)*fs)
    erds_smooth = np.array([erds[0:window_len].mean()])
    for ind_loop,window_ind in enumerate(np.arange(np.int(0.5*fs), erds.shape[1],np.int(0.5*fs))):
        if window_ind+window_len>erds.shape[1]:
            window_len = erds.shape[1] - window_ind
        curr_sum = erds[:,window_ind: window_ind+window_len].mean()
        erds_smooth = np.concatenate([erds_smooth,np.array([(curr_sum + erds_smooth[ind_loop])/2])])

    axis_smooth = np.arange(time_vec[0],time_vec[-1],1/new_freq)
    axis_smooth = np.concatenate([axis_smooth,np.array([time_vec[-1]])])
    if plot==True:
        fig,[ax1,ax2] = plt.subplots(2,1)
        ax1.plot(time_vec, erds.T)
        ax2.plot(axis_smooth,erds_smooth)
        plt.show()
    return erds_smooth, axis_smooth 

def extract_erds_signal(signal, time_vec, ref, fs, new_freq):
    # extract erds. Signal must be previously filtered in the desired band
    power_signal = np.power(signal,2)
    smooth_power_signal, smooth_time = smooth_signal(power_signal, fs, new_freq, time_vec, plot=False)
    erds = (smooth_power_signal - ref)/ref
    return erds

def extract_erds_epochs(epochs, time_vec, ref, fs, new_freq, lims_time_event):
    # extract erds. Signal must be previously filtered in the desired band
    epochs_event_data = np.power(epochs.get_data(),2)
    epochs_mean = epochs_event_data.mean(axis=0)
    smooth_array = None
    for ch in range(epochs_mean.shape[0]):
        smooth_power, smooth_time = smooth_signal(epochs_mean[ch,:].reshape([1,epochs_mean.shape[-1]]), fs, new_freq, time_vec, plot=False)
        if smooth_array is not None:
            smooth_array = np.concatenate([smooth_array, smooth_power.reshape([1, smooth_power.shape[0]])], axis = 0)
        else: smooth_array = smooth_power.reshape([1, smooth_power.shape[0]])
    ref = ref.reshape([ref.shape[0],1])
    erds = (smooth_array - ref)/ref
    smooth_ind_0 = np.where(np.abs(smooth_time-lims_time_event[0])<1e-5)[0][0]
    smooth_ind_1 = np.where(np.abs(smooth_time-lims_time_event[1])<1e-5)[0][0]
    ind_intv = np.array([smooth_ind_0,smooth_ind_1])
    erds = erds[:,ind_intv[0]:ind_intv[1]]
    return erds


def extract_ref(epochs, time_ind):
    epochs_event_data = np.power(epochs.get_data(),2)
    epochs_mean = epochs_event_data.mean(axis=0)
    ref = epochs_mean[:,time_ind[0]:time_ind[1]].mean(axis=1)
    return ref


def plot_erds_topomap(erds_events, y,  array_pos, band, events_dict, vmin, vmax):
    fig, axs = plt.subplots(ncols=4, nrows = 1, figsize=(6, 2), gridspec_kw=dict(top=0.9),
                       sharex=True, sharey=True)
    for event_name, event_id in events_dict.items():
        ind_obs_event = np.where(y==event_id)[0]
        erds_event =  erds_events[:, ind_obs_event]
        array_plot = erds_event.mean(axis=1) #pega a média no tempo
        im,cn = plt_topomap(array_plot, pos=array_pos[3:,:], axes = axs[event_id], show = False, vmin = vmin, vmax = vmax);
        time_txt = "{}-{}s".format(8, 36)
        axs[event_id].set_title("{} {}".format(band, time_txt), fontsize=6)
    
