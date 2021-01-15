# Authors: Raffaela Cunha <raffaelacunha@gmail.com>

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
from scipy import stats

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
    filt_raw.load_data()
    filt_raw.filter(0.1,45, verbose=False) 
    filt_raw.notch_filter(np.arange(60, 241, 60), verbose=False)
    filt_csd = mne.preprocessing.compute_current_source_density(filt_raw)
    if plot_flag ==True:
        filt_csd.plot(duration=2960, start = 316,n_channels=20, remove_dc=False);
        filt_csd.plot_psd(fmax=100);
    return filt_csd

def get_events_sequence(seq_events_sect, num_sect, time_events ):    
    seq_events = np.repeat(seq_events_sect,num_sect, axis=0)
    seq_events = seq_events.flatten()
    seq_events = np.reshape(seq_events,[len(seq_events),1])
    zero_column = np.zeros([len(seq_events),1],dtype='int32')
    array_events = np.concatenate([time_events[1:],zero_column], axis=1)
    array_events = np.concatenate([array_events,seq_events],axis=1)
    ind_crop = np.arange(8,72,9)
    array_events = np.delete(array_events, ind_crop, axis=0)
    return array_events

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

def extract_features_signal(epochs_test, ref_train, events_intv, new_freq, base_power=None):#ref_train is an array (for a given band)
    fs = epochs_test.info['sfreq']
    selection = epochs_test.selection
    y_test = None
    erds_test = None
    rpower_test = None
    for ind_ep in range(selection.size):
        ep_test = epochs_test[ind_ep]
        ep_event_id = ep_test.event_id
        ep_event_id = ep_event_id[next(iter(ep_event_id))]
        ep_test.crop(events_intv[ep_event_id,0],events_intv[ep_event_id,1])
        signal_ep = ep_test.get_data()
        signal_ep = signal_ep.reshape([signal_ep.shape[1], signal_ep.shape[2]])
        time_vec_ep = np.arange(0, signal_ep.size/fs,signal_ep.size/(signal_ep.size*fs))
        erds_ep, rpower_ep = extract_erds_signal(signal_ep, time_vec_ep, ref_train, fs, new_freq, base_power)
        
        y_ep = np.repeat(ep_event_id, erds_ep.shape[1])
            
        if erds_test is not None:
            erds_test = np.concatenate([erds_test, erds_ep], axis=1)
            rpower_test = np.concatenate([rpower_test, rpower_ep], axis=1)
            y_test = np.concatenate([y_test, y_ep])
        else: 
            erds_test = erds_ep
            rpower_test = rpower_ep
            y_test = y_ep
    return erds_test, y_test, rpower_test


def extract_erds_signal(signal, time_vec, ref, fs, new_freq, base_power):
    # extract erds. Signal must be previously filtered in the desired band
    power_signal = np.power(signal,2)
    smooth_array = None
    for ch in range(signal.shape[0]):
        smooth_power, smooth_time = smooth_signal(power_signal[ch,:].reshape([1,signal.shape[-1]]), fs, new_freq, time_vec, plot=False)
        if smooth_array is not None:
            smooth_array = np.concatenate([smooth_array, smooth_power.reshape([1, smooth_power.shape[0]])], axis = 0)
        else: smooth_array = smooth_power.reshape([1, smooth_power.shape[0]])
    ref = ref.reshape([ref.shape[0],1])
    erds = (smooth_array - ref)/ref
    rel_power = smooth_array/base_power.reshape([base_power.shape[0],1])
    return erds, rel_power

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
    smooth_ind_0 = np.where(np.abs(smooth_time-lims_time_event[0])<1e-5)[0][0]
    smooth_ind_1 = np.where(np.abs(smooth_time-lims_time_event[1])<1e-5)[0][0]
    ind_intv = np.array([smooth_ind_0,smooth_ind_1])
    smooth_array = smooth_array[:,ind_intv[0]:ind_intv[1]]    
    ref = ref.reshape([ref.shape[0],1])
    erds = (smooth_array - ref)/ref
    return erds

def extract_base_power(epochs,filt_freqs, events_intv):
    epochs.filter(filt_freqs[0], filt_freqs[1])
    selection = epochs.selection
    base_power = None
    for ind_ep in range(selection.size): # acha a potencia de cada epoca separadamente(devido aos diferentes tempos de corte)
        ep = epochs[ind_ep]
        ep_event_id = ep.event_id
        ep_event_id = ep_event_id[next(iter(ep_event_id))]
        ep.crop(events_intv[ep_event_id,0], events_intv[ep_event_id,1]) 
        ep_event_data = np.power(ep.get_data(),2)
        ep_base_power = ep_event_data.mean(axis=2) #media no tempo (shape (63,1))
        if base_power is not None:
            base_power = np.concatenate([base_power,ep_base_power.reshape([ep_base_power.shape[-1],1])], axis=1)#each epoch in a column
        else: base_power = ep_base_power.reshape([ep_base_power.shape[-1],1])
    val_base_power = base_power.mean(axis=1)  # media ao longo das epocas

    return val_base_power

def extract_ref(epochs, time_vec, lims_time_event, fs, new_freq):
    epochs_event_data = np.power(epochs.get_data(),2)
    epochs_mean = epochs_event_data.mean(axis=0)
    # acrescentado smooth na referencia
    smooth_array = None
    for ch in range(epochs_mean.shape[0]):
        smooth_power, smooth_time = smooth_signal(epochs_mean[ch,:].reshape([1,epochs_mean.shape[-1]]), fs, new_freq, time_vec, plot=False)
        if smooth_array is not None:
            smooth_array = np.concatenate([smooth_array, smooth_power.reshape([1, smooth_power.shape[0]])], axis = 0)
        else: smooth_array = smooth_power.reshape([1, smooth_power.shape[0]])
        smooth_ind_0 = np.where(np.abs(smooth_time-lims_time_event[0])<1e-5)[0][0]
    smooth_ind_1 = np.where(np.abs(smooth_time-lims_time_event[1])<1e-5)[0][0]
    ind_intv = np.array([smooth_ind_0,smooth_ind_1])
    smooth_array = smooth_array[:,ind_intv[0]:ind_intv[1]]  
    # remove outliers
    ref = np.array([])
    for ch in range(epochs_mean.shape[0]):
        ref_ch = smooth_array[ch,:]
        z_score = stats.zscore(smooth_array[ch,:])
        ind_outlier = np.where(np.abs(z_score)>3)[0]
        if ind_outlier.size!=0:
            ref_ch = np.delete(ref_ch,ind_outlier).mean()
        ref = np.concatenate([ref,np.array([ref_ch.mean()])])
    return ref


def plot_erds_topomap(erds_events, y,  info, band, events_dict, vmin, vmax, ch_names):
    fig, axs = plt.subplots(ncols=4, nrows = 1, figsize=(6, 2), gridspec_kw=dict(top=0.9),
                       sharex=True, sharey=True)
    for event_name, event_id in events_dict.items():
        #if event_name!="neutro":
            ind_obs_event = np.where(y==event_id)[0]
            erds_event =  erds_events[:, ind_obs_event]
            array_plot = erds_event.mean(axis=1) #pega a média no tempo
            im,cn = plt_topomap(array_plot, pos=info, axes = axs[event_id], show = False, vmin = vmin, vmax = vmax, show_names = True, names = ch_names);
            time_txt = "{}-{}s".format(8, 36)
            axs[event_id].set_title("{} {}".format(event_name, band), fontsize=10)
    cbar = plt.colorbar(im)  
    cbar.ax.tick_params(labelsize=6)
    axs[3].axis('off')
    plt.show()
    
def get_channels_stats(erds,y, events_dict, ch_names):
    #describe
    for event_name, ind_event in events_dict.items():
            df = pd.DataFrame([])
            ind_obs_event = np.where(y==ind_event)[0]
            for ch in range(erds.shape[1]):
                obs_events = erds[ind_obs_event,ch]
                df[ch_names[ch]]= obs_events
            print(event_name)
            print(df.describe())
    #boxplot    
    n_rows = int(np.ceil(np.sqrt(len(ch_names))))
    n_cols = n_rows
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)    
    ind_ch = 0
    for ch in range(erds.shape[1]): 
         df = pd.DataFrame([])
         for event_name, ind_event in events_dict.items():
             ind_obs_event = np.where(y==ind_event)[0]
             obs_events = erds[ind_obs_event,ch]
             df[event_name]= pd.Series(obs_events)
         col = int(np.floor(ch/axs.shape[0]))
         row = np.mod(ch,axs.shape[0])
         df.boxplot(ax=axs[row,col])
         axs[row,col].set_title(ch_names[ch], fontsize=7, fontweight='bold')
         axs[row,col].tick_params(labelsize=7)
    ind_ch+=1
    plt.show()

    
    
    