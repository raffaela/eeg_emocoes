# Authors: Raffaela Cunha <raffaelacunha@gmail.com>

import os
import numpy as np
import mne
from utils import read_eeginfo, eeg_preprocess, get_events_sequence
from utils import extract_erds_epochs, extract_ref, extract_base_power
from utils import extract_features_signal, plot_erds_topomap, get_channels_stats
import config
from classify import train_knn, test_model, train_randomforest, report_average, print_confusion_matrix
from collections import deque
import pandas as pd
from excel_write import append_df_to_excel
import matplotlib.pyplot as plt


def analyze_mean(epochs, ch_names):
    erds_dict, y_dict, ref_dict = extract_ERDS(epochs)
    for band, lims in config.freq_bands.items(): # frequency band loop
        get_channels_stats(erds_dict[band].T,y_dict[band], config.events_id, ch_names)
        plot_erds_topomap(erds_dict[band],y_dict[band], raw_filt.info, band, config.events_id, -0.5, 0.5, ch_names)


def extract_ERDS(epochs, freq_bands = config.freq_bands):
    # extract ERDS 
    fs = epochs.info['sfreq']
    time_vec = epochs.times
    ind_0 = np.where(np.abs(epochs.times-config.lims_time_neutro[0])<1e-5)[0][0]
    ind_1 = np.where(np.abs(epochs.times-config.lims_time_neutro[1])<1e-5)[0][0]
    lims_ind_neutro = [ind_0, ind_1]
    erds_band = None
    ref_dict = {}
    erds_dict = {}
    y_dict = {}
    for band, lims in freq_bands.items(): # frequency band loop
        erds_band = None
        y_band = None
        #epochs.filter(lims[0],lims[1],phase="zero-double", method = "iir", iir_params= dict(order=4, ftype='cheby1', rp=0.5), verbose=False);
        epochs_band = epochs.copy()
        epochs_band.filter(lims[0],lims[1], verbose=False);
        ref = extract_ref(epochs_band['neutro'].copy(), time_vec, config.lims_time_neutro, fs, config.new_freq) 
        for event_type, ind_event in config.events_id.items(): # event type loop
            epochs_event = epochs_band[event_type].copy()
            if event_type=='neutro':
                time_ind_event = config.lims_time_neutro # indices para intervalo de tempo a considerar na extracao do erds para neutro
            else: time_ind_event = config.lims_time_event # indices para intervalo de tempo a considerar na extracao do erds para angustia e ternura   

            erds = extract_erds_epochs(epochs_event,time_vec,ref,fs,config.new_freq, time_ind_event)
            if y_band is not None:
                y_band = np.concatenate([y_band, np.repeat(ind_event, erds.shape[1])], axis=0)
            else: y_band =  np.repeat(ind_event, erds.shape[1])
            if erds_band is not None:
                erds_band = np.concatenate([erds_band, erds], axis=1)
            else: erds_band = erds

        ref_dict[band] = ref    
        erds_dict[band] = erds_band
        y_dict[band] = y_band
    return erds_dict, y_dict, ref_dict      

if __name__ == '__main__':
    raw, time_events = read_eeginfo(config.eeg_filepath, config.marc_filepath)
    raw_filt = eeg_preprocess(raw)
    array_events = get_events_sequence(config.seq_events, config.num_sections , time_events)
    ch_names = raw_filt.info['ch_names']
    # transform data to Epochs
    epochs = mne.Epochs(raw_filt.copy(), array_events, event_id=config.events_id, 
    tmin=config.tmin_epoch, tmax=config.tmax_epoch, preload=True, baseline=None)

    # Plot mean ERDS topomaps and boxplots for each frequency
    analyze_mean(epochs.copy(), ch_names)
    
    # Exploratory analysis of selected features 
    raw_filt.pick_channels(config.channels_pick)
    band = 'beta_gama'
    fs = epochs.info['sfreq']
    base_power = extract_base_power(epochs.copy(), config.base_freq, config.events_intv)
    lims_freq = config.freq_bands[band]
    epochs_band = epochs.copy()
    epochs_band.filter(lims_freq[0], lims_freq[1],verbose=False)
    ref_band = extract_ref(epochs_band['neutro'].copy(), epochs.times, config.lims_time_neutro, fs, config.new_freq) 
    erds_band,y_band, rpower_band = extract_features_signal(epochs_band.copy(), ref_band,config.events_intv, config.new_freq, base_power) 
    
    #Get statistics (for train data)        
    get_channels_stats(erds_band.T,y_band, config.events_id, ch_names)
    get_channels_stats(rpower_band.T,y_band, config.events_id, ch_names)
       
    model_fullrf,  acc_fullrf = train_randomforest(erds_band.T, y_band, config.events_id, importance=True, ch_names = ch_names)
 

    
