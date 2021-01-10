# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 02:01:56 2021

@author: Raffaela
"""

import numpy as np
import mne
from utils import read_eeginfo, eeg_preprocess, get_events_sequence
from utils import extract_erds_epochs, extract_ref
from utils import extract_erds_signal, plot_erds_topomap, get_channels_stats
import config
from classify import train_knn, test_knn, train_randomforest, test_randomforest
from collections import deque
from sklearn.decomposition import PCA 



def extract_ERDS_signal(epochs_test, ref_train):#ref_train is an array (for a given band)
    fs = epochs.info['sfreq']
    selection = epochs_test.selection
    y_test = None
    erds_test = None
    for ind_ep in range(selection.size):
        ep_test = epochs_test[ind_ep]
        ep_event_id = ep_test.event_id
        ep_event_id = ep_event_id[next(iter(ep_event_id))]
        ep_test = ep_test.crop(config.events_intv[ep_event_id,0], config.events_intv[ep_event_id,1])
        signal_ep = ep_test.get_data()
        signal_ep = signal_ep.reshape([signal_ep.shape[1], signal_ep.shape[2]])
        time_vec_ep = np.arange(0, signal_ep.size/fs,signal_ep.size/(signal_ep.size*fs))
        erds_ep= extract_erds_signal(signal_ep, time_vec_ep, ref_train, fs, config.new_freq)
        y_ep = np.repeat(ep_event_id, erds_ep.shape[1])
            
        if erds_test is not None:
            erds_test = np.concatenate([erds_test, erds_ep], axis=1)
            y_test = np.concatenate([y_test, y_ep])
        else: 
            erds_test = erds_ep
            y_test = y_ep
    return erds_test, y_test

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

def run_kfold(folds, epochs, array_split, pos, freq_bands=config.train_band, plot_flag = False):
    fs = epochs.info['sfreq']
    ch_names = epochs.info['ch_names']
    band = "beta_gama" 
    time_vec = epochs.times
    for ep_test in folds:
        print("Fold {}".format(ep_test)) 
        folds_train = np.delete(folds.copy(), ep_test)
        folds_test = folds[ep_test]
        array_train = None
        for f in folds_train:
            if array_train is not None:
                array_train = np.concatenate([array_train, array_split[f]])
            else:
                array_train = array_split[f]
        
        array_test = array_split[folds_test]       
        epochs_train = epochs[array_train]
        epochs_test = epochs[array_test]

        # extract ERDS from test epochs
        lims_freq = freq_bands[band]
        epochs_train.filter(lims_freq[0], lims_freq[1],verbose=False)
        ref_train = extract_ref(epochs_train['neutro'].copy(), time_vec, config.lims_time_neutro, fs, config.new_freq) 
        erds_train,y_train = extract_ERDS_signal(epochs_train.copy(), ref_train)

        
        # plot mean ERDS topomaps 
        if plot_flag == True:
            plot_erds_topomap(erds_train,y_train, pos, band, config.events_id, -0.5, 0.5, ch_names)
        
        #Apply PCA
        erds_train = erds_train.T
        
        # train model
        model = train_knn(erds_train, y_train, config.events_id)
        model2 = train_randomforest(erds_train, y_train, config.events_id)

        # extract ERDS from test epochs
        lims_freq = freq_bands[band]
        #epochs_test.filter(lims_freq[0], lims_freq[1], phase="zero-double", method = "iir", iir_params= dict(order=4, ftype='cheby1', rp=0.5), verbose=False)
        epochs_test.filter(lims_freq[0], lims_freq[1],verbose=False)
        
        erds_test,y_test = extract_ERDS_signal(epochs_test.copy(), ref_train)


        erds_test = erds_test.T    

        #Get statistics (from train)        
        get_channels_stats(erds_train,y_train, config.events_id, ch_names)
        
        # test model
        acc_test = test_knn(erds_test, y_test, model, config.events_id)
        acc_test2 = test_knn(erds_test, y_test, model2, config.events_id)

        

if __name__ == '__main__':

    # read files 
    erds_file = "ERDS.npz"
    y_file = "y_true.npz"
    ref_file = "ref.npz"
    eeg_filepath ="dados\\SUBJ002\\SUBJ0002.vhdr"
    marc_filepath = "dados\\SUBJ002\\Marcsubject2.mat"
    raw, time_events = read_eeginfo(eeg_filepath, marc_filepath)
    raw_filt = eeg_preprocess(raw)
    array_events = get_events_sequence(config.seq_events, config.num_sections , time_events)
    #channels_train = ['Oz','O1','O2']
    channels_drop = ['F7','F8','FC2','Pz','AF4','F6','AF7','AF8','FT7']
    #raw_filt.pick_channels(channels_train)
    raw_filt.drop_channels(channels_drop)
    ch_names = raw_filt.info['ch_names']
    # transform data to Epochs
    epochs = mne.Epochs(raw_filt.copy(), array_events, event_id=config.events_id, 
    tmin=config.tmin_epoch, tmax=config.tmax_epoch, preload=True, baseline=None)
    
    # # Exploratory Data Analysis
    # erds_dict, y_dict, ref_dict = extract_ERDS(epochs.copy())
    # band = 'gama'
    # get_channels_stats(erds_dict[band].T,y_dict[band], config.events_id, ch_names)
    # np.savez(erds_file, **erds_dict)
    # np.savez(y_file,**y_dict)
    # np.savez(ref_file,**ref_dict)
    # np.savez("epochs.npz", epochs)
    # ch_names = epochs.info['ch_names']
    #for band, lims in config.freq_bands.items(): # frequency band loop
    #plot_erds_topomap(erds_dict[band],y_dict[band], raw_filt.info, band, config.events_id, -0.5, 0.5, ch_names)
    
    #kfold cross-validation
    k = 8
    epochs_ind = epochs.selection
    epochs_ind = deque(epochs_ind)
    ep_start = 4
    epochs_ind.rotate(ep_start)
    epochs_ind = np.array(epochs_ind)
    array_split = np.split(epochs_ind, k)
    folds = np.arange(k)
    run_kfold(folds, epochs, array_split, raw_filt.info, plot_flag = False)


