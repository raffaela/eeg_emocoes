# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 02:01:56 2021

@author: Raffaela
"""

import numpy as np
import mne
from utils import read_eeginfo, eeg_preprocess, get_events_sequence
from utils import get_sensor_positions, extract_erds_epochs, extract_ref, plot_erds_topomap
import config
from classify import train_knn, test_knn




def extract_ERDS(epochs):
    # extract ERDS 
    fs = epochs.info['sfreq']
    ind_0 = np.where(np.abs(epochs.times-config.lims_time_neutro[0])<1e-5)[0][0]
    ind_1 = np.where(np.abs(epochs.times-config.lims_time_neutro[1])<1e-5)[0][0]
    lims_ind_neutro = [ind_0, ind_1]
    erds_band = None
    erds_dict = {}
    y_dict = {}
    for band, lims in config.freq_bands.items(): # frequency band loop
        erds_band = None
        y_band =np.array([])
        epochs.filter(lims[0],lims[1]);
        ref = extract_ref(epochs['neutro'], lims_ind_neutro) 
        for event_type, ind_event in config.events_id.items(): # event type loop
            epochs_event = epochs[event_type]
            if event_type=='neutro':
                time_ind_event = config.lims_time_neutro # indices para intervalo de tempo a considerar na extracao do erds para neutro
            else: time_ind_event = config.lims_time_event # indices para intervalo de tempo a considerar na extracao do erds para angustia e ternura   
            erds = extract_erds_epochs(epochs_event,time_vec,ref,fs,config.new_freq, time_ind_event)
            if y_band is not None:
                y_band = np.concatenate([y_band, np.repeat(ind_event, erds.shape[1])], axis=0)
            else: y_band =  np.repeat(ind_event, erds.shape[1])
            #print("erds shape",erds.shape)
            if erds_band is not None:
                erds_band = np.concatenate([erds_band, erds], axis=1)
            else: erds_band = erds
    
    erds_dict[band] = erds_band
    y_dict[band] = y_band
    return erds_dict, y_dict




if __name__ == '__main__':
    
    # read files 
    eeg_filepath ="dados\\SUBJ002\\SUBJ0002.vhdr"
    marc_filepath = "dados\\SUBJ002\\Marcsubject2.mat"
    raw, time_events = read_eeginfo(eeg_filepath, marc_filepath)
    raw_filt = eeg_preprocess(raw)
    array_events = get_events_sequence(config.seq_events, config.num_sections , time_events)
    
    # transform data to Epochs
    epochs = mne.Epochs(raw_filt.copy(), array_events, event_id=config.events_id, 
    tmin=config.tmin_epoch, tmax=config.tmax_epoch, preload=True, baseline=None)
    time_vec = epochs.times
    pos = get_sensor_positions(raw.info)
    
    # Exploratory Data Analysis
    erds_dict, y_dict = extract_ERDS(epochs)
    for band, lims in config.freq_bands.items(): # frequency band loop
        plot_erds_topomap(erds_dict[band],y_dict[band], pos, band, config.events_id, -0.5, 0.5)
    #kfold cross-validation
    k = 8
    plot_flag = True
    epochs_ind = epochs.selection
    epochs_split = np.split(epochs_ind, k)
    folds = np.arange(k)
    for ep_test in folds:
        folds_train = np.delete(folds, ep_test)
        folds_test = folds[ep_test]
        epochs_train = None
        for f in folds_train:
            if epochs_train is not None:
                epochs_train = np.concateate(epochs_train, epochs_split[f])
            else:
                epochs_train = epochs_split[f]
                
        epochs_test = epochs_split[folds_test]
    
        # extract ERDS from train epochs
        erds_train, y_train = extract_ERDS(epochs_train)
        
        # plot mean ERDS topomaps 
        band = "alpha" # este teste será feito todo baseado na banda alpha
        #for band, lims in config.freq_bands.items(): # frequency band loop
        if plot_flag == True:
            plot_erds_topomap(erds_train[band],y_train[band], pos, band, config.events_id, -0.5, 0.5)
        model = train_knn(erds_train[band].T, y_train[band], config.events_id)
        
        # extract ERDS from test epochs
        erds_test, y_test = extract_ERDS(epochs_test)
        acc_test = test_knn(erds_test, y_test, model)
    # treinamento do modelo KNN
    
    
    # k-fold cross validation
    #k = 8
    #train_epochs = 

