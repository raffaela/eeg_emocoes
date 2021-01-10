# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 02:01:56 2021

@author: Raffaela
"""

import numpy as np
import mne
from utils import read_eeginfo, eeg_preprocess, get_events_sequence
from utils import extract_erds_epochs, extract_ref
from utils import extract_erds_signal, plot_erds_topomap
import config
from classify import train_knn, test_knn, train_randomforest, test_randomforest
from collections import deque
from sklearn.decomposition import PCA 



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
        ref = extract_ref(epochs_band['neutro'].copy(), lims_ind_neutro, time_vec, fs, config.new_freq) 
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
            #remove bug do neutro 
            # TODO: fazer o processamento do neutro em separado
            # if event_type == "neutro":
            #     erds_band = np.zeros(erds_band.shape)
        ref_dict[band] = ref    
        erds_dict[band] = erds_band
        y_dict[band] = y_band
    return erds_dict, y_dict, ref_dict

def run_kfold(folds, epochs, array_split, pos, freq_bands=config.train_band, plot_flag = False):
    fs = epochs.info['sfreq']
    for ep_test in folds:
        print("Fold {}".format(ep_test)) 
        folds_train = np.delete(folds, ep_test)
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

        # extract ERDS from train epochs
        erds_train, y_train, ref_train = extract_ERDS(epochs_train, freq_bands=config.train_band)
        
        # plot mean ERDS topomaps 
        band = "beta_gama" # este teste será feito todo baseado na banda alpha
        #for band, lims in config.freq_bands.items(): # frequency band loop
        if plot_flag == True:
            plot_erds_topomap(erds_train[band],y_train[band], pos, band, config.events_id, -0.5, 0.5)
        
        #Apply PCA
        pca = PCA(n_components=7)
        erds_t = erds_train[band].T
        pca.fit(erds_t)
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)
        erds_t = pca.transform(erds_t)
        
        # train model
        model = train_knn(erds_t, y_train[band], config.events_id)
        model2 = train_randomforest(erds_t, y_train[band], config.events_id)
        # signal_test = np.moveaxis(epochs_test.get_data(), 0, -1)
        # signal_test = signal_test.reshape([signal_test.shape[0],-1])
        
        # extract ERDS from test epochs
        lims_freq = freq_bands[band]
        #epochs_test.filter(lims_freq[0], lims_freq[1], phase="zero-double", method = "iir", iir_params= dict(order=4, ftype='cheby1', rp=0.5), verbose=False)
        epochs_test.filter(lims_freq[0], lims_freq[1],verbose=False)
        
        y_test = None
        erds_test = None
        for ind_ep in range(array_test.size):
            ep_test = epochs_test[ind_ep]
            ep_event_id = ep_test.event_id
            ep_event_id = ep_event_id[next(iter(ep_event_id))]
            ep_test = ep_test.crop(config.events_intv[ep_event_id,0], config.events_intv[ep_event_id,1])
            signal_ep = ep_test.get_data()
            signal_ep = signal_ep.reshape([signal_ep.shape[1], signal_ep.shape[2]])
            time_vec_ep = np.arange(0, signal_ep.size/fs,signal_ep.size/(signal_ep.size*fs))
            erds_ep= extract_erds_signal(signal_ep, time_vec_ep, ref_train[band], fs, config.new_freq)
            y_ep = np.repeat(ep_event_id, erds_ep.shape[1])
            
            if erds_test is not None:
                erds_test = np.concatenate([erds_test, erds_ep], axis=1)
                y_test = np.concatenate([y_test, y_ep])
            else: 
                erds_test = erds_ep
                y_test = y_ep
        
        #Apply PCA
        pca = PCA(n_components=7)
        erds_t2 = erds_test.T
        pca.fit(erds_t2)
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)
        erds_t2 = pca.transform(erds_t2)       
        
        # test model
        acc_test = test_knn(erds_t2, y_test, model, config.events_id)
        acc_test2 = test_knn(erds_t2, y_test, model2, config.events_id)
        
        

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
    
    # transform data to Epochs
    epochs = mne.Epochs(raw_filt.copy(), array_events, event_id=config.events_id, 
    tmin=config.tmin_epoch, tmax=config.tmax_epoch, preload=True, baseline=None)
    #fs = epochs.info['sfreq']
    # montage = raw.get_montage()
    # pos = get_sensor_positions(montage)
    
    # Exploratory Data Analysis
    erds_dict, y_dict, ref_dict = extract_ERDS(epochs.copy())
    np.savez(erds_file, **erds_dict)
    np.savez(y_file,**y_dict)
    np.savez(ref_file,**ref_dict)
    np.savez("epochs.npz", epochs)
    ch_names = epochs.info['ch_names']
    for band, lims in config.freq_bands.items(): # frequency band loop
          plot_erds_topomap(erds_dict[band],y_dict[band], raw.info, band, config.events_id, -0.5, 0.5, ch_names)
    #kfold cross-validation
    k = 8
    epochs_ind = epochs.selection
    epochs_ind = deque(epochs_ind)
    ep_start = 4
    epochs_ind.rotate(ep_start)
    epochs_ind = np.array(epochs_ind)
    array_split = np.split(epochs_ind, k)
    folds = np.arange(k)

    #run_kfold(folds, epochs, array_split, raw.info)

    

        
        
        
    # treinamento do modelo KNN
    
    
    # k-fold cross validation
    #k = 8
    #train_epochs = 

