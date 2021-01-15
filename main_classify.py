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


def get_average_train_acc(train_acc_knn,train_acc_rf):
    dict_train_knn = {}
    dict_train_rf = {}
    divisor = len(train_acc_knn)
    dict_train_knn['neutro']= sum(item['neutro'] for item in train_acc_knn)/divisor
    dict_train_knn['ternura']=sum(item['ternura'] for item in train_acc_knn)/divisor
    dict_train_knn['angustia']=sum(item['angustia'] for item in train_acc_knn)/divisor
    dict_train_rf['neutro']=sum(item['neutro'] for item in train_acc_rf)/divisor
    dict_train_rf['ternura']=sum(item['ternura'] for item in train_acc_rf)/divisor
    dict_train_rf['angustia']=sum(item['angustia'] for item in train_acc_rf)/divisor
    return dict_train_knn, dict_train_rf


def run_kfold(folds, epochs, array_split, pos, band=config.classify_band, freq_bands=config.freq_bands, plot_flag = False):
    fs = epochs.info['sfreq']
    ch_names = epochs.info['ch_names']
    time_vec = epochs.times
    train_acc_knn =[]
    train_acc_rf =[]
    reports_knn = []
    cf_matrices_knn = np.array([])
    reports_rf = []
    cf_matrices_rf = np.array([])
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

        # extract base power for relative power calc
        base_power = extract_base_power(epochs_train.copy(), config.base_freq, config.events_intv)
       
        # extract ERDS and relative power from train epochs
        lims_freq = freq_bands[band]
        epochs_train.filter(lims_freq[0], lims_freq[1],verbose=False)
        ref_train = extract_ref(epochs_train['neutro'].copy(), time_vec, config.lims_time_neutro, fs, config.new_freq) 
        erds_train,y_train, rpower_train = extract_features_signal(epochs_train.copy(), ref_train,config.events_intv,config.new_freq, base_power) 
        #Transpose to enter classifier
        erds_train = erds_train.T
        
        # train model
        model_knn, scaler_knn, acc_knn = train_knn(erds_train, y_train, config.events_id)
        model_rf, acc_rf = train_randomforest(erds_train, y_train, config.events_id)
        
        train_acc_knn.append(acc_knn)
        train_acc_rf.append(acc_rf)

        # extract ERDS from test epochs
        lims_freq = freq_bands[band]
        epochs_test.filter(lims_freq[0], lims_freq[1],verbose=False)
        erds_test,y_test, rpower_test = extract_features_signal(epochs_test.copy(), ref_train, config.events_intv,config.new_freq, base_power)
        # Transpose to enter classifier
        erds_test = erds_test.T    
        
        # test model
        report_knn, cf_matrix_knn = test_model(erds_test, y_test, model_knn, config.events_id, scaler_knn)
        report_rf, cf_matrix_rf = test_model(erds_test, y_test, model_rf, config.events_id)
        # append reports for averaging
        reports_knn.append(report_knn)
        reports_rf.append(report_rf)
        
        if cf_matrices_knn.size==0:
            cf_matrices_knn = cf_matrix_knn
        else: cf_matrices_knn += cf_matrix_knn
        if cf_matrices_rf.size==0:
            cf_matrices_rf = cf_matrix_rf
        else: cf_matrices_rf += cf_matrix_rf
        
    report_knn_av = report_average(reports_knn)
    report_rf_av = report_average(reports_rf)
    
    dict_train_knn, dict_train_acc = get_average_train_acc(train_acc_knn,train_acc_rf)
    
    return report_knn_av, report_rf_av, cf_matrices_knn, cf_matrices_rf, dict_train_knn, dict_train_acc
        

if __name__ == '__main__':

    raw, time_events = read_eeginfo(config.eeg_filepath, config.marc_filepath)
    raw_filt = eeg_preprocess(raw)
    array_events = get_events_sequence(config.seq_events, config.num_sections , time_events)
    raw_filt.pick_channels(config.channels_pick)
    #raw_filt.drop_channels(channels_drop)
    ch_names = raw_filt.info['ch_names']
    # transform data to Epochs
    epochs = mne.Epochs(raw_filt.copy(), array_events, event_id=config.events_id, 
    tmin=config.tmin_epoch, tmax=config.tmax_epoch, preload=True, baseline=None)
    
    #kfold cross-validation
    k = 8
    epochs_ind = epochs.selection
    epochs_ind = deque(epochs_ind)
    ep_start = 4
    epochs_ind.rotate(ep_start)
    epochs_ind = np.array(epochs_ind)
    array_split = np.split(epochs_ind, k)
    folds = np.arange(k)
    report_knn_av, report_rf_av, cf_matrices_knn, cf_matrices_rf, dict_train_knn, dict_train_rf = run_kfold(folds, epochs, array_split, raw_filt.info, plot_flag = False)

    cf_fig_knn = print_confusion_matrix(cf_matrices_knn, ['neutro','ternura','angustia'])
    cf_fig_rf = print_confusion_matrix(cf_matrices_rf, ['neutro','ternura','angustia'])
    
    # record results
    fig_file_knn = os.path.join(config.results_folder,"{}_cf_knn".format(config.results_file))
    cf_fig_knn.savefig(fig_file_knn)
    fig_file_rf = os.path.join(config.results_folder,"{}_cf_rf".format(config.results_file))
    cf_fig_rf.savefig(fig_file_rf)
    
    xlsx_file = os.path.join(config.results_folder,"{}.xlsx".format(config.results_file))
    report_knn_av.to_excel(xlsx_file , sheet_name='report_knn')
    append_df_to_excel(xlsx_file , report_rf_av, sheet_name='report_rf')
    
    append_df_to_excel(xlsx_file ,pd.DataFrame(cf_matrices_knn), sheet_name='cf_matrix_knn')
    append_df_to_excel(xlsx_file , pd.DataFrame(cf_matrices_rf), sheet_name='cf_matrix_rf')
    
    df_train_acc_knn = pd.DataFrame.from_dict(dict_train_knn, orient='index')
    df_train_acc_rf = pd.DataFrame.from_dict(dict_train_rf, orient='index')
    
    append_df_to_excel(xlsx_file , df_train_acc_knn, sheet_name='train_acc_knn')
    append_df_to_excel(xlsx_file , df_train_acc_rf, sheet_name='train_acc_rf')
    
    