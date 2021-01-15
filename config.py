# Authors: Raffaela Cunha <raffaelacunha@gmail.com>

import numpy as np

seq_events = np.array([[0,1,0,1,0,1,0,1,0,0,2,0,2,0,2,0,2,0]],dtype='int32')
num_sections = 4
tmin_epoch = -10
tmax_epoch = 45
events_id = {'neutro':0,'ternura':1,'angustia':2}
events_intv = np.array([[0,12],[0,45],[0,45]])
base_freq = [4,45]
train_band= {'beta_gama':[13,42]}
#train_band= {'alpha_theta':[4,13]}
freq_bands = {'delta':[0.5,4],'theta':[4,8],'alpha':[8,13],'beta':[13,30],'gama':[30,45], 'beta_gama':[13,42]}
#ref_time_erds = [-10, 0 ]
lims_time_neutro = [2, 12]
lims_time_event = [8, 36]
new_freq = 2 
inv_events_id = {v: k for k, v in events_id.items()}
classify_band = "beta_gama"
results_folder = "results"
results_file = "classification_results" 
eeg_filepath ="dados\\SUBJ002\\SUBJ0002.vhdr"
marc_filepath = "dados\\SUBJ002\\Marcsubject2.mat"
channels_pick =['Fp2','O2','T7','T8','Oz','FC6','CP5','TP9','TP10','FC4','P5','TP7','TP8','FT9','FT10','Fpz']
#channels_drop = ['Fpz','FT7','C6','FC5','T8','FC6','Fp1','Fp2','F4','C3','C4','FT9', 'AF7','AF8','Fz','F2','F1']