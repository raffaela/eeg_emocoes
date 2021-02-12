# Authors: Raffaela Cunha <raffaelacunha@gmail.com>
import os
import numpy as np
from datetime import datetime as dt
import yaml

now = dt.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")

current_version = "0.0.1_beta"

seq_events = np.array([[0,1,0,1,0,1,0,1,0,0,2,0,2,0,2,0,2,0]],dtype='int32')
num_sections = 4
tmin_epoch = -10
tmax_epoch = 45
events_id = {'neutro':0,'ternura':1,'angustia':2}
inv_events_id = {v: k for k, v in events_id.items()}
events_intv = np.array([[0,12],[0,45],[0,45]])
base_freq = [4,45]
freq_bands = {'delta':[0.5,4],'theta':[4,8],'alpha':[8,13],'beta':[13,30],'gama':[30,45], 'beta_gama':[13,42]}
new_freq = 2 
results_folder = os.path.join("results",dt_string)
results_file = "classification_results" 
#lims_time_neutro = [2, 12]
#channels_drop = ['Fpz','FT7','C6','FC5','T8','FC6','Fp1','Fp2','F4','C3','C4','FT9', 'AF7','AF8','Fz','F2','F1']

config_path = os.path.dirname( __file__ ) #Pasta raiz do projeto
config_filepath = os.path.join(config_path, 'user_config.yaml')

with open(config_filepath) as f:
    user_config = yaml.load(f, Loader=yaml.FullLoader)
    
eeg_filepath = user_config['eeg_filepath']
marc_filepath = user_config['marc_filepath']
channels_pick = user_config['channels_pick']
classify_band = user_config['classify_band']
lims_time_neutro = user_config['lims_time_neutro']
lims_time_event = user_config['lims_time_event']
k_neighbours = user_config['k_neighbours']


