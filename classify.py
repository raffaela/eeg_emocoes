# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:44:25 2021

@author: Raffaela
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#erds_all = erds.reshape([erds_array_all.shape[2],erds_array_all.shape[0], erds_array_all.shape[1]])
def train_knn(erds, y, events_dict):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(erds, y)
    for event_name, event_id in events_dict.items():
        ind_obs_event = np.where(y==event_id)[0]
        erds_event_test =  erds[ind_obs_event,:]
        predicted = model.predict(erds_event_test) # 0:Overcast, 2:Mild
        acc_evento = len(np.where(np.array(predicted)==event_id)[0])/len(predicted)
        print("Train acc {}: {}".format(event_name, acc_evento))
    return model

def test_knn(erds_test, y_test, model, events_dict):
    predicted  = model.predict(erds_test)
    acc = len(np.where(predicted == y_test)==1)/len(predicted)
    print("Test acc: {}".format(acc))
    
    
    
# ind_0 = np.where(np.abs(epochs.times-config.lims_time_event[0])<1e-5)[0][0]
# ind_1 = np.where(np.abs(epochs.times-config.lims_time_event[1])<1e-5)[0][0]
# lims_ind_event = [ind_0, ind_1]