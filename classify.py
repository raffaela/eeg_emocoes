# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:44:25 2021

@author: Raffaela
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

#erds_all = erds.reshape([erds_array_all.shape[2],erds_array_all.shape[0], erds_array_all.shape[1]])
def train_knn(erds, y, events_dict):
    k_vizinhos = 10
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

    predicted  = model.predict(erds_test).astype(int)
    # for event_name, event_id in events_dict.items():
    #     pred_event = np.where(predicted==event_id)[0]
    #     true_event = y_test[pred_event].astype(int)
    #     correct = np.where((true_event==event_id)==True)[0]
    #     acc_event = correct.size/pred_event.size
    #     print("Test acc {}: {}".format(event_name, acc_event))
    print(classification_report(y_test,predicted, [0,1,2] ))
    
def train_randomforest(erds_test, y_test, events_dict):
    model = RandomForestClassifier(max_depth=4, random_state=0)
    model.fit(erds_test, y_test)
    for event_name, event_id in events_dict.items():
        ind_obs_event = np.where(y_test==event_id)[0]
        erds_event_test =  erds_test[ind_obs_event,:]
        predicted = model.predict(erds_event_test) # 0:Overcast, 2:Mild
        acc_evento = len(np.where(np.array(predicted)==event_id)[0])/len(predicted)
        print("Train acc {}: {}".format(event_name, acc_evento))
    return model

def test_randomforest(erds_test, y_test,model, events_dict):
    predicted  = model.predict(erds_test).astype(int)
    print(classification_report(y_test,predicted, [0,1,2] ))
# ind_0 = np.where(np.abs(epochs.times-config.lims_time_event[0])<1e-5)[0][0]
# ind_1 = np.where(np.abs(epochs.times-config.lims_time_event[1])<1e-5)[0][0]
# lims_ind_event = [ind_0, ind_1]