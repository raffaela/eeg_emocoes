# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:44:25 2021

@author: Raffaela
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

#erds_all = erds.reshape([erds_array_all.shape[2],erds_array_all.shape[0], erds_array_all.shape[1]])
def train_knn(erds, y, events_dict):
    k_vizinhos = 10
    model = KNeighborsClassifier(n_neighbors=k_vizinhos)
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
    labels = [0,1,2]
    display_labels = ['neutro','ternura','angustia']
    print(classification_report(y_test,predicted, labels=labels, target_names = display_labels))
    plot_confusion_matrix(model, erds_test, y_test, labels=labels, display_labels = display_labels)  
    
def train_randomforest(erds_test, y_test, events_dict):
    model = RandomForestClassifier(random_state=0)
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
    labels = [0,1,2]
    display_labels = ['neutro','ternura','angustia']
    print(classification_report(y_test,predicted, labels=labels, target_names = display_labels ))
    plot_confusion_matrix(model, erds_test, y_test, labels=labels, display_labels = display_labels)  
