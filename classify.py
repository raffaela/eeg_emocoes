# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:44:25 2021

@author: Raffaela
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from sklearn import preprocessing as prep
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd
import seaborn as sns

#erds_all = erds.reshape([erds_array_all.shape[2],erds_array_all.shape[0], erds_array_all.shape[1]])
def train_knn(erds, y, events_dict):
    erds_scaled, scaler = scale_data(erds)
    k_vizinhos = 10
    model = KNeighborsClassifier(n_neighbors=k_vizinhos,metric="manhattan")
    model.fit(erds_scaled, y)
    acc = {}
    for event_name, event_id in events_dict.items():
        ind_obs_event = np.where(y==event_id)[0]
        erds_event_test =  erds[ind_obs_event,:]
        predicted = model.predict(erds_event_test) 
        acc_evento = len(np.where(np.array(predicted)==event_id)[0])/len(predicted)
        print("Train acc {}: {}".format(event_name, acc_evento))
        acc[event_name]=acc_evento
    return model, scaler, acc

def train_randomforest(erds, y, events_dict):
    model = RandomForestClassifier(random_state=0)
    model.fit(erds, y)
    acc = {}
    for event_name, event_id in events_dict.items():
        ind_obs_event = np.where(y==event_id)[0]
        erds_event_test =  erds[ind_obs_event,:]
        predicted = model.predict(erds_event_test)
        acc_evento = len(np.where(np.array(predicted)==event_id)[0])/len(predicted)
        print("Train acc {}: {}".format(event_name, acc_evento))
        acc[event_name]=acc_evento
    return model, acc

def test_model(erds_test, y_test, model, events_dict, scaler=None):
    if scaler is not None:
        erds_test = scaler.transform(erds_test)
    predicted  = model.predict(erds_test).astype(int)
    labels = [0,1,2]
    display_labels = ['neutro','ternura','angustia']
    report = classification_report(y_test,predicted, labels=labels, target_names = display_labels)
    print(report)
    plot_confusion_matrix(model, erds_test, y_test, labels=labels, display_labels = display_labels)  
    cf_matrix = confusion_matrix(y_test,predicted, labels = labels)
    return report, cf_matrix





def report_average(reports):
    report_list = list()
    for report in reports:
        report = report.split('accuracy')[0]
        splited = [' '.join(x.split()) for x in report.split('\n\n')]
        header = [x for x in splited[0].split(' ')]
        data = np.array(splited[1].split(' ')).reshape(-1, len(header) + 1)
        data = np.delete(data, 0, 1).astype(float)
        avg_total = np.array([x for x in splited[2].split(' ')][3:]).astype(float).reshape(-1, len(header))
        df = pd.DataFrame(np.concatenate((data, avg_total)), columns=header)
        report_list.append(df)
        res = reduce(lambda x, y: x.add(y, fill_value=0), report_list) / len(report_list)
    return res


def scale_data(X_train):
    min_max_scaler = prep.RobustScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    return X_train_minmax, min_max_scaler

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    divisor = np.sum(df_cm,axis=1).to_numpy()
    divisor = divisor.reshape([df_cm.shape[0],1])

    try:
        heatmap = sns.heatmap(df_cm/divisor, annot=True, fmt=".0%")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig