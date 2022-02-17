import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from scipy import signal
import pywt

import os
import time
import datetime
import random
import h5py
import pickle

from ggs import *
import ipywidgets as widgets

from platform import python_version
print(python_version())




def readData(accDir, annotFile):
    files = os.listdir(accDir)
    files_csv = [f for f in files if f[-3:] == 'csv']
    empatica_dict = dict()
    for f in files_csv:
        data = np.genfromtxt(accDir+f, delimiter=',') # creates numpy array for each Empatica acc csv file
        key = int(float(f.strip("ACC.csv")))
        empatica_dict[key] = data
    tmp = pd.read_excel(annotFile, sheet_name=None)
    annot_dict = dict(zip(tmp.keys(), [i.dropna() for i in tmp.values()])) # Remove the rows with NaN values (some with ladder 2 missing)
    return empatica_dict, annot_dict

def getLabeledDict(empatica_dict, annot_dict, subject_ids):
    labeled_dict = {}; taskInd_dict = {}
    for id in subject_ids:
        start_time = int(empatica_dict[id][0,0])
        acc = empatica_dict[id][2:,:]
        label = list(map(lambda i: i.replace("_end", "").replace("_start", ""), annot_dict['P'+ str(id)].taskName.tolist()))
        task_time= list(map(lambda i: time.mktime(datetime.datetime.strptime(i[:6] + '20' + i[6:], "%m/%d/%Y %H:%M:%S").timetuple()),
                            annot_dict['P'+ str(id)].startTime_global.tolist()))
        task_ind = [int(x - start_time)*SR for x in task_time]
        taskInd_dict[id] = task_ind
        label_tmp = np.empty(acc.shape[0], dtype=object)
        for i, (j, k) in enumerate(zip(task_ind[0::2], task_ind[1::2])):
            tmpInd = 2*i
            label_tmp[j:k] = label[tmpInd]
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))[:,None]
        accel = np.hstack((acc, acc_mag))
        labeled_dict[id] = pd.DataFrame(np.hstack((accel, label_tmp.reshape(label_tmp.shape[0],1))), columns=['X', 'Y', 'Z', 'Mag', 'label'])
    return labeled_dict, taskInd_dict




if __name__ == '__main__':
    sepAccDict, sepAnnotDict = readData(accDir='./Data/Acc Data/', annotFile='./Data/Annotation Data/separate.xlsx')
    SR=int(sepAccDict[8][1,0])

    sepSubIDs = list(range(8,45))
    sepLabeledDict_, sepTaskIndDict = getLabeledDict(sepAccDict, sepAnnotDict, sepSubIDs)


    # Apply Filter on All Subjects
    n=4; fc=2; w=fc/(SR/2)
    b, a = signal.butter(n, w, 'low')
    sepLabeledDict_filtered = dict(map(lambda key: (key, signal.filtfilt(b, a, x=sepLabeledDict_[key].drop(columns='label'), axis=0)), sepLabeledDict_.keys()))
    # back to DF and add label
    sepLabeledDict_filtered_dfs = dict(map(lambda key: (
                                                            key, pd.DataFrame(sepLabeledDict_filtered[key],columns=['X', 'Y', 'Z', 'Mag']).assign(label=sepLabeledDict_[key].label)
                                                        ), sepLabeledDict_filtered.keys()))
    # Remove data without label
    filt_noNA_dict = dict(map(lambda key: (key, sepLabeledDict_filtered_dfs[key].dropna()), sepLabeledDict_filtered_dfs.keys()))



    start_time = time.time()
    # tasks = ['sit', 'stand', 'walk', 'hoist', 'lift', 'ladder1']
    tasks = ['sit', 'stand', 'walk', 'hoist', 'lift', 'push', 'type', 'ladder1', 'ladder2', 'electricPanel', 'overhead']
    seg_dict = {}

    with open('test_subs.pickle', 'rb') as infile:
        test_subs = pickle.load(infile)

    for sub in test_subs:
        sig__ = filt_noNA_dict[sub]
        sig_ = sig__[sig__.label.isin(tasks)]

        sig_.loc[sig_.label.isin(['ladder1', 'ladder2']), 'label'] = 'ladder'
        
        groups = [df for _, df in sig_.groupby('label')]
        random.seed(sub)
        random.shuffle(groups)
        sig_ = pd.concat(groups).reset_index(drop=True)

        sig = sig_.values[:,:3].astype(np.float).T
        bps, objectives = GGS(sig, Kmax=20, lamb=1e4)
        seg_dict[sub] = bps

    with open('segments_RandomTaskOrder.pickle', 'wb') as outfile:
        pickle.dump(seg_dict, outfile)

    print('Elapsed time = {}'.format(time.time() - start_time))
