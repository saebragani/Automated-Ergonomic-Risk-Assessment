################################################## Load Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import signal
import pywt

import os
import time
import datetime
import random
import h5py
import pickle

import tensorflow as tf
keras = tf.keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import History

from sklearn.preprocessing import LabelEncoder

##################################################
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

##################################################
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
	##################################################
	sepAccDict, sepAnnotDict = readData(accDir='./Data/50_subs/Acc Data/separate/', annotFile='./Data/50_subs/Annotation Data/separate.xlsx')
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

	################################################## Windowing
	winLen = 320
	window_dict = {}
	label_dict = {}
	for key in filt_noNA_dict.keys():
	    window_list = []
	    labels=[]
	    for g1, df1 in filt_noNA_dict[key].groupby('label'):
	        for g2, df2 in df1.groupby(np.arange(df1.shape[0]) // winLen):
	            if df2.shape[0]==winLen:
	                window_list.append(df2.drop(columns=['Mag', 'label']).values)
	                labels.append(g1)
	    window_dict[key] = np.array(window_list)
	    label_dict[key] = labels

	################################################## Split train-test
	random.seed(2021)
	percentTrain = 80
	all_subs = list(label_dict.keys())
	train_subs = random.sample(all_subs, k=int(len(all_subs)*(percentTrain/100)))
	test_subs = list(set(all_subs) - set(train_subs))

	with open('test_subs.pickle', 'wb') as outfile:
		pickle.dump(test_subs, outfile)

	train_array_list = [window_dict[key] for key in train_subs]
	test_array_list = [window_dict[key] for key in test_subs]
	train_np = np.concatenate(train_array_list)
	test_np = np.concatenate(test_array_list)

	train_label__ = [label_dict[key] for key in train_subs]
	train_label_ = [item for sublist in train_label__ for item in sublist]
	train_label = [item.replace('1', '').replace('2', '') for item in train_label_]
	test_label__ = [label_dict[key] for key in test_subs]
	test_label_ = [item for sublist in test_label__ for item in sublist]
	test_label = [item.replace('1', '').replace('2', '') for item in test_label_]


	################################################## read and prepare input data for tf
	SR = 1/32
	scales = range(1,200)
	waveletname = 'morl'

	x_train = np.zeros((train_np.shape[0], max(scales), winLen, train_np.shape[2]), np.float)
	for i in range(train_np.shape[0]):
		for j in range(train_np.shape[2]):
			sig = train_np[i,:,j]
			coeff, freq = pywt.cwt(sig, scales, waveletname, SR)
			x_train[i, :, :, j] = coeff
	scaler = np.max(x_train)
	with open('scaler.pickle', 'wb') as outfile:
		pickle.dump(scaler, outfile)
	x_train = x_train/scaler

	x_test = np.zeros((test_np.shape[0], max(scales), winLen, test_np.shape[2]), np.float)
	for i in range(test_np.shape[0]):
		for j in range(test_np.shape[2]):
			sig = test_np[i,:,j]
			coeff, freq = pywt.cwt(sig, scales, waveletname, SR)
			x_test[i, :, :, j] = coeff
	x_test = x_test/scaler

	y_train_integer_encoded = LabelEncoder().fit_transform(train_label)
	y_test_integer_encoded = LabelEncoder().fit_transform(test_label)
	y_train = keras.utils.to_categorical(y_train_integer_encoded, 10)
	y_test = keras.utils.to_categorical(y_test_integer_encoded, 10)

	################################################## construct the NN
	input_shape = (x_train.shape[1:])
	num_classes = 10
	batch_size = 20
	epochs = 10

	##############
	model = models.Sequential()
	model.add(layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
	model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(layers.Conv2D(64, (5, 5), activation='relu'))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))

	model.add(layers.Flatten())
	model.add(layers.Dense(100, activation='relu'))
	model.add(layers.Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])#AUC Precision Recall

	history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

	model.save('./tf_model/')

	with open('keras_history.pickle', 'wb') as outfile:
		pickle.dump(history.history, outfile)
