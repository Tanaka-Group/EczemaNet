#!/usr/bin/env python
# coding: utf-8

# # Model Training: Baseline Study - Ordinal VS Categorical

# In[ ]:


import sys
sys.path.append("..")
sys.path.append("../lib")

import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import itertools
from datetime import datetime

import matplotlib.pyplot as plt
import scikitplot as skplt
import scipy
from scipy import ndimage

from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score)
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from math import sqrt

import keras
from keras.models import model_from_json
from keras.preprocessing import image
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard
from keras import optimizers

# Confirm Keras sees the GPU
from keras import backend as K
assert len(K.tensorflow_backend._get_available_gpus()) > 0

from EczemaNet_helper import *


# ## Load Data

# In[ ]:


ECZEMANET_MODEL="eczemanet_models.EczemaNet_MobileNet"
PATH_TO_DATASET = "../data"
OUTPUT_PATH = "../output/Baseline_Categorical/"
BRANCHES_LIST = ['sassad_cra','sassad_dry','tiss_ery','tiss_exc','sassad_exu','sassad_lic','tiss_oed']
PATH_TO_CROPS_DATA = "<PATH_TO_CROPS_DATA>"


# In[ ]:

# Load data:
meta_data = pd.read_csv(os.path.join(PATH_TO_DATASET,"meta_data.csv"))

# Get cross validation IDs:
cvid_df = pd.read_csv(os.path.join(PATH_TO_DATASET,"patient_cvid.csv"))
print("Total number of unique cases: " + str(len(cvid_df)))


# In[ ]:


def label_transformer(inputs, k_lim=4):
    """ Label transformer

    Converts multiclass label (0,1,2,...) for k-classes into one-hot categorical representation

    # Arguments
        inputs: input labels.
            - e.g. [0,2,1,3,...]
    # Returns
        output labels in one-hot categorical form.
            - e.g.[[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1],...]

    """
    sampleLabels = [0,1,2,3]
    labels = np.unique(inputs)
    binarizedLabels = [];
    if all(labels[i] == sampleLabels[i] for i, v in enumerate(labels)):
        for idx, value in enumerate(inputs):
            if value == 0:
                binarizedLabels.append([1,0,0,0])
            elif value == 1:
                binarizedLabels.append([0,1,0,0])
            elif value == 2:
                binarizedLabels.append([0,0,1,0])
            elif value == 3:
                binarizedLabels.append([0,0,0,1])
            else:
                raise Exception("Error: Illegal Values!")
        return np.array(binarizedLabels)
    else:
        raise Exception("Issue with labels, not [0,1,2,3]! Cannot be converted to ordinary labels.")

def predict(x_data, meta_data, model, branchlist=BRANCHES_LIST):
    """ Make severity predictions and return outputs

    # Arguments
        x_data: input (image) data
        meta_data: meta data (with label information)
    # Returns
        Y_true: true label data
        Y_pred: predicted labels
        Y_proba: predicted probabilities

    """

    Y_true = pd.DataFrame()
    Y_proba = pd.DataFrame()
    Y_pred = pd.DataFrame()

    # 1. Make predictions
    y_crops_proba = model.predict(x_data)

    # 2. Combine crops into whole images
    meta_grouped = meta_data.groupby(["refno","visno"])
    for group_name, df_group in meta_grouped:
        (refno, visno) = group_name
        indexes = df_group.index.values

        y_whole_proba_categorical = {}
        y_whole_pred = {}

        # -- Average over crops and convert to categorical --
        for idx, branchname in enumerate(branchlist):
            y_whole_proba_categorical[branchname] = np.mean(y_crops_proba[idx][indexes],axis=0)
            y_whole_pred[branchname] = np.argmax(y_whole_proba_categorical[branchname])

        # -- Calculate combined scores --
        for score_name in ALL_COMBINED_KEYS:
            y_tmp_probas = []
            for idx, branch in enumerate(ALL_COMBINED_KEYS[score_name]):
                y_tmp_probas.append(y_whole_proba_categorical[branch])
            y_tmp_probas = np.stack(y_tmp_probas)
            y_whole_proba_categorical[score_name] = np.squeeze(conv_sum(input_probas=[y_tmp_probas], branchlist=ALL_COMBINED_KEYS[score_name]))
            custm = scipy.stats.rv_discrete(name='custm', values=(np.arange( 3*len(ALL_COMBINED_KEYS[score_name])+1 ), y_whole_proba_categorical[score_name]))
            y_whole_pred[score_name] = custm.mean()

        # -- Save everything --
        Y_true = Y_true.append(meta_data.loc[indexes[0]],ignore_index=True)
        Y_proba = Y_proba.append(y_whole_proba_categorical,ignore_index=True)
        Y_pred = Y_pred.append(y_whole_pred,ignore_index=True)

    for score_name in ALL_COMBINED_KEYS:
        Y_true[score_name] = np.sum( [ Y_true[key] for key in ALL_COMBINED_KEYS[score_name]], axis=0)

    return Y_true, Y_pred, Y_proba


# In[ ]:


stat_df = pd.DataFrame()
Y_FINAL_TURE = pd.DataFrame()
Y_FINAL_PRED = pd.DataFrame()
Y_FINAL_PROBA = pd.DataFrame()

for run in range(0,10):

    # ------------------------------------------------
    # Spliting indexes:
    # ------------------------------------------------

    train_refnos = cvid_df[cvid_df['cv_id'] != run]['refno']
    test_refnos = cvid_df[cvid_df['cv_id'] == run]['refno']

    meta_train = pd.DataFrame()
    meta_test = pd.DataFrame()

    for refno in train_refnos.values:
        meta_train = meta_train.append(meta_data[meta_data['refno'] == refno])
    for refno in test_refnos.values:
        meta_test = meta_test.append(meta_data[meta_data['refno'] == refno])

    # ------------------------------------------------
    # Preparing inputs/labels to the right format:
    # ------------------------------------------------
    print("Preparing inputs/labels...")

    y_train = {}
    y_test = {}
    classWeights = {}

    # Training set:
    for branch in BRANCHES_LIST:
        y_train[branch+"_output"] = label_transformer(meta_train[branch])
    x_train = load_images(meta_train['filepath'], image_size=224)
    meta_train = meta_train.reset_index()

    # Test set:
    for branch in BRANCHES_LIST:
        y_test[branch+"_output"] = label_transformer(meta_test[branch])
    x_test = load_images(meta_test['filepath'], image_size=224)
    meta_test = meta_test.reset_index()

    # Class weights (Balancing classes):
    for branch in BRANCHES_LIST:
        for idx in range(3):
            col = [row[idx] for row in y_train[branch+"_output"]]
            classWeights[branch+"_"+str(idx)+"_output"] = weight_ratio(col)
    print("Training/Test inputs & labels prepared.")

    # ------------------------------------------------
    # Model Training:
    # ------------------------------------------------
    module_name=ECZEMANET_MODEL
    today_str = datetime.today().strftime('%Y-%m-%d')
    tensorboard_log_dir =  os.path.join(OUTPUT_PATH,"tensorboard_log/"+ str(today_str) + "/run_" + str(run))
    model_fn = os.path.join(OUTPUT_PATH, str(today_str) + "_run-" + str(run) + "_model_weights.h5")
    model_arch_fn = os.path.join(OUTPUT_PATH, str(today_str) + "_run-" + str(run) + "_model_architecture.json")
    model_statsreport_fn = os.path.join(OUTPUT_PATH,"model_stats_report.txt")
    stat_df_fn = os.path.join(OUTPUT_PATH,"model_stats.pkl")
    statistics_fn = os.path.join(OUTPUT_PATH,"model_stats.bin")
    y_true_final_fn = os.path.join(OUTPUT_PATH,"y_true.csv")
    y_pred_final_fn = os.path.join(OUTPUT_PATH,"y_pred.csv")
    y_proba_final_fn = os.path.join(OUTPUT_PATH,"y_proba.csv")

    eps = 50 # (<=Maximum, default with EarlyStop)
    print("Model training begins, run " + str(run) + " with " + str(eps) + " total epoches...")
    print("Tensorboard log-dir: " + tensorboard_log_dir)
    (H, model) = train_model(
        eczemanetModule=module_name,
        x_data = (x_train, x_test),
        y_data = (y_train, y_test),
        classWeights = classWeights,
        branchlist = BRANCHES_LIST,
        verbose = True,
        batchsize = 32,
        catagories = 4,
        itr = eps,
        earlystopping = True,
        withGenerator = False,
        lossType = "categorical_crossentropy",
        finalAct = "softmax",
        tensor_logpath=tensorboard_log_dir)

    # ------------------------------------------------
    # Export trained models:
    # ------------------------------------------------
    print("Model trained. Saving model...")
    model.save_weights(model_fn)
    with open(model_arch_fn, 'w') as f:
        f.write(model.to_json())

    # ------------------------------------------------
    # Model Evaluation:
    # ------------------------------------------------
    # 1. Make predictions:
    Y_true, Y_pred, Y_proba = predict(x_data=x_test, meta_data=meta_test, model=model, branchlist = BRANCHES_LIST)

    # 2. Evaluate results:
    stat = calculate_statistics(Y_true, Y_pred, Y_proba)
    stat["run"] = run
    stat_df = stat_df.append(stat, ignore_index=True)

    # 3. Saving all labels & outputs:
    # + Export statistics as pickle file:
    Y_FINAL_TURE = Y_FINAL_TURE.append(Y_true)
    Y_FINAL_PRED = Y_FINAL_PRED.append(Y_pred)
    Y_FINAL_PROBA = Y_FINAL_PROBA.append(Y_proba)

    print("[INFO] Storing all statistics...")
    stat_df.to_pickle(stat_df_fn)
    Y_FINAL_TURE.to_csv(y_true_final_fn, index=False)
    Y_FINAL_PRED.to_csv(y_pred_final_fn, index=False)
    Y_FINAL_PROBA.to_csv(y_proba_final_fn, index=False)


    # End of run (cross-validation), adding counter:
    print("----------------------------------------------")
    run += 1
    del x_train, x_test

print("===========================================")
plot_model(model, to_file=os.path.join(OUTPUT_PATH,"model.png"))


# In[ ]:




