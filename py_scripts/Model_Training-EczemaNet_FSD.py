#!/usr/bin/env python
# coding: utf-8

# ## Model Training: EczemaNet Full-suite with Fully Sign Dependency (FSD)
# In this notebook, the goal is to load pre-trained EczemaNet_VGG16 Full-suite that has proven a good performance, and use it as the secondary transfer learning to learn the dependency between each signs.

# In[ ]:


# Make sure the ability to import custom libraries
import sys
sys.path.append("..")
sys.path.append("../lib")

# Import libraries as usual
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

# SciKit-Learn specific functions:
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

# Keras specific functions:
import keras
from keras.models import Model, model_from_json
from keras.preprocessing import image
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Concatenate, concatenate, Dropout
from keras.constraints import max_norm
from keras import optimizers
from keras.utils import plot_model
from keras.regularizers import l2

# Confirm Keras sees the GPU
from keras import backend as K
# assert len(K.tensorflow_backend._get_available_gpus()) > 0

from EczemaNet_helper import *


# ### 1. Define some static paths

# In[ ]:


ECZEMANET_MODEL="eczemanet_models.EczemaNet_MobileNet"
PATH_TO_DATASET = "../data"
OUTPUT_PATH = "../output/EczemaNet_FSD/"
PRETRAINED_OUTPUT_PATH = "../output/EczemaNet/"
PRETRAINED_MODEL_PREFIX = "2020-03-11_run-"
BRANCHES_LIST = ['sassad_cra','sassad_dry','tiss_ery','tiss_exc','sassad_exu','sassad_lic','tiss_oed']


# ### 2. Load pre-processed data and Cross-validation information

# In[ ]:


# Load data:
meta_data = pd.read_csv(os.path.join(PATH_TO_DATASET,"meta_data.csv"))

# Get cross validation IDs:
cvid_df = pd.read_csv(os.path.join(PATH_TO_DATASET,"patient_cvid.csv"))
print("Total number of unique cases: " + str(len(cvid_df)))


# In[ ]:


stat_df = pd.DataFrame()
Y_FINAL_TURE = pd.DataFrame()
Y_FINAL_PRED = pd.DataFrame()
Y_FINAL_PROBA = pd.DataFrame()

# Cross validation using pre-defined folds
for run in range(0,10):
    print("[INFO] Start Run: "+str(run+1)+" of 10")
    # ------------------------------------------------
    # 0. Define Variables
    # ------------------------------------------------
    meta_train = pd.DataFrame()
    meta_test = pd.DataFrame()
    y_train = {}
    y_test = {}
    classWeights = {}
    losses = {}
    module_name = ECZEMANET_MODEL
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
    # ------------------------------------------------
    # 1. Spliting indexes
    # ------------------------------------------------
    train_refnos = cvid_df[cvid_df['cv_id'] != run]['refno'] # <-- Obtain patient refnos for training set
    test_refnos = cvid_df[cvid_df['cv_id'] == run]['refno']  # <-- Obtain patient refnos for test set
    # Sort all labels (meta data) according to the refno:
    for refno in train_refnos.values:
        meta_train = meta_train.append(meta_data[meta_data['refno'] == refno])
    for refno in test_refnos.values:
        meta_test = meta_test.append(meta_data[meta_data['refno'] == refno])
    # ------------------------------------------------
    # 2. Preparing CV specific Inputs(X) and labels(Y)
    # ------------------------------------------------
    print("[INFO] Preparing inputs/labels...")
    # -- 2.1 Prepare Training set --
    for branch in BRANCHES_LIST:
        y_train[branch+"_output"] = np.array(label_ordinariser(meta_train[branch]))
    x_train = load_images(meta_train['filepath'], image_size=224)
    meta_train = meta_train.reset_index()
    # -- 2.2 Prepare Test set --
    for branch in BRANCHES_LIST:
        y_test[branch+"_output"] = np.array(label_ordinariser(meta_test[branch]))
    x_test = load_images(meta_test['filepath'], image_size=224)
    meta_test = meta_test.reset_index()
    # -- 2.3 Prepare Classweights --
    for branch in BRANCHES_LIST:
        for idx in range(3):
            col = [row[idx] for row in y_train[branch+"_output"]]
            classWeights[branch+"_"+str(idx)+"_output"] = weight_ratio(col)
    print("Training/Test inputs & labels prepared.")
    # -- 2.4 Define losses --
    for branch in BRANCHES_LIST:
        losses[branch+"_output"] = "binary_crossentropy"
    # ------------------------------------------------
    # 3. Load Pre-trained EczemaNet model
    # ------------------------------------------------
    print("[INFO] Load pre-trained EczemaNet model and modify...")
    pretrained_model_fn = os.path.join(PRETRAINED_OUTPUT_PATH, PRETRAINED_MODEL_PREFIX + str(run) + "_model_weights.h5")
    pretrained_model_arch_fn = os.path.join(PRETRAINED_OUTPUT_PATH, PRETRAINED_MODEL_PREFIX + str(run) + "_model_architecture.json")
    with open(pretrained_model_arch_fn, 'r') as f:
        pretrained_model = model_from_json(f.read())
    pretrained_model.load_weights(pretrained_model_fn)
    # ------------------------------------------------
    # 4. Modify Pre-trained model
    # ------------------------------------------------
    # -- 4.1 Remove last output+dropout layers --
    for branch in BRANCHES_LIST:
        pretrained_model.layers.pop()
        pretrained_model.layers.pop()
    # -- 4.2 Add new layers --
    intermediate_output = []
    for layer in pretrained_model.layers:
        if "tiss_ery_dense512_2" in layer.name:
            x = Dropout(rate=0.5, name="ery_dropout")(layer.output)
            intermediate_output.append(Dense(1, name=str("tiss_ery_preoutput"), kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(x))
        elif "tiss_exc_dense512_2" in layer.name:
            x = Dropout(rate=0.5, name="exc_dropout")(layer.output)
            intermediate_output.append(Dense(1, name=str("tiss_exc_preoutput"), kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(x))
        elif "tiss_oed_dense512_2" in layer.name:
            x = Dropout(rate=0.5, name="oed_dropout")(layer.output)
            intermediate_output.append(Dense(1, name=str("tiss_oed_preoutput"), kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(x))
        elif "sassad_lic_dense512_2" in layer.name:
            x = Dropout(rate=0.5, name="lic_dropout")(layer.output)
            intermediate_output.append(Dense(1, name=str("sassad_lic_preoutput"), kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(x))
        elif "sassad_exu_dense512_2" in layer.name:
            x = Dropout(rate=0.5, name="exu_dropout")(layer.output)
            intermediate_output.append(Dense(1, name=str("sassad_exu_preoutput"), kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(x))
        elif "sassad_dry_dense512_2" in layer.name:
            x = Dropout(rate=0.5, name="dry_dropout")(layer.output)
            intermediate_output.append(Dense(1, name=str("sassad_dry_preoutput"), kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(x))
        elif "sassad_cra_dense512_2" in layer.name:
            x = Dropout(rate=0.5, name="cra_dropout")(layer.output)
            intermediate_output.append(Dense(1, name=str("sassad_cra_preoutput"), kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(x))
        else:
            pass
    final_output = []
    for branch_name in BRANCHES_LIST:
        final_output.append(
            Dense(3, activation='sigmoid', name=str(branch_name + "_output"), kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1))(concatenate(intermediate_output)))
    # -- 4.2 Compile model --
    model = Model(inputs=pretrained_model.input, outputs=final_output, name="eczemanet")
    # -- 4.3 Freeze pretrained layers --
    for layer in model.layers:
            layer.trainable = True
    # ------------------------------------------------
    # 5. Train new model
    # ------------------------------------------------
    print("[INFO] Start training Model ...")
    opt = optimizers.SGD(lr=1e-4, momentum=0.9)
    tb_callBack = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=0, write_graph=True, write_images=True)
    es_callBack = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=3, verbose=1)
    model.compile(optimizer=opt, loss=losses, metrics=['mae','accuracy'])
    H = model.fit(x = x_train, y = y_train,
              validation_data = (x_test, y_test),
              class_weight = classWeights,
              epochs = 50, # <= Maximum, Early stopping
              batch_size = 32,
              verbose = 1,
              callbacks = [tb_callBack, es_callBack])
    # ------------------------------------------------
    # 6. Export/Save trained models
    # ------------------------------------------------
    print("[INFO] Model trained. Saving model ...")
    model.save_weights(model_fn)
    with open(model_arch_fn, 'w') as f:
        f.write(model.to_json())
    # ------------------------------------------------
    # 7. Model Evaluation:
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

print("===========================================")
plot_model(model, to_file=os.path.join(OUTPUT_PATH,"model.png"))


# In[ ]:





# In[ ]:




