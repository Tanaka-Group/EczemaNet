"""
Helper functions for EczemaNet
"""
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../lib")

import os
import pandas as pd
import numpy as np
import fnmatch
import itertools as itertools
import math as math
import sklearn as sk
import scipy as scipy
import scikitplot as skplt
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import backend as K
from keras import optimizers
from keras import metrics
from keras.models import model_from_json
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import multi_gpu_model

import six.moves.urllib as urllib
import PIL as pil
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO

from ImagePreprocessing import ImageDataGenerator
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

# Confirm Keras sees the GPU:
print("[INFO] # of GPUs (Keras): " + str(len(K.tensorflow_backend._get_available_gpus())))
# Check tensorflow version:
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

"""
#############################################################
---------------------- Static Variables ---------------------
#############################################################
"""

BRANCHES_LIST = ['sassad_cra','sassad_dry','tiss_ery','tiss_exc','sassad_exu','sassad_lic','tiss_oed']
ALL_COMBINED_KEYS = {
        "sassad_tot": ['sassad_cra', 'sassad_dry', 'tiss_ery', 'tiss_exc', 'sassad_exu', 'sassad_lic'],
        "scorad_tot": ['tiss_ery', 'tiss_oed', 'sassad_exu', 'tiss_exc', 'sassad_lic', 'sassad_dry'],
        "easi_tot": ['tiss_ery', 'tiss_oed', 'tiss_exc', 'sassad_lic'],
        "tiss_tot": ['tiss_ery', 'tiss_oed', 'tiss_exc']
    }
SWET_SORTED_DIR = "<PATHTO>/images"
PATH_TO_ROI_FROZEN_GRAPH = '<PATHTO>/frozen_inference_graph.pb'
PATH_TO_ROI_LABELS = '<PATHTO>/label_map.pbtxt'

"""
#############################################################
--------------------------- Utils ---------------------------
#############################################################
"""

def label_ordinariser(inputs, k_lim=4):
    """ Label ordinarizer

    Converts multiclass label (0,1,2,...) for k-classes into ordinal representation

    # Arguments
        inputs: input labels.
            - e.g. [0,2,1,3,...]
        k_lim: integer, the number of classes.
    # Returns
        output labels in ordinal form.
            - e.g.[[0,0,0],[1,1,0],[1,0,0],[1,1,1],...]

    """
    outputs = [ [ 0 for i in range(k_lim-1) ] for j in range(len(inputs)) ]

    for i, val in enumerate(inputs):
        if val > k_lim or val < 0:
            raise Exception("[ERROR] Value out of range!")
        for j in range(val):
            outputs[i][j] = 1

    return outputs

def label_ordinariser_reversed(inputs):
    """ Reverse from ordinary labelling to multiclass labels

    Revese the ordinal labels back to multiclass labels

    # Arguments
        inputs: ordinal labels.
            - e.g.[[0,0,0],[1,1,0],...]
    # Returns
        output labels in multiclass label
            - e.g.[0,2,...]

    """
    outputs = []

    for i, val in enumerate(inputs):
        out = np.sum(val, dtype=np.int32)
        if out > k_lim or out < 0:
            raise Exception("[ERROR] Value out of range!")
        outputs.append(out)

    return outputs

def proba_ordinal_to_categorical(inputs):
    """ Orinal -> Categorical Probability convertion

    Convert ordinal probas (k-1) to categorical (k-classes) probas

    # Arguments
        inputs: ordinal probabilities (k-1 classes)
            - e.g.[0.123,0.2321,0.888]
    # Returns
        output categorical probabilities (k classes)
            - e.g.[0.877, 0.0944517, 0.0031974095999999994, 0.0253508904]

    """
    k = len(inputs[0]) + 1
    outputs = []
    for input in inputs:
        output = [0 for i in range(k)]
        for c in range(0, k):
            if c is 0:
                output[0] = 1.0 - input[0]
                p_cond = input[0]
            elif c is (k-1):
                output[k-1] = p_cond
            else:
                output[c] = (1.0 - input[c]) * p_cond
                p_cond = input[c] * p_cond
        outputs.append(output)
    return outputs


def conv_sum(input_probas, comb_array=None, branchlist=None):
    """ Calculate convolutional sum

    Calculate convolutional sum

    # Arguments
        input_probas: ordinal probabilities (k-1 classes)
            - e.g. [[9.57781076e-02, 8.76802418e-01, 2.65899727e-02, 8.29501252e-04],
                   [1.32381320e-02, 1.59786726e-01, 3.27317506e-01, 4.99657661e-01],
                   [1.66641474e-02, 3.03159765e-01, 6.74962319e-01, 5.21376077e-03],
                   [1.13785565e-01, 6.65654127e-01, 2.15435103e-01, 5.12520969e-03],
                   [9.94189308e-01, 5.76749335e-03, 4.29769659e-05, 2.21664067e-07],
                   [2.44857550e-01, 5.82501496e-01, 3.58530823e-02, 1.36787862e-01]]
        comb_array: combinatorial array
            - Can reduce computational resources, requires # of branches information.
        branchlist: Branch list
    # Returns
        output convolutional probabilities
            - e.g. [[5.852559121119898e-07,
                    2.7888531375250724e-05,
                    0.0005283264189413422,
                    0.005184462832191,
                    0.028991045518549308,
                    0.09748364466277436,
                    0.2062091543271896,
                    0.2785957022807078,
                    0.22573941135322664,
                    0.10103699726044552,
                    0.04470941632757998,
                    0.010814878298372545,
                    0.0006554969918371807,
                    2.2541639726847054e-05,
                    4.4404435849156057e-07,
                    4.230472073992656e-09,
                    2.620563658292333e-11,
                    1.3376977240757644e-13,
                    3.358112077856369e-16]]

    """
    # Generate all array combinations, if not passed on.
    # Can reduce computational resources, requires # of branches information:
    if comb_array is None:
        if branchlist is None:
            raise Exception("[ERROR] 'branchlist' is required if combinatorial array is not passed on!")
        num_repeats = len(branchlist)
        comb_array = [[] for i in range( num_repeats*3+1 )]
        for i in list( itertools.product([0, 1, 2, 3], repeat=num_repeats) ):
            comb_array[ np.sum(np.array(i)) ].append(np.array(i))
    else:
        num_repeats = int( (len(comb_array) - 1) / 3 )

    # Calculation of Convolutional Sum:
    output_array = []
    for pred_proba in input_probas:
        prob_array = [0.0 for i in range( num_repeats*3+1 )]
        for i in range( 0, len(comb_array) ):
            for j in comb_array[i]:
                pr = 1
                for k in range(0, len(j)):
                    pr = pr * pred_proba[k][j[k]]
                prob_array[i] += pr
        norm_prob_array = [float(i)/sum(prob_array) for i in prob_array]
        output_array.append(norm_prob_array)
    return output_array

def calculate_rps(probabilities,Y):
    """ Calculate rps

    TODO: Code refactoring required

    # Arguments
        probabilities: probabilities
        Y: true label
    # Returns
        output RPS score

    """
    rps=[]
    [numb_of_images,numb_of_classes]=np.shape(probabilities)
    for element in range (0,numb_of_images):
        temp_rps=0
        for element2 in range (0,numb_of_classes):
            yk=np.sum(probabilities[element,0:(element2+1):1])
            real=int(Y[element])
            if real<=element2:
                ok=1
            else:
                ok=0
            temp_rps=temp_rps+((yk-ok)**2)
        temp_rps=temp_rps/(numb_of_classes-1)
        rps.append(temp_rps)
    return rps

def calculate_average_rps(y_true,y_proba):
    """ Calculate average rps

    # Arguments
        y_true: true labels
        y_proba: predicted probabilities
    # Returns
        output average RPS score

    """
    rps = calculate_rps(probabilities=y_proba, Y=y_true)
    av_rps = float(np.mean(rps))
    return av_rps

def load_images(image_fps, dir_prefix=None, image_size=224):

    x_data = []

    for imagefp in image_fps:

        if dir_prefix is not None:
            imagefp = os.path.join(dir_prefix, imagefp)

        img = keras.preprocessing.image.load_img(imagefp, target_size=(image_size, image_size))
        img = keras.preprocessing.image.img_to_array(img)

        x_data.append(img)

    return np.array(x_data)

"""
#############################################################
----------------------- Preprocessing -----------------------
#############################################################
"""

def get_swet_img_fps(refno, visno, swet_images_dir=SWET_SORTED_DIR):
    """ Get SWET Image Filepaths

    Get SWET image filenpaths by refno and visno

    # Arguments
        refno: (int) Patient reference number
            - e.g. 1001
        visno: (int) visit number
            - e.g. 1
        swet_images_dir: absolute path to the SWET images
    # Returns
        absolute image filepaths
            - e.g. ['/<ABSPATH>/wk00.jpg','/<ABSPATH>/wk00_2.jpg']
    """

    refno_dn = str(refno).zfill(5)
    refno_dp = os.path.join(swet_images_dir,refno_dn)

    if os.path.exists(refno_dp) == False:
        return [];

    image_prefix = "wk*";
    if visno == 1:
        image_prefix = "wk00*";
    elif visno == 2:
        image_prefix = "wk04*";
    elif visno == 3:
        image_prefix = "wk12*";
    elif visno == 4:
        image_prefix = "wk16*";
    else:
        print(visno)

    images_fns = fnmatch.filter(os.listdir(refno_dp), image_prefix)
    images_fns = [os.path.join(refno_dp,x) for x in images_fns]

    return images_fns

"""
#############################################################
--------------------------- Model ---------------------------
#############################################################
"""

def weight_ratio (in_col):
    """ weight ratio calculation

    Calculate the weight ratio for each classes

    # Arguments
        in_col: input column
            - e.g.[0,1,2,2,2,2,2,1,0] (k=3 classes)
    # Returns
        output weight ratio
            - e.g.[1,1,2][0.25,0.25,0.5]

    """
    outputs = []
    cw = sk.utils.class_weight.compute_class_weight('balanced', np.unique(in_col), in_col)
    w_min = min(cw)

    for w in cw:
        outputs.append( w/w_min )
    return outputs


def train_model(eczemanetModule,
                x_data,
                y_data,
                classWeights,
                branchlist,
                verbose=True,
                batchsize=32,
                catagories=3,
                itr=5,
                earlystopping=True,
                withGenerator=False,
                inputSize=224,
                opt=None,
                lossType="categorical_crossentropy",
                finalAct="softmax",
                tensor_logpath='./tensorboard_log'):
    """ EczemaNet Model Building and Training

    # Arguments
        eczemanetModule: name of EzemaNet Module to import
            e.g. "eczemanet_models.EczemaNet_VGG16"
        x_data: input (image) data, consists of "(x_train, x_test)"
        y_data: output (label) data, consists of "(y_train, y_test)"
        classWeights: class weights
        branchlist: list of all branches
        verbose: (Boolean) print out training messages
        batchsize: batchsize for training
        catagories: number of output catagories (nodes)
        itr: number of iterations (epochs) to train
        earlystopping: (Boolean) en/disable early stopping
        withGenerator: (Boolean) en/disable input (image) distortion
        inputSize: Input size of the image
        opt: training optimizer
        lossType: loss function for training
            - e.g. "binary_crossentropy", "categorical_crossentropy"
        finalAct: final layer activation function
            - e.g. "sigmoid", "softmax"
        tensor_logpath: log filepath for tensorflow
    # Returns
        (H, model)
        H: keras training history
        model: keras model

    """

    EczemaNet = getattr(__import__(eczemanetModule, fromlist=["EczemaNet"]), "EczemaNet")

    (x_train, x_test) = x_data
    (y_train, y_test) = y_data

    if withGenerator is True:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest')
        datagen.fit(x_train)

    model = EczemaNet.build(
        branches_list=branchlist,
        catagory_per_branch=catagories,
        input_size=inputSize,
        finalOperation=finalAct)

    losses={}
    for branch in branchlist:
        losses[branch+"_output"] = lossType

    if opt is None:
        model_opt = optimizers.SGD(lr=1e-4, momentum=0.9)
    else:
        model_opt = opt

    try:
        model = multi_gpu_model(model)
    except:
        pass
    model.compile(optimizer=model_opt, loss=losses, metrics=['mae','accuracy'])

    tb_callBack = TensorBoard(log_dir=tensor_logpath, histogram_freq=0, write_graph=True, write_images=True)
    es_callBack = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=3, verbose=verbose)

    cbs = [tb_callBack, es_callBack]
    if earlystopping is False:
        cbs = [tb_callBack]

    if withGenerator is True:
        H = model.fit_generator(
                datagen.flow(x_train, y_train, batch_size=batchsize),
                validation_data=(x_test, y_test),
                class_weight=classWeights,
                epochs=itr,
                steps_per_epoch=len(x_train) / batchsize,
                verbose=verbose,
                callbacks=cbs)
    else:
        H = model.fit(
                x=x_train,
                y=y_train,
                validation_data=(x_test, y_test),
                class_weight=classWeights,
                epochs=itr,
                batch_size=batchsize,
                verbose=verbose,
                callbacks=cbs)

    return H, model

def predict(x_data, meta_data, model, branchlist=BRANCHES_LIST, weighted=False):
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
    y_crops_proba_ordinal = model.predict(x_data)

    # 2. Combine crops into whole images
    meta_grouped = meta_data.groupby(["refno","visno"])
    for group_name, df_group in meta_grouped:
        (refno, visno) = group_name
        indexes = df_group.index.values

        y_crops_proba_categorial = {}
        y_whole_proba_categorical = {}
        y_whole_pred = {}

        # -- Calculate weights for each crops if possible --
        if weighted is True:
            weights = meta_data.loc[indexes]['detect_score']
            weights_norm = [ weight if weight > 1.0/len(weights) else 1.0/len(weights) for weight in weights]
            weights_norm = [weight/np.sum(weights_norm) for weight in weights_norm]

        # -- Average over crops and convert to categorical --
        for idx, branchname in enumerate(branchlist):
            y_crops_proba_categorial[branchname] = proba_ordinal_to_categorical(y_crops_proba_ordinal[idx][indexes])
            if weighted is True:
                y_whole_proba_categorical[branchname] = np.average(y_crops_proba_categorial[branchname],axis=0, weights=weights_norm)
            else:
                y_whole_proba_categorical[branchname] = np.mean(y_crops_proba_categorial[branchname],axis=0)
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
    # return Y_pred, Y_proba



def calculate_statistics(Y_true, Y_pred, Y_proba=None, branchlist=BRANCHES_LIST):
    """ Produce all statistics

    # Arguments
        Y_true: true label data
        Y_pred: predicted labels
        Y_proba: predicted probabilities
        branchlist: branches
    # Returns
        Pandas Dataframe with all statistics

    """
    stat = {}
    # -- Combined severity scores --
    for score_name in ALL_COMBINED_KEYS:
        try:
            stat[score_name+"_mae"] = [sk.metrics.mean_absolute_error( Y_true[score_name], Y_pred[score_name])]
            stat[score_name+"_mse"] = [sk.metrics.mean_squared_error( Y_true[score_name], Y_pred[score_name])]
            stat[score_name+"_rmse"] = [math.sqrt(sk.metrics.mean_squared_error( Y_true[score_name], Y_pred[score_name]))]
            stat[score_name+"_r2"] = [sk.metrics.r2_score( Y_true[score_name], Y_pred[score_name])]
            try:
                stat[score_name+"_rps"] = [calculate_average_rps( np.array(Y_true[score_name]), np.stack(Y_proba[score_name]))]
            except:
                stat[score_name+"_rps"] = [None]
        except:
            pass
    # -- Individual sign evaluations --
    for idx, branch_name in enumerate(branchlist):
        try:
            stat[branch_name+"_rps"] = [calculate_average_rps( np.array(Y_true[branch_name]), np.stack(Y_proba[branch_name]))]
        except:
            stat[branch_name+"_rps"] = [None]
        stat[branch_name+"_acc"] = [sk.metrics.accuracy_score( Y_true[branch_name], np.rint(Y_pred[branch_name]))]
        stat[branch_name+"_f1"] = [sk.metrics.f1_score( Y_true[branch_name], np.rint(Y_pred[branch_name]), average='weighted' )]
    return pd.DataFrame(stat)

"""
#############################################################
------------------------- ROI Model -------------------------
#############################################################
"""

class ROI:
    """ EzemaNet Region-of-Interest (ROI) model
    """

    def __init__(self):

        # Load frozen graph:
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_ROI_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Load label map:
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_ROI_LABELS, use_display_name=True)

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        np_image = np.array(image.getdata())
        np_image = np_image.reshape(im_height, im_width, 3).astype(np.uint8)
        return np_image

    def run_inference_for_single_image(self, image):

        with self.detection_graph.as_default():

            with tf.Session() as sess:

                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}

                for key in ['num_detections', 'detection_boxes', 'detection_scores', 
                            'detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    def visualise_detection(self, image, image_dict, min_thresh=.5):

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            image_dict['detection_boxes'],
            image_dict['detection_classes'],
            image_dict['detection_scores'],
            self.category_index,
            instance_masks = image_dict.get('detection_masks'),
            min_score_thresh = min_thresh,
            use_normalized_coordinates = True,
            line_thickness = 16)

        return image


