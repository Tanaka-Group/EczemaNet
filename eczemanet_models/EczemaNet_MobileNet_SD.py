from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Concatenate
from keras import regularizers
from keras.constraints import max_norm
from keras.optimizers import Adam
from keras.applications.mobilenet import preprocess_input
from keras.applications.mobilenet import MobileNet

import tensorflow as tf

class EczemaNet:

    @staticmethod
    def build_branch(inputs, branchName, numCategories, finalAct="softmax"):
        # Add additional layers for each branch
        x = Dense(numCategories, activation=finalAct, name=str(branchName + "_output"),
                  kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(inputs)
        return x

    @staticmethod
    def build(branches_list, catagory_per_branch, numVggDenseNodes=512,  input_size=224, finalOperation="softmax"):

        # create the base pre-trained model
        base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(input_size,input_size,3))

        # add branches to the pre-trained model
        intermediate_output = base_model.output
        x = Dense(numVggDenseNodes, activation='relu', name=str("FC_dense" + str(numVggDenseNodes) + "_1"),
                  kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(intermediate_output)
        x = Dropout(rate=0.5)(x)
        x = Dense(numVggDenseNodes, activation='relu', name=str("FC_dense" + str(numVggDenseNodes) + "_2"),
                  kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(x)
        x = Dropout(rate=0.5)(x)

        network_branches = []
        numCategories = len(branches_list) * catagory_per_branch
        for branch_name in branches_list:
            network_branches.append(
                EczemaNet.build_branch(
                    inputs=x,
                    branchName=branch_name,
                    numCategories=catagory_per_branch,
                    finalAct=finalOperation))

        # put things together
        model = Model(inputs=base_model.input, outputs=network_branches, name="eczemanet")

        for layer in model.layers:
            layer.trainable = True

        # return the constructed network architecture
        return model