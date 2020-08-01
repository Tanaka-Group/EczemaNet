from keras import backend as K
from keras.models import Model
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
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3

import tensorflow as tf

class EczemaNet:
    @staticmethod
    def build_branch(inputs, branchName, numCategories, numVggDenseNodes=512, finalAct="softmax"):
        # Add additional layers for each branch
        # DENSE(512) => DENSE(512) => OUTPUT
        x = Dense(numVggDenseNodes, activation='relu', name=str(branchName + "_dense" + str(numVggDenseNodes) + "_1"),
                  kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(inputs)
        x = Dropout(rate=0.5)(x)
        x = Dense(numVggDenseNodes, activation='relu', name=str(branchName + "_dense" + str(numVggDenseNodes) + "_2"),
                  kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(numCategories, activation=finalAct, name=str(branchName + "_output"),
                  kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(x)
        return x

    @staticmethod
    def build(branches_list, catagory_per_branch, input_size=224, finalOperation="softmax"):
        # create the base pre-trained model
        base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(input_size,input_size,3))

        # add branches to the pre-trained model
        intermediate_output = base_model.output
        network_branches = []
        for branch_name in branches_list:
            network_branches.append(
                EczemaNet.build_branch(
                    inputs=intermediate_output,
                    branchName=branch_name,
                    numCategories=catagory_per_branch,
                    finalAct=finalOperation))

        # put things together
        model = Model(inputs=base_model.input, outputs=network_branches, name="eczemanet")

        for layer in model.layers:
            layer.trainable = True

        # return the constructed network architecture
        return model