from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D, Reshape, Dense, Flatten, Dropout, Subtract, BatchNormalization, GlobalAveragePooling3D
from keras.layers.core import Lambda
from keras.engine import Model
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
from keras import backend as K
from keras import regularizers

from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss

import tensorflow as tf

try:
    #from keras_contrib.layers.normalization import InstanceNormalization
    from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
except ImportError:
    raise ImportError("Install keras_contrib in order to use instance normalization."
                          "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")



create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


'''
def testnet_backbone(inputs_1, n_base_filters):
    in_conv1 = create_convolution_block(inputs_1, n_base_filters)
    in_conv2 = create_convolution_block(in_conv1, n_base_filters*2, strides=(2, 2, 2))
    in_conv3 = create_convolution_block(in_conv2, n_base_filters*4, strides=(2, 2, 2))
    in_conv4 = create_convolution_block(in_conv3, n_base_filters*8, strides=(2, 2, 2))
    in_conv5 = create_convolution_block(in_conv4, n_base_filters*16, strides=(2, 2, 2))
    pool1 = GlobalAveragePooling3D()(in_conv5)
    return pool1

'''


def siam3dunet_backbone(inputs, n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4, activation_name="sigmoid", mask_name='mask1'):

    return_layers = []
   
    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

        #if level_number in [2, 4]:
        #    return_layers.append(current_layer)

    return current_layer

    '''
    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))
    
    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number == 1:
               return_layers.append(output_layer)

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name, name=mask_name)(output_layer)
    #return_layers.append(activation_block)
    return_layers.append(output_layer)

    #model = Model(inputs=inputs, outputs=activation_block)
    #model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    #return model
    return return_layers
    '''






def testnet_model(input_shape=(4, 128, 128, 128), optimizer=Adam, initial_learning_rate=5e-4, activation_name="sigmoid", **kwargs):


    inputs_1 = Input(input_shape)
    #inputs_2 = Input(input_shape)

    #pool_1 = testnet_backbone(inputs_1, n_base_filters)
    #pool_2 = testnet_backbone(inputs_2, n_base_filters)
    #sf_add = concatenate([pool_1, pool_2], axis=1)



    sf_return = siam3dunet_backbone(inputs_1, mask_name='mask1', **kwargs)
    sf_add = GlobalAveragePooling3D()(sf_return)

    out_pred_score = Dense(1, activation=None)(sf_add)
    out_pred_score = Lambda(print_output, arguments={'msg':' output'})(out_pred_score)
    out_pred_score = Activation(activation_name)(out_pred_score)
    out_pred_score = Lambda(print_output, arguments={'msg':' output sigmoid'}, name='score')(out_pred_score)


    print (initial_learning_rate)
    print (activation_name)

    model = Model(inputs=inputs_1, outputs=out_pred_score)
    #model = Model(inputs=[inputs_1, inputs_2], outputs=out_pred_score)
    model.compile(optimizer=SGD(lr=initial_learning_rate, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=optimizer(lr=initial_learning_rate), loss={'score':'binary_crossentropy'}, metrics=['accuracy'])

    #model.metrics_tensors += model.outputs
    return model


def print_output(x, msg):
    return K.print_tensor(x, message=f"{msg} is: ")

def print_output_max(x, msg):
    return K.print_tensor(K.max(x), message=f"{msg} is: ")

def loss_func(y_true, y_pred):
    return weighted_dice_coefficient_loss(y_true, y_pred)

def loss_(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)



def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2



