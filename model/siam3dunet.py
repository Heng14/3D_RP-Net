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

from keras.utils.training_utils import multi_gpu_model

import tensorflow as tf

try:
    #from keras_contrib.layers.normalization import InstanceNormalization
    from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
except ImportError:
    raise ImportError("Install keras_contrib in order to use instance normalization."
                          "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")



create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def siam3dunet_backbone(inputs, n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4, activation_name="sigmoid", mask_name='mask1'):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """

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

        if level_number in [2, 4]:
            return_layers.append(current_layer)

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
    return_layers.append(activation_block)
    #return_layers.append(output_layer)

    #model = Model(inputs=inputs, outputs=activation_block)
    #model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    #return model
    return return_layers

#def kernel_init(sf):
#    def kernel_init_(shape):
#        #return tf.Variable(sf,validate_shape=False)
#        return sf
#    return kernel_init_



def sf_squeeze(sf1):
    return K.squeeze(sf1, axis=0)

def sf_permute_dimensions(squeezed_tensor):
    return K.permute_dimensions(squeezed_tensor, (1,2,3,0))

def sf_expand_dims(transpose_tensor):
    return K.expand_dims(transpose_tensor, -1)

def sf_conv3d_base(x):
    sf2, expand_tensor = x

    return K.conv3d(sf2, expand_tensor, strides=[1, 1, 1, 1, 1], padding="valid", data_format="channels_first")

def sf_conv3d(x):
    sf2, sf1 = x

    #squeezed_tensor = K.squeeze(sf1, axis=0)
    #transpose_tensor = K.permute_dimensions(squeezed_tensor, (1,2,3,0))
    #expand_tensor = K.expand_dims(transpose_tensor, -1)
    #out_sf = K.conv3d(sf2, expand_tensor, strides=[1, 1, 1, 1, 1], padding="valid", data_format="channels_first")

    squeezed_tensor = Lambda(sf_squeeze)(sf1)
    transpose_tensor = Lambda(sf_permute_dimensions)(squeezed_tensor)
    expand_tensor = Lambda(sf_expand_dims)(transpose_tensor)
    out_sf = Lambda(sf_conv3d_base)([sf2, expand_tensor])
    return out_sf

def sf_module(sf1, sf2, n_filters):
    sf1 = create_convolution_block(sf1, n_filters)
    sf2 = create_convolution_block(sf2, n_filters)
    #sf1 = Lambda(print_output, arguments={'msg':' sf1'})(sf1)
    #sf2 = Lambda(print_output, arguments={'msg':' sf2'})(sf2)

    x = Lambda(sf_conv3d)([sf2, sf1])

    x = InstanceNormalization(axis=1)(x)
    #x = Lambda(print_output, arguments={'msg':' x'})(x)
    return x

def print_output(x, msg):
    return K.print_tensor(x, message=f"{msg} is: ")

def print_output_max(x, msg):
    return K.print_tensor(K.max(x), message=f"{msg} is: ")

def loss_func(y_true, y_pred):
    return weighted_dice_coefficient_loss(y_true, y_pred)

def loss_(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

def siam3dunet_model(input_shape=(4, 128, 128, 128), optimizer=Adam, initial_learning_rate=5e-4, activation_name="sigmoid", **kwargs):
    inputs_1 = Input(input_shape)
    inputs_2 = Input(input_shape)

    #input_m_shape = tuple([1] + list(input_shape[1:]))
    #inputs_m1 = Input(input_m_shape)
    #inputs_m2 = Input(input_m_shape)

    return_layers_1 = siam3dunet_backbone(inputs_1, mask_name='mask1', **kwargs)
    return_layers_2 = siam3dunet_backbone(inputs_2, mask_name='mask2', **kwargs)

    sf1_0 = return_layers_1[0] #sf:select feature
    sf2_0 = return_layers_2[0]
    #sf1_0 = BatchNormalization(axis=1)(sf1_0)
    #sf2_0 = BatchNormalization(axis=1)(sf2_0)
    #sf_0 = sf_module(sf1_0, sf2_0, 64)
    #sf_0 = sf_module(sf1_0, sf2_0, 1)
    #sf_0 = Add()([sf1_0, sf2_0]
    sf_0 = Subtract()([sf1_0, sf2_0])
    #sf_0 = concatenate([sf1_0, sf2_0], axis=1)
    sf_0 = create_convolution_block(sf_0, 32)


    #out_sf = Conv3D(1, (36, 36, 36), kernel_initializer=kernel_init(sf1_1))(sf2_1)

    sf1_1 = return_layers_1[1]
    sf2_1 = return_layers_2[1]
    #sf1_1 = InstanceNormalization(axis=1)(sf1_1)
    #sf2_1 = InstanceNormalization(axis=1)(sf2_1)
    #sf_1 = sf_module(sf1_1, sf2_1, 256)
    #sf_1 = sf_module(sf1_1, sf2_1, 1)
    #sf_1 = Add()([sf1_1, sf2_1])
    sf_1 = Subtract()([sf1_1, sf2_1])
    sf_1 = create_convolution_block(sf_1, 32)


    sf1_2 = return_layers_1[2] 
    sf2_2 = return_layers_2[2]
    #sf_2 = sf_module(sf1_2, sf2_2, 3)
    #sf_2 = sf_module(sf1_2, sf2_2, 1)
    sf_2 = Subtract()([sf1_2, sf2_2])
    sf_2 = create_convolution_block(sf_2, 32)


    out_pred_mask_1 = return_layers_1[-1]
    out_pred_mask_2 = return_layers_2[-1]


    sf_0 = GlobalAveragePooling3D()(sf_0)
    sf_1 = GlobalAveragePooling3D()(sf_1)
    sf_2 = GlobalAveragePooling3D()(sf_2)


    #sf_add = Add()([sf_0, sf_1, sf_2])
    sf_add = concatenate([sf_0, sf_1, sf_2], axis=1)

    #sf_0 = GlobalAveragePooling3D()(sf_0)
    #sf_1 = GlobalAveragePooling3D()(sf_1)



    #sf_0 = Lambda(print_output, arguments={'msg':' sf_0'})(sf_0)
    #sf_1 = Lambda(print_output, arguments={'msg':' sf_1'})(sf_1)

    #sf_0_max = Lambda(print_output_max, arguments={'msg':' sf_0_max'})(sf_0)
    #sf_1_max = Lambda(print_output_max, arguments={'msg':' sf_1_max'})(sf_1)


    #sf_add = concatenate([sf_0, sf_1], axis=1)
    #sf_add = Lambda(print_output, arguments={'msg':' sf_add'})(sf_add)

    #out_m1 = GlobalAveragePooling3D()(out_pred_mask_1)

    #sf_add = GlobalAveragePooling3D()(sf1_1)


    #out_pred_score = Activation(activation_name)(sf_add)

    #for i in range(len(out_pred_score.shape)-1):
    #    out_pred_score = K.squeeze(out_pred_score, axis=-1)

    #out_pred_score = Reshape(target_shape=())(out_pred_score)

    #sf_add = Flatten()(sf_add)



    #out_pred_score = Dense(1, activation=activation_name, name='score')(sf_add)
    #out_pred_score = Dense(1, activation=activation_name)(sf_add)

    #sf_add = Dense(10, activation='relu', kernel_initializer='Ones')(sf_add)
    #sf_add = Dense(10, activation='relu')(sf_add)
    #sf_add = Lambda(print_output, arguments={'msg':' sf_add'})(sf_add)

    #sf_add = Dropout(0.5)(sf_add)

    #out_pred_score = Dense(1, activation=None, kernel_initializer='Ones')(sf_add)
    #out_pred_score = Dense(1, activation=None, kernel_regularizer=regularizers.l2(0.01))(sf_add)
    out_pred_score = Dense(1, activation=None)(sf_add)
    #out_pred_score = Lambda(print_output, arguments={'msg':' output'})(out_pred_score)
    out_pred_score = Activation(activation_name, name='score')(out_pred_score)
    #out_pred_score = Lambda(print_output, arguments={'msg':' output sigmoid'}, name='score')(out_pred_score)


    #tf.Print(out_pred_score, [out_pred_score], 'out_pred_score= ')
    #tf.print(out_pred_score)
    #print_layer = Lambda((lambda x: tf.Print(x, [x], 'out_pred_score= ')), name='print')(out_pred_score)
    #print_layer = Lambda((lambda x: tf.Print(x, [x], message='out_pred_score= ', first_n=-1, summarize=1024)), name='print')(out_pred_score)

    print (initial_learning_rate)
    print (activation_name)

    model = Model(inputs=[inputs_1, inputs_2], outputs=[out_pred_score, out_pred_mask_1, out_pred_mask_2])

    #parallel_model = multi_gpu_model(model, gpus=2)

    model.compile(optimizer=SGD(lr=initial_learning_rate, momentum=0.9), loss={'score':'binary_crossentropy','mask1':loss_func,'mask2':loss_func}, loss_weights={'score': 1., 'mask1': 0.2, 'mask2': 0.2}, metrics=['accuracy'])
    #model.compile(optimizer=optimizer(lr=initial_learning_rate), loss={'score':'binary_crossentropy'}, metrics=['accuracy'])

    #model.metrics_tensors += model.outputs
    return model


    #return siam3dunet_backbone(inputs_1, **kwargs)
    #siam3dunet_backbone(inputs_2, **kwargs)



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



