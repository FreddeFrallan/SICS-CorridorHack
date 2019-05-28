import numpy as np
import keras.layers as KL
import keras.models as KM
import keras.backend as K

import keras
import Demo.Img2Fashion.keras_applications as KA

KA.set_keras_submodules(
    backend=keras.backend,
    layers=keras.layers,
    models=keras.models,
    engine=keras.engine,
    utils=keras.utils,
)

def get_backbone(backbone_name, weights='imagenet', input_shape=None):
    backbones = {
        'densenet121': KA.densenet.DenseNet121,
        'densenet169': KA.densenet.DenseNet169,
        'densenet201': KA.densenet.DenseNet201,
        'inceptionresnetv2': KA.inception_resnet_v2.InceptionResNetV2,
        'inceptionv3': KA.inception_v3.InceptionV3,
        'nasnet': KA.nasnet.NASNetLarge,
        'nasnetmobile': KA.nasnet.NASNetMobile,
        'resnext101': KA.resnext.ResNeXt101,
        'vgg16': KA.vgg16.VGG16,
    }
    backbone_model = backbones[backbone_name](weights='imagenet', input_shape=input_shape, include_top=False)
    return backbone_model

def get_preprocessing(backbone_name):
    preprocessing_fns = {
        'densenet121': KA.densenet.preprocess_input,
        'densenet169': KA.densenet.preprocess_input,
        'densenet201': KA.densenet.preprocess_input,
        'inceptionresnetv2': KA.inception_resnet_v2.preprocess_input,
        'inceptionv3': KA.inception_v3.preprocess_input,
        'nasnet': KA.nasnet.preprocess_input,
        'nasnetmobile': KA.nasnet.preprocess_input,
        'resnext101': KA.resnext.preprocess_input,
        'vgg16': KA.vgg16.preprocess_input,
    }
    return preprocessing_fns[backbone_name]

def get_feature_pyramid_layers(backbone_name):
    pyramid_layers = {
        'densenet121': (-1, 311, 139, 51),
        'densenet169': (-1, 367, 139, 51),
        'densenet201': (-1, 479, 139, 51),
        'inceptionresnetv2': (-1, 594, 260, 16),
        'inceptionv3': (-1, 228, 86, 16),
        'nasnet': (-1, 710, 383, 56),
        'nasnetmobile': (767, 530, 293, 53),
        'resnext101': (476, 432, 108, 50),
        'vgg16': (18, 14, 10, 6),
    }

    return pyramid_layers[backbone_name]

def FPN(backbone_name = 'densenet169',
        input_shape = (None, None, 3),
        encoder_weights = "imagenet",
        freeze_encoder = False,
        use_batchnorm = True,
        pyramid_block_filters = 256,
        segmentation_block_filters = 128,
        upsample_rates = (1, 2, 2, 2),
        last_upsample = 4,
        interpolation = 'bilinear',
        classes = 23,
        activation = 'softmax'):

    backbone = get_backbone(
        backbone_name, 
        weights=encoder_weights, 
        input_shape=input_shape)

    fpn_layers = get_feature_pyramid_layers(backbone_name)

    assert(len(upsample_rates) == len(fpn_layers))
    # extract feature pyramid layers from backbone
    outputs = [backbone.layers[i].output for i in fpn_layers]

    m = None
    pyramid = []
    for i, c in enumerate(outputs):
        m, p = pyramid_block(pyramid_filters=pyramid_block_filters,
                             segmentation_filters=segmentation_block_filters,
                             upsample_rate=upsample_rates[i],
                             use_batchnorm=use_batchnorm,
                             stage=i)(c, m)
        pyramid.append(p)

    # upsample and concatenate all pyramid layer
    upsampled_pyramid = []

    for i, p in enumerate(pyramid[::-1]):
        if upsample_rates[i] > 1:
            upsample_rate = to_tuple(np.prod(upsample_rates[:i+1]))
            p = KL.UpSampling2D(size=upsample_rate, data_format='channels_last', interpolation=interpolation)(p)
        upsampled_pyramid.append(p)

    x = KL.Concatenate()(upsampled_pyramid)

    # final convolution
    filters = segmentation_block_filters * len(pyramid)
    x = conv2d_block(filters, (3, 3), use_batchnorm=use_batchnorm, padding='same')(x)

    x = KL.Conv2D(classes, (3, 3), padding='same', name='score_map')(x)

    # upsampling to original spatial resolution
    x = KL.UpSampling2D(size=to_tuple(last_upsample), data_format='channels_last', interpolation=interpolation)(x)

    # activation
    segmentation_output = KL.Activation(activation)(x)

    model = KM.Model(backbone.input, segmentation_output)

    if freeze_encoder:
        for layer in backbone.layers:
            layer.trainable = False

    model.name = 'fpn-{}'.format(backbone.name)

    return model

def pyramid_block(pyramid_filters=256, segmentation_filters=128, upsample_rate=2,
                  use_batchnorm=False, stage=0):
    def layer(c, m=None):

        x = conv2d_block(pyramid_filters, (1, 1),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        name='pyramid_stage_{}'.format(stage))(c)

        if m is not None:
            up = KL.UpSampling2D(size=to_tuple(upsample_rate), data_format='channels_last', interpolation='nearest')(m)
            x = KL.Add()([x, up])

        # segmentation head
        p = conv2d_block(segmentation_filters, (3, 3),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        name='segm1_stage_{}'.format(stage))(x)

        p = conv2d_block(segmentation_filters, (3, 3),
                        padding='same',
                        use_batchnorm=use_batchnorm,
                        name='segm2_stage_{}'.format(stage))(p)
        m = x

        return m, p
    return layer

def conv2d_block(filters, kernel_size,
                activation='relu',
                use_batchnorm=True,
                name='conv_block',
                **kwargs):
    def layer(input_tensor):

        x = KL.Conv2D(filters, kernel_size, use_bias=not(use_batchnorm), name=name+'_conv', **kwargs)(input_tensor)
        if use_batchnorm:
            x = KL.BatchNormalization(name=name+'_bn',)(x)
        x = KL.Activation(activation, name=name+'_'+activation)(x)

        return x
    return layer

def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

    raise ValueError('Value should be tuple of length 2 or int value, got "{}"'.format(x))
