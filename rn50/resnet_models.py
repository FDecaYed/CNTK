# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk.initializer import he_normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu

#
# Resnet building blocks
#
def conv_bn(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False)(c)
    return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init)
    return relu(r)

def resnet_basic(input, num_filters):
    c1 = conv_bn_relu(input, (3,3), num_filters)
    c2 = conv_bn(c1, (3,3), num_filters)
    p  = c2 + input
    return relu(p)

def resnet_basic_inc(input, num_filters, strides=(2,2)):
    c1 = conv_bn_relu(input, (3,3), num_filters, strides)
    c2 = conv_bn(c1, (3,3), num_filters)
    s  = conv_bn(input, (1,1), num_filters, strides)
    p  = c2 + s
    return relu(p)

def resnet_basic_stack(input, num_stack_layers, num_filters):
    assert (num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = resnet_basic(l, num_filters)
    return l

def resnet_bottleneck(input, num_filters, inter_num_filters):
    c1 = conv_bn_relu(input, (1,1), inter_num_filters)
    c2 = conv_bn_relu(c1, (3,3), inter_num_filters)
    c3 = conv_bn(c2, (1,1), num_filters)
    p = c3 + input
    return relu(p)

def resnet_bottleneck_inc(input, num_filters, inter_num_filters, stride1x1, stride3x3):
    c1 = conv_bn_relu(input, (1,1), inter_num_filters, stride1x1)
    c2 = conv_bn_relu(c1, (3,3), inter_num_filters, stride3x3)
    c3 = conv_bn(c2, (1,1), num_filters)
    s = conv_bn(input, (1,1), num_filters, tuple(map(lambda x, y: x * y, stride1x1, stride3x3)))
    p = c3 + s
    return relu(p)

def resnet_bottleneck_stack(input, num_stack_layers, num_filters, inter_num_filters):
    assert (num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = resnet_bottleneck(l, num_filters, inter_num_filters)
    return l

#
# Defines the residual network model for classifying images
#
def create_cifar10_model(input, num_stack_layers, num_classes):
    c_map = [16, 32, 64]

    conv = conv_bn_relu(input, (3,3), c_map[0])
    r1 = resnet_basic_stack(conv, num_stack_layers, c_map[0])

    r2_1 = resnet_basic_inc(r1, c_map[1])
    r2_2 = resnet_basic_stack(r2_1, num_stack_layers-1, c_map[1])

    r3_1 = resnet_basic_inc(r2_2, c_map[2])
    r3_2 = resnet_basic_stack(r3_1, num_stack_layers-1, c_map[2])

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(8,8))(r3_2)
    z = Dense(num_classes)(pool)
    return z

#
# Defines the resnet-50 model
#
def create_resnet50_model(input, num_classes):
    c_map = [64, 128, 256, 512, 1024, 2048]
    num_layers = [2, 3, 5, 2]
    stride1x1 = (2,2)
    stride3x3 = (1,1)

    conv1 = conv_bn_relu(input, (7,7), c_map[0], (2,2))
    pool1 = MaxPooling((3,3), (2,2), True)(conv1)

    conv2_1 = resnet_bottleneck_inc(pool1, c_map[2], c_map[0], (1,1), (1,1))
    conv2_x = resnet_bottleneck_stack(conv2_1, num_layers[0], c_map[2], c_map[0])

    conv3_1 = resnet_bottleneck_inc(conv2_x, c_map[3], c_map[1], stride1x1, stride3x3)
    conv3_x = resnet_bottleneck_stack(conv3_1, num_layers[1], c_map[3], c_map[1])

    conv4_1 = resnet_bottleneck_inc(conv3_x, c_map[4], c_map[2], stride1x1, stride3x3)
    conv4_x = resnet_bottleneck_stack(conv4_1, num_layers[2], c_map[4], c_map[2])

    conv5_1 = resnet_bottleneck_inc(conv4_x, c_map[5], c_map[3], stride1x1, stride3x3)
    conv5_x = resnet_bottleneck_stack(conv5_1, num_layers[3], c_map[5], c_map[3])

    pool2 = AveragePooling((7,7), (1,1))(conv5_x)
    z = Dense(num_classes)(pool2)
    return z
