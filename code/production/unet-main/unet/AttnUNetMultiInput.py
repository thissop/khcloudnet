# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:54:57 2018

@author: Nabila Abraham
@Edited by Jesse Meyer, NASA, added prelu, he_normal init, and post train pruning
"""

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, add, multiply, Lambda, AveragePooling2D, UpSampling2D, BatchNormalization, SpatialDropout2D, PReLU

import keras as K
from keras import mixed_precision

kinit = "he_normal"#"glorot_normal"

consta = K.initializers.Constant(0.25) #NOTE(Jesse): This is the number used in the original paper
conv_act_default = PReLU

def AttnGatingBlock(x, g, inter_shape, name):
    ''' take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same feature channels (theta_x)
    then, upsample g to be same size as x 
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''
    
    shape_x = x.shape#K.int_shape(x)  # 32
    shape_g = g.shape#K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same', name=name + '_xl', dtype='float32')(x)  # 16
    shape_theta_x = theta_x.shape# K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), kernel_initializer=kinit, padding='same', name=name + "_phi_g", dtype='float32')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), kernel_initializer=kinit, padding='same', name=name+'_g_up', dtype='float32')(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x], dtype='float32')
    #act_xg = Activation('relu')(concat_xg)
    act_xg = conv_act_default(consta, name=name + "_prelu", shared_axes=[1, 2])(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same', name=name+'_psi', dtype='float32')(act_xg)
    sigmoid_xg = Activation('sigmoid', name=name + "_sig_xg")(psi)

    shape_sigmoid = sigmoid_xg.shape#K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]), interpolation="bilinear", dtype='float32')(sigmoid_xg)  # 32
    upsample_psi = Lambda(lambda x, repnum: K.ops.repeat(x, repnum, axis=3), arguments={'repnum': shape_x[3]}, name=name+'_psi_up', dtype='float32')(upsample_psi)
    
    y = multiply([upsample_psi, x], name=name+'_q_attn', dtype='float32')

    result = Conv2D(shape_x[3], (1, 1), kernel_initializer=kinit, padding='same', name=name+'_q_attn_conv', dtype='float32')(y)
    result_bn = BatchNormalization(name=name+'_q_attn_bn', dtype='float32')(result)
    return result_bn

def UnetConv2D(input, outdim, name):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_1_conv')(input)
    x = BatchNormalization(name=name + '_1_bn')(x)
    x = conv_act_default(consta, name=name + "_1_prelu", shared_axes=[1, 2])(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_2_conv')(x)
    x = BatchNormalization(name=name + '_2_bn')(x)
    x = conv_act_default(consta, name=name + "_2_prelu", shared_axes=[1, 2])(x)

    return x

def UnetGatingSignal(input, name):
    ''' this is simply 1x1 convolution, bn, activation '''
    shape = input.shape#K.int_shape(input)
    x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same", kernel_initializer=kinit, name=name + '_gate_conv', dtype='float32')(input)
    x = BatchNormalization(name=name + '_gate_bn', dtype='float32')(x)
    x = conv_act_default(consta, name=name + "_prelu", shared_axes=[1, 2])(x)
    
    return x

def attn_reg_train(input_shape, layer_count=32, weights_file=None):
    mixed_precision.set_global_policy("mixed_bfloat16")
    
    img_input = Input(shape=input_shape, name='input_scale1', dtype='float32')
    scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
    scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
    scale_img_4 = AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)

    conv1 = UnetConv2D(img_input, 1 * layer_count, name='conv1')
    #conv1 = SpatialDropout2D(0.2, name='1_sdo2d')(conv1) #NOTE(Jesse): The notion to use dropout before maxpooling is to regulate the attention input later on.
    pool1 = MaxPooling2D(name='mp1')(conv1)
    
    input2 = Conv2D(2 * layer_count, (3, 3), kernel_initializer=kinit, padding='same', name='conv_scale2')(scale_img_2)
    input2 = conv_act_default(consta, name="i2_prelu", shared_axes=[1, 2])(input2)

    input2 = concatenate([input2, pool1], axis=3)
    conv2 = UnetConv2D(input2, 2 * layer_count, name='conv2')
    #conv2 = SpatialDropout2D(0.2, name='1_sdo2d')(conv2)
    pool2 = MaxPooling2D(name='mp2')(conv2)
    
    input3 = Conv2D(4 * layer_count, (3, 3), kernel_initializer=kinit, padding='same', name='conv_scale3')(scale_img_3)
    input3 = conv_act_default(consta, name="i3_prelu", shared_axes=[1, 2])(input3)

    input3 = concatenate([input3, pool2], axis=3)
    conv3 = UnetConv2D(input3, 4 * layer_count, name='conv3')
    #conv3 = SpatialDropout2D(0.2, name='3_sdo2d')(conv3)
    pool3 = MaxPooling2D(name='mp3')(conv3)
    
    input4 = Conv2D(8 * layer_count, (3, 3), kernel_initializer=kinit, padding='same', name='conv_scale4')(scale_img_4)
    input4 = conv_act_default(consta, name="i4_prelu", shared_axes=[1, 2])(input4)

    input4 = concatenate([input4, pool3], axis=3)
    conv4 = UnetConv2D(input4, 2 * layer_count, name='conv4')
    #conv4 = SpatialDropout2D(0.2, name='4_sdo2d')(conv4)
    pool4 = MaxPooling2D(name='mp4')(conv4)
        
    center = UnetConv2D(pool4, 16 * layer_count, name='center') #NOTE(Jesse): End of encoding stage
    
    g1 = UnetGatingSignal(center,  name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 4 * layer_count, 'attn_1')

    center_t = Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', kernel_initializer=kinit)(center)
    center_t = conv_act_default(consta, name="cT_prelu", shared_axes=[1, 2])(center_t)
    up1 = concatenate([center_t, attn1], name='up1')

    g2 = UnetGatingSignal(up1, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 2 * layer_count, 'attn_2')

    up1_t = Conv2DTranspose(2 * layer_count, (3,3), strides=(2,2), padding='same', kernel_initializer=kinit)(up1)
    up1_t = conv_act_default(consta, name="u1T_prelu", shared_axes=[1, 2])(up1_t)
    up2 = concatenate([up1_t, attn2], name='up2')

    g3 = UnetGatingSignal(up2, name='g3') #up1 ???
    attn3 = AttnGatingBlock(conv2, g3, 1 * layer_count, 'attn_3')

    up2_t = Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', kernel_initializer=kinit)(up2)
    up2_t = conv_act_default(consta, name="u2T_prelu", shared_axes=[1, 2])(up2_t)
    up3 = concatenate([up2_t, attn3], name='up3')

    g4 = UnetGatingSignal(up3, name='g4')
    attn4 = AttnGatingBlock(conv1, g4, 1 * layer_count, 'attn_4')

    up3_t = Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', kernel_initializer=kinit)(up3)
    up3_t = conv_act_default(consta, name="u3T_prelu", shared_axes=[1, 2])(up3_t)
    #up4 = concatenate([up3_t, conv1], name='up4')
    up4 = concatenate([up3_t, attn4], name='up4')
    
    conv6 = UnetConv2D(up1, 8 * layer_count, name='conv6')
    conv7 = UnetConv2D(up2, 4 * layer_count, name='conv7')
    conv8 = UnetConv2D(up3, 2 * layer_count, name='conv8')
    conv9 = UnetConv2D(up4, 1 * layer_count, name='conv9')

    out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1', dtype='float32')(conv6)
    out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2', dtype='float32')(conv7)
    out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3', dtype='float32')(conv8)
    out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final', dtype='float32')(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])
    model.build(input_shape)
    if weights_file:
        model.load_weights(weights_file)
 
    return model

def attn_reg(input_shape, layer_count=32, weights_file=None):
    mixed_precision.set_global_policy("mixed_bfloat16")
    
    img_input = Input(shape=input_shape, name='input_scale1', dtype='float32')
    scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
    scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
    scale_img_4 = AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)

    conv1 = UnetConv2D(img_input, 1 * layer_count, name='conv1')
    pool1 = MaxPooling2D(name='mp1')(conv1)
    
    input2 = Conv2D(2 * layer_count, (3, 3), kernel_initializer=kinit, padding='same', name='conv_scale2')(scale_img_2)
    input2 = conv_act_default(consta, name="i2_prelu", shared_axes=[1, 2])(input2)

    input2 = concatenate([input2, pool1], axis=3)
    conv2 = UnetConv2D(input2, 2 * layer_count, name='conv2')
    pool2 = MaxPooling2D(name='mp2')(conv2)
    
    input3 = Conv2D(4 * layer_count, (3, 3), kernel_initializer=kinit, padding='same', name='conv_scale3')(scale_img_3)
    input3 = conv_act_default(consta, name="i3_prelu", shared_axes=[1, 2])(input3)

    input3 = concatenate([input3, pool2], axis=3)
    conv3 = UnetConv2D(input3, 4 * layer_count, name='conv3')
    pool3 = MaxPooling2D(name='mp3')(conv3)
    
    input4 = Conv2D(8 * layer_count, (3, 3), kernel_initializer=kinit, padding='same', name='conv_scale4')(scale_img_4)
    input4 = conv_act_default(consta, name="i4_prelu", shared_axes=[1, 2])(input4)

    input4 = concatenate([input4, pool3], axis=3)
    conv4 = UnetConv2D(input4, 2 * layer_count, name='conv4')
    pool4 = MaxPooling2D(name='mp4')(conv4)
        
    center = UnetConv2D(pool4, 16 * layer_count, name='center')
    
    g1 = UnetGatingSignal(center,  name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 4 * layer_count, 'attn_1')

    center_t = Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', kernel_initializer=kinit)(center)
    center_t = conv_act_default(consta, name="cT_prelu", shared_axes=[1, 2])(center_t)
    up1 = concatenate([center_t, attn1], name='up1')

    g2 = UnetGatingSignal(up1, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 2 * layer_count, 'attn_2')

    up1_t = Conv2DTranspose(2 * layer_count, (3,3), strides=(2,2), padding='same', kernel_initializer=kinit)(up1)
    up1_t = conv_act_default(consta, name="u1T_prelu", shared_axes=[1, 2])(up1_t)
    up2 = concatenate([up1_t, attn2], name='up2')

    g3 = UnetGatingSignal(up2, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 1 * layer_count, 'attn_3')

    up2_t = Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', kernel_initializer=kinit)(up2)
    up2_t = conv_act_default(consta, name="u2T_prelu", shared_axes=[1, 2])(up2_t)
    up3 = concatenate([up2_t, attn3], name='up3')

    g4 = UnetGatingSignal(up3, name='g4')
    attn4 = AttnGatingBlock(conv1, g4, 1 * layer_count, 'attn_4')

    up3_t = Conv2DTranspose(1 * layer_count, (3,3), strides=(2,2), padding='same', kernel_initializer=kinit)(up3)
    up3_t = conv_act_default(consta, name="u3T_prelu", shared_axes=[1, 2])(up3_t)
    #up4 = concatenate([up3_t, conv1], name='up4')
    up4 = concatenate([up3_t, attn4], name='up4')

    conv9 = UnetConv2D(up4, 1 * layer_count, name='conv9')
    out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final', dtype='float32')(conv9)

    model = Model(inputs=[img_input], outputs=[out9])
    model.build(input_shape)
    if weights_file:
        model.load_weights(weights_file)
 
    return model

def peel_trained_model(src_model, input_shape):
    #NOTE(Jesse): The attn_reg model for training has extra outputs and weights that are not necessary for inferrence and incur a substantial performance cost.
    # So we "peel" them here.
    #
    # Also, TF / Keras globally namespace layer names, so if two identical models are created, they _do not_ share the same layer names
    # if layer names are not explicit provided.  This is stupid and causes all this dumb code to exist for no reason.
    # These models have over a hundred layers and they continue to grow so I don't think it's reasonable to solve it on a per layer basis
    
    blacklisted_lyrs = ("pred1", "pred2", "pred3", "conv6", "conv7", "conv8")
    dst_model = attn_reg(input_shape)
    base_idx = 0
    for lyr_idx, dst_lyr in enumerate(dst_model.layers):
        while True:
            src_lyr = src_model.get_layer(index=lyr_idx + base_idx)
            for bl_lyrn in blacklisted_lyrs:
                if src_lyr.name.startswith(bl_lyrn):
                    base_idx += 1
                    break
            else:
                break

        src_lyr_wghts = src_lyr.get_weights()
        if len(src_lyr_wghts) == 0:
            continue

        assert src_lyr.output.shape == dst_lyr.output.shape
        
        dst_lyr.set_weights(src_lyr_wghts)

    return dst_model
