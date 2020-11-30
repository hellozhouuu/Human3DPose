from typing import Tuple
import tensorflow as tf
import tensorflow.contrib as tc

import pickle
import numpy as np

def vnect_model(inputs,is_training=True):

    # Conv
    conv1 = tc.layers.conv2d(inputs, kernel_size=7, num_outputs=64, stride=2, scope='conv1')
    pool1 = tc.layers.max_pool2d(conv1, kernel_size=3, padding='same', scope='pool1')

    # Residual block 2a
    res2a_branch2a = tc.layers.conv2d(pool1, kernel_size=1, num_outputs=64, scope='res2a_branch2a')
    res2a_branch2b = tc.layers.conv2d(res2a_branch2a, kernel_size=3, num_outputs=64, scope='res2a_branch2b')
    res2a_branch2c = tc.layers.conv2d(res2a_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch2c')
    res2a_branch1 = tc.layers.conv2d(pool1, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch1')
    res2a = tf.add(res2a_branch2c, res2a_branch1, name='res2a_add')
    res2a = tf.nn.relu(res2a, name='res2a')

    # Residual block 2b
    res2b_branch2a = tc.layers.conv2d(res2a, kernel_size=1, num_outputs=64, scope='res2b_branch2a')
    res2b_branch2b = tc.layers.conv2d(res2b_branch2a, kernel_size=3, num_outputs=64, scope='res2b_branch2b')
    res2b_branch2c = tc.layers.conv2d(res2b_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2b_branch2c')
    res2b = tf.add(res2b_branch2c, res2a, name='res2b_add')
    res2b = tf.nn.relu(res2b, name='res2b')

    # Residual block 2c
    res2c_branch2a = tc.layers.conv2d(res2b, kernel_size=1, num_outputs=64, scope='res2c_branch2a')
    res2c_branch2b = tc.layers.conv2d(res2c_branch2a, kernel_size=3, num_outputs=64, scope='res2c_branch2b')
    res2c_branch2c = tc.layers.conv2d(res2c_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2c_branch2c')
    res2c = tf.add(res2c_branch2c, res2b, name='res2c_add')
    res2c = tf.nn.relu(res2c, name='res2c')

    # Residual block 3a
    res3a_branch2a = tc.layers.conv2d(res2c, kernel_size=1, num_outputs=128, stride=2, scope='res3a_branch2a')
    res3a_branch2b = tc.layers.conv2d(res3a_branch2a, kernel_size=3, num_outputs=128, scope='res3a_branch2b')
    res3a_branch2c = tc.layers.conv2d(res3a_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3a_branch2c')
    res3a_branch1 = tc.layers.conv2d(res2c, kernel_size=1, num_outputs=512, activation_fn=None, stride=2, scope='res3a_branch1')
    res3a = tf.add(res3a_branch2c, res3a_branch1, name='res3a_add')
    res3a = tf.nn.relu(res3a, name='res3a')

    # Residual block 3b
    res3b_branch2a = tc.layers.conv2d(res3a, kernel_size=1, num_outputs=128, scope='res3b_branch2a')
    res3b_branch2b = tc.layers.conv2d(res3b_branch2a, kernel_size=3, num_outputs=128,scope='res3b_branch2b')
    res3b_branch2c = tc.layers.conv2d(res3b_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3b_branch2c')
    res3b = tf.add(res3b_branch2c, res3a, name='res3b_add')
    res3b = tf.nn.relu(res3b, name='res3b')

    # Residual block 3c
    res3c_branch2a = tc.layers.conv2d(res3b, kernel_size=1, num_outputs=128, scope='res3c_branch2a')
    res3c_branch2b = tc.layers.conv2d(res3c_branch2a, kernel_size=3, num_outputs=128,scope='res3c_branch2b')
    res3c_branch2c = tc.layers.conv2d(res3c_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3c_branch2c')
    res3c = tf.add(res3c_branch2c, res3b, name='res3c_add')
    res3c = tf.nn.relu(res3c, name='res3c')

    # Residual block 3d
    res3d_branch2a = tc.layers.conv2d(res3c, kernel_size=1, num_outputs=128, scope='res3d_branch2a')
    res3d_branch2b = tc.layers.conv2d(res3d_branch2a, kernel_size=3, num_outputs=128,scope='res3d_branch2b')
    res3d_branch2c = tc.layers.conv2d(res3d_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3d_branch2c')
    res3d = tf.add(res3d_branch2c, res3c, name='res3d_add')
    res3d = tf.nn.relu(res3d, name='res3d')

    # Residual block 4a
    res4a_branch2a = tc.layers.conv2d(res3d, kernel_size=1, num_outputs=256, stride=2, scope='res4a_branch2a')
    res4a_branch2b = tc.layers.conv2d(res4a_branch2a, kernel_size=3, num_outputs=256,scope='res4a_branch2b')
    res4a_branch2c = tc.layers.conv2d(res4a_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None,scope='res4a_branch2c')
    res4a_branch1 = tc.layers.conv2d(res3d, kernel_size=1, num_outputs=1024, activation_fn=None, stride=2, scope='res4a_branch1')
    res4a = tf.add(res4a_branch2c, res4a_branch1, name='res4a_add')
    res4a = tf.nn.relu(res4a, name='res4a')

    # Residual block 4b
    res4b_branch2a = tc.layers.conv2d(res4a, kernel_size=1, num_outputs=256, scope='res4b_branch2a')
    res4b_branch2b = tc.layers.conv2d(res4b_branch2a, kernel_size=3, num_outputs=256, scope='res4b_branch2b')
    res4b_branch2c = tc.layers.conv2d(res4b_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4b_branch2c')
    res4b = tf.add(res4b_branch2c, res4a, name='res4b_add')
    res4b = tf.nn.relu(res4b, name='res4b')

    # Residual block 4c
    res4c_branch2a = tc.layers.conv2d(res4b, kernel_size=1, num_outputs=256, scope='res4c_branch2a')
    res4c_branch2b = tc.layers.conv2d(res4c_branch2a, kernel_size=3, num_outputs=256, scope='res4c_branch2b')
    res4c_branch2c = tc.layers.conv2d(res4c_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4c_branch2c')
    res4c = tf.add(res4c_branch2c, res4b, name='res4c_add')
    res4c = tf.nn.relu(res4c, name='res4c')

    # Residual block 4d
    res4d_branch2a = tc.layers.conv2d(res4c, kernel_size=1, num_outputs=256, scope='res4d_branch2a')
    res4d_branch2b = tc.layers.conv2d(res4d_branch2a, kernel_size=3, num_outputs=256, scope='res4d_branch2b')
    res4d_branch2c = tc.layers.conv2d(res4d_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4d_branch2c')
    res4d = tf.add(res4d_branch2c, res4c, name='res4d_add')
    res4d = tf.nn.relu(res4d, name='res4d')

    # Residual block 4e
    res4e_branch2a = tc.layers.conv2d(res4d, kernel_size=1, num_outputs=256, scope='res4e_branch2a')
    res4e_branch2b = tc.layers.conv2d(res4e_branch2a, kernel_size=3, num_outputs=256, scope='res4e_branch2b')
    res4e_branch2c = tc.layers.conv2d(res4e_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4e_branch2c')
    res4e = tf.add(res4e_branch2c, res4d, name='res4e_add')
    res4e = tf.nn.relu(res4e, name='res4e')

    # Residual block 4f
    res4f_branch2a = tc.layers.conv2d(res4e, kernel_size=1, num_outputs=256, scope='res4f_branch2a')
    res4f_branch2b = tc.layers.conv2d(res4f_branch2a, kernel_size=3, num_outputs=256, scope='res4f_branch2b')
    res4f_branch2c = tc.layers.conv2d(res4f_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4f_branch2c')
    res4f = tf.add(res4f_branch2c, res4e, name='res4f_add')
    res4f = tf.nn.relu(res4f, name='res4f')

    # Residual block 5a
    res5a_branch2a_new = tc.layers.conv2d(res4f, kernel_size=1, num_outputs=512, scope='res5a_branch2a_new')
    res5a_branch2b_new = tc.layers.conv2d(res5a_branch2a_new, kernel_size=3, num_outputs=512, scope='res5a_branch2b_new')
    res5a_branch2c_new = tc.layers.conv2d(res5a_branch2b_new, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch2c_new')
    res5a_branch1_new = tc.layers.conv2d(res4f, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch1_new')
    res5a = tf.add(res5a_branch2c_new, res5a_branch1_new, name='res5a_add')
    res5a = tf.nn.relu(res5a, name='res5a')

    # Residual block 5b
    res5b_branch2a_new = tc.layers.conv2d(res5a, kernel_size=1, num_outputs=256, scope='res5b_branch2a_new')
    res5b_branch2b_new = tc.layers.conv2d(res5b_branch2a_new, kernel_size=3, num_outputs=128, scope='res5b_branch2b_new')
    res5b_branch2c_new = tc.layers.conv2d(res5b_branch2b_new, kernel_size=1, num_outputs=256, scope='res5b_branch2c_new')

    # Transpose Conv
    res5c_branch1a = tf.layers.conv2d_transpose(res5b_branch2c_new, kernel_size=4, filters=63, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch1a')
    res5c_branch2a = tf.layers.conv2d_transpose(res5b_branch2c_new, kernel_size=4, filters=128, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch2a')
    bn5c_branch2a = tc.layers.batch_norm(res5c_branch2a, scale=True, is_training=is_training, scope='bn5c_branch2a')
    bn5c_branch2a = tf.nn.relu(bn5c_branch2a)

    res5c_delta_x, res5c_delta_y, res5c_delta_z = tf.split(res5c_branch1a, num_or_size_splits=3, axis=3)
    res5c_branch1a_sqr = tf.multiply(res5c_branch1a, res5c_branch1a, name='res5c_branch1a_sqr')
    res5c_delta_x_sqr, res5c_delta_y_sqr, res5c_delta_z_sqr = tf.split(res5c_branch1a_sqr, num_or_size_splits=3, axis=3)
    res5c_bone_length_sqr = tf.add(tf.add(res5c_delta_x_sqr, res5c_delta_y_sqr), res5c_delta_z_sqr)
    res5c_bone_length = tf.sqrt(res5c_bone_length_sqr)

    res5c_branch2a_feat = tf.concat([bn5c_branch2a, res5c_delta_x, res5c_delta_y, res5c_delta_z, res5c_bone_length],
                                            axis=3, name='res5c_branch2a_feat')

    res5c_branch2b = tc.layers.conv2d(res5c_branch2a_feat, kernel_size=3, num_outputs=128, scope='res5c_branch2b')
    res5c_branch2c = tf.layers.conv2d(res5c_branch2b, kernel_size=1, filters=68, activation=None, use_bias=False, name='res5c_branch2c')
    heapmap, x_heatmap, y_heatmap, z_heatmap = tf.split(res5c_branch2c, num_or_size_splits=4, axis=3)

    return heapmap,x_heatmap,y_heatmap,z_heatmap

def vnect_model_v1(inputs,is_training=True):

    # Conv
    conv1 = tc.layers.conv2d(inputs, kernel_size=7, num_outputs=64, stride=2, scope='conv1')
    pool1 = tc.layers.max_pool2d(conv1, kernel_size=3, padding='same', scope='pool1')

    # Residual block 2a
    res2a_branch2a = tc.layers.conv2d(pool1, kernel_size=1, num_outputs=64, scope='res2a_branch2a')
    res2a_branch2b = tc.layers.conv2d(res2a_branch2a, kernel_size=3, num_outputs=64, scope='res2a_branch2b')
    res2a_branch2c = tc.layers.conv2d(res2a_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch2c')
    res2a_branch1 = tc.layers.conv2d(pool1, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch1')
    res2a = tf.add(res2a_branch2c, res2a_branch1, name='res2a_add')
    res2a = tf.nn.relu(res2a, name='res2a')

    # Residual block 2b
    res2b_branch2a = tc.layers.conv2d(res2a, kernel_size=1, num_outputs=64, scope='res2b_branch2a')
    res2b_branch2b = tc.layers.conv2d(res2b_branch2a, kernel_size=3, num_outputs=64, scope='res2b_branch2b')
    res2b_branch2c = tc.layers.conv2d(res2b_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2b_branch2c')
    res2b = tf.add(res2b_branch2c, res2a, name='res2b_add')
    res2b = tf.nn.relu(res2b, name='res2b')

    # Residual block 2c
    res2c_branch2a = tc.layers.conv2d(res2b, kernel_size=1, num_outputs=64, scope='res2c_branch2a')
    res2c_branch2b = tc.layers.conv2d(res2c_branch2a, kernel_size=3, num_outputs=64, scope='res2c_branch2b')
    res2c_branch2c = tc.layers.conv2d(res2c_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2c_branch2c')
    res2c = tf.add(res2c_branch2c, res2b, name='res2c_add')
    res2c = tf.nn.relu(res2c, name='res2c')

    # Residual block 3a
    res3a_branch2a = tc.layers.conv2d(res2c, kernel_size=1, num_outputs=128, stride=2, scope='res3a_branch2a')
    res3a_branch2b = tc.layers.conv2d(res3a_branch2a, kernel_size=3, num_outputs=128, scope='res3a_branch2b')
    res3a_branch2c = tc.layers.conv2d(res3a_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3a_branch2c')
    res3a_branch1 = tc.layers.conv2d(res2c, kernel_size=1, num_outputs=512, activation_fn=None, stride=2, scope='res3a_branch1')
    res3a = tf.add(res3a_branch2c, res3a_branch1, name='res3a_add')
    res3a = tf.nn.relu(res3a, name='res3a')

    # Residual block 3b
    res3b_branch2a = tc.layers.conv2d(res3a, kernel_size=1, num_outputs=128, scope='res3b_branch2a')
    res3b_branch2b = tc.layers.conv2d(res3b_branch2a, kernel_size=3, num_outputs=128,scope='res3b_branch2b')
    res3b_branch2c = tc.layers.conv2d(res3b_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3b_branch2c')
    res3b = tf.add(res3b_branch2c, res3a, name='res3b_add')
    res3b = tf.nn.relu(res3b, name='res3b')

    # Residual block 3c
    res3c_branch2a = tc.layers.conv2d(res3b, kernel_size=1, num_outputs=128, scope='res3c_branch2a')
    res3c_branch2b = tc.layers.conv2d(res3c_branch2a, kernel_size=3, num_outputs=128,scope='res3c_branch2b')
    res3c_branch2c = tc.layers.conv2d(res3c_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3c_branch2c')
    res3c = tf.add(res3c_branch2c, res3b, name='res3c_add')
    res3c = tf.nn.relu(res3c, name='res3c')

    # Residual block 3d
    res3d_branch2a = tc.layers.conv2d(res3c, kernel_size=1, num_outputs=128, scope='res3d_branch2a')
    res3d_branch2b = tc.layers.conv2d(res3d_branch2a, kernel_size=3, num_outputs=128,scope='res3d_branch2b')
    res3d_branch2c = tc.layers.conv2d(res3d_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3d_branch2c')
    res3d = tf.add(res3d_branch2c, res3c, name='res3d_add')
    res3d = tf.nn.relu(res3d, name='res3d')

    # Residual block 4a
    res4a_branch2a = tc.layers.conv2d(res3d, kernel_size=1, num_outputs=256, stride=2, scope='res4a_branch2a')
    res4a_branch2b = tc.layers.conv2d(res4a_branch2a, kernel_size=3, num_outputs=256,scope='res4a_branch2b')
    res4a_branch2c = tc.layers.conv2d(res4a_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None,scope='res4a_branch2c')
    res4a_branch1 = tc.layers.conv2d(res3d, kernel_size=1, num_outputs=1024, activation_fn=None, stride=2, scope='res4a_branch1')
    res4a = tf.add(res4a_branch2c, res4a_branch1, name='res4a_add')
    res4a = tf.nn.relu(res4a, name='res4a')

    # Residual block 4b
    res4b_branch2a = tc.layers.conv2d(res4a, kernel_size=1, num_outputs=256, scope='res4b_branch2a')
    res4b_branch2b = tc.layers.conv2d(res4b_branch2a, kernel_size=3, num_outputs=256, scope='res4b_branch2b')
    res4b_branch2c = tc.layers.conv2d(res4b_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4b_branch2c')
    res4b = tf.add(res4b_branch2c, res4a, name='res4b_add')
    res4b = tf.nn.relu(res4b, name='res4b')

    # Residual block 4c
    res4c_branch2a = tc.layers.conv2d(res4b, kernel_size=1, num_outputs=256, scope='res4c_branch2a')
    res4c_branch2b = tc.layers.conv2d(res4c_branch2a, kernel_size=3, num_outputs=256, scope='res4c_branch2b')
    res4c_branch2c = tc.layers.conv2d(res4c_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4c_branch2c')
    res4c = tf.add(res4c_branch2c, res4b, name='res4c_add')
    res4c = tf.nn.relu(res4c, name='res4c')

    # Residual block 4d
    res4d_branch2a = tc.layers.conv2d(res4c, kernel_size=1, num_outputs=256, scope='res4d_branch2a')
    res4d_branch2b = tc.layers.conv2d(res4d_branch2a, kernel_size=3, num_outputs=256, scope='res4d_branch2b')
    res4d_branch2c = tc.layers.conv2d(res4d_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4d_branch2c')
    res4d = tf.add(res4d_branch2c, res4c, name='res4d_add')
    res4d = tf.nn.relu(res4d, name='res4d')

    # Residual block 4e
    res4e_branch2a = tc.layers.conv2d(res4d, kernel_size=1, num_outputs=256, scope='res4e_branch2a')
    res4e_branch2b = tc.layers.conv2d(res4e_branch2a, kernel_size=3, num_outputs=256, scope='res4e_branch2b')
    res4e_branch2c = tc.layers.conv2d(res4e_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4e_branch2c')
    res4e = tf.add(res4e_branch2c, res4d, name='res4e_add')
    res4e = tf.nn.relu(res4e, name='res4e')

    # Residual block 4f
    res4f_branch2a = tc.layers.conv2d(res4e, kernel_size=1, num_outputs=256, scope='res4f_branch2a')
    res4f_branch2b = tc.layers.conv2d(res4f_branch2a, kernel_size=3, num_outputs=256, scope='res4f_branch2b')
    res4f_branch2c = tc.layers.conv2d(res4f_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4f_branch2c')
    res4f = tf.add(res4f_branch2c, res4e, name='res4f_add')
    res4f = tf.nn.relu(res4f, name='res4f')

    # Residual block 5a
    res5a_branch2a_new = tc.layers.conv2d(res4f, kernel_size=1, num_outputs=512, scope='res5a_branch2a_new')
    res5a_branch2b_new = tc.layers.conv2d(res5a_branch2a_new, kernel_size=3, num_outputs=512, scope='res5a_branch2b_new')
    res5a_branch2c_new = tc.layers.conv2d(res5a_branch2b_new, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch2c_new')
    res5a_branch1_new = tc.layers.conv2d(res4f, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch1_new')
    res5a = tf.add(res5a_branch2c_new, res5a_branch1_new, name='res5a_add')
    res5a = tf.nn.relu(res5a, name='res5a')

    # Residual block 5b
    res5b_branch2a_new = tc.layers.conv2d(res5a, kernel_size=1, num_outputs=256, scope='res5b_branch2a_new')
    res5b_branch2b_new = tc.layers.conv2d(res5b_branch2a_new, kernel_size=3, num_outputs=128, scope='res5b_branch2b_new')
    res5b_branch2c_new = tc.layers.conv2d(res5b_branch2b_new, kernel_size=1, num_outputs=256, scope='res5b_branch2c_new')

    # Transpose Conv
    res5c_branch1a = tf.layers.conv2d_transpose(res5b_branch2c_new, kernel_size=4, filters=63, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch1a')
    res5c_branch2a = tf.layers.conv2d_transpose(res5b_branch2c_new, kernel_size=4, filters=128, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch2a')
    bn5c_branch2a = tc.layers.batch_norm(res5c_branch2a, scale=True, is_training=is_training, scope='bn5c_branch2a')
    bn5c_branch2a = tf.nn.relu(bn5c_branch2a)

    res5c_delta_x, res5c_delta_y, res5c_delta_z = tf.split(res5c_branch1a, num_or_size_splits=3, axis=3)
    res5c_branch1a_sqr = tf.multiply(res5c_branch1a, res5c_branch1a, name='res5c_branch1a_sqr')
    res5c_delta_x_sqr, res5c_delta_y_sqr, res5c_delta_z_sqr = tf.split(res5c_branch1a_sqr, num_or_size_splits=3, axis=3)
    res5c_bone_length_sqr = tf.add(tf.add(res5c_delta_x_sqr, res5c_delta_y_sqr), res5c_delta_z_sqr)
    res5c_bone_length = tf.sqrt(res5c_bone_length_sqr)

    res5c_branch2a_feat = tf.concat([bn5c_branch2a, res5c_delta_x, res5c_delta_y, res5c_delta_z, res5c_bone_length],
                                            axis=3, name='res5c_branch2a_feat')

    res5c_branch2b = tc.layers.conv2d(res5c_branch2a_feat, kernel_size=3, num_outputs=128, scope='res5c_branch2b')
    res5c_branch2c = tf.layers.conv2d(res5c_branch2b, kernel_size=1, filters=51, activation=None, use_bias=False, name='res5c_branch2c')
    x_heatmap, y_heatmap, z_heatmap = tf.split(res5c_branch2c, num_or_size_splits=3, axis=3)

    return x_heatmap,y_heatmap,z_heatmap

def vnect_model_v2(inputs,is_training=True):

    # Conv
    conv1 = tc.layers.conv2d(inputs, kernel_size=7, num_outputs=64, stride=2, scope='conv1')
    pool1 = tc.layers.max_pool2d(conv1, kernel_size=3, padding='same', scope='pool1')

    # Residual block 2a
    res2a_branch2a = tc.layers.conv2d(pool1, kernel_size=1, num_outputs=64, scope='res2a_branch2a')
    res2a_branch2b = tc.layers.conv2d(res2a_branch2a, kernel_size=3, num_outputs=64, scope='res2a_branch2b')
    res2a_branch2c = tc.layers.conv2d(res2a_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch2c')
    res2a_branch1 = tc.layers.conv2d(pool1, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2a_branch1')
    res2a = tf.add(res2a_branch2c, res2a_branch1, name='res2a_add')
    res2a = tf.nn.relu(res2a, name='res2a')

    # Residual block 2b
    res2b_branch2a = tc.layers.conv2d(res2a, kernel_size=1, num_outputs=64, scope='res2b_branch2a')
    res2b_branch2b = tc.layers.conv2d(res2b_branch2a, kernel_size=3, num_outputs=64, scope='res2b_branch2b')
    res2b_branch2c = tc.layers.conv2d(res2b_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2b_branch2c')
    res2b = tf.add(res2b_branch2c, res2a, name='res2b_add')
    res2b = tf.nn.relu(res2b, name='res2b')

    # Residual block 2c
    res2c_branch2a = tc.layers.conv2d(res2b, kernel_size=1, num_outputs=64, scope='res2c_branch2a')
    res2c_branch2b = tc.layers.conv2d(res2c_branch2a, kernel_size=3, num_outputs=64, scope='res2c_branch2b')
    res2c_branch2c = tc.layers.conv2d(res2c_branch2b, kernel_size=1, num_outputs=256, activation_fn=None, scope='res2c_branch2c')
    res2c = tf.add(res2c_branch2c, res2b, name='res2c_add')
    res2c = tf.nn.relu(res2c, name='res2c')

    # Residual block 3a
    res3a_branch2a = tc.layers.conv2d(res2c, kernel_size=1, num_outputs=128, stride=2, scope='res3a_branch2a')
    res3a_branch2b = tc.layers.conv2d(res3a_branch2a, kernel_size=3, num_outputs=128, scope='res3a_branch2b')
    res3a_branch2c = tc.layers.conv2d(res3a_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3a_branch2c')
    res3a_branch1 = tc.layers.conv2d(res2c, kernel_size=1, num_outputs=512, activation_fn=None, stride=2, scope='res3a_branch1')
    res3a = tf.add(res3a_branch2c, res3a_branch1, name='res3a_add')
    res3a = tf.nn.relu(res3a, name='res3a')

    # Residual block 3b
    res3b_branch2a = tc.layers.conv2d(res3a, kernel_size=1, num_outputs=128, scope='res3b_branch2a')
    res3b_branch2b = tc.layers.conv2d(res3b_branch2a, kernel_size=3, num_outputs=128,scope='res3b_branch2b')
    res3b_branch2c = tc.layers.conv2d(res3b_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3b_branch2c')
    res3b = tf.add(res3b_branch2c, res3a, name='res3b_add')
    res3b = tf.nn.relu(res3b, name='res3b')

    # Residual block 3c
    res3c_branch2a = tc.layers.conv2d(res3b, kernel_size=1, num_outputs=128, scope='res3c_branch2a')
    res3c_branch2b = tc.layers.conv2d(res3c_branch2a, kernel_size=3, num_outputs=128,scope='res3c_branch2b')
    res3c_branch2c = tc.layers.conv2d(res3c_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3c_branch2c')
    res3c = tf.add(res3c_branch2c, res3b, name='res3c_add')
    res3c = tf.nn.relu(res3c, name='res3c')

    # Residual block 3d
    res3d_branch2a = tc.layers.conv2d(res3c, kernel_size=1, num_outputs=128, scope='res3d_branch2a')
    res3d_branch2b = tc.layers.conv2d(res3d_branch2a, kernel_size=3, num_outputs=128,scope='res3d_branch2b')
    res3d_branch2c = tc.layers.conv2d(res3d_branch2b, kernel_size=1, num_outputs=512, activation_fn=None,scope='res3d_branch2c')
    res3d = tf.add(res3d_branch2c, res3c, name='res3d_add')
    res3d = tf.nn.relu(res3d, name='res3d')


    # Residual block 4a
    res4a_branch2a = tc.layers.conv2d(res3d, kernel_size=1, num_outputs=256, stride=2, scope='res4a_branch2a')
    res4a_branch2b = tc.layers.conv2d(res4a_branch2a, kernel_size=3, num_outputs=256,scope='res4a_branch2b')
    res4a_branch2c = tc.layers.conv2d(res4a_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None,scope='res4a_branch2c')
    res4a_branch1 = tc.layers.conv2d(res3d, kernel_size=1, num_outputs=1024, activation_fn=None, stride=2, scope='res4a_branch1')
    res4a = tf.add(res4a_branch2c, res4a_branch1, name='res4a_add')
    res4a = tf.nn.relu(res4a, name='res4a')

    # Residual block 4b
    res4b_branch2a = tc.layers.conv2d(res4a, kernel_size=1, num_outputs=256, scope='res4b_branch2a')
    res4b_branch2b = tc.layers.conv2d(res4b_branch2a, kernel_size=3, num_outputs=256, scope='res4b_branch2b')
    res4b_branch2c = tc.layers.conv2d(res4b_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4b_branch2c')
    res4b = tf.add(res4b_branch2c, res4a, name='res4b_add')
    res4b = tf.nn.relu(res4b, name='res4b')

    # Residual block 4c
    res4c_branch2a = tc.layers.conv2d(res4b, kernel_size=1, num_outputs=256, scope='res4c_branch2a')
    res4c_branch2b = tc.layers.conv2d(res4c_branch2a, kernel_size=3, num_outputs=256, scope='res4c_branch2b')
    res4c_branch2c = tc.layers.conv2d(res4c_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4c_branch2c')
    res4c = tf.add(res4c_branch2c, res4b, name='res4c_add')
    res4c = tf.nn.relu(res4c, name='res4c')

    # Residual block 4d
    res4d_branch2a = tc.layers.conv2d(res4c, kernel_size=1, num_outputs=256, scope='res4d_branch2a')
    res4d_branch2b = tc.layers.conv2d(res4d_branch2a, kernel_size=3, num_outputs=256, scope='res4d_branch2b')
    res4d_branch2c = tc.layers.conv2d(res4d_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4d_branch2c')
    res4d = tf.add(res4d_branch2c, res4c, name='res4d_add')
    res4d = tf.nn.relu(res4d, name='res4d')

    # Residual block 4e
    res4e_branch2a = tc.layers.conv2d(res4d, kernel_size=1, num_outputs=256, scope='res4e_branch2a')
    res4e_branch2b = tc.layers.conv2d(res4e_branch2a, kernel_size=3, num_outputs=256, scope='res4e_branch2b')
    res4e_branch2c = tc.layers.conv2d(res4e_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4e_branch2c')
    res4e = tf.add(res4e_branch2c, res4d, name='res4e_add')
    res4e = tf.nn.relu(res4e, name='res4e')

    # Residual block 4f
    res4f_branch2a = tc.layers.conv2d(res4e, kernel_size=1, num_outputs=256, scope='res4f_branch2a')
    res4f_branch2b = tc.layers.conv2d(res4f_branch2a, kernel_size=3, num_outputs=256, scope='res4f_branch2b')
    res4f_branch2c = tc.layers.conv2d(res4f_branch2b, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res4f_branch2c')
    res4f = tf.add(res4f_branch2c, res4e, name='res4f_add')
    res4f = tf.nn.relu(res4f, name='res4f')



    # Residual block 5a
    res5a_branch2a_new = tc.layers.conv2d(res4f, kernel_size=1, num_outputs=512, scope='res5a_branch2a_new')
    res5a_branch2b_new = tc.layers.conv2d(res5a_branch2a_new, kernel_size=3, num_outputs=512, scope='res5a_branch2b_new')
    res5a_branch2c_new = tc.layers.conv2d(res5a_branch2b_new, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch2c_new')
    res5a_branch1_new = tc.layers.conv2d(res4f, kernel_size=1, num_outputs=1024, activation_fn=None, scope='res5a_branch1_new')
    res5a = tf.add(res5a_branch2c_new, res5a_branch1_new, name='res5a_add')
    res5a = tf.nn.relu(res5a, name='res5a')
    #
    # # Residual block 5b
    res5b_branch2a_new = tc.layers.conv2d(res5a, kernel_size=1, num_outputs=256, scope='res5b_branch2a_new')
    res5b_branch2b_new = tc.layers.conv2d(res5b_branch2a_new, kernel_size=3, num_outputs=128, scope='res5b_branch2b_new')
    res5b_branch2c_new = tc.layers.conv2d(res5b_branch2b_new, kernel_size=1, num_outputs=256, scope='res5b_branch2c_new')
    #
    #
    transpose_out=tf.layers.conv2d_transpose(res5b_branch2c_new, kernel_size=4, filters=51, activation=None, strides=2, padding='same', use_bias=False, name='transpose_conv1')
    bn_output = tc.layers.batch_norm(transpose_out, scale=True, is_training=is_training, scope='bn1')
    final_output=tf.nn.relu(bn_output)
    # # output heatmap
    x_heatmap, y_heatmap, z_heatmap = tf.split(final_output, num_or_size_splits=3, axis=3)

    # Transpose Conv
    # res5c_branch1a = tf.layers.conv2d_transpose(res5b_branch2c_new, kernel_size=4, filters=63, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch1a')
    # res5c_branch2a = tf.layers.conv2d_transpose(res5b_branch2c_new, kernel_size=4, filters=128, activation=None, strides=2, padding='same', use_bias=False, name='res5c_branch2a')
    # bn5c_branch2a = tc.layers.batch_norm(res5c_branch2a, scale=True, is_training=is_training, scope='bn5c_branch2a')
    # bn5c_branch2a = tf.nn.relu(bn5c_branch2a)
    #
    # res5c_delta_x, res5c_delta_y, res5c_delta_z = tf.split(res5c_branch1a, num_or_size_splits=3, axis=3)
    # res5c_branch1a_sqr = tf.multiply(res5c_branch1a, res5c_branch1a, name='res5c_branch1a_sqr')
    # res5c_delta_x_sqr, res5c_delta_y_sqr, res5c_delta_z_sqr = tf.split(res5c_branch1a_sqr, num_or_size_splits=3, axis=3)
    # res5c_bone_length_sqr = tf.add(tf.add(res5c_delta_x_sqr, res5c_delta_y_sqr), res5c_delta_z_sqr)
    # res5c_bone_length = tf.sqrt(res5c_bone_length_sqr)
    #
    # res5c_branch2a_feat = tf.concat([bn5c_branch2a, res5c_delta_x, res5c_delta_y, res5c_delta_z, res5c_bone_length],
    #                                         axis=3, name='res5c_branch2a_feat')
    #
    # res5c_branch2b = tc.layers.conv2d(res5c_branch2a_feat, kernel_size=3, num_outputs=128, scope='res5c_branch2b')
    # res5c_branch2c = tf.layers.conv2d(res5c_branch2b, kernel_size=1, filters=51, activation=None, use_bias=False, name='res5c_branch2c')
    # x_heatmap, y_heatmap, z_heatmap = tf.split(res5c_branch2c, num_or_size_splits=3, axis=3)

    return x_heatmap,y_heatmap,z_heatmap
if __name__ == '__main__':
    # import hiddenlayer as hl
    # import hiddenlayer.transforms as ht
    import os
    # Hide GPUs. Not needed for this demo.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    with tf.Session() as sess:
        with tf.Graph().as_default() as tf_graph:
            # Setup input placeholder
            inputs = tf.placeholder(tf.float32, shape=(1, 368, 368, 3))
            # Build model
            hm,xm,ym,zm = vnect_model(inputs,is_training=True)
            # Build HiddenLayer graph
            
            print(xm.shape)
    # hl_graph.save('lightweight', format='png')
