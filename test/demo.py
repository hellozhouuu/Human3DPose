#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import os
import sys
sys.path.extend([os.path.dirname(os.path.abspath(__file__))])
import cv2
import time
import numpy as np
import tensorflow as tf
import utils
# from OneEuroFilter import OneEuroFilter


class VNectEstimator:

    # the side length of the CNN input box
    box_size = 368
    # the input box size is 8 times the side length of the output heatmaps
    hm_factor = 8
    # sum of the joints to be detected
    joints_sum = 21
    # parent joint indexes of each joint (for plotting the skeletal lines)
    joint_parents = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13]

    def __init__(self):
        print('Initializing VNect Estimator...')
        # the scale factors to zoom down the input image crops
        # put different scales to get better average performance
        # for faster loops, use less scales e.g. [1], [1, 0.7]
        self.scales = [1, 0.85, 0.7]
        # load pretrained VNect model
        self.sess = tf.Session()
        if os.getcwd().endswith('src'):
            saver = tf.train.import_meta_graph('../models/tf_model/vnect_tf.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint('../models/tf_model/'))
        else:
            saver = tf.train.import_meta_graph('./models/tf_model/vnect_tf.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint('./models/tf_model/'))
        graph = tf.get_default_graph()
        self.input_crops = graph.get_tensor_by_name('Placeholder:0')
        self.heatmap_xy = graph.get_tensor_by_name('split_2:0')
        self.heatmap_xz = graph.get_tensor_by_name('split_2:1')
        self.heatmap_yz = graph.get_tensor_by_name('split_2:2')

        print('VNect Estimator initialized.')

    @staticmethod
    def gen_input_batch(img_input, box_size, scales):
        # input image --> sqrared image acceptable for the model
        img_square, scaler, [offset_x, offset_y] = utils.img_scale_squarify(img_input, box_size)
        # generate multi-scale image batch
        input_batch = []
        for scale in scales:
            img = utils.img_scale_padding(img_square, scale, box_size) if scale < 1 else img_square
            input_batch.append(img)
        # image value range: [0, 255) --> [-0.4, 0.6)
        input_batch = np.asarray(input_batch, dtype=np.float32) / 255 - 0.4
        return input_batch, scaler, [offset_x, offset_y]

    # def joint_filter(self, joints, dim=2):
    #     t = time.time()
    #     if dim == 2:
    #         for i in range(self.joints_sum):
    #             joints[i, 0] = self.filter_2d[i][0](joints[i, 0], t)
    #             joints[i, 1] = self.filter_2d[i][1](joints[i, 1], t)
    #     else:
    #         for i in range(self.joints_sum):
    #             joints[i, 0] = self.filter_3d[i][0](joints[i, 0], t)
    #             joints[i, 1] = self.filter_3d[i][1](joints[i, 1], t)
    #             joints[i, 2] = self.filter_3d[i][2](joints[i, 2], t)

    #     return joints

    # 获取的坐标是heatmap内的坐标
    def extract_3d_joints(self, heatmap_xy, heatmap_xz, heatmap_yz):
        x_hat = [i for i in range(cfg.output_hm_shape[0])]
        y_hat = [i for i in range(cfg.output_hm_shape[1])]
        z_hat = [i for i in range(cfg.output_hm_shape[2])]
        joint_3d = np.zeros((self.joints_sum,3))
        for i in range(self.joints_sum):
            joint_3d[i,0] = np.mean(heatmap_xy.dot(x_hat))
            joint_3d[i,1] = np.mean(heatmap_xy.T.dot(y_hat))
            joint_3d[i,2] = (np.mean(heatmap_yz.T.dot(z_hat))+np.mean(heatmap_xz.T.dot(z_hat)))/2
        return joint_3d

    def __call__(self, img_input):
        t0 = time.time()
        img_batch, scaler, [offset_x, offset_y] = self.gen_input_batch(img_input, self.box_size, self.scales)
        h_xy, h_xz, h_yz = self.sess.run([self.heatmap_xy,
                                        self.heatmap_xz,
                                        self.heatmap_yz],
                                       {self.input_crops: img_batch})

        
        joints_3d = self.extract_3d_joints(h_xy, h_xz, h_yz, self.hm_factor, scaler)
        # joints_3d = self.joint_filter(joints_3d, dim=3)

        joints_3d_cam = np.zeros((self.joints_sum,3))
        for i in range(self.joints_sum):
            # ha
            joints_3d_cam[i] = joints_3d[i] / heatmap_size * box3d_size * scale

        print('FPS: {:>2.2f}'.format(1 / (time.time() - t0)))
        return joints_3d


if __name__ == '__main__':
    estimator = VNectEstimator()
    j_3d = estimator(cv2.imread('../pic/test_pic.jpg'))
    print('\njoints_3d')
    for i, j in enumerate(j_3d):
        print(i, j)
