# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for p in joints:
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


class ScaleAwareHeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints

    def get_gaussian_kernel(self, sigma):
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return g

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        for p in joints:
            sigma = p[0, 3]
            g = self.get_gaussian_kernel(sigma)
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or \
                       x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb, cc:dd] = np.maximum(
                        hms[idx, aa:bb, cc:dd], g[a:b, c:d])
        return hms


class JointsGenerator():
    def __init__(self, max_num_people, num_joints, output_res, tag_per_joint):
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        self.output_res = output_res
        self.tag_per_joint = tag_per_joint

    def __call__(self, joints):
        visible_nodes = np.zeros((self.max_num_people, self.num_joints, 2))
        output_res = self.output_res
        for i in range(len(joints)):
            tot = 0
            for idx, pt in enumerate(joints[i]):
                x, y = int(pt[0]), int(pt[1])
                if pt[2] > 0 and x >= 0 and y >= 0 \
                   and x < self.output_res and y < self.output_res:
                    if self.tag_per_joint:
                        visible_nodes[i][tot] = \
                            (idx * output_res**2 + y * output_res + x, 1)
                    else:
                        visible_nodes[i][tot] = \
                            (y * output_res + x, 1)
                    tot += 1
        return visible_nodes


def get_heatmap(keypoints, ori_height, ori_width, heatmap_height, heatmap_width, heatmap_channels, sigma, scales = None):
    '''
    function that create gaussian filter heatmap based keypoints.
    :param keypoints: ndarray with shape [person_num, joints_num, 3], each joint contains three attribute, [x, y, v]
    :param ori_height: ori_img height
    :param ori_width: ori_img width
    :param heatmap_height: heatmap_height
    :param heatmap_width: heatmap_width
    :param heatmap_channels: number of joints
    :param sigma: parameter about gaussian function
    :param scales: optional. if scales not none, it means each every single point has a scale attribute,
                  scale == -1 or scale == 3 means this point is can not define or has regular size.
                  scale == 2 means this point has middle size.
                  scale == 1 means thi point has small size.
    :return: heatmap
            A ndarray with shape [heatmap_height, heatmap_width, heatmap_channels]
    '''
    factorx = heatmap_width / ori_width
    factory = heatmap_height / ori_height

    heatmap = np.zeros((heatmap_height, heatmap_width, heatmap_channels), dtype=np.float32)

    for i in range(heatmap_channels):

        single_heatmap = np.zeros(shape=(heatmap_height, heatmap_width), dtype=np.float32)
        for j in range(keypoints.shape[0]):
            people = keypoints[j]
            center_x = people[i][0] * factorx
            center_y = people[i][1] * factory

            if center_x >= heatmap_width or center_y >= heatmap_height:
                continue
            if center_x < 0 or center_y < 0:
                continue
            if center_x == 0 and center_y == 0:
                continue
            if people[i][2] == 3:
                continue
            if scales is not None:
                scale = scales[j][i][0]
                if scale == -1 or scale == 3:
                    sigma = 1. * sigma
                elif scale == 2:
                    sigma = 0.8 * sigma
                else:
                    sigma = 0.5 * sigma

            single_heatmap = gaussian(single_heatmap, center_x, center_y, sigma=sigma)

        heatmap[:, :, i] = single_heatmap

    return heatmap

def gaussian(heatmap, center_x, center_y, sigma):
    # sigma = 1.0 , 半径范围为3.5个像素
    # sigma = 2.0, 半径范围为6.5个像素
    # sigma = 0.5, 半径范围为2.0个像素


    th = 4.6052
    delta = math.sqrt(th * 2)

    height = heatmap.shape[0]
    width = heatmap.shape[1]

    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))

    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

    ## fast way
    arr_heat = heatmap[y0:y1 + 1, x0:x1 + 1]
    exp_factor = 1 / 2.0 / sigma / sigma
    x_vec = (np.arange(x0, x1 + 1) - center_x) ** 2
    y_vec = (np.arange(y0, y1 + 1) - center_y) ** 2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0

    heatmap[y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heat, arr_exp)

    ## slow way
    # for y in range(y0, y1):
    #     for x in range(x0, x1):
    #         d = (x - center_x) ** 2 + (y - center_y) ** 2
    #         exp = d / 2.0 / sigma / sigma
    #         if exp > th:
    #             continue
    #
    #         heatmap[y][x] = max(heatmap[y][x], math.exp(-exp))
    #         heatmap[y][x] = min(heatmap[y][x], 1.0)

    return heatmap




def gen_heatmap(height, width, center_x, center_y, sigma=1):
    heatmap = np.zeros((height, width), dtype=np.float32)
    th = 4.6052
    delta = math.sqrt(th * 2)
    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))
    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))
    for y in range(y0, y1):
        for x in range(x0, x1):
            d = (x - center_x) ** 2 + (y - center_y) ** 2
            exp = d / 2.0 / sigma / sigma
            if exp > th:
                continue
            heatmap[y, x] = math.exp(-exp)
    return heatmap