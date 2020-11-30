#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


import os
import os.path as osp
import re
import cv2
import copy
import math
# import h5py
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import json
# import torchvision.transforms as tv_transforms
from .preprocessing import load_img, process_bbox, augmentation
from .transforms import world2cam, cam2pixel, transform_joint_to_other_db
from .config import cfg
class Human36M:
    # joint indexes used in vnect (totally 21 joints)
    vnect_ids = [i for i in range(17)]

    def __init__(self, bpath,parameters, data_split='train', subjects=None, if_train_set=True):
        # select training set or test set
        # self.transform = tv_transforms.ToTensor()
        self.if_train_set = if_train_set
        self.h36m_joints_name = ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',
                    'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top',
                    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
        self.h36m_joint_num = 17
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.h36m_flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
        # load data index path
        self.img_dir = osp.join(bpath,  'Human36M', 'images')
        self.annot_path = osp.join(bpath, 'Human36M', 'annotations')
        self.parameter=parameters

        # # load frame path list
        # self.df = pd.read_csv(self.list_path, sep=' ', header=None)  # load path data
        # self.df = self.df.loc[self.df[0].isin(self.subjects), 1].sample(frac=1)  # select and suffle
        # frame annotation
        # self.annots = h5py.File(self.annot_path, 'r')
        self.data_split = data_split
        if self.data_split =='train':
            self.subject=parameters['subject']
            self.subsampling_ratio=parameters['train_subsampling_ratio']
        else:
            self.subject=parameters['test_subjects']
            self.subsampling_ratio = parameters['test_subsampling_ratio'] 
        self.datalist = self.load_data_list()

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            # subject = [1,5,6,7,8]
            subject=self.parameter['subject']
        elif self.data_split == 'test':
            subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject

    # def _load_data_list(self):
    #     subject_list = self.get_subject()
    #     sampling_ratio = self.get_subsampling_ratio()
        
    def load_data_list(self):
        # subject_list = self.get_subject()
        # sampling_ratio = self.get_subsampling_ratio()
        subject_list = self.subject
        sampling_ratio = self.subsampling_ratio
        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
        db.createIndex()


        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            
            # check subject and frame_idx
            frame_idx = img['frame_idx']
            if frame_idx % sampling_ratio != 0:
                continue

            # check smpl parameter exist
            subject = img['subject']; action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx']

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
            
            # only use frontal camera following previous works (HMR and SPIN)
            if self.data_split == 'test' and str(cam_idx) != '4':
                continue
                
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)
            joint_valid = np.ones((self.h36m_joint_num,1))
        
            # if self.data_split == 'train':
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue
            # else:
            #     bbox=None
                
            datalist.append({
                'img_path': img_path,
                'img_id': image_id,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'cam_param': cam_param})
            
        return datalist



    # def __getitem__(self, idx):
    #     input_img_shape=[parameter['height'],parameter['width']]
    #     output_hm_shape=parameter['output_hm_shape']
    #     bbox_3d_size=parameter['box_3d_size']
    #     data = copy.deepcopy(self.datalist[idx])
    #     img_path, img_shape, bbox, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['cam_param']
        
    #     # img
    #     img = load_img(img_path)
    #     img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
    #     # img = self.transform(img.astype(np.float32))/255.
        
    #     if self.data_split == 'train':
    #         # h36m gt
    #         h36m_joint_img = data['joint_img']
    #         h36m_joint_cam = data['joint_cam']
    #         h36m_joint_cam = h36m_joint_cam - h36m_joint_cam[self.h36m_root_joint_idx,None,:] # root-relative
    #         h36m_joint_valid = data['joint_valid']
    #         if do_flip:
    #             h36m_joint_cam[:,0] = -h36m_joint_cam[:,0]
    #             h36m_joint_img[:,0] = img_shape[1] - 1 - h36m_joint_img[:,0]
    #             for pair in self.h36m_flip_pairs:
    #                 h36m_joint_img[pair[0],:], h36m_joint_img[pair[1],:] = h36m_joint_img[pair[1],:].copy(), h36m_joint_img[pair[0],:].copy()
    #                 h36m_joint_cam[pair[0],:], h36m_joint_cam[pair[1],:] = h36m_joint_cam[pair[1],:].copy(), h36m_joint_cam[pair[0],:].copy()
    #                 h36m_joint_valid[pair[0],:], h36m_joint_valid[pair[1],:] = h36m_joint_valid[pair[1],:].copy(), h36m_joint_valid[pair[0],:].copy()

    #         h36m_joint_img_xy1 = np.concatenate((h36m_joint_img[:,:2], np.ones_like(h36m_joint_img[:,:1])),1)
    #         h36m_joint_img[:,:2] = np.dot(img2bb_trans, h36m_joint_img_xy1.transpose(1,0)).transpose(1,0)
    #         h36m_joint_img[:,0] = h36m_joint_img[:,0] / input_img_shape[1] * output_hm_shape[2]
    #         h36m_joint_img[:,1] = h36m_joint_img[:,1] / input_img_shape[0] * output_hm_shape[1]
    #         h36m_joint_img[:,2] = h36m_joint_img[:,2] - h36m_joint_img[self.h36m_root_joint_idx][2] # root-relative
    #         h36m_joint_img[:,2] = (h36m_joint_img[:,2] / (bbox_3d_size * 1000  / 2) + 1)/2. * output_hm_shape[0] # change cfg.bbox_3d_size from meter to milimeter
            
    #         # check truncation
    #         h36m_joint_trunc = h36m_joint_valid * ((h36m_joint_img[:,0] >= 0) * (h36m_joint_img[:,0] < output_hm_shape[2]) * \
    #                     (h36m_joint_img[:,1] >= 0) * (h36m_joint_img[:,1] < output_hm_shape[1]) * \
    #                     (h36m_joint_img[:,2] >= 0) * (h36m_joint_img[:,2] < output_hm_shape[0])).reshape(-1,1).astype(np.float32)

    #         # # transform h36m joints to target db joints
    #         # h36m_joint_img = transform_joint_to_other_db(h36m_joint_img, self.h36m_joints_name, self.joints_name)
    #         # h36m_joint_cam = transform_joint_to_other_db(h36m_joint_cam, self.h36m_joints_name, self.joints_name)
    #         # h36m_joint_valid = transform_joint_to_other_db(h36m_joint_valid, self.h36m_joints_name, self.joints_name)
    #         # h36m_joint_trunc = transform_joint_to_other_db(h36m_joint_trunc, self.h36m_joints_name, self.joints_name)

            
    #         # 3D data rotation augmentation
    #         rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
    #         [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    #         [0, 0, 1]], dtype=np.float32)
    #         # h36m coordinate
    #         h36m_joint_cam = np.dot(rot_aug_mat, h36m_joint_cam.transpose(1,0)).transpose(1,0) / 1000 # milimeter to meter


    #         inputs = {'img': img}
    #         targets = {'orig_joint_img': h36m_joint_img,  'orig_joint_cam': h36m_joint_cam}
    #         meta_info = {'orig_joint_valid': h36m_joint_valid, 'orig_joint_trunc': h36m_joint_trunc, 'is_3D': float(True)}
    #         return inputs, targets, meta_info
    #     else:
    #         inputs = {'img': img}
    #         targets = {}
    #         meta_info = {'bb2img_trans': bb2img_trans}
    #         return inputs, targets, meta_info

    @staticmethod
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
        
    @staticmethod
    def visualize_heatmap(heatmap):
        heatmap = heatmap / heatmap.max() *255
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
    

def visualize_heatmap(heatmap):
    heatmap = heatmap / heatmap.max() *255
    return heatmap    

def get_item(data_list,idx,parameter):
    input_img_shape=[parameter['height'],parameter['width']]
    output_hm_shape=parameter['output_hm_shape']
    bbox_3d_size=parameter['box_3d_size']
    h36m_flip_pairs=( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
    data = copy.deepcopy(data_list[idx])
    img_path, img_shape, bbox, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['cam_param']
    img = load_img(img_path)  #
    # x1=int(bbox[0])
    # y1=int(bbox[1])
    # x2=int(bbox[2])
    # y2=int(bbox[3])
    # colors = (0,0,255)
    # cv2.rectangle(img, (x1, y1), (x2,y2), colors, 5)
    # cv2.imwrite('./test.jpg',img)
    # #cut img by bbox
    # print(img.shape)
    
    img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, parameter['mode'])
    # img=img[int(bbox[0]):int(bbox[2]),int(bbox[1]):int(bbox[3]),:]
    # h36m gt
    h36m_joint_img = data['joint_img']
    h36m_joint_cam = data['joint_cam']
    h36m_joint_cam = h36m_joint_cam - h36m_joint_cam[0,None,:] # root-relative
    h36m_joint_valid = data['joint_valid']
    if do_flip:
        h36m_joint_cam[:,0] = -h36m_joint_cam[:,0]
        h36m_joint_img[:,0] = img_shape[1] - 1 - h36m_joint_img[:,0]
        for pair in h36m_flip_pairs:
            h36m_joint_img[pair[0],:], h36m_joint_img[pair[1],:] = h36m_joint_img[pair[1],:].copy(), h36m_joint_img[pair[0],:].copy()
            h36m_joint_cam[pair[0],:], h36m_joint_cam[pair[1],:] = h36m_joint_cam[pair[1],:].copy(), h36m_joint_cam[pair[0],:].copy()
            h36m_joint_valid[pair[0],:], h36m_joint_valid[pair[1],:] = h36m_joint_valid[pair[1],:].copy(), h36m_joint_valid[pair[0],:].copy()

    h36m_joint_img_xy1 = np.concatenate((h36m_joint_img[:,:2], np.ones_like(h36m_joint_img[:,:1])),1)
    h36m_joint_img[:,:2] = np.dot(img2bb_trans, h36m_joint_img_xy1.transpose(1,0)).transpose(1,0)
    h36m_joint_img[:,0] = h36m_joint_img[:,0] / input_img_shape[1] * output_hm_shape[2]
    h36m_joint_img[:,1] = h36m_joint_img[:,1] / input_img_shape[0] * output_hm_shape[1]
    h36m_joint_img[:,2] = h36m_joint_img[:,2] - h36m_joint_img[0][2] # root-relative
    h36m_joint_img[:,2] = (h36m_joint_img[:,2] / (bbox_3d_size * 1000  / 2) + 1)/2. * output_hm_shape[0] # change cfg.bbox_3d_size from meter to milimeter
    
    # check truncation
    h36m_joint_trunc = h36m_joint_valid * ((h36m_joint_img[:,0] >= 0) * (h36m_joint_img[:,0] < output_hm_shape[2]) * \
                (h36m_joint_img[:,1] >= 0) * (h36m_joint_img[:,1] < output_hm_shape[1]) * \
                (h36m_joint_img[:,2] >= 0) * (h36m_joint_img[:,2] < output_hm_shape[0])).reshape(-1,1).astype(np.float32)

    # # transform h36m joints to target db joints
    # h36m_joint_img = transform_joint_to_other_db(h36m_joint_img, self.h36m_joints_name, self.joints_name)
    # h36m_joint_cam = transform_joint_to_other_db(h36m_joint_cam, self.h36m_joints_name, self.joints_name)
    # h36m_joint_valid = transform_joint_to_other_db(h36m_joint_valid, self.h36m_joints_name, self.joints_name)
    # h36m_joint_trunc = transform_joint_to_other_db(h36m_joint_trunc, self.h36m_joints_name, self.joints_name)

    
    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    [0, 0, 1]], dtype=np.float32)
    # h36m coordinate
    h36m_joint_cam = np.dot(rot_aug_mat, h36m_joint_cam.transpose(1,0)).transpose(1,0) / 1000 # milimeter to meter


    # inputs = {'img': img}
    targets = {'orig_joint_img': h36m_joint_img,  'orig_joint_cam': h36m_joint_cam}
    meta_info = {'orig_joint_valid': h36m_joint_valid, 'orig_joint_trunc': h36m_joint_trunc, 'is_3D': float(True)}
    # return inputs, targets, meta_info
    # count=1
    img_input = copy.deepcopy(img)

    # img_input=img_input[int(bbox[0]):int(bbox[0])+int(bbox[2]),int(bbox[1]):int(bbox[1])+int(bbox[3]),:]
    img_pro = cv2.resize(img, (46, 46))
    xy=[]
    xz=[]
    yz=[]
    for point in targets['orig_joint_img']:
        # print(point[:2])
        hm_xy=gen_heatmap(output_hm_shape[0],output_hm_shape[1],point[0],point[1])
        hm_xz=gen_heatmap(output_hm_shape[0],output_hm_shape[2],point[0],point[2])
        hm_yz=gen_heatmap(output_hm_shape[1],output_hm_shape[2],point[1],point[2])
        heatmap_xy = visualize_heatmap(hm_xy)
        heatmap_xz = visualize_heatmap(hm_xz)
        heatmap_yz = visualize_heatmap(hm_yz)
        heatmap_xy = np.array([heatmap_xy for i in range(3)]).transpose(1,2,0)
        heatmap_xz = np.array([heatmap_xz for i in range(3)]).transpose(1,2,0)
        heatmap_yz = np.array([heatmap_yz for i in range(3)]).transpose(1,2,0)
        # img_pro = img_ori.copy()
        cv2.circle(img_pro, (point[0], point[1]), 1, (0,0,255), 1)
        # print(img_pro.shape)
        # print(heatmap_xy.shape)
        # img1 = np.concatenate([img_pro, heatmap_xy], axis=1)
        # img2 = np.concatenate([heatmap_xz, heatmap_yz], axis=1)
        # img3 = np.vstack((img1, img2))
        # img3 = cv2.resize(img3, (368, 368))
        # cv2.imwrite('heatmap_xyz_'+str(count)+'.jpg',img3)

        # cv2.circle(img, (int(point[0]*256/64),int(point[1]*256/64)), 1, (0,0,255), 1)
        # count += 1
        xy.append(hm_xy)
        xz.append(hm_xz)
        yz.append(hm_yz)
    np_xy=np.dstack(xy)
    np_xz=np.dstack(xz)
    np_yz=np.dstack(yz)

    labels=np.concatenate([np_xy,np_xy,np_xz,np_yz],axis=2)

    
    return img_input,labels,img_pro

    


# if __name__ == '__main__':
#     import time
#     m = Human36M('/home/zml/workspace_2013/I2L-MeshNet')
#     start = time.time()
#     img, targets, meta_info  = m.__getitem__(1)
#     img = img['img']
#     print('loading time: %.3fs' % (time.time() - start))

#     # heatmap = heatmaps[0, ..., 0]  # head
#     # heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
#     # overlay = img * 0.5 + cv2.resize(heatmap_bgr, (368, 368)) * 255 * 0.5
#     print('targets',targets)
#     print('img',img)
#     print('img.shape',img.shape)
#     img_ori = img.copy()
#     img_ori = cv2.resize(img_ori, (64, 64))
#     # cv2.imshow('image', img)
#     # 'orig_joint_img': h36m_joint_img
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     count = 1
#     for point in targets['orig_joint_img']:
#         print(point[:2])
#         heatmap_xy = m.visualize_heatmap(m.gen_heatmap(cfg.output_hm_shape[0],cfg.output_hm_shape[1],point[0],point[1]))
#         heatmap_xz = m.visualize_heatmap(m.gen_heatmap(cfg.output_hm_shape[0],cfg.output_hm_shape[2],point[0],point[2]))
#         heatmap_yz = m.visualize_heatmap(m.gen_heatmap(cfg.output_hm_shape[1],cfg.output_hm_shape[2],point[1],point[2]))
#         heatmap_xy = np.array([heatmap_xy for i in range(3)]).transpose(1,2,0)
#         heatmap_xz = np.array([heatmap_xz for i in range(3)]).transpose(1,2,0)
#         heatmap_yz = np.array([heatmap_yz for i in range(3)]).transpose(1,2,0)
#         img_pro = img_ori.copy()
#         cv2.circle(img_pro, (point[0], point[1]), 1, (0,0,255), 1)
#         # print(img_pro.shape)
#         # print(heatmap_xy.shape)
#         img1 = np.concatenate([img_pro, heatmap_xy], axis=1)
#         img2 = np.concatenate([heatmap_xz, heatmap_yz], axis=1)
#         img3 = np.vstack((img1, img2))
#         img3 = cv2.resize(img3, (368, 368))
#         cv2.imwrite('heatmap_xyz_'+str(count)+'.jpg',img3)

#         cv2.circle(img, (int(point[0]*256/64),int(point[1]*256/64)), 1, (0,0,255), 1)
#         count += 1
#     cv2.imwrite('image.jpg',img)
