# from cv2 import data
import tensorflow as tf
import os
import json
import cv2
import numpy as np
import sys

# from tools.target_generators import get_heatmap
sys.path.append('..')
from tools.human36m import Human36M,get_item
# from .img_aug import img_aug_fuc


os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

id_kps_dict = {}
parameters = {}
id_body_annos = {}

# own
image_annos={}
id_2d_kps_dict={}
id_3d_kps_dict={}
data_list=[]

def set_params(params):
    global parameters
    parameters = params


def _prepare(json_file):

    global id_kps_dict
    global id_body_annos

    img_ids   = []

    with open(json_file) as f:
        annos = json.load(f)
        for anno in annos:
            img_ids.append(anno['image_id'])
            kps = []
            for key, val in anno['keypoint_annotations'].items():
                kps += val
            kps = np.reshape(np.asarray(kps), (-1, 14, 3))
            id_kps_dict[anno['image_id']] = kps
            id_body_annos[anno['image_id']] = anno['human_annotations']

    return img_ids

def prepare(json_file):
    global id_2d_kps_dict
    global id_3d_kps_dict
    global image_annos
    global parameters

    img_ids=[]
    with open(json_file) as f:
        label=json.load(f)
        image_annos=label["images"]
        annos=label["annotations"]
        for anno in annos:
            img_ids.append(anno["img_id"])
            kps_2d=np.reshape(anno["2D-keypoints"],(-1,parameters["num_kps"],2))
            kps_3d=np.reshape(anno["3D-keypoints"],(-1,parameters["num_kps"],3))
            ## add extra items in here
            id_2d_kps_dict[anno["img_id"]]=kps_2d
            id_3d_kps_dict[anno["img_id"]]=kps_3d
    return img_ids
            
def prepare_data(mode):
    global parameters
    global data_list

    dataset_path=parameters['data_path']
    
    data=Human36M(dataset_path,parameters,data_split=mode)

    len_data=len(data.datalist)
    
    data_list=data.datalist

    return len_data


    
def get_dataset_num(parameters):
    dataset_path=parameters['data_path']
    data=Human36M(dataset_path,parameters)
    len_data=len(data.datalist)
    return len_data



def _parse_function(img_id, mode='train'):

    global id_kps_dict, parameters

    if type(img_id) == type(b'123'):
        img_id = str(img_id, encoding='utf-8')
    if type(mode) == type(b'123'):
        mode   = str(mode, encoding='utf-8')

    # read img_data and convert BGR to RGB
    if mode == 'train':
        img_data = cv2.imread(os.path.join(parameters['train_data_path'], img_id + '.jpg'))
        data_aug = True
        sigma = parameters['sigma']
    elif mode == 'valid':
        img_data = cv2.imread(os.path.join(parameters['valid_data_path'], img_id + '.jpg'))
        data_aug = False
        sigma = 1.
    else:
        img_data = None
        data_aug = None
        sigma    = None
        print('parse_function mode must be train or valid.')
        exit(-1)

    h, w, _ = img_data.shape

    # get kps
    kps_channels = parameters['num_kps']
    paf_channels = parameters['paf']
    keypoints = id_kps_dict[img_id]

    keypoints = np.reshape(np.asarray(keypoints), newshape=(-1, kps_channels, 3))

    bboxs = []
    for key, value in id_body_annos[img_id].items():
        bboxs.append(value)
    if data_aug:
        # print('run data aug')
        img_data, keypoints, bboxs = img_aug_fuc(img_data, keypoints, bboxs)

    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

    img      = cv2.resize(img_data, (parameters['width'], parameters['height']))
    img      = np.asarray(img, dtype=np.float32) / 255.
    heatmap_height = parameters['height'] // parameters['input_scale']
    heatmap_width  = parameters['width'] // parameters['input_scale']

    heatmap = get_heatmap(keypoints, h, w, heatmap_height, heatmap_width, kps_channels, sigma)
    paf     = get_paf(keypoints, h, w, heatmap_height, heatmap_width, paf_channels, parameters['paf_width_thre'])

    # add head mask info
    mask = np.zeros((heatmap_height, heatmap_width, 1), dtype=np.float32)
    for value in bboxs:
        body_box = value
        factorX = w  / heatmap_width
        factorY = h  / heatmap_height
        body_box[0] /= factorX
        body_box[1] /= factorY
        body_box[2] /= factorX
        body_box[3] /= factorY

        minx = int(max(1, body_box[0] - 5))
        miny = int(max(1, body_box[1] - 5))
        maxx = int(min(heatmap_width - 1, body_box[2] + 5))
        maxy = int(min(heatmap_height - 1, body_box[3] + 5))

        mask[miny:maxy, minx:maxx, :] = True
    
    labels = np.concatenate([heatmap, paf, mask], axis=-1)
    return img, labels

# def parse_function(img_id, mode='train'):

#     global image_annos # img_id:image_name
#     global id_3d_kps_dict,id_2d_kps_dict,parameters

#     kps_channels=parameters["num_kps"]
#     if mode == 'train':
#         img_data = cv2.imread(os.path.join(parameters['train_data_path'], image_annos[img_id]))
#         # data_aug = True # add aug in here 
#         sigma = parameters['sigma']
#     elif mode == 'valid':
#         img_data = cv2.imread(os.path.join(parameters['valid_data_path'], image_annos[img_id]))
#         data_aug = False
#         sigma = 1.
#     else:
#         img_data = None
#         data_aug = None
#         sigma    = None
#         print('parse_function mode must be train or valid.')
#         exit(-1)

#     h, w, _ = img_data.shape

#     #get gt keypoints
#     kps_2d=id_2d_kps_dict[img_id]
#     kps_3d=id_3d_kps_dict[img_id]

#     # dataset aug
#     # if data_aug:
#         #augment dataset

#     #convert BGR to RGB 
#     img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    
#     img      = cv2.resize(img_data, (parameters['width'], parameters['height']))
#     img      = np.asarray(img, dtype=np.float32) / 255.
#     heatmap_height = parameters['height'] // parameters['input_scale']
#     heatmap_width  = parameters['width'] // parameters['input_scale']
#     hm=get_heatmap(kps_2d, h, w, heatmap_height, heatmap_width, kps_channels, sigma)
#     xm=get_heatmap(kps_3d, h, w, heatmap_height, heatmap_width, kps_channels, sigma)
#     ym=get_heatmap(kps_3d, h, w, heatmap_height, heatmap_width, kps_channels, sigma)
#     zm=get_heatmap(kps_3d, h, w, heatmap_height, heatmap_width, kps_channels, sigma)
#     labels = np.concatenate([hm, xm,ym,zm], axis=-1)
#     return img,labels

def parse_function(img_id, mode='train'):
    global data_list
    global parameters
    img,labels,img_vis=get_item(data_list,img_id,parameters)
    # count=1
    # input_img_shape=[parameters['height'],parameters['width']]
    # output_hm_shape=parameters['output_hm_shape']
    # bbox_3d_size=parameter['box_3d_size']
    

    if parameters['vis_input']:
        img_re = cv2.resize(img_vis, (368, 368))
        cv2.imwrite(os.path.join(parameters['vis_path'],str(img_id)+'_test.jpg'),img_re)
        # cv2.imwrite('./test.jpg',img_vis)
    # cv2.imwrite(os.path.join(parameters['vis_path'],str(img_id)+'_input.jpg'),img)
    img      = np.asarray(img, dtype=np.float32) / 255.
    
    return img,labels
    
    




def get_dataset_pipeline(parameters, epochs=1, mode='train'):

    set_params(parameters)
    # if mode == 'train':
    #     json_file = parameters['train_json_file']
    #     batch_size = parameters['batch_size']
    # elif mode == 'valid':
    #     json_file = parameters['valid_json_file']
    #     batch_size = parameters['valid_batch_size']
    # else:
    #     json_file = None
    #     batch_size = None
    #     print('Dataset mode must be train or valid.')
    #     exit(-1)
    if mode=='train':
        batch_size = parameters['batch_size']
    # if mode=='train':
    else:
        batch_size = parameters['test_batch_size']
        
    len_dataset = prepare_data(mode)
    
    img_ids=np.arange(len_dataset)
    np.random.shuffle(img_ids)
    dataset = tf.data.Dataset.from_tensor_slices(img_ids)

    dataset = dataset.map(
        lambda  img_id: tuple(
            tf.py_func(
                func=parse_function,
                inp = [img_id, mode],
                Tout=[tf.float32, tf.float32])),
        num_parallel_calls=12)

    dataset = dataset.batch(batch_size, drop_remainder=True).repeat(epochs)
    # dataset = dataset.batch(batch_size).repeat(epochs)
    dataset = dataset.prefetch(buffer_size=batch_size*12*4)

    return dataset

# from configs.configs import train_config 


# parameters=train_config
# # global parameters
# set_params(train_config)
# print(prepare_data('train'))
# img,labels=parse_function(1)
# print(labels.shape)
