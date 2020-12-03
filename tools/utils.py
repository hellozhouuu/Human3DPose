#@cvhadessun
import numpy as np
import cv2
import tensorflow as tf

def img_scale(img, scale):
    """
    Resize a image by s scaler in both x and y directions.

    :param img: input image
    :param scale: scale  factor, new image side length / raw image side length
    :return: the scaled image
    """
    return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def img_padding(img, box_size, color='black'):
    """
    Given the input image and side length of the box, put the image into the center of the box.

    :param img: the input color image, whose longer side is equal to box size
    :param box_size: the side length of the square box
    :param color: indicating the padding area color
    :return: the padded image
    """
    h, w = img.shape[:2]
    offset_x, offset_y = 0, 0
    if color == 'black':
        pad_color = [0, 0, 0]
    elif color == 'grey':
        pad_color = [128, 128, 128]
    img_padded = np.ones((box_size, box_size, 3), dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
    if h > w:
        offset_x = box_size // 2 - w // 2
        img_padded[:, offset_x: box_size // 2 + int(np.ceil(w / 2)), :] = img
    else:  # h <= w
        offset_y = box_size // 2 - h // 2
        img_padded[offset_y: box_size // 2 + int(np.ceil(h / 2)), :, :] = img
    return img_padded, [offset_x, offset_y]

def img_scale_squarify(img, box_size):
    """
    To scale and squarify the input image into a square box with fixed size.

    :param img: the input color image
    :param box_size: the length of the square box
    :return: box image, scaler and offsets
    """
    h, w = img.shape[:2]
    scaler = box_size / max(h, w)
    img_scaled = img_scale(img, scaler)
    img_padded, [offset_x, offset_y] = img_padding(img_scaled, box_size)
    assert img_padded.shape == (box_size, box_size, 3), 'padded image shape invalid'
    return img_padded, scaler, [offset_x, offset_y]


def img_scale_padding(img, scaler, box_size, color='black'):
    """
    For a box image, scale down it and then pad the former area.

    :param img: the input box image
    :param scaler: scale factor, new image side length / raw image side length, < 1
    :param box_size: side length of the square box
    :param color: the padding area color
    """
    img_scaled = img_scale(img, scaler)
    if color == 'black':
        pad_color = (0, 0, 0)
    elif color == 'grey':
        pad_color = (128, 128, 128)
    pad_h = (box_size - img_scaled.shape[0]) // 2
    pad_w = (box_size - img_scaled.shape[1]) // 2
    pad_h_offset = (box_size - img_scaled.shape[0]) % 2
    pad_w_offset = (box_size - img_scaled.shape[1]) % 2
    img_scale_padded = np.pad(img_scaled,
                              ((pad_w, pad_w + pad_w_offset),
                               (pad_h, pad_h + pad_h_offset),
                               (0, 0)),
                              mode='constant',
                              constant_values=(
                                  (pad_color[0], pad_color[0]),
                                  (pad_color[1], pad_color[1]),
                                  (pad_color[2], pad_color[2])))
    return img_scale_padded

def gen_input_batch(img_input, box_size, scales):
    # input image --> sqrared image acceptable for the model
    img_square, scaler, [offset_x, offset_y] = img_scale_squarify(img_input, box_size)
    # generate multi-scale image batch
    input_batch = []
    for scale in scales:
        img = img_scale_padding(img_square, scale, box_size) if scale < 1 else img_square
        input_batch.append(img)
    # image value range: [0, 255) --> [-0.4, 0.6)
    input_batch = np.asarray(input_batch, dtype=np.float32) / 255 - 0.4
    return input_batch, scaler, [offset_x, offset_y]


def extract_3d_joints(heatmap_xy, heatmap_xz, heatmap_yz,param):
    heatmap_xy = tf.squeeze(tf.transpose(heatmap_xy,perm=[0,3,1,2]))
    heatmap_xy_t = tf.transpose(heatmap_xy,perm=[0,2,1])
    heatmap_xz = tf.squeeze(tf.transpose(heatmap_xz,perm=[0,3,1,2]))
    heatmap_yz = tf.squeeze(tf.transpose(heatmap_yz,perm=[0,3,1,2]))
    x_hat = tf.to_float(tf.range(0, param['output_hm_shape'][0]))
    y_hat = tf.to_float(tf.range(0, param['output_hm_shape'][1]))
    z_hat = tf.to_float(tf.range(0, param['output_hm_shape'][2]))

    joint_3d_x = tf.reduce_mean(tf.tensordot(heatmap_xy,x_hat,axes=1),axis=1)
    joint_3d_y = tf.reduce_mean(tf.tensordot(heatmap_xy_t,y_hat,axes=1),axis=1)
    joint_3d_z = (tf.reduce_mean(tf.tensordot(heatmap_yz,z_hat,axes=1),axis=1) +  tf.reduce_mean(tf.tensordot(heatmap_xz,z_hat,axes=1),axis=1))/2
    joint_3d_x, joint_3d_y , joint_3d_z = tf.expand_dims(joint_3d_x,1), tf.expand_dims(joint_3d_y,1), tf.expand_dims(joint_3d_z,1)
    joint_3d = tf.concat([joint_3d_x,joint_3d_y,joint_3d_z],axis=1)
    print(joint_3d.shape)
    return joint_3d

def decode_pose(xm,ym,zm,param):
    scale=1
    # box_size=param['width']
    # img_batch, scaler, [offset_x, offset_y]=gen_input_batch(img,box_size,scales)
    
    joints_3d=extract_3d_joints(xm,ym,zm,param)

    # joints_3d_cam = np.zeros((param['num_kps'],3))

    # for i in range(param['num_kps']):
        # 
    joints_3d_cam = joints_3d / param['output_hm_shape'][0] * param['width'] * scale
    
    return joints_3d_cam