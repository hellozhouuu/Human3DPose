import tensorflow as tf
import os
import time
import cv2
import numpy as np
import sys
import vis 
sys.path.append('../')
from model.vnect import vnect_model_v1
from tools.utils import decode_pose

# params = {}
# params['test_model'] = '/media/hsw/E/ckpt/test/model.ckpt-15309'
# # params['video_path'] = '/media/hsw/E/video/bank/jiachaojian.mp4'
# params['img_path']   = '/media/hsw/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
# # params['img_path']   = '/media/ulsee/E/yuncong/yuncong_data/our/test/0/'

# params['thre1'] = 0.1
# params['thre2'] = 0.0
from configs.configs import train_config
params = train_config

use_gpu = True

def main():
    # use_gpu = False

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    input_img = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    # _1, _2, cpm, paf = light_openpose(input_img, is_training=False)
    xm,ym,zm = vnect_model_v1(input_img,is_training=False)
    saver = tf.train.Saver()

    total_img = 0
    total_time = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, params['test_model'])
        print('#---------Successfully loaded trained model.---------#')
        if 'video_path'in params.keys() and params['video_path'] is not None:
            # video_capture = cv2.VideoCapture('rtsp://admin:youwillsee!@10.24.1.238')
            video_capture = cv2.VideoCapture(params['video_path'])
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            start_second = 0
            start_frame = fps * start_second
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            while True:
                retval, img_ori = video_capture.read()
                if not retval:
                    break

                img_data = cv2.cvtColor(img_ori, code=cv2.COLOR_BGR2RGB)
                img_data = cv2.resize(img_data, (params['height'], params['width']))
                img = img_data / 255.

                start_time = time.time()
                out_xm,out_ym,out_zm= sess.run([xm,ym,zm], feed_dict={input_img: [img]})
                end_time = time.time()

                joints_3d = decode_pose(out_xm,out_ym,out_zm, params)
                decode_time = time.time()
                print ('inference + decode time == {}'.format(decode_time - start_time))

                total_img += 1
                total_time += (end_time-start_time)

                #vis
                
                # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                # cv2.imshow('result', canvas)
                # cv2.waitKey(1)
        elif params['img_path'] is not None:
            for img_name in os.listdir(params['img_path']):
                if img_name.split('.')[-1] != 'jpg':
                    continue
                img_ori = cv2.imread(os.path.join(params['img_path'], img_name))
                img_copy = img_ori.copy()  
                img_data = cv2.cvtColor(img_ori, code=cv2.COLOR_BGR2RGB)
                img_data = cv2.resize(img_data, (params['height'], params['width']))
                img = img_data / 255.

                start_time = time.time()
                out_xm,out_ym,out_zm= sess.run([xm,ym,zm], feed_dict={input_img: [img]})
                end_time = time.time()
                print('out_xm.shape',out_xm.shape)
                joints_3d = decode_pose(out_xm,out_ym,out_zm, params)
                decode_time = time.time()
                print('inference + decode time == {}'.format(decode_time - start_time))

                total_img += 1

                total_time += (end_time - start_time)

                out_xm = out_xm.squeeze()
                out_xm = np.transpose(out_xm,(2,0,1))
                print(out_xm.shape)
                for idx in range(len(out_xm)):
                    tmp = cv2.resize(out_xm[idx]*255, (1000, 1002))
                    loc = np.argmax(tmp)
                    y,x = loc //1000,loc %1000
                    print(x,y)
                    cv2.circle(img_copy,(x,y),5,(0, 0, 255),2)
                    # img_copy = cv2.addWeighted(img_copy, 1, tmp, 0.5, 0, dtype = cv2.CV_32F)
                cv2.imwrite('images/colored_'+img_name,img_copy)
                # joints = sess.run(joints_3d)
                # skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
                # print(joints[:,:2].T.shape)
                # img_out = vis.vis_keypoints(img_ori, joints[:,:2]/2*1000/46)
                # cv2.imwrite('images/out_'+img_name,img_out)
                
                #vis
                # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                # cv2.imshow('result', canvas)
                # cv2.waitKey(0)

        else:
            print('Nothing to process.')



main()