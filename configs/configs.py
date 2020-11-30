train_config = {}
# valid_config= {}

# dataset parameters
# train_config['train_json_file'] = '/home/hsw/hswData/ai_train.json'
# train_config['train_data_path'] = '/home/hsw/hswData/ai_train'
train_config['data_path']='/home/swh/exercise_work'
train_config['valid_batch_size']=1
train_config['train_nums']      = 210000
train_config['valid_nums']      = 30000
# train_config['valid_json_file'] = '/home/hsw/hswData/ai_valid_1000.json'
# train_config['valid_data_path'] = '/home/hsw/hswData/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
train_config['sub_sample_ratio']= 5 #
train_config['subject']=[1,5]
# train params
train_config['gpu'] = '1' # index of gpu for training
train_config['batch_size'] = 32
train_config['height']     = 368
train_config['width']      = 368
train_config['output_hm_shape']=[46,46,46]
train_config['box_3d_size']=2
train_config['num_kps']    = 17
train_config['paf']        = 13*2
train_config['sigma']      = 1.
train_config['mode']= 'train'
train_config['train_subsampling_ratio']=5  # subsampling dataset for training

# train_config['paf_width_thre']  = 1.
train_config['save_models']     = 100
train_config['input_scale']     = 8

train_config['finetuning'] = False
train_config['checkpoint_path'] = '/home/swh/exercise_work/Vnect_project/ckpt_v2'
train_config['vis_path']='/home/swh/exercise_work/Vnect_project/results'
train_config['vis_input']=False

#test
train_config['test_batch_size']=1
train_config['test_subjects']=[9]
train_config['test_subsampling_ratio']=64  # subsampling dataset for training
train_config['scales']=[1]
train_config['test_model']='/home/swh/exercise_work/ckpt/model.ckpt-21756'
train_config['img_path']='/home/swh/exercise_work/Vnect_project/test/test.jpg'
