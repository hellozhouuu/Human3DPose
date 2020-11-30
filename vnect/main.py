import tensorflow as tf
import sys

sys.path.append('..')
from trainer import model_fn,train_input_fn
from configs.configs import train_config
from data.dataset import get_dataset_num

def main():
    # Creates session config. allow_soft_placement = True, is required for
    # multi-GPU and is not harmful for other modes
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = train_config['gpu']

    session_config = tf.ConfigProto(
        log_device_placement=True,
        inter_op_parallelism_threads=20,
        intra_op_parallelism_threads=20,
        allow_soft_placement=True)

    session_config.gpu_options.allow_growth = True

    # distribution_strategy = tf.contrib.distribute.OneDeviceStrategy(device='/gpu:6')

    steps_per_epoch = get_dataset_num(train_config) // train_config['batch_size']   #only for training

    run_config = tf.estimator.RunConfig(
        # train_distribute=distribution_strategy,
        session_config=session_config,
        save_checkpoints_steps=steps_per_epoch,
        save_summary_steps=100,
        log_step_count_steps=100,
        keep_checkpoint_max=200
    )


    # if train_config['finetuning']:
    #     ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=train_config['finetuning'])
    #     model_dir = train_config['finetuning']
    # else:
    ws = None
    model_dir = train_config['checkpoint_path']

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=run_config, model_dir=model_dir,
        params={
            'batch_size': train_config['batch_size'],
            'train_nums': steps_per_epoch,
            'lr':4e-5,
            'height': train_config['height'],
            'width': train_config['width'],
            'num_kps': train_config['num_kps']
        },
        # warm_start_from=ws
    )

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('start training')

    eval_spec  = tf.estimator.EvalSpec(input_fn=lambda : train_input_fn(parameters=train_config, epochs=1, mode='valid'))
    train_spec = tf.estimator.TrainSpec(input_fn=lambda : train_input_fn(parameters=train_config, epochs=200, mode='train'))
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    # tf.estimator.train_and_evaluate(estimator, eval_spec, eval_spec)


# if __name__ == '__main__':
main()
