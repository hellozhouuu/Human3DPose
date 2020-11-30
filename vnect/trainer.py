import tensorflow as tf
import sys

from model.vnect import vnect_model,vnect_model_v1,vnect_model_v2
from data.dataset import get_dataset_pipeline

# from model import Vnect



def train_input_fn(parameters, epochs, mode='train'):

    dataset = get_dataset_pipeline(parameters, epochs, mode)
    return  dataset

def model_fn(features, labels, mode, params):

    # get model output
    features = tf.reshape(features, [-1, params['height'],params['width'], 3])
    # gt_hms = labels[..., :params['num_kps']]
    # gt_pafs = labels[..., params['num_kps']:params['num_kps'] + params['paf']]
    gt_xms=labels[..., :params['num_kps']]
    gt_yms=labels[..., params['num_kps']:2*params['num_kps']]
    gt_zms=labels[..., 2*params['num_kps']:3*params['num_kps']]
    # mask = labels[..., params['num_kps'] + params['paf']:]
    # mask = tf.reshape(mask, [-1, params['height']//params['scale'], params['width']//params['scale'], 1])

    # hms, xms ,yms,zms = vnect_model(features, is_training=True)
    xms, yms, zms = vnect_model_v2(features, is_training=True)
    # predictions = {
    #     'pred_heatmap': hms,
    #     'pred_xm': xms,
    #     'pred_ym': yms,
    #     'pred_zm': zms,
    # }
    predictions = {
        'pred_xm': xms,
        'pred_ym': yms,
        'pred_zm': zms,
    }
    tf.summary.image('img', features, max_outputs=3)
    # tf.summary.image('pred_hmap', tf.reduce_sum(hms, axis=3, keepdims=True), max_outputs=3)
    tf.summary.image('pred_xm', tf.reduce_sum(xms, axis=3, keepdims=True), max_outputs=3)
    tf.summary.image('pred_ym', tf.reduce_sum(yms, axis=3, keepdims=True), max_outputs=3)
    tf.summary.image('pred_zm', tf.reduce_sum(zms, axis=3, keepdims=True), max_outputs=3)
    # tf.summary.image('gt_hmap', tf.reduce_sum(gt_cpms, axis=3, keepdims=True), max_outputs=3)
    # tf.summary.image('gt_paf', tf.expand_dims(
    #     (gt_pafs[..., 0] - tf.reduce_min(gt_pafs[..., 0])) / (tf.reduce_max(gt_pafs[..., 0]) - tf.reduce_min(gt_pafs[..., 0])),
    #     axis=3
    # ), max_outputs=3)
    # tf.summary.image('pred_paf', tf.expand_dims(
    #     (paf[..., 0] - tf.reduce_min(paf[..., 0])) / (tf.reduce_max(paf[..., 0]) - tf.reduce_min(paf[..., 0])),
    #     axis=3
    # ), max_outputs=3)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # cpm_mask = tf.concat([mask for i in range(params['num_kps'])], axis=-1)
    # paf_mask = tf.concat([mask for i in range(params['paf'])], axis=-1)
    # cpm = tf.where(cpm_mask > 0, cpm, cpm * 0)
    # paf = tf.where(paf_mask > 0, paf, paf * 0)
    # gt_cpms = tf.where(cpm_mask > 0, gt_cpms, gt_cpms * 0)
    # gt_pafs = tf.where(paf_mask > 0, gt_pafs, gt_pafs * 0)
    # loss = tf.nn.l2_loss(cpm - gt_cpms) + tf.nn.l2_loss(paf - gt_pafs) * 2

    # loss implementation
    # loss=tf.nn.l2_loss(hms-gt_xms) + tf.nn.l2_loss(xms-gt_xms)+ \
    #                 tf.nn.l2_loss(yms-gt_yms)+ tf.nn.l2_loss(zms-gt_zms)
    loss = tf.nn.l2_loss(xms - gt_xms) + \
           tf.nn.l2_loss(yms - gt_yms) + tf.nn.l2_loss(zms - gt_zms)
    # loss=hms
    
    tf.identity(loss, name='loss')
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        # metrics_dict = {
        #     'heatmap': tf.metrics.mean_squared_error(labels=gt_cpms, predictions=predictions['pred_heatmap']),
        #     'paf': tf.metrics.mean_squared_error(labels=gt_pafs, predictions=predictions['pred_paf'])
        # }
        metrics_dict={
            # 'heatmap':tf.metrics.mean_squared_error(labels=gt_hms, predictions=predictions['pred_heatmap']),
            'xm':tf.metrics.mean_squared_error(labels=gt_xms, predictions=predictions['pred_xm']),
            'ym':tf.metrics.mean_squared_error(labels=gt_yms, predictions=predictions['pred_ym']),
            'zm':tf.metrics.mean_squared_error(labels=gt_zms, predictions=predictions['pred_zm'])
        }

        # add tf.summary.scalar in here
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=metrics_dict
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        # step lr
        # values = [params['lr'], 0.1*params['lr'], 0.01*params['lr'], 0.001*params['lr']]
        # boundaries = [params['train_nums']*50, params['train_nums']*100, params['train_nums']*150]
        # learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        # constant lr
        learning_rate = tf.Variable(params['lr'], trainable=False, name='lr')

        tf.identity(learning_rate, name='lr')
        tf.summary.scalar('lr', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op
    )