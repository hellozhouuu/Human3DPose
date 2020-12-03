import tensorflow as tf
import numpy as np


u=tf.reshape(np.arange(0,6),[3,2])
v = tf.to_float(tf.range(0, 2))
# v=tf.Variable(tf.random_uniform([2,3,3]))
mul=tf.reduce_mean(tf.tensordot(tf.cast(u,tf.float32),v,axes=1))
print(mul.shape)
s=tf.Session()
s.run(tf.initialize_all_variables())
joint_3d = tf.Variable(tf.zeros((3,3)))
print (s.run(u))
print (s.run(v))
print (s.run(mul))
joint_3d[0,0] = mul