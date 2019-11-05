from __future__ import print_function, division, unicode_literals
import tensorflow as tf



# USEFUL LAYERS
fc = tf.layers.dense
conv = tf.layers.conv2d
deconv = tf.layers.conv2d_transpose
relu = tf.nn.relu
maxpool = tf.layers.max_pooling2d
dropout_layer = tf.layers.dropout
batchnorm = tf.layers.batch_normalization
winit = tf.contrib.layers.xavier_initializer()
repeat = tf.contrib.layers.repeat
arg_scope = tf.contrib.framework.arg_scope
l2_regularizer = tf.contrib.layers.l2_regularizer

'''
#################################
############ MININET ############
#################################
'''

def module_mininet(inputs, filters, name, dilation_rate=(1, 1), training=True):


    with tf.variable_scope('module_' + name): # Name scope

        x = tf.layers.separable_conv2d(inputs, int(filters), (1,3), padding='same',
            depthwise_initializer=winit, pointwise_initializer=tf.initializers.lecun_normal(),  activation=tf.nn.selu, dilation_rate=(1,1),
            depthwise_regularizer=l2_regularizer(0.0001), pointwise_regularizer=l2_regularizer(0.0001))

        x2 = tf.layers.separable_conv2d(x, int(filters), (3,1), padding='same',
            depthwise_initializer=winit, pointwise_initializer=tf.initializers.lecun_normal(),  activation=None, dilation_rate=(1,1),
            depthwise_regularizer=l2_regularizer(0.0001), pointwise_regularizer=l2_regularizer(0.0001))

        x = tf.nn.selu(x2)
        x = tf.layers.separable_conv2d(x, int(filters), (3,1), padding='same',
            depthwise_initializer=winit, pointwise_initializer=tf.initializers.lecun_normal(),  activation=tf.nn.selu, dilation_rate=dilation_rate,
            depthwise_regularizer=l2_regularizer(0.0001), pointwise_regularizer=l2_regularizer(0.0001))

        x = tf.layers.separable_conv2d(x, int(filters), (1,3), padding='same',
            depthwise_initializer=winit, pointwise_initializer=tf.initializers.lecun_normal(),  activation=None, dilation_rate=dilation_rate,
            depthwise_regularizer=l2_regularizer(0.0001), pointwise_regularizer=l2_regularizer(0.0001))

        x = tf.add(x, x2)
        x = tf.contrib.nn.alpha_dropout(x, keep_prob=0.75)
        x = tf.add(x, inputs[:, :, :, :filters])
        x = tf.nn.selu(x)

    return x




def downsampling_mininet(inputs, filters, name, strides=(2, 2), kernels=(3, 3), training=True):
    with tf.variable_scope('downsampling_' + name):
        x = tf.layers.separable_conv2d(inputs, filters, kernels, strides=strides, padding='same',
            depthwise_initializer=winit, pointwise_initializer=tf.initializers.lecun_normal(),  activation=tf.nn.selu,
            depthwise_regularizer=l2_regularizer(0.0001), pointwise_regularizer=l2_regularizer(0.0001))

        return x


def upsampling_mininet(inputs, filters, name, strides=(2, 2), kernels=(3, 3), training=True, last=False):
     with tf.variable_scope('upsampling_' + name):

        activation = tf.nn.selu
        if last:
            activation=None
        x = tf.layers.conv2d_transpose(inputs, filters, kernels, strides=strides, padding='same', kernel_initializer=tf.initializers.lecun_normal(),  activation=activation,
            kernel_regularizer=l2_regularizer(0.0001)) # there is also dilation_rate!

        return x


def MiniNet(input_x=None, n_classes=20, training=True):
    d1 = downsampling_mininet(input_x, 12, 'd1', strides=(2, 2), kernels=(3, 3), training=training)
    d2 = downsampling_mininet(d1, 24, 'd2', strides=(2, 2), kernels=(3, 3), training=training)
    d3 = downsampling_mininet(d2, 48, 'd2_1', strides=(2, 2), kernels=(3, 3), training=training)
    d4 = downsampling_mininet(d3, 96, 'd3', strides=(2, 2), kernels=(3, 3), training=training)
    m4 = module_mininet(d4, 96, 'm3', dilation_rate=(1, 1), training=training)
    m5 = module_mininet(m4, 96, 'm4', dilation_rate=(1, 2), training=training)
    m6 = module_mininet(m5, 96, 'm5', dilation_rate=(1, 4), training=training)
    m7 = module_mininet(m6, 96, 'm6', dilation_rate=(1, 8), training=training)

    d5 = downsampling_mininet(d4, 192, 'd5', strides=(2, 2), kernels=(3, 3), training=training)
    d5 = module_mininet(d5, 192, 'm7d5', dilation_rate=(1, 1), training=training)
    d6 = downsampling_mininet(d5, 386, 'd6', strides=(2, 2), kernels=(3, 3), training=training)
    d6 = module_mininet(d6, 386, 'm7d6', dilation_rate=(1, 1), training=training)
    d6 = module_mininet(d6, 386, 'm7d6d6', dilation_rate=(1, 1), training=training)
    up_1 = upsampling_mininet(d6, 192, '_up11', training=training)
    up_1 = module_mininet(up_1, 192, 'm7d5up_1', dilation_rate=(1, 1), training=training)
    up_2 = upsampling_mininet(tf.concat([up_1, d5], axis=3), 96, 'new_1', training=training)

    up_concat = upsampling_mininet(tf.concat([m7, up_2, d4], axis=3), 96, 'new_2', training=training)

    m9 = module_mininet(up_concat, 48, 'm8', dilation_rate=(1, 1), training=training)

    up2 = upsampling_mininet(tf.concat((m9, d3), axis=3), 32, 'up2', training=training)
    up3 = upsampling_mininet(tf.concat((up2, d2), axis=3), 16, 'up22', training=training)

    out = upsampling_mininet(tf.concat((up3, d1), axis=3), n_classes, 'up3', training=training, last=True)
    return out
