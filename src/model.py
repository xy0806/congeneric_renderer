from __future__ import division
import os
import time
from glob import glob
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
from ops import *
from utils import *
from seg_eval import *


class unet_2D_xy(object):
    """ Implementation of 2D U-net"""
    def __init__(self, sess, param_set):
        self.sess           = sess
        self.phase          = param_set['phase']
        self.batch_size     = param_set['batch_size']
        self.inputI_size    = param_set['inputI_size']
        self.inputI_chn     = param_set['inputI_chn']
        self.outputI_size   = param_set['outputI_size']
        self.output_chn     = param_set['output_chn']
        self.resize_r       = param_set['resize_r']
        self.traindata_dir  = param_set['traindata_dir']
        self.chkpoint_dir   = param_set['chkpoint_dir']
        self.lr             = param_set['learning_rate']
        self.beta1          = param_set['beta1']
        self.epoch          = param_set['epoch']
        self.model_name     = param_set['model_name']
        self.save_intval    = param_set['save_intval']
        self.testdata_dir   = param_set['testdata_dir']
        self.labeling_dir   = param_set['labeling_dir']

        # build model graph
        self.build_online_segmentor_model()

    # dice loss function
    def dice_loss_fun(self, pred, input_gt):
        # dim = tf.shape(pred)
        # input_gt = tf.one_hot(input_gt, dim[3])
        class_n = 2
        input_gt = tf.one_hot(input_gt, class_n)
        # print(input_gt.shape)
        dice = 0
        for i in range(class_n):
            inse = tf.reduce_mean(pred[:, :, :, i]*input_gt[:, :, :, i])
            l = tf.reduce_sum(pred[:, :, :, i]*pred[:, :, :, i])
            r = tf.reduce_sum(input_gt[:, :, :, i] * input_gt[:, :, :, i])
            dice = dice + 2*inse/(l+r)
        return -dice

    # class-weighted cross-entropy loss function
    def softmax_weighted_loss(self, logits, labels):
        """
        Loss = weighted * -target*log(softmax(logits))
        :param logits: probability score
        :param labels: ground_truth
        :return: softmax-weifhted loss
        """
        class_n = 2
        gt = tf.one_hot(labels, class_n)
        pred = logits
        softmaxpred = tf.nn.softmax(pred)
        loss = 0
        for i in range(class_n):
            gti = gt[:, :, :, i]
            predi = softmaxpred[:, :, :, i]
            weighted = 1-(tf.reduce_sum(gti)/tf.reduce_sum(gt))
            # print("class %d"%(i))
            # print(weighted)
            loss = loss + -tf.reduce_mean(weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1)))
        return loss

    # build online segmentor model graph
    def build_online_segmentor_model(self):
        # input
        self.orig_app = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.inputI_size, self.inputI_size, self.inputI_chn], name='orig_app')
        self.orig_label = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.inputI_size, self.inputI_size], name='orig_label')

        self.desired_app = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.inputI_size, self.inputI_size, self.inputI_chn], name='desired_app')
        self.desired_strct = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.inputI_size, self.inputI_size, self.inputI_chn], name='desired_strct')
        self.desired_pair_app = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.inputI_size, self.inputI_size, self.inputI_chn], name='desired_pair_app')
        self.desired_pair_strct = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.inputI_size, self.inputI_size, self.inputI_chn], name='desired_pair_strct')

        ################ update appearance via app render
        self.updated_app, self.aux0_update_app = self.unet_app_render(self.orig_app)
        self.render_loss = tf.reduce_mean(tf.abs(self.updated_app - self.orig_app)) + tf.reduce_mean(tf.abs(self.aux0_update_app - self.orig_app))

        ################ appearance discriminator
        self.app_D_re, self.app_D_re_logits = self.app_discriminator(self.desired_app, reuse=False)
        self.app_D_fa, self.app_D_fa_logits = self.app_discriminator(self.updated_app, reuse=True)
        # appearance classification loss
        # === generator
        self.app_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.app_D_fa), logits=self.app_D_fa_logits))
        # === discriminator
        self.app_d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.app_D_re), logits=self.app_D_re_logits))
        self.app_d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.app_D_fa), logits=self.app_D_fa_logits))
        self.app_d_loss = self.app_d_loss_real + self.app_d_loss_fake

        ################ segmentor on updated appearance
        self.pred_prob, self.updated_strct, self.pred_label, self.aux1_prob, self.aux2_prob = self.unet_segmentor(self.updated_app)
        self.temp_forground = self.updated_strct[:, :, :, 1]
        self.updated_forground = tf.reshape(self.temp_forground, [self.batch_size, self.inputI_size, self.inputI_size, 1])
        # ========= dice loss
        self.main_dice_loss = self.dice_loss_fun(self.pred_prob, self.orig_label)
        # auxiliary loss
        self.aux1_dice_loss = self.dice_loss_fun(self.aux1_prob, self.orig_label)
        self.aux2_dice_loss = self.dice_loss_fun(self.aux2_prob, self.orig_label)
        #
        self.total_dice_loss = self.main_dice_loss + 0.4*self.aux1_dice_loss + 0.8*self.aux2_dice_loss

        # ========= class-weighted cross-entropy loss
        self.main_wght_loss = self.softmax_weighted_loss(self.pred_prob, self.orig_label)
        self.aux1_wght_loss = self.softmax_weighted_loss(self.aux1_prob, self.orig_label)
        self.aux2_wght_loss = self.softmax_weighted_loss(self.aux2_prob, self.orig_label)
        self.total_wght_loss = self.main_wght_loss + 0.6*self.aux1_wght_loss + 0.9*self.aux2_wght_loss

        self.segmentor_loss = 100.0*self.total_dice_loss + self.total_wght_loss

        ################ structure discriminator
        self.strct_D_re, self.strct_D_re_logits = self.strct_discriminator(self.desired_strct, reuse=False)
        self.strct_D_fa, self.strct_D_fa_logits = self.strct_discriminator(self.updated_forground, reuse=True)
        # appearance classification loss
        # === generator
        self.strct_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.strct_D_fa), logits=self.strct_D_fa_logits))
        # === discriminator
        self.strct_d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.strct_D_re), logits=self.strct_D_re_logits))
        self.strct_d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.strct_D_fa), logits=self.strct_D_fa_logits))
        self.strct_d_loss = self.strct_d_loss_real + self.strct_d_loss_fake

        ################ mixed discriminator
        self.desired_pair = tf.concat([self.desired_pair_app, self.desired_pair_strct], 3)
        self.updated_pair = tf.concat([self.updated_app, self.updated_forground], 3)
        self.pair_D_re, self.pair_D_re_logits = self.pair_discriminator(self.desired_pair, reuse=False)
        self.pair_D_fa, self.pair_D_fa_logits = self.pair_discriminator(self.updated_pair, reuse=True)
        # appearance classification loss
        # === generator
        self.pair_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.pair_D_fa), logits=self.pair_D_fa_logits))
        # === discriminator
        self.pair_d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.pair_D_re), logits=self.pair_D_re_logits))
        self.pair_d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.pair_D_fa), logits=self.pair_D_fa_logits))
        self.pair_d_loss = self.pair_d_loss_real + self.pair_d_loss_fake

        # trainable variables
        self.u_vars = tf.trainable_variables()

        self.rend_vars = [var for var in self.u_vars if 'rend_' in var.name]
        self.segmentor_vars = [var for var in self.u_vars if 'seg_' in var.name]
        self.app_disc_vars = [var for var in self.u_vars if 'D_app_' in var.name]
        self.strct_disc_vars = [var for var in self.u_vars if 'D_strct_' in var.name]
        self.pair_disc_vars = [var for var in self.u_vars if 'D_pair_' in var.name]

        # create model saver
        self.saver = tf.train.Saver()

    def unet_app_render(self, inputI):
        """2D U-net"""
        phase_flag = (self.phase == 'train')
        concat_dim = 3

        with tf.variable_scope("unet_app_render") as scope:
            conv1_1 = conv2D_bn_relu(input=inputI, output_chn=32, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='rend_Conv1_1')
            conv1_2 = conv2D_bn_relu(input=conv1_1, output_chn=32, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='rend_Conv1_2')
            pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=2, strides=2, name='pool1')
            #
            conv2_1 = conv2D_bn_relu(input=pool1, output_chn=64, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='rend_Conv2_1')
            conv2_2 = conv2D_bn_relu(input=conv2_1, output_chn=64, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='rend_Conv2_2')
            pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=2, strides=2, name='pool2')
            #
            conv3_1 = conv2D_bn_relu(input=pool2, output_chn=128, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='rend_Conv3_1')

            # up-sampling path
            deconv1_1 = deconv2D_bn_relu(input=conv3_1, output_chn=128, is_training=phase_flag, name='rend_deconv1_1')
            #
            concat_1 = tf.concat([deconv1_1, conv2_2], axis=concat_dim, name='concat_1')
            deconv1_2 = conv2D_bn_relu(input=concat_1, output_chn=64, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='rend_deconv1_2')
            deconv2_1 = deconv2D_bn_relu(input=deconv1_2, output_chn=64, is_training=phase_flag, name='rend_deconv2_1')
            #
            concat_2 = tf.concat([deconv2_1, conv1_2], axis=concat_dim, name='concat_2')
            deconv2_2 = conv2D_bn_relu(input=concat_2, output_chn=32, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='rend_deconv2_2')

            # prediction
            updated_app = conv2d(input=deconv2_2, output_chn=1, kernel_size=1, stride=1, use_bias=True, name='rend_updated_app_conv')

            # ======================
            # auxiliary prediction 0
            aux0_conv = conv2d(input=deconv1_2, output_chn=16, kernel_size=1, stride=1, use_bias=True, name='rend_aux0_conv')
            aux0_prob = Deconv2d(input=aux0_conv, output_chn=1, name='rend_aux0_prob')

            return updated_app, aux0_prob


    # 2D unet graph
    def unet_segmentor(self, inputI):
        """2D U-net"""
        phase_flag = (self.phase=='train')
        concat_dim = 3
        with tf.variable_scope("unet_segmentor") as scope:
            conv1_1 = conv2D_bn_relu(input=inputI, output_chn=32, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='seg_Conv1')
            pool1 = tf.layers.max_pooling2d(inputs=conv1_1, pool_size=2, strides=2, name='pool1')
            #
            conv2_1 = conv2D_bn_relu(input=pool1, output_chn=64, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='seg_Conv2')
            pool2 = tf.layers.max_pooling2d(inputs=conv2_1, pool_size=2, strides=2, name='pool2')
            #
            conv3_1 = conv2D_bn_relu(input=pool2, output_chn=128, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='seg_Conv3a')
            conv3_2 = conv2D_bn_relu(input=conv3_1, output_chn=128, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='seg_Conv3b')
            pool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=2, strides=2, name='pool3')
            #
            conv5_1 = conv2D_bn_relu(input=pool3, output_chn=256, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='seg_conv5_1')

            # up-sampling path
            deconv1_1 = deconv2D_bn_relu(input=conv5_1, output_chn=256, is_training=phase_flag, name='seg_deconv1_1')
            #
            concat_2 = tf.concat([deconv1_1, conv3_2], axis=concat_dim, name='concat_2')
            deconv2_2 = conv2D_bn_relu(input=concat_2, output_chn=128, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='seg_deconv2_2')
            deconv3_1 = deconv2D_bn_relu(input=deconv2_2, output_chn=128, is_training=phase_flag, name='seg_deconv3_1')
            #
            concat_3 = tf.concat([deconv3_1, conv2_1], axis=concat_dim, name='concat_3')
            deconv3_2 = conv2D_bn_relu(input=concat_3, output_chn=64, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='seg_deconv3_2')
            deconv4_1 = deconv2D_bn_relu(input=deconv3_2, output_chn=64, is_training=phase_flag, name='seg_deconv4_1')
            #
            concat_4 = tf.concat([deconv4_1, conv1_1], axis=concat_dim, name='concat_4')
            deconv4_2 = conv2D_bn_relu(input=concat_4, output_chn=32, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='seg_deconv4_2')
            # predicted probability
            pred_prob = conv2d(input=deconv4_2, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True, name='seg_pred_prob')

            # ======================
            # auxiliary prediction 1
            aux1_conv = conv2d(input=deconv2_2, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True, name='seg_aux1_conv')
            aux1_deconv_1 = Deconv2d(input=aux1_conv, output_chn=self.output_chn, name='seg_aux1_deconv_1')
            aux1_prob = Deconv2d(input=aux1_deconv_1, output_chn=self.output_chn, name='seg_aux1_prob')
            # auxiliary prediction 2
            aux2_conv = conv2d(input=deconv3_2, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True, name='seg_aux2_conv')
            aux2_prob = Deconv2d(input=aux2_conv, output_chn=self.output_chn, name='seg_aux2_prob')

            # predicted labels
            soft_prob = tf.nn.softmax(pred_prob, name='pred_soft')
            pred_label = tf.argmax(soft_prob, axis=3, name='argmax')

            return pred_prob, soft_prob, pred_label, aux1_prob, aux2_prob

    # appearance discriminator
    def app_discriminator(self, img, reuse=False):
        phase_flag = (self.phase == 'train')

        with tf.variable_scope("app_discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = conv2D_bn_relu(input=img, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_app_h0_conv')
            h1 = conv2D_bn_relu(input=h0, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_app_h1_conv')
            h2 = conv2D_bn_relu(input=h1, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_app_h2_conv')
            h3 = conv2D_bn_relu(input=h2, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_app_h3_conv')
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'D_app_h3_lin')

            return tf.nn.sigmoid(h4), h4

    # appearance discriminator
    def strct_discriminator(self, img, reuse=False):
        phase_flag = (self.phase == 'train')

        with tf.variable_scope("strct_discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = conv2D_bn_relu(input=img, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_strct_h0_conv')
            h1 = conv2D_bn_relu(input=h0, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_strct_h1_conv')
            h2 = conv2D_bn_relu(input=h1, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_strct_h2_conv')
            h3 = conv2D_bn_relu(input=h2, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_strct_h3_conv')
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'D_strct_h3_lin')

            return tf.nn.sigmoid(h4), h4

    # appearance discriminator
    def pair_discriminator(self, img_pair, reuse=False):
        phase_flag = (self.phase == 'train')

        with tf.variable_scope("pair_discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = conv2D_bn_relu(input=img_pair, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_pair_h0_conv')
            h1 = conv2D_bn_relu(input=h0, output_chn=32, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_pair_h1_conv')
            h2 = conv2D_bn_relu(input=h1, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_pair_h2_conv')
            h3 = conv2D_bn_relu(input=h2, output_chn=64, kernel_size=3, stride=1, use_bias=True, is_training=phase_flag, name='D_pair_h3_conv')
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'D_pair_h3_lin')

            return tf.nn.sigmoid(h4), h4

    # train function
    def train_online_seg(self):
        """Train 2D U-net"""
        rend_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.render_loss, var_list=self.rend_vars)
        #
        rend2disc_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.app_g_loss, var_list=self.rend_vars)
        app_disc_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.app_d_loss, var_list=self.app_disc_vars)
        #
        segmentor_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.segmentor_loss, var_list=[self.segmentor_vars, self.rend_vars])
        #
        segmtor2strct_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.strct_g_loss, var_list=[self.segmentor_vars, self.rend_vars])
        strct_disc_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.strct_d_loss, var_list=self.strct_disc_vars)
        #
        pair_discG_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.pair_g_loss, var_list=[self.rend_vars, self.segmentor_vars])
        pair_discD_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1).minimize(self.pair_d_loss, var_list=self.pair_disc_vars)

        # initialization
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        # load model
        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # load training image files
        tr_img_clec, tr_label_clec = load_2d_img_pairs(self.traindata_dir, '.png', rend=0)
        # temporary file to save loss
        loss_log = open("loss.txt", "w")

        counter = 1
        for epoch in np.arange(self.epoch):
            start_time = time.time()
            # train batch
            rend_batch_img, rend_batch_label = get_batch_pairs(tr_img_clec, tr_label_clec, self.inputI_size, self.batch_size, chn=1)
            app_D_batch_img, app_D_batch_label = get_batch_pairs(tr_img_clec, tr_label_clec, self.inputI_size, self.batch_size, chn=1)
            strct_D_batch_img, strct_D_batch_label = get_batch_pairs(tr_img_clec, tr_label_clec, self.inputI_size, self.batch_size, chn=1)
            strct_D_batch_label = strct_D_batch_label.astype('float32')
            strct_D_batch_label = np.reshape(strct_D_batch_label, [self.batch_size, self.inputI_size, self.inputI_size, 1])

            pair_D_batch_img, pair_D_batch_label = get_batch_pairs(tr_img_clec, tr_label_clec, self.inputI_size, self.batch_size, chn=1)
            pair_D_batch_label = pair_D_batch_label.astype('float32')
            pair_D_batch_label = np.reshape(pair_D_batch_label, [self.batch_size, self.inputI_size, self.inputI_size, 1])

            # Update appearance render
            _, cur_rend_loss = self.sess.run([rend_optimizer, self.render_loss], feed_dict={self.orig_app: rend_batch_img})
            #
            _, cur_appG_loss = self.sess.run([rend2disc_optimizer, self.app_g_loss], feed_dict={self.orig_app: rend_batch_img})
            _, cur_appD_loss = self.sess.run([app_disc_optimizer, self.app_d_loss], feed_dict={self.orig_app: rend_batch_img, self.desired_app: app_D_batch_img})

            _, cur_seg_loss = self.sess.run([segmentor_optimizer, self.segmentor_loss], feed_dict={self.orig_app: rend_batch_img, self.orig_label: rend_batch_label})

            _, cur_strctG_loss = self.sess.run([segmtor2strct_optimizer, self.strct_g_loss], feed_dict={self.orig_app: rend_batch_img})
            _, cur_strctD_loss = self.sess.run([strct_disc_optimizer, self.strct_d_loss], feed_dict={self.orig_app: rend_batch_img, self.desired_strct: strct_D_batch_label})

            _, cur_pairG_loss = self.sess.run([pair_discG_optimizer, self.pair_g_loss], feed_dict={self.orig_app: rend_batch_img})
            _, cur_pairD_loss = self.sess.run([pair_discD_optimizer, self.pair_d_loss], feed_dict={self.orig_app: rend_batch_img, self.desired_pair_app: pair_D_batch_img, self.desired_pair_strct: pair_D_batch_label})

            loss_log.write("%s    %s    %s    %s    %s    %s    %s    %s\n" % (cur_rend_loss, cur_appG_loss, cur_appD_loss, cur_seg_loss, cur_strctG_loss, cur_strctD_loss, cur_pairG_loss, cur_pairD_loss))

            counter += 1
            print("Epoch: [%2d] time: %4.4f" % (epoch, time.time() - start_time))
            print("cur_rend_loss: %.8f, cur_appG_loss: %.8f, cur_appD_loss: %.8f, cur_seg_loss: %.8f" % (cur_rend_loss, cur_appG_loss, cur_appD_loss, cur_seg_loss))
            print("cur_strctG_loss: %.8f, cur_strctD_loss: %.8f, cur_pairG_loss: %.8f, cur_pairD_loss: %.8f" % (cur_strctG_loss, cur_strctD_loss, cur_pairG_loss, cur_pairD_loss))

            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)

        loss_log.close()


    #################################################
    ################### TEST HERE ###################
    #################################################
    # train function
    def test_online_seg(self):
        #######################
        # render the training dataset with the self-play-trained model to
        # get a more uniformed training dataset
        self.test_ini_rend_train()
        #######################
        rend_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr/10, beta1=self.beta1).minimize(self.render_loss / 2, var_list=self.rend_vars)
        #
        rend2disc_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr/10, beta1=self.beta1).minimize(self.app_g_loss, var_list=self.rend_vars)
        app_disc_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr/10, beta1=self.beta1).minimize(self.app_d_loss, var_list=self.app_disc_vars)
        #
        segmtor2strct_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr/10, beta1=self.beta1).minimize(self.strct_g_loss, var_list=[self.segmentor_vars, self.rend_vars])
        strct_disc_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr/10, beta1=self.beta1).minimize(self.strct_d_loss, var_list=self.strct_disc_vars)
        #
        pair_discG_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr/10, beta1=self.beta1).minimize(self.pair_g_loss, var_list=[self.rend_vars])
        pair_discD_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr/10, beta1=self.beta1).minimize(self.pair_d_loss, var_list=self.pair_disc_vars)
        # load all image files
        tr_img_clec, tr_label_clec = load_2d_img_pairs(self.traindata_dir, '.png', rend=1)
        te_img_clec, te_label_clec = load_2d_img_pairs(self.testdata_dir, '.png', rend=0)

        """generate a batch of paired images for training"""
        rend_batch_img = np.zeros([self.batch_size, self.inputI_size, self.inputI_size, 1]).astype('float32')
        rend_batch_label = np.zeros([self.batch_size, self.inputI_size, self.inputI_size]).astype('int32')
        # iteration number
        test_iteration = 25
        test_N = len(te_img_clec)
        #################################################
        ########## SET Test start and end HERE ##########
        #################################################
        test_s = 0
        test_e = test_N
        for t in np.arange(test_s, test_e):
            print("processing #%d image..." % t)
            # create folder for each image
            img_epoch_path = self.labeling_dir + "/" + str(t)
            if not os.path.exists(img_epoch_path):
                os.makedirs(img_epoch_path)
            curve_epoch_path = self.labeling_dir + "/curve"
            if not os.path.exists(curve_epoch_path):
                os.makedirs(curve_epoch_path)

            dice_delta_log = open((curve_epoch_path + "/" + str(t) + "_dif.txt"), "w")
            ################## initialization
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            if self.load_chkpoint(self.chkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            ##################

            test_dice = []
            ######################## test image
            ### randomly couple an image to increase the batch size for better training
            temp_test_img = (te_img_clec[t]).astype('float32')
            temp_test_label = (te_label_clec[t]).astype('int32')
            for k in range(self.batch_size):
                if k == 0:
                    rend_batch_img[0, :, :, 0] = temp_test_img
                    rend_batch_label[0, :, :] = temp_test_label
                else:
                    # randomly flip
                    if np.random.random() > 0.5:
                        rend_batch_img[k, :, :, 0] = (np.fliplr(temp_test_img)).astype('float32')
                        rend_batch_label[k, :, :] = (np.fliplr(temp_test_label)).astype('int32')
                    else:
                        rend_batch_img[k, :, :, 0] = (np.flipud(temp_test_img)).astype('float32')
                        rend_batch_label[k, :, :] = (np.flipud(temp_test_label)).astype('int32')

            ##################
            # evaluate the initial segmentation
            cube_label = self.sess.run(self.pred_label, feed_dict={self.orig_app: rend_batch_img})
            ini_dice_c = []
            for c in range(self.output_chn):
                ints = np.sum(((rend_batch_label[0, :, :] == c) * 1) * ((cube_label[0, :, :] == c) * 1))
                union = np.sum(((rend_batch_label[0, :, :] == c) * 1) + ((cube_label[0, :, :] == c) * 1)) + 0.0001
                ini_dice_c.append((2.0 * ints) / union)
            # save initial prediction
            temp_seg = cube_label[0, :, :]
            temp_seg = np.reshape(temp_seg, [self.inputI_size, self.inputI_size])
            temp_seg = (temp_seg * 255).astype('uint8')
            save_path = img_epoch_path + "/" + "_00.png"
            cv2.imwrite(save_path, temp_seg)

            ##################
            for epoch in np.arange(test_iteration):
                start_time = time.time()
                temp_rend_batch_img = rend_batch_img
                #######
                app_D_batch_img, app_D_batch_label = get_batch_pairs(tr_img_clec, tr_label_clec, self.inputI_size, self.batch_size, chn=1)
                #
                strct_D_batch_img, strct_D_batch_label = get_batch_pairs(tr_img_clec, tr_label_clec, self.inputI_size, self.batch_size, chn=1)
                strct_D_batch_label = strct_D_batch_label.astype('float32')
                strct_D_batch_label = np.reshape(strct_D_batch_label, [self.batch_size, self.inputI_size, self.inputI_size, 1])
                #
                pair_D_batch_img, pair_D_batch_label = get_batch_pairs(tr_img_clec, tr_label_clec, self.inputI_size, self.batch_size, chn=1)
                pair_D_batch_label = pair_D_batch_label.astype('float32')
                pair_D_batch_label = np.reshape(pair_D_batch_label, [self.batch_size, self.inputI_size, self.inputI_size, 1])
                # Update appearance render
                _, cur_rend_loss = self.sess.run([rend_optimizer, self.render_loss], feed_dict={self.orig_app: temp_rend_batch_img})

                _, cur_appG_loss = self.sess.run([rend2disc_optimizer, self.app_g_loss], feed_dict={self.orig_app: temp_rend_batch_img})
                _, cur_appG_loss = self.sess.run([rend2disc_optimizer, self.app_g_loss], feed_dict={self.orig_app: temp_rend_batch_img})
                _, cur_appD_loss = self.sess.run([app_disc_optimizer, self.app_d_loss], feed_dict={self.orig_app: temp_rend_batch_img, self.desired_app: app_D_batch_img})

                _, cur_strctG_loss = self.sess.run([segmtor2strct_optimizer, self.strct_g_loss], feed_dict={self.orig_app: temp_rend_batch_img})
                _, cur_strctG_loss = self.sess.run([segmtor2strct_optimizer, self.strct_g_loss], feed_dict={self.orig_app: temp_rend_batch_img})
                _, cur_strctG_loss = self.sess.run([segmtor2strct_optimizer, self.strct_g_loss], feed_dict={self.orig_app: temp_rend_batch_img})
                _, cur_strctD_loss = self.sess.run([strct_disc_optimizer, self.strct_d_loss], feed_dict={self.orig_app: temp_rend_batch_img, self.desired_strct: strct_D_batch_label})

                _, cur_pairG_loss = self.sess.run([pair_discG_optimizer, self.pair_g_loss], feed_dict={self.orig_app: temp_rend_batch_img})
                _, cur_pairG_loss = self.sess.run([pair_discG_optimizer, self.pair_g_loss], feed_dict={self.orig_app: temp_rend_batch_img})
                _, cur_pairD_loss = self.sess.run([pair_discD_optimizer, self.pair_d_loss], feed_dict={self.orig_app: temp_rend_batch_img, self.desired_pair_app: pair_D_batch_img, self.desired_pair_strct: pair_D_batch_label})

                # self.log_writer.add_summary(summary_str, counter)
                cube_label = self.sess.run(self.pred_label, feed_dict={self.orig_app: temp_rend_batch_img})
                updated_soft_prob = self.sess.run(self.updated_forground, feed_dict={self.orig_app: temp_rend_batch_img})
                dice_c = []
                for c in range(self.output_chn):
                    ints = np.sum(((rend_batch_label[0, :, :]==c)*1)*((cube_label[0, :, :]==c)*1))
                    union = np.sum(((rend_batch_label[0, :, :]==c)*1) + ((cube_label[0, :, :]==c)*1)) + 0.0001
                    dice_c.append((2.0*ints)/union)

                print("initial dice:")
                print ini_dice_c
                print("updated dice:")
                print dice_c
                print (dice_c[1] - ini_dice_c[1])*100

                test_dice.append(dice_c[1])

                dice_delta_log.write("%.8f\n" % ((dice_c[1] - ini_dice_c[1])*100))

                print("Epoch: [%2d] time: %4.4f" % (epoch, time.time() - start_time))
                print("cur_rend_loss: %.8f, cur_appG_loss: %.8f, cur_appD_loss: %.8f" % (cur_rend_loss, cur_appG_loss, cur_appD_loss))
                print("cur_strctG_loss; %.8f, cur_strctD_loss: %.8f, cur_pairG_loss: %.8f, cur_pairD_loss: %.8f" % (cur_strctG_loss, cur_strctD_loss, cur_pairG_loss, cur_pairD_loss))

                # plot dice curve
                self.plot_curve(test_dice, ini_dice_c[1], (curve_epoch_path + "/" + str(t) + "_dice.png"))

                # save prediction
                temp_seg = cube_label[0, :, :]
                temp_seg = np.reshape(temp_seg, [self.inputI_size, self.inputI_size])
                temp_seg = (temp_seg * 255).astype('uint8')
                seg_save_path = img_epoch_path + "/" + str(epoch) + ".png"
                cv2.imwrite(seg_save_path, temp_seg)
            dice_delta_log.close()


    def test_ini_rend_train(self):
        #########
        # rend the training dataset to a more uniform intensity distribution
        #########
        # create folder
        update_path = self.traindata_dir + "/rend_us"
        if not os.path.exists(update_path):
            os.makedirs(update_path)

        print "### rendering training dataset to a more uniform appearance..."
        print ("Saving to %s..." % update_path)

        # load all image files
        tr_img_clec, tr_label_clec = load_2d_img_pairs(self.traindata_dir, '.png', rend=0)

        # temporary file to save loss
        """generate a batch of paired images for training"""
        rend_batch_img = np.zeros([self.batch_size, self.inputI_size, self.inputI_size, 1]).astype('float32')

        ###### initialization
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        ######

        ######################## test image
        ### randomly couple an image to increase the batch size as 2 for better training
        test_N = len(tr_img_clec)
        for t in np.arange(test_N):
            print("processing # %d image..." % t)

            temp_test_img = (tr_img_clec[t]).astype('float32')
            for k in range(self.batch_size):
                if k == 0:
                    rend_batch_img[0, :, :, 0] = temp_test_img
                else:
                    # randomly flip
                    if np.random.random() > 0.5:
                        rend_batch_img[k, :, :, 0] = (np.fliplr(temp_test_img)).astype('float32')
                    else:
                        rend_batch_img[k, :, :, 0] = (np.flipud(temp_test_img)).astype('float32')
            ########################

            ######
            cube_img = self.sess.run(self.updated_app, feed_dict={self.orig_app: rend_batch_img})
            update_img = cube_img[0, :, :, 0]
            update_img = np.reshape(update_img, [self.inputI_size, self.inputI_size])
            update_img = (update_img - np.max(update_img)) / (np.max(update_img) - np.min(update_img))

            update_img = (update_img*255).astype('uint8')

            # cv2.imshow('img', np.concatenate(((orig_img*255).astype('uint8'), update_img.astype('uint8')), axis=1))
            # cv2.waitKey(0)
            #
            save_path = update_path + "/" + str(t) + ".png"
            cv2.imwrite(save_path, update_img)

        print "Done!"


    # save checkpoint file
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        model_dir = "%s_%s" % (self.batch_size, self.outputI_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    # load checkpoint file
    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.batch_size, self.outputI_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def plot_curve(self, metric_arr, ini_val, fig_name):
        t = np.arange(len(metric_arr))
        ini_v = np.ones(len(metric_arr))*ini_val
        # plt.title('%s' % fig_name)
        plt.xlabel('epochs')
        plt.ylabel('dice')
        plt.plot(metric_arr)
        plt.plot(t, ini_v, 'r--')
        plt.grid()
        plt.savefig(fig_name)
        # plt.show()
        plt.close()