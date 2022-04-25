import numpy as np
import json
import warnings

warnings.filterwarnings('ignore')
import os

import tensorflow as tf
from tensorflow_core.contrib.rnn.python.ops.rnn_cell import WeightNormLSTMCell
from gym.utils import seeding

MODE_Z = 1
MODE_Z_HIDDEN = 2

class missVAE(object):
    def __init__(self, hps, step, scope, gpu_mode=True, reuse=False):
        self.hps = hps
        self.step_est = step
        self.scope = scope
        self._seed()

        self.input_x = None
        self.input_domain_index = None
        self.input_sign = None
        self.seq_length = None

        self.mu = None
        self.logvar = None

        self.z = None
        self.z_map = None
        self.y_o = None  # VAE output
        self.y_o_next = None
        self.y_r = None

        self.input_z = None
        self.output_z = None
        self.input_a = None  # RNN input placeholder for action
        self.input_a_prev = None
        self.input_r = None  # RNN input placeholder for reward
        self.input_r_prev = None
        self.cell = None

        self.initial_state = None
        self.final_state = None

        self.out_logmix = None
        self.out_mean = None
        self.out_logstd = None
        self.z_out_logmix = None
        self.z_out_mean = None
        self.z_out_logstd = None

        self.global_step = None
        self.r_obs_loss = None  # reconstruction loss for observation
        self.r_next_obs_loss = None
        self.r_reward_loss = None  # reconstruction loss for reward
        self.kl_loss = None
        self.vae_loss = None
        self.vae_pure_loss = None
        self.transition_loss = None
        self.vae_causal_filter_loss = None
        self.theta_loss = None
        self.causal_filter_loss = None
        self.total_loss = None

        self.lr = None
        self.train_op = None
        self.vae_lr = None
        self.vae_train_op = None
        self.transition_lr = None
        self.transition_train_op = None

        self.init = None
        self.assign_ops = None
        self.weightnorm_ops = None
        self.sess = None

        self.merged = None

        self.SSL_A = None  # parameter C_{s->o} for o_t = f(A * s_t, e_t)
        self.SSL_B = None  # parameter C_{s->r} for r_t = g(B * s_t, C * a_t, epsilon_t)
        self.SSL_C = None  # parameter C_{a->r} for r_t = g(B * s_t, C * a_t, epsilon_t)
        self.SSL_D = None  # parameter C_{s->s} for s_{t+1} = h(D * s_t, E * a_t, eta_t)
        self.SSL_E = None  # parameter C_{a->s} for s_{t+1} = h(D * s_t, E * a_t, eta_t)
        self.SSL_F = None  # the parameter before theta_s

        self.theta_o = None  # characterize changing factors [dimension of o, number of domains]
        self.theta_s = None
        self.theta_r = None
        self.theta = None  # theta = [theta_o,theta_s,theta_r]

        if self.hps.is_training == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            self.gpu_config = tf.ConfigProto(device_count={'GPU': 4},
                                             allow_soft_placement=True,
                                             log_device_placement=False)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.gpu_config = tf.ConfigProto(device_count={'GPU': 0},
                                             allow_soft_placement=True,
                                             log_device_placement=False)

        self.gpu_config.gpu_options.allow_growth = True

        with tf.variable_scope(self.scope, reuse=reuse):
            if not gpu_mode:
                print("model using cpu")
                self.g = tf.Graph()
                with self.g.as_default():
                    self.build_model()
            else:
                print("model using gpu")
                self.g = tf.Graph()
                with self.g.as_default():
                    self.build_model()
        self.init_session()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def build_model(self):
        if self.step_est == 2:
            # parameters for structured state learning (SSL)
            with tf.variable_scope('SSL', reuse=tf.AUTO_REUSE):
                self.SSL_D = tf.get_variable('D', [self.hps.z_size, self.hps.z_size], tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
                self.SSL_E = tf.get_variable('E', [self.hps.action_size, self.hps.z_size], tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
                self.SSL_F = tf.get_variable('F', [self.hps.theta_s_size, self.hps.z_size], tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
            self.theta_s = tf.get_variable('theta_s', [self.hps.theta_s_size, self.hps.domain_size], tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer())

            self.input_a = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.action_size])
            self.input_z = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.z_size])
            self.output_z = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.z_size])
            self.input_domain_index = tf.placeholder(tf.int32, shape=[self.hps.batch_size])
            lossfunc = self.markovian_tran()
        else:
            self.seq_length = tf.placeholder(tf.int32)
            self.input_sign = tf.placeholder(tf.int32)

            # parameters for structured state learning (SSL)
            with tf.variable_scope('SSL', reuse=tf.AUTO_REUSE):
                self.SSL_A = tf.get_variable('A', [self.hps.z_size], tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
                self.SSL_B = tf.get_variable('B', [self.hps.z_size], tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
                self.SSL_C = tf.get_variable('C', [self.hps.action_size], tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
                self.SSL_D = tf.get_variable('D', [self.hps.z_size, self.hps.z_size], tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
                self.SSL_E = tf.get_variable('E', [self.hps.action_size, self.hps.z_size], tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer())
                if self.step_est == 3:
                    self.SSL_F = tf.get_variable('F', [self.hps.theta_s_size, self.hps.z_size], tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer())

            with tf.variable_scope('theta', reuse=tf.AUTO_REUSE):
                self.theta_o = tf.get_variable('theta_o', [self.hps.theta_o_size, self.hps.domain_size], tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer())
                self.theta_s = tf.get_variable('theta_s', [self.hps.theta_s_size, self.hps.domain_size], tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer())
                self.theta_r = tf.get_variable('theta_r', [self.hps.theta_r_size, self.hps.domain_size], tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer())

            ############################################# Encoder ##########################################################
            #################################### q(s_t | o_{<=t}, a_{<t}, r_{<t}) ##########################################
            # input of VAE
            self.input_x = tf.placeholder(tf.float32, shape=[self.hps.batch_size, None, 128, 128, 1])
            self.input_a_prev = tf.placeholder(dtype=tf.float32,
                                               shape=[self.hps.batch_size, None, self.hps.action_size])
            self.input_a = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, None, self.hps.action_size])
            self.input_r_prev = tf.placeholder(dtype=tf.float32,
                                               shape=[self.hps.batch_size, None, self.hps.reward_size])
            self.input_r = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, None, self.hps.reward_size])

            self.input_domain_index = tf.placeholder(dtype=tf.int32, shape=[self.hps.batch_size])

            # Encoder: s_t = q(o_t, a_{t-1}, r_{t-1})
            obs_x = tf.reshape(self.input_x, [-1, 128, 128, 1])
            obs_x_next = tf.reshape(self.input_x[:, 1:, :, :, :], [-1, 128, 128, 1])
            obs_a_prev = tf.reshape(self.input_a_prev, [-1, self.hps.action_size])
            obs_a = tf.reshape(self.input_a, [-1, self.hps.action_size])
            obs_r_prev = tf.reshape(self.input_r_prev, [-1, self.hps.reward_size])
            obs_r = tf.reshape(self.input_r, [-1, self.hps.reward_size])

            with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
                with tf.device('gpu:0'):
                    # obs_x
                    hx = tf.layers.conv2d(obs_x, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
                    hx = tf.layers.conv2d(hx, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
                    hx = tf.layers.conv2d(hx, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
                    hx = tf.layers.conv2d(hx, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
                    hx = tf.reshape(hx, [-1, 6 * 6 * 256])

                    # obs_a
                    ha = tf.layers.dense(obs_a_prev, 1 * 128, activation=tf.nn.relu, name="enc_action_fc1")

                    # obs_r
                    hr = tf.layers.dense(obs_r_prev, 1 * 128, activation=tf.nn.relu, name="enc_reward_fc1")

                    theta0_o = []
                    theta0_s = []
                    theta0_r = []
                    for i in range(self.hps.batch_size):
                        d_idx = tf.slice(self.input_domain_index, [i], [1])
                        tmp_o = tf.transpose(self.theta_o[:, d_idx[0]])
                        tmp_o = tf.stack([tmp_o] * self.hps.max_seq_len)
                        theta0_o.append(tmp_o)
                        tmp_s = tf.transpose(self.theta_s[:, d_idx[0]])
                        tmp_s = tf.stack([tmp_s] * self.hps.max_seq_len)
                        theta0_s.append(tmp_s)
                        tmp_r = tf.transpose(self.theta_r[:, d_idx[0]])
                        tmp_r = tf.stack([tmp_r] * self.hps.max_seq_len)
                        theta0_r.append(tmp_r)
                    theta0_o = tf.convert_to_tensor(theta0_o)
                    theta0_o = tf.reshape(theta0_o, [-1, self.hps.theta_o_size])
                    theta0_s = tf.convert_to_tensor(theta0_s)
                    theta0_s = tf.reshape(theta0_s, [-1, self.hps.theta_s_size])
                    theta0_r = tf.convert_to_tensor(theta0_r)
                    theta0_r = tf.reshape(theta0_r, [-1, self.hps.theta_r_size])
                    theta0 = tf.concat([theta0_o, theta0_s, theta0_r], 1)
                    hd = tf.layers.dense(theta0, 3 * 128, activation=tf.nn.relu, name="enc_domain_fc1")
                    h_xard = tf.concat([hx, ha, hr, hd], 1)
                ################################################### LSTM ###################################################
                with tf.device('gpu:0'):
                    cell = WeightNormLSTMCell(self.hps.rnn_size, norm=True)
                    self.cell = cell
                    input_h = tf.reshape(h_xard, [self.hps.batch_size, self.seq_length, 6 * 6 * 256 + 5 * 128])
                    self.initial_state = cell.zero_state(batch_size=self.hps.batch_size, dtype=tf.float32)
                    NOUT = self.hps.output_seq_width * self.hps.num_mixture * 3  # bh: 3 means out_logmix, out_mean, out_logstd

                    with tf.variable_scope('RNN'):
                        output_w = tf.get_variable("output_w", [self.hps.rnn_size, NOUT],
                                                   initializer=tf.contrib.layers.xavier_initializer())
                        output_w = tf.nn.l2_normalize(output_w, [0])
                        output_b = tf.get_variable("output_b", [NOUT],
                                                   initializer=tf.contrib.layers.xavier_initializer())

                    output, last_state = tf.nn.dynamic_rnn(cell, input_h, initial_state=self.initial_state,
                                                           time_major=False, swap_memory=True, dtype=tf.float32,
                                                           scope="RNN")

                    output = tf.reshape(output, [-1, self.hps.rnn_size])
                    output = tf.nn.xw_plus_b(output, output_w, output_b)
                    output = tf.reshape(output, [-1, self.hps.num_mixture * 3])
                    self.final_state = last_state

                    ########################################## MDN-RNN ################################################
                    out_logmix, out_mean, out_logstd = self.get_mdn_coef(output)
                    self.out_logmix = out_logmix
                    self.out_mean = out_mean
                    self.out_logstd = out_logstd

                    if self.step_est == 3:
                        if self.hps.is_training == 0:
                            # the index of the cluster which has the largest probability
                            logmix_map_idx = tf.argmax(out_logmix, 1)
                            out_mean_map = []
                            for i in range(out_logmix.shape[0]):
                                out_mean_map.append(out_mean[i, logmix_map_idx[i]])
                            out_mean_map = tf.convert_to_tensor(out_mean_map)
                            self.z_map = tf.reshape(out_mean_map, [-1, self.hps.output_seq_width])

                    logmix2 = out_logmix / self.hps.temperature
                    logmix2 -= tf.reduce_max(logmix2)
                    logmix2 = tf.exp(logmix2)
                    logmix2 /= tf.reshape(tf.reduce_sum(logmix2, 1), [-1, 1])

                    mixture_len = self.hps.batch_size * self.seq_length * self.hps.output_seq_width

                    ########################################## Sampling from MDN-RNN ###########################################
                    logmix2_list = [logmix2[:, 0]]
                    for j in range(self.hps.num_mixture - 1):
                        logmix2_list.append(logmix2[:, j + 1] + logmix2_list[j])

                    logmix2 = tf.stack(logmix2_list, axis=1)

                    mixture_rand_idx = tf.tile(tf.random_uniform([mixture_len, 1]), [1, self.hps.num_mixture])
                    zero_ref = tf.zeros_like(mixture_rand_idx)

                    idx = tf.argmax(tf.cast(tf.less_equal(mixture_rand_idx - logmix2, zero_ref), tf.int32),
                                    axis=1, output_type=tf.int32)

                    indices = tf.range(0, mixture_len) * self.hps.num_mixture + idx
                    chosen_mean = tf.gather(tf.reshape(out_mean, [-1]), indices)
                    chosen_logstd = tf.gather(tf.reshape(out_logstd, [-1]), indices)

                    rand_gaussian = tf.random_normal([mixture_len]) * np.sqrt(self.hps.temperature)
                    sample_z = chosen_mean + tf.exp(chosen_logstd) * rand_gaussian

                    self.z = tf.reshape(sample_z, [-1, self.hps.output_seq_width])

            ############################################# Decoder ##########################################################
            with tf.device("gpu:1"):
                # Decoder for Observation: o_t = f(A * s_t, theta_o, e_t)
                ssl_zo = tf.multiply(self.z, self.SSL_A)  # SSL from state to observation
                with tf.variable_scope('ObsDecoder', reuse=tf.AUTO_REUSE):
                    h1 = tf.layers.dense(ssl_zo, 6 * 6 * 256, kernel_constraint=self.deconv_weightnorm, name="dec_fc")
                    # theta_o MLP and then concatenate , notice the shape. flexible when to concatenate with h, need tuning
                    h2 = tf.layers.dense(theta0_o, 1 * 256, kernel_constraint=self.deconv_weightnorm,
                                         name="dec_theta_o")
                    h = tf.concat([h1, h2], 1)
                    h = tf.reshape(h, [-1, 1, 1, 37 * 256])

                    with tf.device("gpu:2"):
                        h = tf.layers.conv2d_transpose(h, 256, 5, strides=2, activation=tf.nn.relu,
                                                       kernel_constraint=self.deconv_weightnorm, name="dec_deconv1")
                        h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu,
                                                       kernel_constraint=self.deconv_weightnorm, name="dec_deconv2")
                        h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu,
                                                       kernel_constraint=self.deconv_weightnorm, name="dec_deconv3")

                    with tf.device("gpu:1"):
                        h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu,
                                                       kernel_constraint=self.deconv_weightnorm, name="dec_deconv4")
                        self.y_o = tf.layers.conv2d_transpose(h, 1, 6, strides=2, activation=tf.nn.tanh,
                                                              kernel_constraint=self.deconv_weightnorm,
                                                              name="dec_deconv5")
                # Decoder for Next Observation
                z_next = tf.reshape(self.z, [self.hps.batch_size, self.seq_length, self.hps.output_seq_width])

                if self.hps.is_training == 1:
                    z_next = tf.reshape(z_next[:, :-1, :], [-1, self.hps.output_seq_width])
                else:
                    z_next = tf.reshape(z_next, [-1, self.hps.output_seq_width])
            with tf.device("gpu:1"):
                with tf.variable_scope('NextObsDecoder', reuse=tf.AUTO_REUSE):
                    nh = tf.layers.dense(z_next, 6 * 6 * 256, kernel_constraint=self.deconv_weightnorm, name="dec_fc")
                    with tf.device("gpu:3"):
                        nh = tf.reshape(nh, [-1, 1, 1, 6 * 6 * 256])
                        nh = tf.layers.conv2d_transpose(nh, 256, 5, strides=2, activation=tf.nn.relu,
                                                        kernel_constraint=self.deconv_weightnorm, name="dec_deconv1")
                        nh = tf.layers.conv2d_transpose(nh, 128, 5, strides=2, activation=tf.nn.relu,
                                                        kernel_constraint=self.deconv_weightnorm, name="dec_deconv2")
                        nh = tf.layers.conv2d_transpose(nh, 64, 5, strides=2, activation=tf.nn.relu,
                                                        kernel_constraint=self.deconv_weightnorm, name="dec_deconv3")
                    with tf.device("gpu:0"):
                        nh = tf.layers.conv2d_transpose(nh, 32, 6, strides=2, activation=tf.nn.relu,
                                                        kernel_constraint=self.deconv_weightnorm, name="dec_deconv4")

                        self.y_o_next = tf.layers.conv2d_transpose(nh, 1, 6, strides=2, activation=tf.nn.tanh,
                                                                   kernel_constraint=self.deconv_weightnorm,
                                                                   name="dec_deconv5")

                # Decoder for Reward: r_t = g(B * s_t, C * a_t, theta_r, epsilon_t)
                ssl_zr = tf.multiply(self.z, self.SSL_B)  # SSL for state to reward
                ssl_ar = tf.multiply(obs_a, self.SSL_C)  # SSL for action to reward
                ssl_zar = tf.concat([ssl_zr, ssl_ar], 1)
                with tf.variable_scope('RewardDecoder', reuse=tf.AUTO_REUSE):
                    lin_h1 = tf.layers.dense(ssl_zar, 4 * 128, activation=tf.nn.relu,
                                             kernel_constraint=self.deconv_weightnorm, name="dec_fc1")
                    lin_h2 = tf.layers.dense(theta0_r, 1 * 128, activation=tf.nn.relu,
                                             kernel_constraint=self.deconv_weightnorm, name="dec_theta_r1")
                    lin_h = tf.concat([lin_h1, lin_h2], 1)
                    lin_h = tf.layers.dense(lin_h, 1 * 128, activation=tf.nn.relu,
                                            kernel_constraint=self.deconv_weightnorm, name="dec_fc2")
                    self.y_r = tf.layers.dense(lin_h, 1, kernel_constraint=self.deconv_weightnorm, name="dec_fc3")
            if self.step_est == 3:
                lossfunc = self.markovian_tran()
        ######################################## Loss Function #########################################################
        if self.hps.is_training == 1:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if self.step_est != 2:
                # VAE Loss
                # reconstruction loss for observation
                self.r_obs_loss = tf.reduce_sum(
                    tf.square(obs_x - self.y_o),
                    reduction_indices=[1, 2, 3]
                )
                self.r_obs_loss = tf.reduce_mean(self.r_obs_loss)
                # reconstruction loss for next observation
                self.r_next_obs_loss = tf.reduce_sum(
                    tf.square(obs_x_next - self.y_o_next),
                    reduction_indices=[1, 2, 3]
                )
                self.r_next_obs_loss = tf.reduce_mean(self.r_next_obs_loss)
                # reconstruction loss for reward
                self.r_reward_loss = tf.reduce_sum(
                    tf.square(obs_r - self.y_r),
                    reduction_indices=[1]
                )
                self.r_reward_loss = tf.reduce_mean(self.r_reward_loss)
                # KL loss
                self.out_logmix = out_logmix
                self.out_mean = out_mean
                self.out_logstd = out_logstd
                self.kl_loss = 0
                for g_idx in range(self.hps.num_mixture):
                    g_logmix = tf.reshape(self.out_logmix[:, g_idx], [-1, self.hps.output_seq_width])
                    g_mean = tf.reshape(self.out_mean[:, g_idx], [-1, self.hps.output_seq_width])
                    g_logstd = tf.reshape(self.out_logstd[:, g_idx], [-1, self.hps.output_seq_width])
                    self.kl_loss += self.kl_gmm(g_logmix, g_mean, g_logstd)
                self.kl_loss = tf.log(1 / (self.kl_loss + 1e-10) + 1e-10)
                self.kl_loss = tf.reduce_mean(self.kl_loss)

                # Theta Loss for sparse constraint
                theta_o_loss = 0
                theta_s_loss = 0
                theta_r_loss = 0
                for i in range(self.hps.domain_size):
                    for j in range(i + 1, self.hps.domain_size):
                        theta_o_loss += .1 * tf.reduce_sum(tf.abs(self.theta_o[:, i] - self.theta_o[:, j]))
                        theta_r_loss += .1 * tf.reduce_sum(tf.abs(self.theta_r[:, i] - self.theta_r[:, j]))
                        theta_s_loss += 0 * tf.reduce_sum(tf.abs(self.theta_s[:, i] - self.theta_s[:, j]))
                    theta_o_loss += 0.1 * tf.reduce_sum(tf.abs(self.theta_o[:, i]))
                    theta_r_loss += 0.1 * tf.reduce_sum(tf.abs(self.theta_r[:, i]))
                    theta_s_loss += 0 * tf.reduce_sum(tf.abs(self.theta_s[:, i]))
                self.theta_loss = theta_o_loss + theta_r_loss + theta_s_loss

                # VAE Causal Filter L1 Loss for sparse constraint
                if self.step_est == 3:
                    self.vae_causal_filter_loss = \
                        .3 * tf.reduce_mean(tf.abs(self.SSL_A)) + \
                        0.2 * tf.reduce_mean(tf.abs(self.SSL_B)) + \
                        0.2 * tf.reduce_mean(tf.abs(self.SSL_C))
                else:
                    self.vae_causal_filter_loss = \
                        0.1 * tf.reduce_mean(tf.abs(self.SSL_A)) + \
                        0.1 * tf.reduce_mean(tf.abs(self.SSL_B)) + \
                        0.1 * tf.reduce_mean(tf.abs(self.SSL_C))

                self.vae_loss = self.r_obs_loss + self.r_next_obs_loss + self.r_reward_loss + self.kl_loss + \
                                self.vae_causal_filter_loss + self.theta_loss
                if self.step_est == 3:
                    self.vae_pure_loss = self.r_obs_loss + self.r_next_obs_loss + self.r_reward_loss + self.kl_loss + self.theta_loss
                else:
                    self.vae_pure_loss = self.r_obs_loss + self.r_next_obs_loss + self.r_reward_loss + self.kl_loss
            if self.step_est == 2:
                self.transition_loss = tf.reduce_mean(lossfunc)

                self.causal_filter_loss = \
                    1.6 * (tf.reduce_sum(tf.abs(self.SSL_D)) - tf.reduce_sum(tf.abs(tf.matrix_diag_part(self.SSL_D)))) \
                    / (self.hps.z_size * self.hps.z_size - self.hps.z_size) + \
                    0.1 * tf.reduce_mean(tf.abs(tf.matrix_diag_part(self.SSL_D))) + \
                    0.01 * tf.reduce_mean(tf.abs(self.SSL_E)) + \
                    0.01 * tf.reduce_mean(tf.abs(self.SSL_F))

                self.total_loss = self.transition_loss + self.causal_filter_loss
            if self.step_est == 3:
                self.transition_loss = tf.reduce_mean(lossfunc)
                # Causal Filter L1 Loss for sparse constraint
                self.causal_filter_loss = \
                    0.1 * tf.reduce_mean(tf.abs(self.SSL_A)) + \
                    0.1 * tf.reduce_mean(tf.abs(self.SSL_B)) + \
                    0.1 * tf.reduce_mean(tf.abs(self.SSL_C)) + \
                    1 * (tf.reduce_sum(tf.abs(self.SSL_D)) - tf.reduce_sum(tf.abs(tf.matrix_diag_part(self.SSL_D)))) \
                    / (self.hps.z_size * self.hps.z_size - self.hps.z_size) + \
                    0.1 * tf.reduce_mean(tf.abs(tf.matrix_diag_part(self.SSL_D))) + \
                    0.01 * tf.reduce_mean(tf.abs(self.SSL_E)) + \
                    0.01 * tf.reduce_mean(tf.abs(self.SSL_F))

                self.total_loss = self.vae_pure_loss + self.transition_loss + self.causal_filter_loss
            ############################################## Three Optimizers ##########################################
            if self.step_est == 2:
                # Total Optimizer
                self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
                optimizer = tf.train.AdamOptimizer(self.lr)
                gvs = optimizer.compute_gradients(self.total_loss)
                capped_gvs = [(tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip), var) for grad, var in
                              gvs]
                self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')
            else:
                self.vae_lr = tf.Variable(self.hps.learning_rate, trainable=False)
                # VAE Optimizer
                vae_optimizer = tf.train.AdamOptimizer(self.vae_lr)
                vae_gvs = vae_optimizer.compute_gradients(self.vae_loss, colocate_gradients_with_ops=True)
                capped_vae_gvs = []
                if self.step_est == 3:
                    for grad, var in vae_gvs:
                        if not (var.name.startswith('SSL/D') or var.name.startswith('SSL/E') or var.name.startswith(
                                'SSL/F')
                                or var.name.startswith('theta_s') or var.name.startswith('Dynamics')):
                            capped_vae_gvs.append(
                                (tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip), var))
                else:
                    for grad, var in vae_gvs:
                        if not (var.name.startswith('SSL/D') or var.name.startswith('SSL/E') or var.name.startswith(
                                'Dynamics')):
                            def f1(): return grad + tf.random_normal(tf.shape(grad, name=None))

                            def f2(): return grad

                            grad_tmp = tf.case([(tf.reduce_mean(grad) > 0.01, f2), (tf.equal(self.input_sign, 0), f2)],
                                               default=f1)
                            capped_vae_gvs.append(
                                (tf.clip_by_value(grad_tmp, -self.hps.grad_clip, self.hps.grad_clip), var))
                self.vae_train_op = vae_optimizer.apply_gradients(capped_vae_gvs,
                                                                  global_step=self.global_step,
                                                                  name='vae_train_step')
                if self.step_est == 3:
                    # RNN Optimizer
                    self.transition_lr = tf.Variable(self.hps.learning_rate, trainable=False)
                    transition_optimizer = tf.train.AdamOptimizer(self.transition_lr)

                    transition_gvs = transition_optimizer.compute_gradients(self.transition_loss)
                    self.transition_train_op = transition_optimizer.apply_gradients(transition_gvs,
                                                                                    name='rnn_train_step')
                    # Total Optimizer
                    self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
                    optimizer = tf.train.AdamOptimizer(self.lr)

                    gvs = optimizer.compute_gradients(self.total_loss)
                    capped_gvs = []
                    for grad, var in gvs:
                        tf.summary.histogram("%s-grad" % var.name, grad)

                        def f1(): return grad + tf.random_normal(tf.shape(grad, name=None))

                        def f2(): return grad

                        grad_tmp = tf.case([(tf.reduce_mean(grad) > 0.01, f2), (tf.equal(self.input_sign, 0), f2)],
                                           default=f1)
                        capped_gvs.append((tf.clip_by_value(grad_tmp, -self.hps.grad_clip, self.hps.grad_clip), var))
                    self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step,
                                                              name='train_step')

        # initialize vars
        self.init = tf.global_variables_initializer()
        self.merged = tf.summary.merge_all()
        t_vars = tf.trainable_variables()
        self.assign_ops = {}
        self.weightnorm_ops = {}

        for var in t_vars:
            pshape = var.get_shape()

            pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + '_placeholder')
            assign_op = var.assign(pl)
            self.assign_ops[var] = (assign_op, pl)

            weightnorm_op = var.assign(pl)

            self.weightnorm_ops[var] = (weightnorm_op, pl)

    # Markovian Transition
    # p(s_{t+1} | s_t, a_t) for step 3 of model estimation
    # p(s_{t+1} | s_t, a_t, theta_s) for step 2 of model estimation
    def markovian_tran(self):
        D_NOUT = 1 * self.hps.num_mixture * 3
        theta0_s = []
        for i in range(self.hps.batch_size):
            d_idx = tf.slice(self.input_domain_index, [i], [1])
            tmp_s = tf.transpose(self.theta_s[:, d_idx[0]])
            if self.step_est == 3:
                tmp_s = tf.stack([tmp_s] * (self.hps.max_seq_len - 1))
            theta0_s.append(tmp_s)
        theta0_s = tf.convert_to_tensor(theta0_s)
        theta0_s = tf.reshape(theta0_s, [-1, self.hps.theta_s_size])

        if self.hps.is_training == 1:
            if self.step_est == 3:
                self.input_z = tf.reshape(self.z, [self.hps.batch_size, self.seq_length, self.hps.z_size])
                tmp_input_z = tf.reshape(self.input_z[:, :-1, :], [-1, self.hps.z_size])
            ssl_zz = []  # SSL for state to state
            for i in range(self.hps.z_size):
                if self.step_est == 2:
                    ssl_zz.append(tf.multiply(self.input_z[:, :], self.SSL_D[:, i]))
                if self.step_est == 3:
                    ssl_zz.append(tf.multiply(tmp_input_z, self.SSL_D[:, i]))
            ssl_zz = tf.convert_to_tensor(ssl_zz)  # ssl_zz: z_size x batch_size x z_size
            ssl_zz = tf.reshape(ssl_zz, [-1, self.hps.z_size])  # ssl_zz: (z_size x batch_size) x z_size

            if self.step_est == 3:
                tmp_input_a = tf.reshape(self.input_a[:, :-1, :], [-1, self.hps.action_size])
            ssl_az = []  # SSL for action to state
            for i in range(self.hps.z_size):
                if self.step_est == 2:
                    ssl_az.append(tf.multiply(self.input_a[:, :], self.SSL_E[:, i]))
                if self.step_est == 3:
                    ssl_az.append(tf.multiply(tmp_input_a, self.SSL_E[:, i]))
            ssl_az = tf.convert_to_tensor(ssl_az)  # ssl_zz: z_size x batch_size x action_size
            ssl_az = tf.reshape(ssl_az, [-1, self.hps.action_size])  # ssl_zz: (z_size x batch_size) x action_size
            ssl_dz = []  # SSL for theta_s to state
            for i in range(self.hps.z_size):
                ssl_dz.append(tf.multiply(theta0_s[:, :], self.SSL_F[:, i]))
            ssl_dz = tf.convert_to_tensor(ssl_dz)
            ssl_dz = tf.reshape(ssl_dz, [-1, self.hps.theta_s_size])
            ssl_za = tf.reshape(tf.concat([ssl_zz, ssl_az, ssl_dz], 1),
                                [-1, self.hps.z_size + self.hps.action_size + self.hps.theta_s_size])
            if self.step_est == 3:
                self.output_z = tf.reshape(self.input_z[:, 1:, :], [-1, self.hps.z_size])

            random_ssl_za = ssl_za
            random_output_z = self.output_z
        else:
            self.input_z = tf.placeholder(dtype=tf.float32,
                                          shape=[self.hps.batch_size, None, self.hps.z_size])

            tmp_input_z = tf.reshape(self.input_z, [-1, self.hps.z_size])
            # s_{t+1} = h(D * s_t, E * a_t, eta_t)
            ssl_zz = []  # SSL for state to state
            for i in range(self.hps.z_size):
                ssl_zz.append(tf.multiply(tmp_input_z, self.SSL_D[:, i]))
            ssl_zz = tf.convert_to_tensor(ssl_zz)  # ssl_zz: z_size x batch_size x z_size
            ssl_zz = tf.reshape(ssl_zz, [-1, self.hps.z_size])  # ssl_zz: (z_size x batch_size) x z_size

            tmp_input_a = tf.reshape(self.input_a, [-1, self.hps.action_size])
            ssl_az = []  # SSL for action to state
            for i in range(self.hps.z_size):
                ssl_az.append(tf.multiply(tmp_input_a, self.SSL_E[:, i]))
            ssl_az = tf.convert_to_tensor(ssl_az)  # ssl_zz: z_size x batch_size x action_size
            ssl_az = tf.reshape(ssl_az, [-1, self.hps.action_size])  # ssl_zz: (z_size x batch_size) x action_size

            ssl_dz = []  # SSL for theta_s to state
            for i in range(self.hps.z_size):
                ssl_dz.append(tf.multiply(theta0_s[:, :], self.SSL_F[:, i]))
            ssl_dz = tf.convert_to_tensor(ssl_dz)
            ssl_dz = tf.reshape(ssl_dz, [-1, self.hps.theta_s_size])

            ssl_za = tf.reshape(tf.concat([ssl_zz, ssl_az, ssl_dz], 1),
                                [-1, self.hps.z_size + self.hps.action_size + self.hps.theta_s_size])

            random_ssl_za = ssl_za

        with tf.variable_scope('Dynamics', reuse=tf.AUTO_REUSE):
            hd = tf.layers.dense(random_ssl_za, 6 * 128, activation=tf.nn.relu,
                                 kernel_constraint=self.deconv_weightnorm, name="fc")
            hd = tf.layers.dense(hd, 4 * 128, activation=tf.nn.relu,
                                 kernel_constraint=self.deconv_weightnorm, name="fc11")
            hd = tf.layers.dense(hd, 2 * 128, activation=tf.nn.relu,
                                 kernel_constraint=self.deconv_weightnorm, name="fc2")
            hd = tf.layers.dense(hd, 128, activation=tf.nn.relu,
                                 kernel_constraint=self.deconv_weightnorm, name="fc3")

            d_output_w = tf.get_variable("output_w", [128, D_NOUT])
            d_output_w = tf.nn.l2_normalize(d_output_w, [0])
            d_output_b = tf.get_variable("output_b", [D_NOUT])

        z_output = tf.nn.xw_plus_b(hd, d_output_w, d_output_b)
        z_output = tf.reshape(z_output, [self.hps.z_size, -1, D_NOUT])
        z_output = tf.transpose(z_output, perm=[1, 0, 2])
        z_output = tf.reshape(z_output, [-1, self.hps.num_mixture * 3])

        z_out_logmix, z_out_mean, z_out_logstd = self.get_mdn_coef(z_output)

        self.z_out_logmix = z_out_logmix
        self.z_out_mean = z_out_mean
        self.z_out_logstd = z_out_logstd

        if self.hps.is_training == 1:
            # reshape target data so that it is compatible with prediction shape
            z_flat_target_data = tf.reshape(random_output_z, [-1, 1])

            lossfunc = self.get_lossfunc(z_out_logmix, z_out_mean, z_out_logstd, z_flat_target_data)
            return lossfunc

    def init_session(self):
        self.sess = tf.Session(graph=self.g, config=self.gpu_config)
        self.sess.run(self.init)

    def reset(self):
        state_init = self.sess.run(self.initial_state)
        action_init = np.zeros((self.hps.batch_size, 1, self.hps.action_size))
        reward_init = np.zeros((self.hps.batch_size, 1, self.hps.reward_size))
        return action_init, reward_init, state_init

    def close_sess(self):
        self.sess.close()

    # bh: after training, encode an observation to hidden state
    def encode(self, x, a_prev, r_prev, domain_index, state_prev=None, seq_len=1):
        if self.step_est == 2:
            return self.sess.run(self.z, feed_dict={self.input_x: x})
        else:
            seq_len = np.int32(seq_len)
            if state_prev is None:
                state_prev = self.sess.run(self.initial_state)
            cwm_vae_feed = {self.input_x: x,
                            self.input_a_prev: a_prev,
                            self.input_r_prev: r_prev,
                            self.initial_state: state_prev,
                            self.input_domain_index: domain_index,
                            self.seq_length: seq_len}
            (z, final_state) = self.sess.run([self.z, self.final_state], feed_dict=cwm_vae_feed)
            return z, final_state

    def encode_new(self, x, a_prev, r_prev, domain_index, state_prev=None, seq_len=1):
        seq_len = np.int32(seq_len)
        if state_prev is None:
            state_prev = self.sess.run(self.initial_state)
        cwm_vae_feed = {self.input_x: x,
                        self.input_a_prev: a_prev,
                        self.input_r_prev: r_prev,
                        self.initial_state: state_prev,
                        self.input_domain_index: domain_index,
                        self.seq_length: seq_len}
        (z_map, final_state) = self.sess.run([self.z_map, self.final_state], feed_dict=cwm_vae_feed)
        return z_map, final_state

    def encode_mu_logvar(self, x, a_prev, r_prev, domain_index, seq_len=1):
        if self.step_est == 2:
            (mu, logvar) = self.sess.run([self.mu, self.logvar], feed_dict={self.input_x: x})
            return mu, logvar
        else:
            seq_len = np.int32(seq_len)
            cwm_vae_feed = {self.input_x: x,
                            self.input_a_prev: a_prev,
                            self.input_r_prev: r_prev,
                            self.input_domain_index: domain_index,
                            self.seq_length: seq_len}
            (logmix, mu, logstd) = self.sess.run([self.out_logmix, self.out_mean, self.out_logstd],
                                                 feed_dict=cwm_vae_feed)
            return logmix, mu, logstd

    def decode(self, z, domain_index, seq_len=1):
        if self.step_est:
            return self.sess.run(self.y_o, feed_dict={self.z: z})
        else:
            return self.sess.run(self.y_o, feed_dict={self.z: z, self.input_domain_index: domain_index,
                                                      self.seq_length: seq_len})

    def decode_new(self, z_map, domain_index, seq_len=1):
        return self.sess.run(self.y_o, feed_dict={self.z: z_map, self.input_domain_index: domain_index,
                                                  self.seq_length: seq_len})

    # predict next observation
    def predict(self, z, domain_index, seq_len=1):
        seq_len = np.int32(seq_len)
        return self.sess.run(self.y_o_next, feed_dict={self.z: z, self.input_domain_index: domain_index,
                                                       self.seq_length: seq_len})

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p * 10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def tf_lognormal(self, y, mean, logstd):
        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

    def get_lossfunc(self, logmix, mean, logstd, y):
        v = logmix + self.tf_lognormal(y, mean, logstd)
        v = tf.reduce_logsumexp(v, 1, keepdims=True)
        return -tf.reduce_mean(v)

    def get_mdn_coef(self, output):
        logmix, mean, logstd = tf.split(output, 3, 1)
        logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
        return logmix, mean, logstd

    def kl_gmm(self, logmix, mu, logvar):
        kl_loss = - 0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar))
        kl_loss = tf.maximum(kl_loss, self.hps.kl_tolerance * self.hps.z_size)
        kl_loss = tf.multiply(tf.exp(logmix), tf.exp(-kl_loss))
        return kl_loss

    def weight_normalization(self):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                p = self.sess.run(var)
                weightnorm_op, pl = self.weightnorm_ops[var]
                self.sess.run(weightnorm_op, feed_dict={pl.name: p})

    def deconv_weightnorm(self, weight):
        if len(weight.get_shape().as_list()) == 2:
            weight = tf.nn.l2_normalize(weight, [0])
        elif len(weight.get_shape().as_list()) == 4:
            weight = tf.nn.l2_normalize(weight, [0, 1, 3])
        return weight

    def conv_weightnorm(self, weight):
        if len(weight.get_shape().as_list()) == 2:
            weight = tf.nn.l2_normalize(weight, [0])
        elif len(weight.get_shape().as_list()) == 4:
            weight = tf.nn.l2_normalize(weight, [0, 1, 2])
        return weight

    def get_random_model_params(self, stdev=0.5):
        # get random params.
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
            rparam.append(np.random.standard_cauchy(s) * stdev)  # spice things up
        return rparam

    def set_model_params(self, params, is_dyn=False, is_testing=False):
        if is_testing:
            with self.g.as_default():
                t_vars = tf.trainable_variables()
                idx = 0
                for var in t_vars:
                    if var.name.startswith('SSL/A') or \
                            var.name.startswith('SSL/B') or \
                            var.name.startswith('SSL/C') or \
                            var.name.startswith('SSL/D') or \
                            var.name.startswith('SSL/E') or \
                            var.name.startswith('SSL/F') or \
                            var.name.startswith('Encoder') or \
                            var.name.startswith('ObsDecoder') or \
                            var.name.startswith('NextObsDecoder') or \
                            var.name.startswith('RewardDecoder') or \
                            var.name.startswith('Dynamics'):
                        pshape = tuple(var.get_shape().as_list())
                        p = np.array(params[idx])
                        assert pshape == p.shape, "inconsistent shape"
                        assign_op, pl = self.assign_ops[var]
                        self.sess.run(assign_op, feed_dict={pl.name: p / 10000.})
                    idx += 1
        else:
            if self.step_est == 0:
                with self.g.as_default():
                    t_vars = tf.trainable_variables()
                    idx = 0
                    if is_dyn:
                        for var in t_vars:
                            if var.name.startswith('SSL/D') or \
                                    var.name.startswith('SSL/E') or var.name.startswith('Dynamics'):
                                pshape = tuple(var.get_shape().as_list())
                                p = np.array(params[idx])
                                assert pshape == p.shape, "inconsistent shape"
                                assign_op, pl = self.assign_ops[var]
                                self.sess.run(assign_op, feed_dict={pl.name: p / 10000.})
                            idx += 1
                    else:
                        for var in t_vars:
                            if var.name.startswith('SSL/A') or \
                                    var.name.startswith('SSL/B') or \
                                    var.name.startswith('SSL/C') or \
                                    var.name.startswith('SSL/D') or \
                                    var.name.startswith('SSL/E') or \
                                    var.name.startswith('theta') or \
                                    var.name.startswith('Encoder') or \
                                    var.name.startswith('ObsDecoder') or \
                                    var.name.startswith('NextObsDecoder') or \
                                    var.name.startswith('RewardDecoder'):
                                pshape = tuple(var.get_shape().as_list())
                                p = np.array(params[idx])
                                assert pshape == p.shape, "inconsistent shape"
                                assign_op, pl = self.assign_ops[var]
                                self.sess.run(assign_op, feed_dict={pl.name: p / 10000.})
                            idx += 1
            if self.step_est == 2:
                with self.g.as_default():
                    t_vars = tf.trainable_variables()
                    idx = 0
                    for var in t_vars:
                        # if var.name.startswith('conv_vae'):
                        pshape = tuple(var.get_shape().as_list())
                        p = np.array(params[idx])
                        assert pshape == p.shape, "inconsistent shape"
                        assign_op, pl = self.assign_ops[var]
                        self.sess.run(assign_op, feed_dict={pl.name: p / 10000.})
                        idx += 1
            if self.step_est == 3:
                with self.g.as_default():
                    t_vars = tf.trainable_variables()
                    idx = 0
                    if is_dyn:
                        for var in t_vars:
                            if var.name.startswith('SSL/D') or \
                                    var.name.startswith('SSL/E') or \
                                    var.name.startswith('SSL/F') or \
                                    var.name.startswith('theta/theta_s') or \
                                    var.name.startswith('Dynamics'):
                                pshape = tuple(var.get_shape().as_list())
                                p = np.array(params[idx])
                                assert pshape == p.shape, "inconsistent shape"
                                assign_op, pl = self.assign_ops[var]
                                self.sess.run(assign_op, feed_dict={pl.name: p / 10000.})
                            idx += 1
                    else:
                        for var in t_vars:
                            if var.name.startswith('SSL/A') or \
                                    var.name.startswith('SSL/B') or \
                                    var.name.startswith('SSL/C') or \
                                    var.name.startswith('SSL/D') or \
                                    var.name.startswith('SSL/E') or \
                                    var.name.startswith('theta') or \
                                    var.name.startswith('Encoder') or \
                                    var.name.startswith('ObsDecoder') or \
                                    var.name.startswith('NextObsDecoder') or \
                                    var.name.startswith('RewardDecoder'):
                                pshape = tuple(var.get_shape().as_list())
                                p = np.array(params[idx])
                                assert pshape == p.shape, "inconsistent shape"
                                assign_op, pl = self.assign_ops[var]
                                self.sess.run(assign_op, feed_dict={pl.name: p / 10000.})
                            idx += 1

    def load_json(self, jsonfile='vae.json', is_dyn=False, is_tesing=False):
        with open(jsonfile, 'r') as f:
            params = json.load(f)
        self.set_model_params(params, is_dyn, is_tesing)

    def save_json(self, jsonfile='vae.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

    def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)

    def save_model(self, model_save_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(model_save_path, 'vae')
        tf.logging.info('saving model %s.', checkpoint_path)
        saver.save(sess, checkpoint_path, 0)  # just keep one
