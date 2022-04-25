import os
import cv2
from datetime import datetime
import argparse
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import random

import tensorflow as tf

# from xvfbwrapper import Xvfb

from models.missVAE import missVAE
from models.dqn_pol import DoubleDQNAgent
from utils.policy.extract import extract_theta, encode_obs
from utils.policy.min_n_suff_set import min_n_suff_set_state, min_n_suff_set_theta
from utils.misc.data_handler import DataHandler
from utils.misc.hyper_params import default_hps
from utils.misc.env_init import env_init


def infer_theta(n_epoch, n_domain, hps, model, dh, save_p):
    N_data = 1000 * n_domain
    N_batches = int(np.floor(N_data / hps.batch_size))
    sign = 1
    for epoch in range(n_epoch):
        for idx in range(N_batches):
            step = model.sess.run(model.global_step)
            curr_learning_rate = \
                (hps.learning_rate - hps.min_learning_rate) * hps.decay_rate ** step + hps.min_learning_rate
            batch_obs, batch_action, batch_reward, batch_domain_index = dh.next_batch()
            action_init = np.zeros((hps.batch_size, 1, hps.action_size))
            batch_action_prev = np.concatenate((action_init, batch_action[:, :-1, :]), axis=1)
            reward_init = np.zeros((hps.batch_size, 1, hps.reward_size))
            batch_reward_prev = np.concatenate((reward_init, batch_reward[:, :-1, :]), axis=1)

            batch_obs = np.reshape(batch_obs, (hps.batch_size, hps.max_seq_len, 128, 128, 1))
            batch_domain_index = np.reshape(batch_domain_index, [hps.batch_size])
            feed = {model.input_x: batch_obs,
                    model.input_a_prev: batch_action_prev,
                    model.input_a: batch_action,
                    model.input_r_prev: batch_reward_prev,
                    model.input_r: batch_reward,
                    model.input_domain_index: batch_domain_index,
                    model.lr: curr_learning_rate,
                    model.seq_length: hps.max_seq_len,
                    model.input_sign: sign}
            (total_loss,
             vae_loss, vae_r_obs_loss, vae_r_next_obs_loss, vae_r_reward_loss, vae_kl_loss, vae_causal_filter_loss,
             transition_loss, causal_filter_loss, state, _) \
                = model.sess.run([model.total_loss,
                                  model.vae_loss,
                                  model.r_obs_loss,
                                  model.r_next_obs_loss,
                                  model.r_reward_loss,
                                  model.kl_loss,
                                  model.vae_causal_filter_loss,
                                  model.transition_loss,
                                  model.causal_filter_loss,
                                  model.final_state,
                                  model.train_op], feed)
            if vae_loss < 50:
                sign = 0
            if step % 100 == 0:
                SSL_A, SSL_B, SSL_C, SSL_D, SSL_E, SSL_F, theta_o, theta_s, theta_r = model.sess.run(
                    [model.SSL_A, model.SSL_B, model.SSL_C,
                     model.SSL_D, model.SSL_E, model.SSL_F,
                     model.theta_o, model.theta_s, model.theta_r])
                print("theta_o:", theta_o)
                print("theta_r:", theta_r)
                print("theta_s:", theta_s)
            output_log = "step: %d (Epoch: %d idx: %d), " \
                         "total_loss: %.4f, " \
                         "vae_loss: %.4f, " \
                         "vae_r_obs_loss: %.4f, " \
                         "vae_r_next_obs_loss: %.4f, " \
                         "vae_r_reward_loss: %.4f, " \
                         "vae_kl_loss: %.4f, " \
                         "vae_causal_filter_loss: %.4f, " \
                         "transition_loss: %.4f, " \
                         "causal_filter_loss: %.4f," \
                         % (step,
                            epoch,
                            idx,
                            total_loss,
                            vae_loss, vae_r_obs_loss, vae_r_next_obs_loss,
                            vae_r_reward_loss, vae_kl_loss, vae_causal_filter_loss,
                            transition_loss,
                            causal_filter_loss)
            print(output_log)
            f = open(os.path.join(save_p, 'output.txt'), 'a')
            f.write(output_log + '\n')
            f.close()

            save_p_e = os.path.join(save_p, 'epochs')
            if not os.path.exists(save_p_e):
                os.makedirs(save_p_e)

            if step % 500 == 0:
                model.save_json(os.path.join(save_p_e, str(epoch) + '_' + str(step) + 'test.json'))

        model.save_json(os.path.join(save_p, 'test.json'))


def predict(env, agent, model, theta_all, k):
    score = 0
    # initialization
    state_ori = env[k].reset()  # the ground-truth state
    # generate observational image
    obs = env[k].render(mode='rgb_array')
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (128, 128), interpolation=cv2.INTER_CUBIC)
    obs = cv2.normalize(obs, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    a_prev, r_prev, c_prev = mvae.reset()
    state, c = encode_obs(hps, model, obs, a_prev, r_prev, c_prev, k)
    state = np.reshape(state, [1, hps.z_size])

    while 1:
        # get action for the current observation and go one step in environment
        action = agent.get_action(state, theta_all[k])
        next_state_ori, reward, done, info = env[k].step(action)
        next_obs = env[k].render(mode='rgb_array')
        next_obs = cv2.cvtColor(next_obs, cv2.COLOR_RGB2GRAY)
        next_obs = cv2.resize(next_obs, (128, 128), interpolation=cv2.INTER_CUBIC)
        next_obs = cv2.normalize(next_obs, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # infer the states
        next_state, next_c = encode_obs(hps, model, next_obs, action, reward, c, k)
        next_state = np.reshape(next_state, [1, hps.z_size])
        state = next_state
        c = next_c

        score += 1
        if done:
            break
    return score


def testing(game, src_domain_index, hps, model, reduction_set_s, reduction_set_theta, k):
    # gpu setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    gpu_config = tf.ConfigProto(device_count={'GPU': 0},
                                allow_soft_placement=False,
                                log_device_placement=False)
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        scores = []
        env = env_init(game, src_domain_index)

        tf.set_random_seed(int(100))
        theta_all = extract_theta(hps, model)

        for i in range(30):
            # set the random seed
            random.seed(int(i * 100))
            np.random.seed(int(i * 100))
            for j in range(n_domain):
                env[j].seed(int(i * 100))

            state_size = hps.z_size
            action_size = env[0].action_space.n
            theta_size = hps.theta_s_size + hps.theta_r_size

            if reduction_set_theta == None:
                agent = DoubleDQNAgent(len(reduction_set_s[0]), action_size, theta_size,
                                       is_load=True, load_p=policy_save_path)
            else:
                agent = DoubleDQNAgent(len(reduction_set_s[0]), action_size,
                                       len(reduction_set_theta) + hps.theta_r_size,
                                       is_load=True, load_p=policy_save_path)
            score = predict(env, agent, model, theta_all, k)
            scores.append(score)
        print(scores)


parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, required=True, help='data path')
parser.add_argument('-source', type=str, required=True, help='data path')
parser.add_argument('-dest', type=str, required=True, help='data path')
parser.add_argument('-domain', type=str, nargs='+', required=True, help='full domain index')
parser.add_argument('-mvae_p', type=str, required=True, help='data path')
parser.add_argument('-k', type=int, required=False, help='test domain index')
parser.add_argument('-step', type=int, required=True, help='test domain index')

args = parser.parse_args()

game = args.name
source_p = args.source
dest_p = args.dest
if not os.path.exists(dest_p):
    os.makedirs(dest_p)
k = args.k
step = args.step
src_domain_index = args.domain
n_domain = len(src_domain_index)
mvae_p = args.mvae_p

date_format = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
time_now = datetime.now().strftime(date_format)
data_indicator = args.source.split('/', 2)[-1]

model_save_path = './results/' + data_indicator + '/test/' + time_now
policy_save_path = './results/' + data_indicator + '/policy_opt/' + time_now + '/policy.h5'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

n_epoch = 1000
hps = default_hps(game, 'test2', n_domain)
mvae = missVAE(hps, step=3, scope='all')
mvae.load_json(os.path.join(mvae_p), is_tesing=True)
dh = DataHandler(hps, 3, source_p, dest_p, src_domain_index, 1000 * n_domain, is_test=True)

(theta_s, theta_r,
 SSL_A, SSL_B, SSL_C,
 SSL_D, SSL_E, SSL_F) = mvae.sess.run([mvae.theta_s, mvae.theta_r,
                                       tf.math.erf(mvae.SSL_A), tf.math.erf(mvae.SSL_B), tf.math.erf(mvae.SSL_C),
                                       tf.math.erf(mvae.SSL_D), tf.math.erf(mvae.SSL_E), tf.math.erf(mvae.SSL_F)])
# minimal & sufficient set identification
reduction_set_s = min_n_suff_set_state(SSL_B, SSL_D)
reduction_set_theta = min_n_suff_set_theta(reduction_set_s, SSL_F)

# Step 1: inference theta
if step == 1:
    infer_theta(n_epoch, n_domain, hps, mvae, dh, model_save_path)
elif step == 2:
    # Step 2: testing
    mvae_pretrain_p = os.path.join(model_save_path, 'test.json')
    mvae.load_json(mvae_pretrain_p, is_dyn=False)
    SCREEN_X = 128
    SCREEN_Y = 128
    # vdisplay = Xvfb(width=SCREEN_X, height=SCREEN_Y)
    # vdisplay.start()
    testing(game, src_domain_index, hps, mvae, reduction_set_s, reduction_set_theta, k=0)
else:
    Exception('Wrong step number !')
