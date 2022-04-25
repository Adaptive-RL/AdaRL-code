import os
import random
import numpy as np
import argparse
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf

from models.dqn_pol import DoubleDQNAgent
from models.missVAE import missVAE
from utils.policy.min_n_suff_set import min_n_suff_set_state, min_n_suff_set_theta
from utils.policy.extract import extract_theta
from utils.policy.train import train
from utils.misc.hyper_params import default_hps
from utils.misc.env_init import env_init

SCREEN_X = 128
SCREEN_Y = 128
EPISODES = 500

parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, required=True, help='data path')
parser.add_argument('-mvae_p', type=str, required=True, help='data path')
parser.add_argument('-source', type=str, required=True, help='data path')
parser.add_argument('-domain', type=str, nargs='+', required=True, help='full domain index')
args = parser.parse_args()

game = args.name
src_domain_index = args.domain
n_domain = len(src_domain_index)
mvae_p = args.mvae_p

date_format = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
time_now = datetime.now().strftime(date_format)

data_indicator = args.source.split('/', 2)[-1]
save_p = './results/' + data_indicator + '/policy_opt/' + time_now
if not os.path.exists(save_p):
    os.makedirs(save_p)

hps = default_hps(game=game, step='pol_opt')
hps_sample = hps._replace(batch_size=1, is_training=0)
mvae = missVAE(hps_sample, step=3, scope='all', gpu_mode=True, reuse=True)
mvae.load_json(os.path.join(mvae_p))

(theta_s, theta_r,
 SSL_A, SSL_B, SSL_C,
 SSL_D, SSL_E, SSL_F) = mvae.sess.run([mvae.theta_s, mvae.theta_r,
                                       tf.math.erf(mvae.SSL_A), tf.math.erf(mvae.SSL_B), tf.math.erf(mvae.SSL_C),
                                       tf.math.erf(mvae.SSL_D), tf.math.erf(mvae.SSL_E), tf.math.erf(mvae.SSL_F)])
# minimal & sufficient set identification
reduction_set_s = min_n_suff_set_state(SSL_B, SSL_D)
reduction_set_theta = min_n_suff_set_theta(reduction_set_s, SSL_F)

# gpu setup
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpu_config = tf.ConfigProto(device_count={'GPU': 0},
                            allow_soft_placement=False,
                            log_device_placement=False)
gpu_config.gpu_options.allow_growth = True

with tf.Session(config=gpu_config) as sess:
    env = env_init(game, src_domain_index)

    random.seed(int(1000))
    np.random.seed(int(1000))
    tf.set_random_seed(int(1000))
    for k in range(5):
        env[k].seed(int(1000))

    state_size = hps.z_size
    action_size = env[0].action_space.n
    theta_size = hps.theta_s_size + hps.theta_r_size
    theta_threshold = env[0].theta_threshold
    x_threshold = env[0].x_threshold

    if reduction_set_theta == None:
        agent = DoubleDQNAgent(len(reduction_set_s[0]), action_size, theta_size)
    else:
        agent = DoubleDQNAgent(len(reduction_set_s[0]), action_size, len(reduction_set_theta) + hps.theta_r_size)

    theta_all = extract_theta(hps, mvae)

    if reduction_set_theta != None:
        theta_all = theta_all[reduction_set_theta]
        for i in range(len(theta_all)):
            tmp = np.column_stack(
                (theta_all[i][:hps.theta_s_size][reduction_set_theta], theta_all[i][hps.theta_s_size:]))

    train(env, agent, mvae, hps, theta_all, reduction_set_s, EPISODES, n_domain, save_p)
