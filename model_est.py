import os
import argparse
import warnings

warnings.filterwarnings('ignore')
import time
from datetime import datetime
import random
import numpy as np

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from utils.misc.hyper_params import default_hps
from utils.misc.data_handler import DataHandler
from utils.misc.data_handler import load_raw_data_list, encode_batch, next_batch
from models.missVAE import missVAE


def model_est_series(hps, model, N_batches, obs_dataset, action_dataset, reward_dataset, domain_dataset, save_p):
    logmix_dataset = []
    mu_dataset = []
    logvar_dataset = []
    domain_dataset2 = []

    dataset_range = np.arange(N_batches * hps.batch_size)
    start_time = time.time()
    for i in range(N_batches):
        batch_obs, batch_action, batch_reward, batch_domain = next_batch(hps,
                                                                         obs_dataset, action_dataset, reward_dataset,
                                                                         domain_dataset,
                                                                         i, dataset_range)
        action_init = np.zeros((hps.batch_size, 1, hps.action_size))
        batch_action_prev = np.concatenate((action_init, batch_action[:, :-1, :]), axis=1)
        reward_init = np.zeros((hps.batch_size, 1, hps.reward_size))
        batch_reward_prev = np.concatenate((reward_init, batch_reward[:, :-1, :]), axis=1)

        batch_logmix, batch_mu, batch_logvar = \
            encode_batch(hps, model, batch_obs, batch_action_prev, batch_reward_prev, batch_domain, hps.max_seq_len)
        logmix_dataset.append(batch_logmix.astype(np.float16))
        mu_dataset.append(batch_mu.astype(np.float16))
        logvar_dataset.append(batch_logvar.astype(np.float16))
        domain_dataset2.append(batch_domain.astype(np.int32))  # N_batches x batch_size

    logmix_dataset = np.reshape(np.array(logmix_dataset), (-1, hps.max_seq_len, hps.z_size, hps.num_mixture))
    mu_dataset = np.reshape(np.array(mu_dataset), (-1, hps.max_seq_len, hps.z_size, hps.num_mixture))
    logvar_dataset = np.reshape(np.array(logvar_dataset), (-1, hps.max_seq_len, hps.z_size, hps.num_mixture))
    domain_dataset2 = np.reshape(np.array(domain_dataset2), (-1))  # (N_batches x batch_size)

    # extend it
    domain_dataset3 = []
    for i in range(N_batches * hps.batch_size):
        d0 = np.stack([domain_dataset2[i]] * (hps.max_seq_len - 1))
        domain_dataset3.append(d0)
    domain_dataset3 = np.reshape(domain_dataset3, [-1])

    at = np.reshape(action_dataset[:, :-1, :], (-1, hps.action_size))
    st_logmix = np.reshape(logmix_dataset[:, :-1, :, :], (-1, hps.z_size, hps.num_mixture))
    st_mu = np.reshape(mu_dataset[:, :-1, :, :], (-1, hps.z_size, hps.num_mixture))
    st_logvar = np.reshape(logvar_dataset[:, :-1, :, :], (-1, hps.z_size, hps.num_mixture))

    st1_logmix = np.reshape(logmix_dataset[:, 1:, :, :], (-1, hps.z_size, hps.num_mixture))
    st1_mu = np.reshape(mu_dataset[:, 1:, :, :], (-1, hps.z_size, hps.num_mixture))
    st1_logvar = np.reshape(logvar_dataset[:, 1:, :, :], (-1, hps.z_size, hps.num_mixture))

    np.savez_compressed(os.path.join(save_p, 'series.npz'),
                        action=at,
                        domain=domain_dataset3,
                        st_logmix=st_logmix, st_mu=st_mu, st_logvar=st_logvar,
                        st1_logmix=st1_logmix, st1_mu=st1_mu, st1_logvar=st1_logvar)

    time_taken = time.time() - start_time
    print("time taken on series: %.4f" % time_taken)


parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, required=True, help='data path')
parser.add_argument('-source', type=str, required=True, help='data path')
parser.add_argument('-dest', type=str, required=True, help='data path')
parser.add_argument('-domain', type=str, nargs='+', required=True, help='full domain index')
args = parser.parse_args()

game = args.name

source_p = args.source
dest_p = args.dest
src_domain_index = args.domain
# Parameters for training
NUM_EPOCH = 1000

date_format = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
time_now = datetime.now().strftime(date_format)

data_indicator = args.source.split('/', 2)[-1]

if not os.path.exists(dest_p):
    os.makedirs(dest_p)

# there 4 steps for model estimation,
# i.e. estimate the VAE components, the series components, the dynamic components, and all components together
N_datas = [10000 * 5, None, 10000 * 5, 10000 * 5]
model_est_steps = ['vae', 'series', 'dynamic', 'all']
for m_step in range(4):
    model_save_path = './results/' + data_indicator + '/' + model_est_steps[m_step] + '/' + time_now
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if m_step != 1:
        model_save_epochs_path = './results/' + data_indicator + '/' + model_est_steps[
            m_step] + '/' + time_now + '/epochs'
        if not os.path.exists(model_save_epochs_path):
            os.makedirs(model_save_epochs_path)
    if m_step == 0:
        structure_save_steps_path = './results/' + data_indicator + '/' + model_est_steps[m_step] + '/' + time_now + \
                                    '/structure_steps'
        if not os.path.exists(structure_save_steps_path):
            os.makedirs(structure_save_steps_path)

    hps = default_hps(game, m_step)
    N_data = N_datas[m_step]
    if N_data == None:
        filelist = os.listdir(dest_p)
        random.shuffle(filelist)
        filelist.sort()

        obs_dataset, action_dataset, reward_dataset, domain_dataset = load_raw_data_list(dest_p, filelist)
        N_data = len(obs_dataset)
    N_batches = int(np.floor(N_data / hps.batch_size))

    mvae = missVAE(hps, m_step, model_est_steps[m_step])
    if m_step == 3:
        vae_p = './results/' + data_indicator + '/' + model_est_steps[0] + '/' + time_now + '/vae.json'
        mvae.load_json(os.path.join(vae_p))
        dyn_p = './results/' + data_indicator + '/' + model_est_steps[2] + '/' + time_now + '/dynamic.json'
        mvae.load_json(os.path.join(dyn_p), is_dyn=True)
    if m_step == 1:
        vae_p = './results/' + data_indicator + '/' + model_est_steps[0] + '/' + time_now + '/vae.json'
        mvae.load_json(os.path.join(vae_p), is_dyn=True)
        model_est_series(hps, mvae, N_batches, obs_dataset, action_dataset, reward_dataset, domain_dataset,
                         model_save_path)
    else:
        print('sodifjsadfa')

        if m_step == 2:
            dest_p = './results/' + data_indicator + '/' + model_est_steps[1] + '/' + time_now
        dh = DataHandler(hps, m_step, source_p, dest_p, src_domain_index, N_data)

        curr_learning_rate = 1
        sign = 1  # 0: not adding random noise to gradient； 1: add

        ##################################### Training #####################################
        for epoch in range(NUM_EPOCH):
            for idx in range(N_batches):
                step = mvae.sess.run(mvae.global_step)
                curr_learning_rate = \
                    (hps.learning_rate - hps.min_learning_rate) * hps.decay_rate ** step + hps.min_learning_rate

                if m_step == 0 or m_step == 3:
                    # note that we also should get domain index
                    batch_obs, batch_action, batch_reward, batch_domain_index = dh.next_batch()

                    action_init = np.zeros((hps.batch_size, 1, hps.action_size))
                    batch_action_prev = np.concatenate((action_init, batch_action[:, :-1, :]), axis=1)
                    reward_init = np.zeros((hps.batch_size, 1, hps.reward_size))
                    batch_reward_prev = np.concatenate((reward_init, batch_reward[:, :-1, :]), axis=1)

                    batch_obs = np.reshape(batch_obs, (hps.batch_size, hps.max_seq_len, 128, 128, 1))
                    batch_domain_index = np.reshape(batch_domain_index, [hps.batch_size])
                    if m_step == 0:
                        vae_feed = {mvae.input_x: batch_obs,
                                    mvae.input_a_prev: batch_action_prev,
                                    mvae.input_a: batch_action,
                                    mvae.input_r_prev: batch_reward_prev,
                                    mvae.input_r: batch_reward,
                                    mvae.input_domain_index: batch_domain_index,
                                    mvae.vae_lr: curr_learning_rate,
                                    mvae.seq_length: hps.max_seq_len,
                                    mvae.input_sign: sign}

                        (z, out_logmix, out_mean, out_logstd, SSL_A, SSL_B, SSL_C, theta_o, theta_s, theta_r,
                         vae_loss, vae_r_obs_loss, vae_r_next_obs_loss, vae_r_reward_loss, vae_kl_loss, _) \
                            = mvae.sess.run([mvae.z,
                                             mvae.out_logmix,
                                             mvae.out_mean,
                                             mvae.out_logstd,
                                             mvae.SSL_A,
                                             mvae.SSL_B,
                                             mvae.SSL_C,
                                             mvae.theta_o,
                                             mvae.theta_s,
                                             mvae.theta_r,
                                             mvae.vae_loss,
                                             mvae.r_obs_loss,
                                             mvae.r_next_obs_loss,
                                             mvae.r_reward_loss,
                                             mvae.kl_loss,
                                             mvae.vae_train_op], vae_feed)

                        if vae_loss < 50:
                            sign = 0

                        output_log = "Step %d (Epoch: %d idx: %d), " \
                                     "vae_loss: %.4f, " \
                                     "vae_r_obs_loss: %.4f, " \
                                     "vae_r_next_obs_loss: %.4f, " \
                                     "vae_r_reward_loss: %.4f," \
                                     "vae_kl_loss: %.4f" \
                                     "learning_rate: %.6f" \
                                     "sign: %d" \
                                     % (step,
                                        epoch,
                                        idx,
                                        vae_loss,
                                        vae_r_obs_loss,
                                        vae_r_next_obs_loss,
                                        vae_r_reward_loss,
                                        vae_kl_loss,
                                        curr_learning_rate,
                                        sign)
                        print(output_log)

                        if step % 100 == 0:
                            SSL_A, SSL_B, SSL_C, theta_o, theta_r, theta_s = mvae.sess.run(
                                [mvae.SSL_A, mvae.SSL_B, mvae.SSL_C, mvae.theta_o, mvae.theta_r, mvae.theta_s])
                            print("SSL_A:", SSL_A)
                            print("SSL_B:", SSL_B)
                            print("SSL_C:", SSL_C)
                            print("theta_o:", theta_o)
                            print("theta_r:", theta_r)
                            print("theta_s:", theta_s)

                            tot_param = {
                                'z': np.array(z),
                                'out_logmix': np.array(out_logmix),
                                'out_mean': np.array(out_mean),
                                'out_logstd': np.array(out_logstd),
                                'SSL_A': np.array(SSL_A),
                                'SSL_B': np.array(SSL_B),
                                'SSL_C': np.array(SSL_C),
                                'theta_o': np.array(theta_o),
                                'theta_s': np.array(theta_s),
                                'theta_r': np.array(theta_r)
                            }

                            np.savez(structure_save_steps_path + '/' + str(step),
                                     z=np.array(z),
                                     out_logmix=np.array(out_logmix),
                                     out_mean=np.array(out_mean),
                                     out_logstd=np.array(out_logstd),
                                     SSL_A=np.array(SSL_A),
                                     SSL_B=np.array(SSL_B),
                                     SSL_C=np.array(SSL_C),
                                     theta_o=np.array(theta_o),
                                     theta_s=np.array(theta_s),
                                     theta_r=np.array(theta_r))
                        if step % 1000 == 0:
                            mvae.save_json(
                                os.path.join(model_save_epochs_path, str(step) + model_est_steps[m_step] + '.json'))
                    else:
                        feed = {mvae.input_x: batch_obs,
                                mvae.input_a_prev: batch_action_prev,
                                mvae.input_a: batch_action,
                                mvae.input_r_prev: batch_reward_prev,
                                mvae.input_r: batch_reward,
                                mvae.input_domain_index: batch_domain_index,
                                mvae.lr: curr_learning_rate,
                                mvae.seq_length: hps.max_seq_len,
                                mvae.input_sign: sign}
                        (total_loss,
                         vae_loss, vae_r_obs_loss, vae_r_next_obs_loss, vae_r_reward_loss, vae_kl_loss,
                         vae_causal_filter_loss,
                         transition_loss, causal_filter_loss, state, _) \
                            = mvae.sess.run([mvae.total_loss,
                                             mvae.vae_loss,
                                             mvae.r_obs_loss,
                                             mvae.r_next_obs_loss,
                                             mvae.r_reward_loss,
                                             mvae.kl_loss,
                                             mvae.vae_causal_filter_loss,
                                             mvae.transition_loss,
                                             mvae.causal_filter_loss,
                                             mvae.final_state,
                                             mvae.train_op], feed)

                        if vae_loss < 50:
                            sign = 0
                        if step % 500 == 0:
                            mvae.save_json(
                                os.path.join(model_save_epochs_path, str(step) + model_est_steps[m_step] + '.json'))
                        if step % 100 == 0:
                            SSL_A, SSL_B, SSL_C, SSL_D, SSL_E, SSL_F, theta_o, theta_s, theta_r = mvae.sess.run(
                                [mvae.SSL_A, mvae.SSL_B,
                                 mvae.SSL_C, mvae.SSL_D,
                                 mvae.SSL_E, mvae.SSL_F,
                                 mvae.theta_o,
                                 mvae.theta_s,
                                 mvae.theta_r])
                            print("SSL_A:", SSL_A)
                            print("SSL_B:", SSL_B)
                            print("SSL_C:", SSL_C)
                            print("SSL_D:", SSL_D)
                            print("SSL_E:", SSL_E)
                            print("SSL_F:", SSL_F)
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

                elif m_step == 2:
                    batch_st, batch_at, batch_st1, batch_domain = dh.next_batch()

                    feed = {mvae.input_a: batch_at,
                            mvae.input_z: batch_st,
                            mvae.output_z: batch_st1,
                            mvae.input_domain_index: batch_domain,
                            mvae.lr: curr_learning_rate}
                    (total_loss, transition_loss, causal_filter_loss, _) \
                        = mvae.sess.run([mvae.total_loss,
                                         mvae.transition_loss,
                                         mvae.causal_filter_loss,
                                         mvae.train_op], feed)

                    output_log = "step: %d (Epoch: %d idx: %d), " \
                                 "total_loss: %.4f, " \
                                 "transition_loss: %.4f, " \
                                 "causal_filter_loss: %.4f," \
                                 % (step,
                                    epoch,
                                    idx,
                                    total_loss,
                                    transition_loss,
                                    causal_filter_loss)
                    print(output_log)

                    if step % 500 == 0:
                        SSL_D, SSL_E, SSL_F, theta_s = mvae.sess.run([mvae.SSL_D, mvae.SSL_E, mvae.SSL_F, mvae.theta_s])
                        print("SSL_D:", SSL_D)
                        print("SSL_E:", SSL_E)
                        print("SSL_F:", SSL_F)
                        print("theta_s:", theta_s)

                    if step % 500 == 0:
                        mvae.save_json(
                            os.path.join(model_save_epochs_path, str(step) + model_est_steps[m_step] + '.json'))
                f = open(model_save_path + '/output.txt', 'a')
                f.write(output_log + '\n')
                f.close()

            mvae.save_json(os.path.join(model_save_path, model_est_steps[m_step] + '.json'))