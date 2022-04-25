import numpy as np


def encode_obs(hps, model, obs, a_prev, r_prev, state_prev, domain_index):
    obs = np.copy(obs).astype(np.float)
    obs = obs.reshape(1, 1, 128, 128, 1)
    a_prev = np.reshape(a_prev, (1, 1, hps.action_size))
    r_prev = np.reshape(r_prev, (1, 1, hps.reward_size))

    domain_index = np.array([domain_index], dtype=np.int32)
    z, final_state = model.encode_new(obs, a_prev, r_prev, domain_index, state_prev)
    return z, final_state


def extract_theta(hps, model):
    theta_all = []
    (theta_s, theta_r) = model.sess.run([model.theta_s, model.theta_r])
    for domain_index in range(hps.domain_size):
        domain_index = np.array([domain_index], dtype=np.int32)
        theta_s_i = np.take(theta_s, domain_index)
        theta_r_i = np.take(theta_r, domain_index)
        theta_i = np.concatenate((theta_s_i, theta_r_i))
        theta_i = np.reshape(theta_i, [1, theta_i.shape[0]])
        theta_all.append(theta_i)
    return theta_all
