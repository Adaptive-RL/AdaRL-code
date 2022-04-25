import os
import numpy as np
import shutil

import tensorflow as tf


def load_raw_data_list(data_dir, filelist):
    obs_list = []
    action_list = []
    reward_list = []
    domain_list = []
    for i in range(len(filelist)):
        filename = filelist[i]
        raw_data = np.load(os.path.join(data_dir, filename))
        obs_list.append(raw_data['obs'])
        action_list.append(raw_data['action'])
        reward_list.append(raw_data['reward'])

        id1 = filename.rfind('_')
        id2 = filename.rfind('.')
        domain_index = int(filename[id1 + 1:id2])
        if domain_index == 11:
            domain_index = 0
        if domain_index == 12:
            domain_index = 1
        if domain_index == 14:
            domain_index = 2
        if domain_index == 16:
            domain_index = 3
        if domain_index == 18:
            domain_index = 4
        domain_list.append(domain_index)

        if (i + 1) % 1000 == 0:
            print("loading file", (i + 1))

    obs_list = np.array(obs_list)
    action_list = np.array(action_list)
    reward_list = np.array(reward_list)
    domain_list = np.array(domain_list)

    obs_list = np.reshape(obs_list, [obs_list.shape[0], obs_list.shape[1], obs_list.shape[2], obs_list.shape[3], 1])
    action_list = np.reshape(action_list, [action_list.shape[0], action_list.shape[1], 1])
    reward_list = np.reshape(reward_list, [reward_list.shape[0], reward_list.shape[1], 1])

    return obs_list, action_list, reward_list, domain_list


def next_batch(hps, obs_dataset, action_dataset, reward_dataset, domain_dataset, index, indices):
    batch_indices = indices[index * hps.batch_size:(index + 1) * hps.batch_size]

    obs = obs_dataset[batch_indices]
    obs = obs.astype(np.float)

    action = action_dataset[batch_indices]
    reward = reward_dataset[batch_indices]
    domain_index = domain_dataset[batch_indices]

    obs = np.reshape(obs, (hps.batch_size, hps.max_seq_len, 128, 128, 1))
    action = np.reshape(action, (hps.batch_size, hps.max_seq_len, hps.action_size))
    reward = np.reshape(reward, (hps.batch_size, hps.max_seq_len, hps.reward_size))
    return obs, action, reward, domain_index


def encode_batch(hps, vae, obs_current, action_prev, reward_prev, domain_index, seq_len):
    logmix, mu, logvar = vae.encode_mu_logvar(obs_current, action_prev, reward_prev, domain_index, seq_len)
    logmix = np.reshape(logmix, (hps.batch_size, hps.max_seq_len, hps.z_size, hps.num_mixture))
    mu = np.reshape(mu, (hps.batch_size, hps.max_seq_len, hps.z_size, hps.num_mixture))
    logvar = np.reshape(logvar, (hps.batch_size, hps.max_seq_len, hps.z_size, hps.num_mixture))
    return logmix, mu, logvar


class DataHandler(object):
    def __init__(self, hps, step_m, source_p, dest_p, src_domain_index, n_data, is_test=False, test_cnt=1000):
        self.hps = hps
        self.step_m = step_m
        self.sample_count = 0
        self.DATA_DIR = dest_p
        self.is_test = is_test
        self.test_cnt = test_cnt
        # bh: only use CPU to generate batch data
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu_config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=False, log_device_placement=False)
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)

        if self.step_m == 2:
            filelist = self.data_dyn(n_data)
        else:
            filelist = self.data_vae(source_p, dest_p, src_domain_index)

        dataset = tf.data.Dataset.from_tensor_slices(filelist)
        if self.step_m == 2:
            bs = n_data
        else:
            bs = 1000
        dataset = dataset.shuffle(buffer_size=bs)
        # bh: use 10 cpu to generate batch

        if self.step_m == 2:
            dataset = dataset.map(self.parse_function_dyn, num_parallel_calls=56)  # apply per-element transformations
        else:
            dataset = dataset.map(self.parse_function_vae, num_parallel_calls=10)
        dataset = dataset.batch(self.hps.batch_size)
        # bh: the number of preprocessed batches
        dataset = dataset.prefetch(10)

        iterator = dataset.make_initializable_iterator()
        self.init_op = iterator.initializer
        self.next_element = iterator.get_next()

        self.sess.run(self.init_op)

    def data_vae(self, source_p, dest_p, src_domain_index):
        for i in src_domain_index:
            source = source_p + '/v' + str(i)
            src_files = os.listdir(source)
            if self.is_test:
                src_files.sort()
                src_files = src_files[:self.test_cnt]
            for file_name in src_files:
                full_file_name = os.path.join(source, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, dest_p)  # copy
                    # rename a file
                    file_name_new = file_name[:-4] + '_' + str(i) + file_name[-4:]
                    os.rename(os.path.join(dest_p, file_name), os.path.join(dest_p, file_name_new))
        filelist = os.listdir(self.DATA_DIR)
        filelist.sort()

        return filelist

    def data_dyn(self, n_data):
        raw_data = np.load(os.path.join(self.DATA_DIR, "series.npz"))
        # load preprocessed data
        self.st_logmix = raw_data["st_logmix"]
        self.st_mu = raw_data["st_mu"]
        self.st_logvar = raw_data["st_logvar"]

        self.st1_logmix = raw_data["st1_logmix"]
        self.st1_mu = raw_data["st1_mu"]
        self.st1_logvar = raw_data["st1_logvar"]

        self.at = raw_data["action"]

        self.domain_index = raw_data["domain"]

        filelist = np.arange(n_data)
        return filelist

    # bh: need to change in the new project. preprocess the image...
    def parse_npz_vae(self, filename):
        raw_data = np.load(os.path.join(self.DATA_DIR, filename.decode("utf-8")))
        obs = np.array(raw_data['obs'])
        obs = obs.astype(np.float32)
        action = np.array(raw_data['action']).astype(np.float32)
        action = np.reshape(action, (self.hps.max_seq_len, self.hps.action_size))
        reward = np.array(raw_data['reward']).astype(np.float32)
        reward = np.reshape(reward, (self.hps.max_seq_len, self.hps.reward_size))
        name = filename.decode("utf-8")

        domain_index = name.split('_')[-1]
        domain_index = domain_index.split('.')[0]

        if self.is_test:
            if 'cartpole' in self.DATA_DIR.lower():
                if domain_index[-1] == '5':
                    domain_index = np.array([0], dtype=np.int32)
                elif domain_index[-1] == '6':
                    domain_index = np.array([1], dtype=np.int32)
                else:
                    raise Exception('wrong domain index')
            elif 'pong' in self.DATA_DIR.lower():
                if domain_index[0] == '0' or domain_index[0] == '2':
                    if domain_index[-1] == '5':
                        domain_index = np.array([0], dtype=np.int32)
                    elif domain_index[-1] == '6':
                        domain_index = np.array([1], dtype=np.int32)
                    else:
                        raise Exception('wrong domain index')
                elif domain_index[0] == '1':
                    if domain_index[-1] == '2':
                        domain_index = np.array([0], dtype=np.int32)
                    elif domain_index[-1] == '3':
                        domain_index = np.array([1], dtype=np.int32)
                    else:
                        raise Exception('wrong domain index')
                elif domain_index[0] == '3':
                    if domain_index[-1] == '3':
                        domain_index = np.array([0], dtype=np.int32)
                    elif domain_index[-1] == '4':
                        domain_index = np.array([1], dtype=np.int32)
                    else:
                        raise Exception('wrong domain index')
                else:
                    raise Exception('wrong domain index')
            else:
                raise Exception('Wrong game')
        else:
            domain_index = np.array([int(domain_index[-1])], dtype=np.int32)
        return obs, action, reward, domain_index

    def parse_npz_dyn(self, idx):
        st_logmix = np.reshape(np.array(self.st_logmix[idx]).astype(np.float32),
                               (self.hps.z_size, self.hps.num_mixture))
        st_mu = np.reshape(np.array(self.st_mu[idx]).astype(np.float32),
                           (self.hps.z_size, self.hps.num_mixture))
        st_logvar = np.reshape(np.array(self.st_logvar[idx]).astype(np.float32),
                               (self.hps.z_size, self.hps.num_mixture))

        st1_logmix = np.reshape(np.array(self.st1_logmix[idx]).astype(np.float32),
                                (self.hps.z_size, self.hps.num_mixture))
        st1_mu = np.reshape(np.array(self.st1_mu[idx]).astype(np.float32),
                            (self.hps.z_size, self.hps.num_mixture))
        st1_logvar = np.reshape(np.array(self.st1_logvar[idx]).astype(np.float32),
                                (self.hps.z_size, self.hps.num_mixture))

        at = np.array(self.at[idx]).astype(np.float32)

        domain_index = np.array(self.domain_index[idx]).astype(np.int32)

        ################################################### sample st ##################################################

        # adjust temperatures
        st_logmix = st_logmix / self.hps.temperature
        st_logmix -= st_logmix.max()
        st_logmix = np.exp(st_logmix)
        st_logmix /= np.reshape(np.sum(st_logmix, 1), (-1, 1))

        mixture_len = self.hps.z_size

        for j in range(self.hps.num_mixture - 1):
            st_logmix[:, j + 1] = st_logmix[:, j + 1] + st_logmix[:, j]

        mixture_rand_idx = np.repeat(np.random.rand(mixture_len, 1), self.hps.num_mixture, axis=1)
        zero_ref = np.zeros_like(mixture_rand_idx)

        idx = np.argmax(
            np.less_equal(mixture_rand_idx - st_logmix, zero_ref).astype(np.int32), axis=1
        ).astype(np.int32)

        indices = np.arange(0, mixture_len) * self.hps.num_mixture + idx
        chosen_mean = np.reshape(st_mu, (-1))[indices]
        chosen_logstd = np.reshape(st_logvar, (-1))[indices]

        rand_gaussian = np.random.rand(mixture_len) * np.sqrt(self.hps.temperature)
        st = chosen_mean + np.exp(chosen_logstd) * rand_gaussian
        st = st.astype(np.float32)

        ################################################### sample st1 #################################################

        # adjust temperatures
        st1_logmix = st1_logmix / self.hps.temperature
        st1_logmix -= st1_logmix.max()
        st1_logmix = np.exp(st1_logmix)
        st1_logmix /= np.reshape(np.sum(st1_logmix, 1), (-1, 1))

        mixture_len = self.hps.z_size

        for j in range(self.hps.num_mixture - 1):
            st1_logmix[:, j + 1] = st1_logmix[:, j + 1] + st1_logmix[:, j]

        mixture_rand_idx = np.repeat(np.random.rand(mixture_len, 1), self.hps.num_mixture, axis=1)
        zero_ref = np.zeros_like(mixture_rand_idx)

        idx = np.argmax(
            np.less_equal(mixture_rand_idx - st1_logmix, zero_ref).astype(np.int32), axis=1
        ).astype(np.int32)

        indices = np.arange(0, mixture_len) * self.hps.num_mixture + idx
        chosen_mean = np.reshape(st1_mu, (-1))[indices]
        chosen_logstd = np.reshape(st1_logvar, (-1))[indices]

        rand_gaussian = np.random.rand(mixture_len) * np.sqrt(self.hps.temperature)
        st1 = chosen_mean + np.exp(chosen_logstd) * rand_gaussian
        st1 = st1.astype(np.float32)

        return st, at, st1, domain_index

    def parse_function_vae(self, filename):
        obs, action, reward, domain_index = tf.py_func(self.parse_npz_vae, [filename],
                                                       [tf.float32, tf.float32, tf.float32, tf.int32])
        return obs, action, reward, domain_index

    def parse_function_dyn(self, idx):
        # st: 100 x 32, at: 100 x 32, st1: 100 x 3
        st, at, st1, domain = tf.py_func(self.parse_npz_dyn, [idx], [tf.float32, tf.float32, tf.float32, tf.int32])
        return st, at, st1, domain

    # bh: return a batch
    def next_batch(self):
        self.sample_count += 1

        if self.sample_count > int(np.floor(10000 / self.hps.batch_size)):
            self.sess.run(self.init_op)
            self.sample_count = 1

        batch_obs, batch_action, batch_reward, batch_domain_index = self.sess.run(self.next_element)

        return batch_obs, batch_action, batch_reward, batch_domain_index
