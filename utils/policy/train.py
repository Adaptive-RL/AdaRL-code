import os.path

import cv2
import numpy as np
import pylab

from utils.policy.extract import encode_obs


def train(env, agent, model, hps, theta_all, rdc_s, tot_episodes, n_domain, save_p):
    # when we train over all source domains,if there is one domain with done=True, then the training is finished
    scores, episodes = [], []
    count = []
    for k in range(n_domain):
        count.append(0)
    for e in range(tot_episodes):
        score = 0
        state_record = []
        state_record_rdc = []
        c_record = []
        state_ori_record = []
        done_record = []
        for k in range(n_domain):
            # initialization
            state_ori = env[k].reset()  # the ground-truth state
            # generate observational image
            obs = env[k].render(mode='rgb_array')
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, (128, 128), interpolation=cv2.INTER_CUBIC)
            obs = cv2.normalize(obs, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            a_prev, r_prev, c_prev = model.reset()
            state, c = encode_obs(hps, model, obs, a_prev, r_prev, c_prev, k)

            state_rdc = state[0][rdc_s[0]]
            state = np.reshape(state, [1, hps.z_size])
            state_rdc = np.reshape(state_rdc, [1, -1])
            state_record.append(state)
            state_record_rdc.append(state_rdc)
            c_record.append(c)
            state_ori_record.append(state_ori)
            done_record.append(False)

        while score < 500:
            for k in range(n_domain):
                # get action for the current observation and go one step in environment
                # action = agent.get_action(state_record[k], theta_all[k])
                action = agent.get_action(state_record_rdc[k], theta_all[k])
                next_state_ori, reward, done, info = env[k].step(action)
                next_obs = env[k].render(mode='rgb_array')
                next_obs = cv2.cvtColor(next_obs, cv2.COLOR_RGB2GRAY)
                next_obs = cv2.resize(next_obs, (128, 128), interpolation=cv2.INTER_CUBIC)
                next_obs = cv2.normalize(next_obs, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                next_state, next_c = encode_obs(hps, model, next_obs, action, reward, c_record[k],
                                                k)  # infer the states
                next_state_rdc = next_state[0][rdc_s[0]]
                next_state = np.reshape(next_state, [1, hps.z_size])
                next_state_rdc = np.reshape(next_state_rdc, [1, -1])
                # if an action make the episode end, then gives penalty of -100
                reward = reward if not done else -100
                # save the sample <s, a, r, s'> to the replay memory
                # agent.append_sample(state_record[k], action, reward, next_state, theta_all[k], done, score)
                agent.append_sample(state_record_rdc[k], action, reward, next_state_rdc, theta_all[k], done, score)
                done_record[k] = done
                state_record_rdc[k] = next_state_rdc
                state_record[k] = next_state
                c_record[k] = next_c
                state_ori_record[k] = next_state_ori
                count[k] += 1

            score += 1
            if any(done_record):
                break

        # every episode update the target model to be same with model
        agent.update_target_model()

        scores.append(score)
        episodes.append(e)
        pylab.plot(episodes, scores, 'b')
        pylab.savefig(os.path.join(save_p, 'score_v_episodes.png'))

        output_log = "episode %d, " \
                     "score: %d, " \
                     "memory length: %d, " \
                     "epsilon: %.8f, " \
                     % (e,
                        score,
                        len(agent.memory),
                        agent.epsilon)
        print(output_log)
        f = open(os.path.join(save_p, 'output.txt'), 'a')
        f.write(output_log + '\n')
        f.close()

        save_p_e = os.path.join(save_p, 'ep')
        if not os.path.exists(save_p_e):
            os.makedirs(save_p_e)

        # save the model
        if e == 0 or (e >= 200 and e % 10 == 0):
            agent.model.save_weights(os.path.join(save_p_e, str(e) + 'policy.h5'))
        if e == tot_episodes - 1:
            agent.model.save_weights(os.path.join(save_p, 'policy.h5'))
