import gym
import gym_cartpole_world
import gym_pong


def env_init(game, src_domain_index):
    env = []
    for domian_idx in src_domain_index:
        version = 'v' + domian_idx
        if 'cartpole' in game.lower():
            env.append(gym.make('CartPoleWorld-{}'.format(version)))
        elif 'pong' in game.lower():
            if 'm' not in game.lower():
                version_orig = 'v0'
                env.append(gym.make('Pong-{}'.format(version_orig)))
            else:
                env.append(gym.make('Pongm-{}'.format(version)))

    if 'cartpole' in game.lower():
        for e in env:
            e.initialize(theta_threshold=15, x_threshold=2.4)

    return env
