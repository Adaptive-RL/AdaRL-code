import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Change in Gravity: v0*
register(
    id='CartPoleWorld-v00',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 5},
)

register(
    id='CartPoleWorld-v01',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 10},
)

register(
    id='CartPoleWorld-v02',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 20},
)

register(
    id='CartPoleWorld-v03',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 30},
)

register(
    id='CartPoleWorld-v04',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 40},
)
# testset
register(
    id='CartPoleWorld-v05',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 15},
)
register(
    id='CartPoleWorld-v06',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'gravity': 55},
)


# Change in CartMass: v1*
register(
    id='CartPoleWorld-v10',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 0.5},
)

register(
    id='CartPoleWorld-v11',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 1.5},
)

register(
    id='CartPoleWorld-v12',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 2.5},
)

register(
    id='CartPoleWorld-v13',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 3.5},
)
register(
    id='CartPoleWorld-v14',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 4.5},
)
# testset
register(
    id='CartPoleWorld-v15',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 1.0},
)
register(
    id='CartPoleWorld-v16',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'masscart': 5.5},
)
# Change in Noise: v2*
register(
    id='CartPoleWorld-v20',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'noises': 0.25},
)
register(
    id='CartPoleWorld-v21',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'noises': 0.75},
)
register(
    id='CartPoleWorld-v22',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'noises': 1.25},
)
register(
    id='CartPoleWorld-v23',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'noises': 1.75},
)
register(
    id='CartPoleWorld-v24',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'noises': 2.25},
)
# testset
register(
    id='CartPoleWorld-v25',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'noises': 0.5},
)
register(
    id='CartPoleWorld-v26',
    entry_point='gym_cartpole_world.envs:CartPoleWorldEnv',
    kwargs={'case': 1, 'noises': 2.75},
)