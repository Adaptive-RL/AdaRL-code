import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Change in Size: v0*
register(
    id='Pongm-v00',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'size': 2.0},
)

register(
    id='Pongm-v01',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'size': 4.0},
)

register(
    id='Pongm-v02',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'size': 6.0},
)
register(
    id='Pongm-v03',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'size': 8.0},
)
# testset
register(
    id='Pongm-v04',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'size': 3.0},
)

register(
    id='Pongm-v05',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'size': 9.0},
)


# Change in Orientation: v1*
register(
    id='Pongm-v10',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'orientation': 0},
)

register(
    id='Pongm-v11',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'orientation': 180},
)
# testset
register(
    id='Pongm-v12',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'orientation': 90},
)

register(
    id='Pongm-v13',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'orientation': 270},
)
# Change in Noise: v2*
register(
    id='Pongm-v20',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'noise': 0.25},
)

register(
    id='Pongm-v21',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'noise': 0.75},
)

register(
    id='Pongm-v22',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'noise': 1.25},
)

register(
    id='Pongm-v23',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'noise':1.75},
)


register(
    id='Pongm-v24',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'noise': 2.25},
)
# testset
register(
    id='Pongm-v25',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'noise':1.0},
)


register(
    id='Pongm-v26',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'noise': 2.75},
)


# Change in Color: v3*
register(
    id='Pongm-v30',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'color': 'default'},
)

register(
    id='Pongm-v31',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'color': 'green'},
)

register(
    id='Pongm-v32',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'color': 'red'},
)
# testset
register(
    id='Pongm-v33',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'color': 'yellow'},
)

register(
    id='Pongm-v34',
    entry_point='gym_pong.envs:AtariEnv',
    kwargs={'color': 'white'},
)