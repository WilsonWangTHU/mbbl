# -----------------------------------------------------------------------------
#   @brief:
#       register the environments here. It's similar to the gym register
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import copy
import importlib

_ENV_INFO = {}

_ENV_INFO.update({
    'gym_reacher': {
        'path': 'mbbl.env.gym_env.reacher',
        'ob_size': 11, 'action_size': 2, 'max_length': 50,
    },

    'gym_cheetah': {
        'path': 'mbbl.env.gym_env.walker',
        'ob_size': 17, 'action_size': 6, 'max_length': 1000
    },

    'gym_walker2d': {
        'path': 'mbbl.env.gym_env.walker',
        'ob_size': 17, 'action_size': 6, 'max_length': 1000
    },

    'gym_fwalker2d': {
        'path': 'mbbl.env.gym_env.fixed_walker',
        'ob_size': 17, 'action_size': 6, 'max_length': 1000
    },
    'gym_dfwalker2d': {
        'path': 'mbbl.env.gym_env.delayed_walker',
        'ob_size': 17, 'action_size': 6, 'max_length': 1000
    },

    'gym_hopper': {
        'path': 'mbbl.env.gym_env.walker',
        'ob_size': 11, 'action_size': 3, 'max_length': 1000
    },
    'gym_fhopper': {
        'path': 'mbbl.env.gym_env.fixed_walker',
        'ob_size': 11, 'action_size': 3, 'max_length': 1000
    },
    'gym_dfhopper': {
        'path': 'mbbl.env.gym_env.delayed_walker',
        'ob_size': 11, 'action_size': 3, 'max_length': 1000
    },

    'gym_swimmer': {
        'path': 'mbbl.env.gym_env.walker',
        'ob_size': 8, 'action_size': 2, 'max_length': 1000
    },
    'gym_fswimmer': {
        'path': 'mbbl.env.gym_env.fixed_swimmer',
        'ob_size': 9, 'action_size': 2, 'max_length': 1000
    },

    'gym_ant': {
        'path': 'mbbl.env.gym_env.walker',
        'ob_size': 27, 'action_size': 8, 'max_length': 1000
    },
    'gym_fant': {
        'path': 'mbbl.env.gym_env.fixed_walker',
        'ob_size': 27, 'action_size': 8, 'max_length': 1000
    },
    'gym_fant2': {
        'path': 'mbbl.env.gym_env.fixed_walker',
        'ob_size': 27, 'action_size': 8, 'max_length': 1000
    },
    'gym_fant5': {
        'path': 'mbbl.env.gym_env.fixed_walker',
        'ob_size': 27, 'action_size': 8, 'max_length': 1000
    },
    'gym_fant10': {
        'path': 'mbbl.env.gym_env.fixed_walker',
        'ob_size': 27, 'action_size': 8, 'max_length': 1000
    },
    'gym_fant20': {
        'path': 'mbbl.env.gym_env.fixed_walker',
        'ob_size': 27, 'action_size': 8, 'max_length': 1000
    },
    'gym_fant30': {
        'path': 'mbbl.env.gym_env.fixed_walker',
        'ob_size': 27, 'action_size': 8, 'max_length': 1000
    },
    'gym_dfant': {
        'path': 'mbbl.env.gym_env.delayed_walker',
        'ob_size': 27, 'action_size': 8, 'max_length': 1000
    },

    'gym_pendulum': {
        'path': 'mbbl.env.gym_env.pendulum',
        'ob_size': 3, 'action_size': 1, 'max_length': 200
    },
    'gym_invertedPendulum': {
        'path': 'mbbl.env.gym_env.invertedPendulum',
        'ob_size': 4, 'action_size': 1, 'max_length': 100
    },
    'gym_acrobot': {
        'path': 'mbbl.env.gym_env.acrobot',
        'ob_size': 6, 'action_size': 1, 'max_length': 200
    },
    'gym_mountain': {
        'path': 'mbbl.env.gym_env.mountain_car',
        'ob_size': 2, 'action_size': 1, 'max_length': 200
    },
    'gym_cartpole': {
        'path': 'mbbl.env.gym_env.cartpole',
        'ob_size': 4, 'action_size': 1, 'max_length': 200
    },

    'gym_petsReacher': {
        'path': 'mbbl.env.gym_env.pets',
        'ob_size': 20, 'action_size': 7, 'max_length': 150
    },
    'gym_petsCheetah': {
        'path': 'mbbl.env.gym_env.pets',
        'ob_size': 18, 'action_size': 6, 'max_length': 1000
    },
    'gym_petsPusher': {
        'path': 'mbbl.env.gym_env.pets',
        'ob_size': 23, 'action_size': 7, 'max_length': 150
    },

    'gym_lunar_lander': {
        'path': 'mbbl.env.gym_env.box2d_lunar_lander',
        'ob_size': 8, 'action_size': 1, 'max_length': 200
    },

    'gym_lunar_lander_continuous': {
        'path': 'mbbl.env.gym_env.box2d_lunar_lander',
        'ob_size': 8, 'action_size': 2, 'max_length': 200
    },
    'gym_point': {
        'path': 'env.gym_env.point',
        'ob_size': 2, 'action_size': 2, 'max_length': 40
    },

    'gym_humanoid': {
        'path': 'mbbl.env.gym_env.humanoid',
        'ob_size': 376, 'action_size': 17, 'max_length': 1000
    },
    'gym_slimhumanoid': {
        'path': 'mbbl.env.gym_env.humanoid',
        'ob_size': 45, 'action_size': 17, 'max_length': 1000
    },
    'gym_nostopslimhumanoid': {
        'path': 'mbbl.env.gym_env.humanoid',
        'ob_size': 45, 'action_size': 17, 'max_length': 1000
    },

    'gym_cheetahO01': {
        'path': 'mbbl.env.gym_env.noise_gym_cheetah',
        'ob_size': 17, 'action_size': 6, 'max_length': 1000
    },
    'gym_cheetahO001': {
        'path': 'mbbl.env.gym_env.noise_gym_cheetah',
        'ob_size': 17, 'action_size': 6, 'max_length': 1000
    },
    'gym_cheetahA01': {
        'path': 'mbbl.env.gym_env.noise_gym_cheetah',
        'ob_size': 17, 'action_size': 6, 'max_length': 1000
    },
    'gym_cheetahA003': {
        'path': 'mbbl.env.gym_env.noise_gym_cheetah',
        'ob_size': 17, 'action_size': 6, 'max_length': 1000
    },

    'gym_pendulumO001': {
        'path': 'mbbl.env.gym_env.noise_gym_pendulum',
        'ob_size': 3, 'action_size': 1, 'max_length': 200
    },
    'gym_cartpoleO001': {
        'path': 'mbbl.env.gym_env.noise_gym_cartpole',
        'ob_size': 4, 'action_size': 1, 'max_length': 200
    },
    'gym_pendulumO01': {
        'path': 'mbbl.env.gym_env.noise_gym_pendulum',
        'ob_size': 3, 'action_size': 1, 'max_length': 200
    },
    'gym_cartpoleO01': {
        'path': 'mbbl.env.gym_env.noise_gym_cartpole',
        'ob_size': 4, 'action_size': 1, 'max_length': 200
    },
})


# The deepmind environments
_ENV_INFO.update({
    'acrobot-swingup_sparse': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 6, 'action_size': 1, 'max_length': 1000
    },
    'acrobot-swingup': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 6, 'action_size': 1, 'max_length': 1000
    },
    'ball_in_cup-catch': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 8, 'action_size': 2, 'max_length': 1000
    },
    'cartpole-swingup_sparse': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 5, 'action_size': 1, 'max_length': 1000
    },
    'cartpole-balance': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 5, 'action_size': 1, 'max_length': 1000
    },
    'cartpole-balance_sparse': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 5, 'action_size': 1, 'max_length': 1000
    },
    'cartpole-swingup': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 5, 'action_size': 1, 'max_length': 1000
    },
    'cheetah-run': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 17, 'action_size': 6, 'max_length': 1000,
        'SOLVED_REWARD': 700
    },

    'finger-turn_easy': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 12, 'action_size': 2, 'max_length': 1000
    },
    'finger-spin': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 9, 'action_size': 2, 'max_length': 1000
    },
    'finger-turn_hard': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 12, 'action_size': 2, 'max_length': 1000
    },
    'fish-upright': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 21, 'action_size': 5, 'max_length': 1000
    },
    'fish-swim': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 24, 'action_size': 5, 'max_length': 1000
    },
    'hopper-stand': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 15, 'action_size': 4, 'max_length': 1000
    },
    'hopper-hop': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 15, 'action_size': 4, 'max_length': 1000
    },
    'humanoid-run': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 67, 'action_size': 21, 'max_length': 1000
    },
    'humanoid-stand': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 67, 'action_size': 21, 'max_length': 1000
    },
    'humanoid-walk': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 67, 'action_size': 21, 'max_length': 1000
    },
    'manipulator-bring_ball': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 37, 'action_size': 5, 'max_length': 1000
    },
    'pendulum-swingup': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 3, 'action_size': 1, 'max_length': 1000
    },
    'point_mass-easy': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 4, 'action_size': 2, 'max_length': 1000
    },

    'reacher-hard': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 7, 'action_size': 2, 'max_length': 1000
    },
    'reacher-easy': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 7, 'action_size': 2, 'max_length': 1000
    },
    'swimmer-swimmer6': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 25, 'action_size': 5, 'max_length': 1000
    },
    'swimmer-swimmer15': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 61, 'action_size': 14, 'max_length': 1000
    },
    'walker-run': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 24, 'action_size': 6, 'max_length': 1000
    },
    'walker-stand': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 24, 'action_size': 6, 'max_length': 1000
    },
    'walker-walk': {
        'path': 'mbbl.env.dm_env.dm_env',
        'ob_size': 24, 'action_size': 6, 'max_length': 1000
    },

    # the envrionments with pos:
    'cheetah-run-pos': {
        'path': 'mbbl.env.dm_env.pos_dm_env',
        'ob_size': 17, 'action_size': 6, 'max_length': 1000,
        'SOLVED_REWARD': 10,
    },

    'dm-humanoid': {
        'path': 'mbbl.env.dm_env.humanoid_env',
        'ob_size': 67, 'action_size': 21, 'max_length': 1000,
        'SOLVED_REWARD': 250
    },
    'dm-humanoid-noise': {
        'path': 'mbbl.env.dm_env.humanoid_env',
        'ob_size': 67, 'action_size': 21, 'max_length': 1000,
        'SOLVED_REWARD': 250
    },
})

_ENV_INFO.update({
    'RoboschoolHumanoid-v1': {
        'path': 'mbbl.env.bullet_env.roboschool',
        'ob_size': 44, 'action_size': 17, 'max_length': 1000,
        'image_height': 100, 'image_width': 150, 'image_channel': 1,
    },
    'RoboschoolHumanoidFlagrun-v1': {
        'path': 'mbbl.env.bullet_env.roboschool',
        'ob_size': 44, 'action_size': 17, 'max_length': 1000,
        'image_height': 100, 'image_width': 150, 'image_channel': 1,
    },
    'RoboschoolHumanoidFlagrunHarder-v1': {
        'path': 'mbbl.env.bullet_env.roboschool',
        'ob_size': 44, 'action_size': 17, 'max_length': 1000,
        'image_height': 100, 'image_width': 150, 'image_channel': 1,
    },
    'RoboschoolAnt-v1': {
        'path': 'mbbl.env.bullet_env.roboschool',
        'ob_size': 44, 'action_size': 17, 'max_length': 1000,
        'image_height': 100, 'image_width': 150, 'image_channel': 1,
    }
})

# DeepMimic Bullet Humanoid
_ENV_INFO.update({
    'Bullet_Humanoid-v1': {
        'path': 'mbbl.env.bullet_env.bullet_humanoid',
        'ob_size': 197, 'action_size': 36, 'max_length': 1000
    }
})

# add the short / middle env variants. for debugging purposes
env_list = ['gym_cheetah', 'gym_walker2d', 'gym_hopper', 'gym_swimmer', 'gym_ant']
len_variant = [10, 100, 200, 333, 500, 1000, 2000]
for env in env_list:
    for variant in len_variant:
        parent_info = _ENV_INFO[env]
        variant_info = copy.deepcopy(parent_info)
        variant_info['max_length'] = variant
        _ENV_INFO[env + '-' + str(variant)] = variant_info

# add the depth image version
env_list = [key for key in _ENV_INFO if key.startswith('Roboschool')]
for env in env_list:
    _ENV_INFO[env]['rgb'] = False
    _ENV_INFO[env]['depth'] = False

    # the rgb environment
    variant_info = copy.deepcopy(_ENV_INFO[env])
    variant_info['rgb'] = True
    variant_info['image_channel'] = 3
    _ENV_INFO[env + '-rgb'] = variant_info

    # the rgb environment
    variant_info = copy.deepcopy(_ENV_INFO[env])
    variant_info['rgb'] = True
    variant_info['depth'] = True
    variant_info['image_channel'] = 4
    _ENV_INFO[env + '-rgbd'] = variant_info

    # the depth environment
    variant_info = copy.deepcopy(_ENV_INFO[env])
    variant_info['rgb'] = True
    variant_info['depth'] = True
    variant_info['image_channel'] = 1
    _ENV_INFO[env + '-rgbd'] = variant_info


def io_information(task_name):

    return _ENV_INFO[task_name]['ob_size'], \
        _ENV_INFO[task_name]['action_size'], \
        _ENV_INFO[task_name]['max_length']


def get_env_info(task_name):
    return _ENV_INFO[task_name]


def make_env(task_name, rand_seed, misc_info={}):
    '''
    render_flag = re.compile(r'__render$')
    if render_flag.search(task_name):
        return render_wrapper(task_name, rand_seed, misc_info), get_env_info(task_name)
    '''

    env_file = importlib.import_module(_ENV_INFO[task_name]['path'])
    return env_file.env(task_name, rand_seed, misc_info), _ENV_INFO[task_name]
