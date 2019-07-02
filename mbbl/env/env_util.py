# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np


class box(object):

    def __init__(self, size, min_val, max_val):
        self.shape = np.array(size).reshape([-1])
        self.high = np.ones(size) * max_val
        self.low = np.ones(size) * min_val


def make_ob_ac_box(env):
    """ @brief: the boxes are sometimes used by gym-style repos
    """
    ob_space = env.observation_spec()
    ob_size = [ob_space[key].shape[0] if ob_space[key].shape is not () else 1
               for key in ob_space]
    observation_space = box(np.sum(ob_size), -np.inf, np.inf)

    action_space = env.action_spec()
    assert action_space.minimum[0] == -1
    assert action_space.maximum[0] == 1
    action_space = box(action_space.shape[0], -1, 1)

    return observation_space, action_space


def get_dm_env_names(task_name):
    return task_name.split('-')


def parse_roboschool_env_names(task_name):
    return task_name.split('-')


def vectorize_ob(ob_dictionary):
    element = [np.array(element).flatten()
               for element in ob_dictionary.values()]
    return np.concatenate(element)


def get_gym_q_info(env, current_version):
    return get_gym_qpos_size(env, current_version), \
        get_gym_qvel_size(env, current_version)


def get_gym_qpos_size(env, current_version):

    if current_version in ['0.7.4', '0.9.4']:
        if hasattr(env, 'env'):
            len_qpos = len(env.env.data.qpos)
        else:
            len_qpos = len(env.data.qpos)
    else:
        len_qpos = len(env.env.sim.data.qpos)

    return len_qpos


def get_gym_qvel_size(env, current_version):

    if current_version in ['0.7.4', '0.9.4']:

        if hasattr(env, 'env'):
            len_qvel = len(env.env.data.qvel)
        else:
            len_qvel = len(env.data.qvel)
    else:
        len_qvel = len(env.env.sim.data.qvel)

    return len_qvel


def play_episode_with_env(env, policy, control_info={}):

    # init the variables
    obs, rewards, action_mus, action_logstds, actions = [], [], [], [], []
    env_infos = []

    # start the env (reset the environment)
    ob, _, _, env_info = env.reset(control_info=control_info)
    obs.append(ob)
    env_infos.append(env_info)

    while True:
        # generate the policy
        action_signal = policy(ob, control_info)
        sampled_action = np.array(action_signal[0], dtype=np.float64)

        # take the action
        ob, reward, done, env_info = env.step(sampled_action)

        # record the stats
        rewards.append((reward))
        obs.append(ob)
        env_infos.append(env_info)
        actions.append(sampled_action)
        action_mus.append(action_signal[1])
        action_logstds.append(action_signal[2])

        if done:  # terminated
            # append one more for the raw_obs
            traj_episode = {
                "obs": np.array(obs, dtype=np.float64),
                "old_action_dist_mus":
                    np.array(np.concatenate(action_mus), dtype=np.float64),
                "old_action_dist_logstds":
                    np.array(np.concatenate(action_logstds), dtype=np.float64),
                "rewards": np.array(rewards, dtype=np.float64),
                "actions":  np.array(actions, dtype=np.float64),
                'control_info': control_info
            }
            break
    return traj_episode


def debug_episode_with_env(obs, actions, env, from_network=False):

    # start the env (reset the environment)
    if from_network is False:
        new_obs = []
        env.reset()
        env.set_state({'start_state': obs[0]})
        new_obs.append(obs[0])

        for sampled_action in actions:
            # take the action
            ob, reward, done, _ = env.step(sampled_action)
            new_obs.append(ob)

            if done:  # terminated
                # append one more for the raw_obs
                traj_episode = {
                    "obs": np.array(new_obs, dtype=np.float64),
                }
                break
    else:
        new_obs = []
        new_obs.append(obs[0])

        pos = 0
        env._env.reset()
        env._env.set_state({'start_state': obs[0]})
        for sampled_action in actions:
            # take the action

            input_data = {'start_state': np.reshape(new_obs[pos], [1, -1]),
                          'action': np.reshape(sampled_action, [1, -1])}

            new_data = env.pred(input_data)[0][0]
            # ob, reward, done, _ = env.step(sampled_action)
            new_obs.append(new_data)
            pos += 1

            if pos >= 100:  # terminated
                # append one more for the raw_obs
                traj_episode = {
                    "obs": np.array(new_obs, dtype=np.float64),
                }
                break
    return traj_episode


def dagger_play_episode_with_env(env, policy, mpc_policy, control_info={}):

    # init the variables
    obs, rewards = [], []
    actions, action_mus, action_logstds = [], [], []
    traj_episode = dict()

    # start the env (reset the environment)
    ob, _, _, _ = env.reset()
    obs.append(ob)

    # from util.common.fpdb import fpdb; fpdb().set_trace()

    while True:
        # generate the policy
        action_signal = policy(ob, control_info)
        mpc_action = mpc_policy(ob)

        # take the action
        ob, reward, done, _ = env.step(action_signal[0])

        # record the stats
        rewards.append((reward))
        obs.append(ob)
        actions.append(mpc_action)

        if done:  # terminated
            # append one more for the raw_obs
            traj_episode = {
                "obs": np.array(obs),
                "rewards": np.array(rewards),
                "actions":  np.array(actions),
                'control_info': control_info
            }
            break
    return traj_episode
