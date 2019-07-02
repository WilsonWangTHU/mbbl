import numpy as np

from mbbl.env import env_register
# from mbbl.env.gym_env import walker
# from mbbl.env.gym_env import reacher
# from mbbl.env.gym_env import invertedPendulum
# from mbbl.env.gym_env import pendulum
# from mbbl.env.gym_env import acrobot
# from mbbl.env.gym_env import cartpole
# from mbbl.env.gym_env import mountain_car
from mbbl.network.dynamics import groundtruth_forward_dynamics
# from mbbl.env.gym_env import pets
from mbbl.env.gym_env import humanoid

TEST = "DYNAMICS_DERIVATIVE"  # 'REWARD_DERIVATIVE'
# TEST = 'REWARD_DERIVATIVE'

# candidate_names = invertedPendulum.env.PENDULUM
# candidate_names = pendulum.env.PENDULUM
# candidate_names = pets.env.ENV
candidate_names = humanoid.env.ENV
if __name__ == '__main__' and TEST == 'REWARD_DERIVATIVE':
    DERIVATIVE_EPS = 1e-6

    # test the walker
    # candidate_names = walker.env.WALKER
    # candidate_names = reacher.env.ARM_2D
    max_error = 0.0
    for env_name in candidate_names:
        env, _ = env_register.make_env(env_name, 123, {})
        env_info = env_register.get_env_info(env_name)
        derivative_env, _ = env_register.make_env(env_name, 234, {})

        data_dict = \
            {'action': np.random.uniform(-1, 1, [1, env_info['action_size']])}
        data_dict['start_state'], _, _, _ = env.reset()
        data_dict['start_state'] = data_dict['start_state'].reshape(1, -1)
        data_dict['start_state'] = data_dict['start_state'].reshape(1, -1)

        r_u = derivative_env.reward_derivative(data_dict, 'action')
        r_uu = derivative_env.reward_derivative(data_dict, 'action-action')
        r_x = derivative_env.reward_derivative(data_dict, 'state')
        r_xx = derivative_env.reward_derivative(data_dict, 'state-state')

        # test the derivative of the reward wrt action
        for i_elem in range(env_info['action_size']):
            for step_size in [0.0, 1e-6, 1e-3, 1e-1, 1.0]:
                delta = np.zeros([env_info['action_size']])
                delta[i_elem] = step_size
                raw_reward = derivative_env.reward(
                    {'start_state': np.reshape(data_dict['start_state'], [-1]),
                     'action': np.reshape(data_dict['action'], [-1])}
                )
                actual_reward = derivative_env.reward(
                    {'start_state': np.reshape(data_dict['start_state'], [-1]),
                     'action': np.reshape(data_dict['action'], [-1]) + delta}
                )
                linear_pred_reward = raw_reward + r_u[0, i_elem] * step_size
                quadratic_pred_reward = linear_pred_reward + \
                    0.5 * r_uu[0, i_elem, i_elem] * (step_size ** 2)

                print('i_elem: ', i_elem)
                print('step_size: ', step_size)
                print('reward: ', actual_reward)
                print('linear_pred diff: ', linear_pred_reward - actual_reward)
                print('quad_pred_diff: ', quadratic_pred_reward - actual_reward)
                max_error = max(max_error, abs(quadratic_pred_reward - actual_reward))
                print('max_error: ', max_error)

        # test the derivative of the reward wrt state
        for i_elem in range(env_info['ob_size']):
            for step_size in [1e-6, 1e-3, 1e-1, 1.0]:
                delta = np.zeros([env_info['ob_size']])
                delta[i_elem] = step_size
                raw_reward = derivative_env.reward(
                    {'start_state': np.reshape(data_dict['start_state'], [-1]),
                     'action': np.reshape(data_dict['action'], [-1])}
                )
                actual_reward = derivative_env.reward(
                    {'start_state': np.reshape(data_dict['start_state'], [-1]) + delta,
                     'action': np.reshape(data_dict['action'], [-1])}
                )
                linear_pred_reward = raw_reward + r_x[0, i_elem] * step_size
                quadratic_pred_reward = linear_pred_reward + \
                    0.5 * r_xx[0, i_elem, i_elem] * (step_size ** 2)

                print('\nPred_state\n')
                print('i_elem: ', i_elem)
                print('step_size: ', step_size)
                print('actual_reward: ', actual_reward)
                print('diff of linear_pred: ', actual_reward - linear_pred_reward)
                print('diff of quad_pred: ', actual_reward - quadratic_pred_reward)
                if abs(quadratic_pred_reward - actual_reward) > 0.1:
                    import pdb; pdb.set_trace()
                max_error = max(max_error, abs(quadratic_pred_reward - actual_reward))
                print('max_error: ', max_error)
                # import pdb; pdb.set_trace()
        print('max_error: ', max_error)

elif TEST == 'DYNAMICS_DERIVATIVE':
    from mbbl.config import base_config
    from mbbl.config import ilqr_config
    parser = base_config.get_base_config()
    parser = ilqr_config.get_ilqr_config(parser)
    args = base_config.make_parser(parser)
    args.gt_dynamics = 1
    # candidate_names = walker.env.WALKER
    # candidate_names = reacher.env.ARM_2D
    for env_name in candidate_names:
        args.task = args.task_name = env_name
        env_info = env_register.get_env_info(env_name)
        # env = pendulum.env(env_name, 123, {})
        env, _ = env_register.make_env(env_name, 123, {})
        env_info = env_register.get_env_info(env_name)
        derivative_env, _ = env_register.make_env(env_name, 123, {})
        env_info = env_register.get_env_info(env_name)
        derivative_env.reset()
        network = groundtruth_forward_dynamics.dynamics_network(
            args, None, None, env_info['ob_size'], env_info['action_size']
        )
        network.build_network()

        data_dict = \
            {'action': np.random.uniform(-1, 1, [1, env_info['action_size']])}
        data_dict['start_state'], _, _, _ = env.reset()
        data_dict['start_state'] = data_dict['start_state'].reshape(1, -1)

        derivative_data = network.get_derivative(data_dict, ['action', 'state'])
        f_u, f_x = derivative_data['action'], derivative_data['state']

        # test the derivative of the reward wrt action
        for i_elem in range(env_info['action_size']):
            for step_size in [0.0, 1e-6, 1e-3, 1e-1, 1.0]:
                delta = np.zeros([env_info['action_size']])
                delta[i_elem] = step_size
                raw_state = derivative_env.fdynamics(
                    {'start_state': np.reshape(data_dict['start_state'], [-1]),
                     'action': np.reshape(data_dict['action'], [-1])}
                )
                actual_state = derivative_env.fdynamics(
                    {'start_state': np.reshape(data_dict['start_state'], [-1]),
                     'action': np.reshape(data_dict['action'], [-1]) + delta}
                )
                linear_pred_state = raw_state + f_u[:, i_elem] * step_size

                print('i_elem: ', i_elem)
                print('step_size: ', step_size)
                print('diff of state: ', actual_state - raw_state,
                      (actual_state - raw_state).max())
                print('diff of linear_pred: ', actual_state - linear_pred_state,
                      (actual_state - linear_pred_state).max())

        # test the derivative of the reward wrt action
        for i_elem in range(env_info['ob_size']):
            for step_size in [0.0, 1e-6, 1e-3, 1e-1, 1.0]:
                delta = np.zeros([env_info['ob_size']])
                delta[i_elem] = step_size
                raw_state = derivative_env.fdynamics(
                    {'start_state': np.reshape(data_dict['start_state'], [-1]),
                     'action': np.reshape(data_dict['action'], [-1])}
                )
                actual_state = derivative_env.fdynamics(
                    {'start_state': np.reshape(data_dict['start_state'], [-1]) + delta,
                     'action': np.reshape(data_dict['action'], [-1])}
                )
                linear_pred_state = raw_state + f_x[:, i_elem] * step_size

                print('i_elem: ', i_elem)
                print('step_size: ', step_size)
                print('diff of state: ', actual_state - raw_state, np.abs(actual_state - raw_state).max())
                print('diff of linear_pred: ', actual_state - linear_pred_state, np.abs(actual_state - linear_pred_state).max())
