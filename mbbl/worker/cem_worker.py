# ------------------------------------------------------------------------------
#   @brief:
# ------------------------------------------------------------------------------
import numpy as np

from .base_worker import base_worker
from mbbl.config import init_path
# from mbbl.util.common import parallel_util


def detect_done(ob, env_name, check_done):
    if env_name == 'gym_fhopper':
        height, ang = ob[:, 0], ob[:, 1]
        done = np.logical_or(height <= 0.7, abs(ang) >= 0.2)

    elif env_name == 'gym_fwalker2d':
        height, ang = ob[:, 0], ob[:, 1]
        done = np.logical_or(
            height >= 2.0,
            np.logical_or(height <= 0.8, abs(ang) >= 1.0)
        )

    elif env_name == 'gym_fant':
        height = ob[:, 0]
        done = np.logical_or(height > 1.0, height < 0.2)

    elif env_name in ['gym_fant2', 'gym_fant5', 'gym_fant10',
                      'gym_fant20', 'gym_fant30']:
        height = ob[:, 0]
        done = np.logical_or(height > 1.0, height < 0.2)
    else:
        done = np.zeros([ob.shape[0]])

    if not check_done:
        done[:] = False

    return done


class worker(base_worker):
    EPSILON = 0.001

    def __init__(self, args, observation_size, action_size,
                 network_type, task_queue, result_queue, worker_id,
                 name_scope='planning_worker'):

        # the base agent
        super(worker, self).__init__(args, observation_size, action_size,
                                     network_type, task_queue, result_queue,
                                     worker_id, name_scope)
        self._base_dir = init_path.get_base_dir()
        self._alpha = args.cem_learning_rate
        self._num_iters = args.cem_num_iters
        self._elites_fraction = args.cem_elites_fraction

    def _plan(self, planning_data):
        num_traj = planning_data['state'].shape[0]
        sample_action = np.reshape(
            planning_data['samples'],
            [num_traj, planning_data['depth'], planning_data['action_size']]
        )
        current_state = planning_data['state']
        total_reward = 0
        done = np.zeros([num_traj])

        for i_depth in range(planning_data['depth']):
            action = sample_action[:, i_depth, :]

            # pred next state
            next_state, _, _ = self._network['dynamics'][0].pred(
                {'start_state': current_state, 'action': action}
            )

            # pred the reward
            reward, _, _ = self._network['reward'][0].pred(
                {'start_state': current_state, 'action': action}
            )

            total_reward += reward * (1 - done)
            current_state = next_state

            # mark the done marker
            this_done = detect_done(next_state, self.args.task, self.args.check_done)
            done = np.logical_or(this_done, done)
            # if np.any(done):
            # from mbbl.util.common.fpdb import fpdb; fpdb().set_trace()

        # Return the cost
        return_dict = {'costs': -total_reward,
                       'sample_id': planning_data['id']}
        return return_dict
