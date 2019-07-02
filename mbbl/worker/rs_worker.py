# ------------------------------------------------------------------------------
#   @brief:
# ------------------------------------------------------------------------------
import numpy as np

from .base_worker import base_worker
from mbbl.config import init_path


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

    def __init__(self, args, observation_size, action_size,
                 network_type, task_queue, result_queue, worker_id,
                 name_scope='planning_worker'):

        # the base agent
        super(worker, self).__init__(args, observation_size, action_size,
                                     network_type, task_queue, result_queue,
                                     worker_id, name_scope)
        self._base_dir = init_path.get_base_dir()

    def _plan(self, planning_data):

        # print(planning_data, self._worker_id)
        num_branch = planning_data['num_branch']
        planned_result = self._random_shooting({
            'state': np.tile(
                np.reshape(planning_data['state'], [-1, self._observation_size]),
                [num_branch, 1]
            ),
            'depth': planning_data['depth']
        })

        # find the best shooting trajectory
        optim_traj_id = np.argmax(planned_result['return'].reshape([-1]))
        # from util.common.fpdb import fpdb; fpdb().set_trace()
        return {
            'action': planned_result['action'][0][optim_traj_id],
            'return': planned_result['return'][optim_traj_id],
            'next_state': planned_result['state'][1][optim_traj_id],
            # 'misc': planned_result
        }

    def _random_shooting(self, planning_data):
        if planning_data['depth'] == 0:
            # end of the planning
            # num_tree_brance = len(planning_data['states'])
            # print planning_data
            return {
                'state': [planning_data['state']],
                'return': np.zeros([len(planning_data['state'])]),
                'reward': [],
                'action': []
            }
        else:
            current_state = planning_data['state']

            # get the control signal
            action, _, _ = self._network['policy'][0].act(
                {'start_state': current_state}
            )

            # pred next state
            next_state, _, _ = self._network['dynamics'][0].pred(
                {'start_state': current_state, 'action': action}
            )

            # pred the reward
            reward, _, _ = self._network['reward'][0].pred(
                {'start_state': current_state, 'action': action, 'next_state': next_state}
            )

            # new planning_data
            next_planning_data = {
                'depth':  planning_data['depth'] - 1, 'state': next_state
            }

            # check the done
            done = detect_done(next_state, self.args.task, self.args.check_done)
            # if np.any(done):
            #     from mbbl.util.common.fpdb import fpdb; fpdb().set_trace()

            planned_result = self._random_shooting(next_planning_data)
            # from util.common.fpdb import fpdb; fpdb().set_trace()
            planned_result['state'].insert(0, current_state)
            planned_result['reward'].insert(0, reward)
            planned_result['return'] = (1 - done) * planned_result['return'] +\
                reward
            planned_result['action'].insert(0, action)
            return planned_result
