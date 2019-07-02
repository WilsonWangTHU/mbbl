# ------------------------------------------------------------------------------
#   @brief:
# ------------------------------------------------------------------------------
from .base_worker import base_worker
from mbbl.config import init_path


class worker(base_worker):

    def __init__(self, args, observation_size, action_size,
                 network_type, task_queue, result_queue, worker_id,
                 name_scope='derivative_worker'):

        # the base agent
        super(worker, self).__init__(args, observation_size, action_size,
                                     network_type, task_queue, result_queue,
                                     worker_id, name_scope)
        self._base_dir = init_path.get_base_dir()

        # build the environments
        self._build_env()

    def _dynamics_derivative(self, data_dict,
                             target=['state', 'action', 'state-action']):

        assert len(self._network['dynamics']) == 1
        return self._network['dynamics'][0].get_derivative(data_dict, target)

    def _reward_derivative(self, data_dict,
                           target=['state', 'action', 'state-state']):

        assert len(self._network['reward']) == 1
        return self._network['reward'][0].get_derivative(data_dict, target)

    def _dynamics(self, data_dict):
        assert len(self._network['dynamics']) == 1
        return {'end_state': self._network['dynamics'][0].pred(data_dict)[0]}

    def _reward(self, data_dict):
        assert len(self._network['reward']) == 1
        return {'reward': self._network['reward'][0].pred(data_dict)[0]}
