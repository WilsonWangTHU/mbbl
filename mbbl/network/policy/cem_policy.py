# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
from .base_policy import base_policy_network
from mbbl.config import init_path


class policy_network(base_policy_network):
    '''
        @brief:
            In this object class, we define the network structure, the restore
            function and save function.
            It will only be called in the agent/agent.py
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):

        super(policy_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        self._base_dir = init_path.get_abs_base_dir()

    def build_network(self):
        pass

    def build_loss(self):
        pass

    def train(self, data_dict, replay_buffer, training_info={}):
        pass

    def act(self, data_dict):
        raise NotImplementedError
