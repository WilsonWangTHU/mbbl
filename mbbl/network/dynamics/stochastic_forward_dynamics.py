# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
# -----------------------------------------------------------------------------
from .base_dynamics import base_dynamics_network
from mbbl.config import init_path
from mbbl.util.common import logger
from mbbl.util.common import tf_networks


class dynamics_network(base_dynamics_network):
    '''
        @brief:
    '''

    def __init__(self, args, session, name_scope,
                 observation_size, action_size):
        '''
            @input:
                @ob_placeholder:
                    if this placeholder is not given, we will make one in this
                    class.

                @trainable:
                    If it is set to true, then the policy weights will be
                    trained. It is useful when the class is a subnet which
                    is not trainable
        '''
        raise NotImplementedError
        super(dynamics_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        self._base_dir = init_path.get_abs_base_dir()
        self._debug_it = 0
