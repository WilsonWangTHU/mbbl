# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       Define the dynamic models for the system, which takes input two
#       adjacent states and output the predicted actions.
# -----------------------------------------------------------------------------
import numpy as np
import tensorflow as tf

from mbbl.config import init_path
from mbbl.util.common import tf_utils
from mbbl.util.common import whitening_util


class base_dynamics_network(object):
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
        self.args = args

        self._session = session
        self._name_scope = name_scope

        self._observation_size = observation_size
        self._action_size = action_size

        self._task_name = args.task_name
        self._network_shape = args.dynamics_network_shape

        self._npr = np.random.RandomState(args.seed)

        self._whitening_operator = {}
        self._whitening_variable = []
        self._base_dir = init_path.get_abs_base_dir()

    def build_network(self):
        pass

    def build_loss(self):
        pass

    def _build_ph(self):

        # initialize the running mean and std (whitening)
        whitening_util.add_whitening_operator(
            self._whitening_operator, self._whitening_variable,
            'state', self._observation_size
        )
        whitening_util.add_whitening_operator(
            self._whitening_operator, self._whitening_variable,
            'diff_state', self._observation_size
        )

        # initialize the input placeholder
        self._input_ph = {
            'start_state': tf.placeholder(
                tf.float32, [None, self._observation_size], name='start_state'
            ),

            'end_state': tf.placeholder(
                tf.float32, [None, self._observation_size], name='end_state'
            ),

            'action': tf.placeholder(
                tf.float32, [None, self._action_size], name='action_state'
            )
        }

    def get_input_placeholder(self):
        return self._input_ph

    def get_weights(self):
        return None

    def set_weights(self, weight_dict):
        pass

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]

        # the weights that actually matter
        self._network_var_list = \
            self._trainable_var_list + self._whitening_variable

        self._set_network_weights = tf_utils.set_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

        self._get_network_weights = tf_utils.get_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

    def load_checkpoint(self, ckpt_path):
        pass

    def save_checkpoint(self, ckpt_path):
        pass

    def get_whitening_operator(self):
        return self._whitening_operator

    def _set_whitening_var(self, whitening_stats):
        whitening_util.set_whitening_var(
            self._session, self._whitening_operator,
            whitening_stats, ['state', 'diff_state']
        )

    def train(self, data_dict, replay_buffer, training_info={}):
        pass

    def eval(self, data_dict):
        raise NotImplementedError

    def pred(self, data_dict):
        raise NotImplementedError

    def use_groundtruth_network(self):
        return False

    def get_derivative(self, data_dict, target):
        raise NotImplementedError
