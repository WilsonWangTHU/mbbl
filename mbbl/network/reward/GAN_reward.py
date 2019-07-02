# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
# -----------------------------------------------------------------------------
import numpy as np

from .base_reward import base_reward_network
from mbbl.config import init_path
from mbbl.util.common import tf_networks
from mbbl.util.common import tf_utils
from mbbl.util.common import logger
from mbbl.util.il import expert_data_util
import tensorflow as tf


class reward_network(base_reward_network):
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
        super(reward_network, self).__init__(
            args, session, name_scope, observation_size, action_size
        )
        self._base_dir = init_path.get_abs_base_dir()
        # load the expert data
        self._expert_trajectory_obs = expert_data_util.load_expert_trajectory(
            self.args.expert_data_name, self.args.traj_episode_num
        )

    def build_network(self):
        self._build_ph()
        self._tensor = {}

        self._tensor['normalized_start_state'] = (
            self._input_ph['start_state'] -
            self._whitening_operator['state_mean']
        ) / self._whitening_operator['state_std']
        self._tensor['net_input'] = self._tensor['normalized_start_state']

        # the mlp for policy
        network_shape = [self._observation_size] + \
            self.args.reward_network_shape + [1]
        num_layer = len(network_shape) - 1
        act_type = \
            [self.args.reward_activation_type] * (num_layer - 1) + [None]
        norm_type = \
            [self.args.reward_normalizer_type] * (num_layer - 1) + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        # init_data[-1]['w_init_para']['stddev'] = 0.01  # the output layer std
        self._MLP = tf_networks.MLP(
            dims=network_shape, scope='discriminator_mlp', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data
        )

        self._tensor['logits'] = self._MLP(self._tensor['net_input'])
        self._tensor['discriminator_output'] = \
            tf.nn.sigmoid(self._tensor['logits'])
        # the self.discriminator_output is the sigmoid(logit)
        self._tensor['logOfD'] = \
            tf.log(self._tensor['discriminator_output'] + 1e-8)
        self._tensor['logOf1minusD'] = \
            tf.log(1 - self._tensor['discriminator_output'] + 1e-8)

        self._tensor['reward_output'] = tf.minimum(
            -self._tensor['logOf1minusD'], self.args.GAN_reward_clip_value
        )

    def build_loss(self):
        self._update_operator = {}
        self._input_ph['if_expert_data'] = tf.placeholder(tf.float32, [None, 1],
                                                          name='observation_gt')
        self._tensor['if_fake_data'] = 1 - self._input_ph['if_expert_data']

        # calculate the entropy
        self._update_operator['entropy'] = tf.reduce_mean(
            tf_utils.logit_bernoulli_entropy(self._tensor['logits'])
        )
        self._update_operator['entropy_loss'] = \
            -self.args.GAN_ent_coeff * self._update_operator['entropy']

        self._update_operator['loss'] = \
            -tf.reduce_mean(self._tensor['if_fake_data'] *
                            self._tensor['logOf1minusD']) + \
            -tf.reduce_mean(self._input_ph['if_expert_data'] *
                            self._tensor['logOfD']) + \
            self._update_operator['entropy_loss']

        # logging stats, real traj should be 1
        self._tensor['expert_traj_accuracy'] = tf.reduce_sum(
            tf.to_float(self._tensor['discriminator_output'] > 0.5)
            * self._input_ph['if_expert_data']
        ) / tf.reduce_sum(self._input_ph['if_expert_data'])

        self._tensor['expert_average_reward'] = tf.reduce_sum(
            self._tensor['reward_output'] * self._input_ph['if_expert_data']
        ) / tf.reduce_sum(self._input_ph['if_expert_data'])

        self._tensor['agent_traj_accuracy'] = tf.reduce_sum(
            tf.to_float(self._tensor['discriminator_output'] < 0.5) *
            self._tensor['if_fake_data']
        ) / tf.reduce_sum(self._tensor['if_fake_data'])  # fake traj should be 0

        self._tensor['agent_average_reward'] = tf.reduce_sum(
            self._tensor['reward_output'] * self._tensor['if_fake_data']
        ) / tf.reduce_sum(self._tensor['if_fake_data'])

        self._update_operator['update_gan_op'] = tf.train.AdamOptimizer(
            learning_rate=self.args.reward_lr
        ).minimize(self._update_operator['loss'])

    def train(self, data_dict, replay_buffer, training_info={}):
        """ @brief:
            The GAN training of the reward function
        """

        agent_obs = data_dict['start_state']
        training_stats = []

        for i_epoch in range(self.args.reward_epochs):
            # we start from 1 since the first state will be fixed to be 0 reward
            agent_sample_id = self._npr.randint(
                1, agent_obs.shape[0], self.args.gan_timesteps_per_epoch
            )
            # sample and process the positive data-samples
            expert_sample_id = self._npr.randint(
                0, self._expert_trajectory_obs.shape[0],
                self.args.positive_negative_ratio *
                self.args.gan_timesteps_per_epoch
            )

            feed_dict = {
                self._input_ph['if_expert_data']: np.reshape(
                    np.concatenate(
                        [np.zeros(len(agent_sample_id)),
                         np.ones(len(expert_sample_id))],
                    ),
                    [-1, 1]
                ),
                self._input_ph['start_state']: np.concatenate(
                    [agent_obs[agent_sample_id],
                     self._expert_trajectory_obs[expert_sample_id]]
                )
            }

            # train the network
            fetch_dict = {
                'update_op': self._update_operator['update_gan_op'],
                'reward_gan_loss': self._update_operator['loss'],
                'expert_traj_accuracy': self._tensor['expert_traj_accuracy'],
                'agent_traj_accuracy': self._tensor['agent_traj_accuracy'],
                'expert_average_reward': self._tensor['expert_average_reward'],
                'agent_average_reward': self._tensor['agent_average_reward'],
                'entropy': self._update_operator['entropy']
            }
            i_training_stats = self._session.run(fetch_dict, feed_dict=feed_dict)
            training_stats.append(i_training_stats)

        training_stats = {
            key: np.mean([training_stats[i_epoch][key]
                          for i_epoch in range(len(training_stats))])
            for key in training_stats[-1] if key != 'update_op'
        }
        self._set_whitening_var(data_dict['whitening_stats'])
        return training_stats

    def eval(self, data_dict):
        pass

    def pred(self, data_dict):
        logger.info('This function should not be used!')
        reward = []
        for i_data in range(len(data_dict['action'])):
            i_reward = self._env.reward(
                {key: data_dict[key][i_data]
                 for key in ['start_state', 'action']}
            )
            reward.append(i_reward)
        return np.stack(reward), -1, -1

    def use_groundtruth_network(self):
        return False

    def generate_rewards(self, rollout_data):
        """@brief:
            This function should be called before _preprocess_data
        """
        for path in rollout_data:
            # the predicted value function (baseline function)
            path["raw_rewards"] = path['rewards']  # preserve the raw reward
            # from mbbl.util.common.fpdb import fpdb; fpdb().set_trace()

            # generate the observation pairs
            ob_pairs = path['obs'][1: len(path['obs'])]

            path["rewards"] = self._session.run(
                self._tensor['reward_output'],
                feed_dict={self._input_ph['start_state']: ob_pairs}
            ).flatten()
            assert len(path["rewards"]) == len(path['raw_rewards'])
        return rollout_data
