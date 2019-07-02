# ------------------------------------------------------------------------------
#   @brief:
#       The optimization agent is responsible for doing the updates.
#   @author:
# ------------------------------------------------------------------------------
from .base_trainer import base_trainer
from mbbl.util.common import logger
import numpy as np
from collections import OrderedDict


class trainer(base_trainer):

    def __init__(self, args, network_type, task_queue, result_queue,
                 name_scope='trainer'):
        # the base agent
        super(trainer, self).__init__(
            args=args, network_type=network_type,
            task_queue=task_queue, result_queue=result_queue,
            name_scope=name_scope
        )

    def _update_parameters(self, rollout_data, training_info):
        # get the observation list
        self._update_whitening_stats(rollout_data)

        # generate the reward from discriminator
        rollout_data = self._network['reward'][0].generate_rewards(rollout_data)

        training_data = self._preprocess_data(rollout_data)
        training_stats = OrderedDict()
        training_stats['avg_reward'] = training_data['avg_reward']
        training_stats['avg_reward_std'] = training_data['avg_reward_std']

        assert 'reward' in training_info['network_to_train']
        # train the policy
        for key in training_info['network_to_train']:
            for i_network in range(self._num_model_ensemble[key]):
                i_stats = self._network[key][i_network].train(
                    training_data, self._replay_buffer, training_info={}
                )
                if i_stats is not None:
                    training_stats.update(i_stats)
        self._replay_buffer.add_data(training_data)

        # record the actual reward (not from the discriminator)
        self._get_groundtruth_reward(rollout_data, training_stats)
        return training_stats

    def _get_groundtruth_reward(self, rollout_data, training_stats):

        for i_episode in rollout_data:
            i_episode['raw_episodic_reward'] = sum(i_episode['raw_rewards'])
        avg_reward = np.mean([i_episode['raw_episodic_reward']
                              for i_episode in rollout_data])
        logger.info('Raw reward: {}'.format(avg_reward))
        training_stats['RAW_reward'] = avg_reward
