# ------------------------------------------------------------------------------
#   @brief:
#       The optimization agent is responsible for doing the updates.
#   @author:
# ------------------------------------------------------------------------------
from .base_trainer import base_trainer


class trainer(base_trainer):

    def __init__(self, args, network_type, task_queue, result_queue,
                 name_scope='trainer'):
        # the base agent
        super(trainer, self).__init__(
            args=args, network_type=network_type,
            task_queue=task_queue, result_queue=result_queue,
            name_scope=name_scope
        )
        # self._base_path = init_path.get_abs_base_dir()

    def _update_parameters(self, rollout_data, training_info):
        # get the observation list
        self._update_whitening_stats(rollout_data)
        training_data = self._preprocess_data(rollout_data)
        training_stats = {'avg_reward': training_data['avg_reward'],
                          'avg_reward_std': training_data['avg_reward_std']}

        # train the policy
        for key in training_info['network_to_train']:
            for i_network in range(self._num_model_ensemble[key]):
                i_stats = self._network[key][i_network].train(
                    training_data, self._replay_buffer, training_info={}
                )
                if i_stats is not None:
                    training_stats.update(i_stats)
        self._replay_buffer.add_data(training_data)
        return training_stats
