# -----------------------------------------------------------------------------
#   @brief: save the true datapoints into a buffer
#   @author: Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np


class replay_buffer(object):

    def __init__(self, use_buffer, buffer_size, rand_seed,
                 observation_size, action_size, save_reward=False):

        self._use_buffer = use_buffer
        self._buffer_size = buffer_size
        self._npr = np.random.RandomState(rand_seed)

        if not self._use_buffer:
            self._buffer_size = 0

        self._observation_size = observation_size
        self._action_size = action_size

        reward_data_size = self._buffer_size if save_reward else 0
        self._data = {
            'start_state': np.zeros(
                [self._buffer_size, self._observation_size],
                dtype=np.float16
            ),

            'end_state': np.zeros(
                [self._buffer_size, self._observation_size],
                dtype=np.float16
            ),

            'action': np.zeros(
                [self._buffer_size, self._action_size],
                dtype=np.float16
            ),

            'reward': np.zeros([reward_data_size], dtype=np.float16)
        }
        self._data_key = [key for key in self._data if len(self._data[key]) > 0]

        self._current_id = 0
        self._occupied_size = 0

    def add_data(self, new_data):
        if self._buffer_size == 0:
            return

        num_new_data = len(new_data['start_state'])

        if num_new_data + self._current_id > self._buffer_size:
            num_after_full = num_new_data + self._current_id - self._buffer_size
            for key in self._data_key:
                # filling the tail part
                self._data[key][self._current_id: self._buffer_size] = \
                    new_data[key][0: self._buffer_size - self._current_id]

                # filling the head part
                self._data[key][0: num_after_full] = \
                    new_data[key][self._buffer_size - self._current_id:]

        else:

            for key in self._data_key:
                self._data[key][self._current_id:
                                self._current_id + num_new_data] = \
                    new_data[key]

        self._current_id = \
            (self._current_id + num_new_data) % self._buffer_size
        self._occupied_size = \
            min(self._buffer_size, self._occupied_size + num_new_data)

    def get_data(self, batch_size):

        # the data from old data
        sample_id = self._npr.randint(0, self._occupied_size, batch_size)
        return {key: self._data[key][sample_id] for key in self._data_key}

    def get_current_size(self):
        return self._occupied_size

    def get_all_data(self):
        return {key: self._data[key][:self._occupied_size]
                for key in self._data_key}
