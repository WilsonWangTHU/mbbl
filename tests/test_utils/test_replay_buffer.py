import unittest

import numpy as np

from mbbl.util.common.replay_buffer import replay_buffer


class TestReplayBuffer(unittest.TestCase):

    def test_init(self):
        replay_buffer_obj = replay_buffer(**self.kwargs)
        self.assertEqual(replay_buffer_obj._use_buffer, self.kwargs["use_buffer"])
        self.assertEqual(replay_buffer_obj._buffer_size,
                         self.kwargs["buffer_size"] if self.kwargs["use_buffer"] else 0)

        self.assertIsInstance(replay_buffer_obj._npr, np.random.RandomState)

        self.assertEqual(replay_buffer_obj._observation_size, self.kwargs["observation_size"])
        self.assertEqual(replay_buffer_obj._action_size, self.kwargs["action_size"])

        self.assertIsInstance(replay_buffer_obj._data, dict)
        self.assertSetEqual(set(replay_buffer_obj._data_key),
                            {"start_state", "end_state", "action"})
        self.assertIsInstance(replay_buffer_obj._data["start_state"], np.ndarray)
        self.assertIsInstance(replay_buffer_obj._data["end_state"], np.ndarray)
        self.assertIsInstance(replay_buffer_obj._data["action"], np.ndarray)
        self.assertIsInstance(replay_buffer_obj._data["reward"], np.ndarray)
        state_shape = (self.kwargs["buffer_size"], self.kwargs["observation_size"])
        action_shape = (self.kwargs["buffer_size"], self.kwargs["action_size"])
        self.assertEqual(replay_buffer_obj._data["start_state"].shape, state_shape)
        self.assertEqual(replay_buffer_obj._data["end_state"].shape, state_shape)
        self.assertEqual(replay_buffer_obj._data["action"].shape, action_shape)
        self.assertEqual(replay_buffer_obj._data["reward"].shape, (0,))

        self.assertEqual(replay_buffer_obj._current_id, 0)
        self.assertEqual(replay_buffer_obj._occupied_size, 0)

    def test_init_with_buffer_size_0(self):
        kwargs = self.kwargs.copy()
        kwargs["use_buffer"] = False
        replay_buffer_obj = replay_buffer(**kwargs)
        self.assertEqual(replay_buffer_obj._buffer_size, 0)

    def test_add_data_not_full(self):
        replay_buffer_obj = replay_buffer(**self.kwargs)
        new_data_size = self.kwargs["buffer_size"] - 2
        new_data = self.generate_new_data(new_data_size)
        replay_buffer_obj.add_data(new_data)
        self.assertEqual(replay_buffer_obj._current_id, new_data_size)
        self.assertEqual(replay_buffer_obj._occupied_size, new_data_size)

        extra_new_data_size = 1
        extra_new_data = self.generate_new_data(extra_new_data_size)
        replay_buffer_obj.add_data(extra_new_data)

        data_size = new_data_size + extra_new_data_size
        self.assertEqual(replay_buffer_obj._current_id, data_size)
        self.assertEqual(replay_buffer_obj._occupied_size, data_size)

        data = {key: np.vstack((new_data[key], extra_new_data[key])) for key in new_data.keys()}

        for data_key in data.keys():
            np.testing.assert_equal(
                replay_buffer_obj._data[data_key][:data_size], data[data_key])

    def test_add_data_full(self):
        replay_buffer_obj = replay_buffer(**self.kwargs)
        data_size = self.kwargs["buffer_size"]
        data = self.generate_new_data(data_size)
        replay_buffer_obj.add_data(data)
        self.assertEqual(replay_buffer_obj._current_id, 0)
        self.assertEqual(replay_buffer_obj._occupied_size, self.kwargs["buffer_size"])

        self.assert_data_dict_equal(replay_buffer_obj._data, data,
                                    exclude_keys=["reward"])

    def test_add_data_overflow(self):
        replay_buffer_obj = replay_buffer(**self.kwargs)
        extra_data_size = 2
        data_size = self.kwargs["buffer_size"] + extra_data_size
        data = self.generate_new_data(data_size)
        replay_buffer_obj.add_data(data)
        self.assertEqual(replay_buffer_obj._current_id, extra_data_size)
        self.assertEqual(replay_buffer_obj._occupied_size, self.kwargs["buffer_size"])

        for data_key in data.keys():
            assert set(map(lambda lis: tuple(lis), replay_buffer_obj._data[data_key])) \
                   <= set(map(lambda lis: tuple(lis), data[data_key][extra_data_size:]))

    def test_add_data_not_full_with_reward(self):
        replay_buffer_obj = replay_buffer(**self.kwargs, save_reward=True)
        data_size = np.random.randint(self.kwargs["buffer_size"])
        data = self.generate_new_data(data_size, include_rewards=True)
        replay_buffer_obj.add_data(data)
        self.assertEqual(replay_buffer_obj._current_id, data_size)
        self.assertEqual(replay_buffer_obj._occupied_size, data_size)
        for data_key in data.keys():
            np.testing.assert_equal(
                replay_buffer_obj._data[data_key][:data_size], data[data_key])

    def test_add_data_overflow_with_reward(self):
        replay_buffer_obj = replay_buffer(**self.kwargs, save_reward=True)
        extra_data_size = 2
        data_size = self.kwargs["buffer_size"] + extra_data_size
        data = self.generate_new_data(data_size, include_rewards=True)
        replay_buffer_obj.add_data(data)
        self.assertEqual(replay_buffer_obj._current_id, extra_data_size)
        self.assertEqual(replay_buffer_obj._occupied_size, self.kwargs["buffer_size"])
        for data_key in data.keys():
            if data_key == "reward":
                assert set(list(replay_buffer_obj._data[data_key])) \
                       <= set(list(data[data_key][extra_data_size:]))
            else:
                assert set(map(lambda lis: tuple(lis), replay_buffer_obj._data[data_key])) \
                       <= set(map(lambda lis: tuple(lis), data[data_key][extra_data_size:]))

    def test_add_data_with_buffer_size_0(self):
        kwargs = self.kwargs.copy()
        kwargs["use_buffer"] = False
        replay_buffer_obj = replay_buffer(**kwargs)
        data = self.generate_new_data(np.random.randint(self.kwargs["buffer_size"] * 2))
        replay_buffer_obj.add_data(data)
        self.assertEqual(replay_buffer_obj._current_id, 0)
        self.assertEqual(replay_buffer_obj._occupied_size, 0)

    def test_get_data(self):
        replay_buffer_obj = replay_buffer(**self.kwargs)
        all_data = self.generate_new_data(self.kwargs["buffer_size"])
        replay_buffer_obj.add_data(all_data)
        batch_size = np.random.randint(1, self.kwargs["buffer_size"])
        data_subset = replay_buffer_obj.get_data(batch_size)
        self.assert_data_dict_equal(data_subset, all_data,
                                    exclude_keys=["reward"], subset=True)
        assert all([val.shape[0] == batch_size for val in data_subset.values()])

    def test_get_data_with_reward(self):
        replay_buffer_obj = replay_buffer(**self.kwargs, save_reward=True)
        all_data = self.generate_new_data(self.kwargs["buffer_size"], include_rewards=True)
        replay_buffer_obj.add_data(all_data)
        batch_size = np.random.randint(1, self.kwargs["buffer_size"])
        data_subset = replay_buffer_obj.get_data(batch_size)
        self.assert_data_dict_equal(data_subset, all_data, subset=True)
        assert all([val.shape[0] == batch_size for val in data_subset.values()])

    def test_get_current_size(self):
        replay_buffer_obj = replay_buffer(**self.kwargs)
        self.assertEqual(replay_buffer_obj.get_current_size(), 0)

        data_size = np.random.randint(1, self.kwargs["buffer_size"])
        data = self.generate_new_data(data_size)
        replay_buffer_obj.add_data(data)
        self.assertEqual(replay_buffer_obj.get_current_size(),
                         replay_buffer_obj._occupied_size)

    def test_get_all_data(self):
        replay_buffer_obj = replay_buffer(**self.kwargs)
        replay_buffer_obj.add_data(self.generate_new_data(self.kwargs["buffer_size"]))
        fetched_data = replay_buffer_obj.get_all_data()
        self.assert_data_dict_equal(fetched_data, replay_buffer_obj._data,
                                    exclude_keys=["reward"])

    def test_get_all_data_with_reward(self):
        replay_buffer_obj = replay_buffer(**self.kwargs, save_reward=True)
        replay_buffer_obj.add_data(
            self.generate_new_data(self.kwargs["buffer_size"], include_rewards=True))
        fetched_data = replay_buffer_obj.get_all_data()
        self.assert_data_dict_equal(fetched_data, replay_buffer_obj._data)

    # ---- Test set up and Tear down ----

    def setUp(self):
        self.kwargs = {
            "use_buffer": True,
            "buffer_size": 20,
            "rand_seed": 0,
            "observation_size": 5,
            "action_size": 3
        }

    def tearDown(self):
        pass

    # ---- Helper methods ----

    def generate_new_data(self, batch_size, include_rewards=False):
        new_state_shape = (batch_size, self.kwargs["observation_size"])
        new_action_shape = (batch_size, self.kwargs["action_size"])
        new_data = {
            "start_state": np.random.rand(*new_state_shape).astype(np.float16),
            "end_state": np.random.rand(*new_state_shape).astype(np.float16),
            "action": np.random.rand(*new_action_shape).astype(np.float16)
        }
        if include_rewards:
            new_data["reward"] = np.random.rand(batch_size).astype(np.float16)
        return new_data

    def assert_data_dict_equal(self, data_dict1, data_dict2,
                               exclude_keys=None, subset=False):
        # if subset == False, assert that data_dict1 and data_dict2 are equal,
        #   excluding exclude_keys
        # if subset == True, assert that each value of data_dict1 is a subset of data_dict2,
        #   excluding exclude_keys
        if not exclude_keys:
            exclude_keys = set()
        else:
            exclude_keys = set(exclude_keys)
        self.assertSetEqual(set(data_dict1.keys()) - exclude_keys,
                            set(data_dict2.keys()) - exclude_keys)
        for key in data_dict1.keys():
            if key not in exclude_keys:
                if subset:
                    if key == "reward":
                        assert set(data_dict1[key]) <= set(data_dict2[key])
                    else:
                        assert set(map(lambda lis: tuple(lis), data_dict1[key])) \
                               <= set(map(lambda lis: tuple(lis), data_dict2[key]))
                else:
                    np.testing.assert_equal(data_dict1[key], data_dict2[key])
