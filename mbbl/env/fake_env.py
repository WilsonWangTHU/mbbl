import numpy as np


class fake_env(object):

    def __init__(self, env, model):
        self._env = env
        self._model = model
        self._state = None
        self._current_step = 0
        self._max_length = self._env._env_info['max_length']
        self._obs_bounds = (-1e5, 1e5)

    def step(self, action):
        self._current_step += 1
        next_state, reward = self._model(self._state, action)
        next_state = np.clip(next_state, *self._obs_bounds)
        self._state = next_state

        if self._current_step > self._max_length:
            done = True
        else:
            done = False
        return next_state, reward, done, None

    def reset(self):
        self._current_step = 0
        ob, reward, done, info = self._env.reset()
        self._state = ob

        return ob, reward, done, info
