# -----------------------------------------------------------------------------
#   @author:
#       Matthew Zhang
#   @brief:
#       Alternate code-up of the box2d lunar lander environment from gym
# -----------------------------------------------------------------------------
import numpy as np

from mbbl.config import init_path
from mbbl.env import base_env_wrapper as bew
from mbbl.env import env_register
from mbbl.env.gym_env.box2d import box2d_make
from mbbl.util.common import logger
import mbbl.env.gym_env.box2d.wrappers


LunarLander = mbbl.env.gym_env.box2d.wrappers.LunarLanderWrapper

VIEWPORT_W = LunarLander.VIEWPORT_W
VIEWPORT_H = LunarLander.VIEWPORT_H
LEG_DOWN = LunarLander.LEG_DOWN
HELIPAD_Y = LunarLander.HELIPAD_Y
FPS = LunarLander.FPS
SCALE = LunarLander.SCALE
FIXED_ANGLE = 0.75

class env(bew.base_env):
    # acrobot has applied sin/cos obs
    LANDER = ['gym_lunar_lander', 'gym_lunar_lander_continuous']

    def __init__(self, env_name, rand_seed, misc_info):
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()

    def step(self, action):
        action = np.clip(action, -1., 1.)
        if not self._env.continuous: # bad way to discretize
            if action.shape == ():
                action = [action]
            if action[0] < -.5:
                action = 0
            elif action[0] < .0:
                action = 1
            elif action[0] < .5:
                action = 2
            else:
                action = 3

        if self._env.env.env.game_over: # prevent movement after ending
            if self._env.continuous:
                action = np.array([0.0, 0.0]) # Nop
            else:
                action = 0 # Nop
        ob, _, _, info = self._env.step(action)

        # get the reward
        reward = self.reward(
            {'end_state': ob, 'start_state': self._old_ob, 'action': action}
        )

        # get the end signal
        self._current_step += 1
        if self._current_step > self._env_info['max_length']:
            done = True
        else:
            done = False
        self._old_ob = np.array(ob)
        return ob, reward, done, info

    def reset(self):
        self._current_step = 0
        self._old_ob = self._env.reset()
        return np.array(self._old_ob), 0.0, False, {}

    def _build_env(self):
        _env_name = {
            'gym_lunar_lander': 'LunarLander-v2',
            'gym_lunar_lander_continuous': 'LunarLanderContinuous-v2'
            }[self._env_name]

        # make the environments
        self._env = box2d_make(_env_name)
        self._env_info = env_register.get_env_info(self._env_name)

    def _set_groundtruth_api(self):
        """ @brief:
                In this function, we could provide the ground-truth dynamics
                and rewards APIs for the agent to call.
                For the new environments, if we don't set their ground-truth
                apis, then we cannot test the algorithm using ground-truth
                dynamics or reward
        """
        self._set_reward_api()
        self._set_dynamics_api()

    def _set_dynamics_api(self):
        '''
        def fdynamics(self, data_dict):
            raise NotImplementedError

        def fdynamics_grad_state(self, data_dict):
            raise NotImplementedError

        def fdynamics_grad_action(self, data_dict):
            raise NotImplementedError

        self.fdynamics = fdynamics
        self.fdynamics_grad_state = fdynamics_grad_state
        self.fdynamics_grad_action = fdynamics_grad_action
        '''

        def fdynamics(data_dict):
            # Recover angles and velocities
            # Batched observations are handled sequentially
            state = data_dict['start_state']
            action = np.clip(data_dict['action'], -1., 1.)
            if not self._env.continuous: # bad way to discretize
                if action.shape == ():
                    action = [action]
                if action[0] < -.5:
                    action = 0
                elif action[0] < .0:
                    action = 1
                elif action[0] < .5:
                    action = 2
                else:
                    action = 3

            x_pos = state[0] * (VIEWPORT_W/SCALE/2) + (VIEWPORT_W/SCALE/2)
            y_pos = state[1] * (VIEWPORT_W/SCALE/2) + (HELIPAD_Y + LEG_DOWN/SCALE)
            x_vel = state[2] * FPS / (VIEWPORT_W/SCALE/2)
            y_vel = state[3] * FPS / (VIEWPORT_H/SCALE/2)
            theta = state[4]
            theta_dot = state[5] * FPS/20

            hull_information = {
                    'x_pos':x_pos,
                    'y_pos':y_pos,
                    'x_vel':x_vel,
                    'y_vel':y_vel,
                    'theta':theta,
                    'theta_dot':theta_dot
                    }

            self._env.reset_dynamically(hull_information)
            return self._env.step(action)[0]

        self.fdynamics = fdynamics

    def _set_reward_api(self):

        # step 1, set the zero-order reward function
        assert self._env_name in self.LANDER

        def reward(data_dict):
            #state = data_dict['end_state']
            old_state = data_dict['start_state']
            action = data_dict['action']
            if not self._env.continuous: # bad way to discretize
                try:
                    if action.shape == ():
                        action = [action]
                    if action[0] < -.5:
                        action = 0
                    elif action[0] < .0:
                        action = 1
                    elif action[0] < .5:
                        action = 2
                    else:
                        action = 3
                except:
                    action = action

            continuous = self._env.continuous

            m_power = 0
            if continuous:
                m_power = 1/(1 + np.exp(-5*action[0]))

            elif not continuous and action==2:
                m_power = 1.0

            s_power = 0
            if self._env.continuous:
                s_power = np.exp(5*action[1] - 2.5)/12

            elif not continuous and action in [1, 3]:
                s_power = 1.0

#            shaping = \
#            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
#            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
#            - 100*abs(state[4]) + 10*state[6] + 10*state[7]

            old_shaping = \
            - (old_state[0]*old_state[0] + old_state[1]*old_state[1]) \
            - (old_state[2]*old_state[2] + old_state[3]*old_state[3]) \
            - (old_state[4]**2) + .1*old_state[6] + .1*old_state[7]

            reward = old_shaping
            reward -= m_power*0.30
            reward -= s_power*0.03

            return reward
            # remove discrete terms

        self.reward = reward

        '''
        # step two, set the first order stats and second order stats

        self.reward_grad_state = reward_grad_state
        self.reward_grad_action = reward_grad_action
        '''
        def reward_derivative(data_dict, target):
            x_ob_pos = 0
            y_ob_pos = 1
            x_vel_pos = 2
            y_vel_pos = 3
            theta_pos = 4
            contact_one_pos = 6
            contact_two_pos = 7

            num_data = len(data_dict['start_state'])
            state = data_dict['start_state']
            action = data_dict['action']

            def discrete():
                if target == 'state':
                    derivative_data = np.zeros([num_data, self._env_info['ob_size']],
                                               dtype=np.float)
                    derivative_data[:,x_ob_pos] = 2*state[:,0]
                    derivative_data[:,y_ob_pos] = 2*state[:,1]
                    derivative_data[:,x_vel_pos] = 2*state[:,2]
                    derivative_data[:,y_vel_pos] = 2*state[:,3]
                    derivative_data[:,theta_pos] = 2*state[:,4]
                    derivative_data[:,contact_one_pos] = .1
                    derivative_data[:,contact_two_pos] = .1

                elif target == 'action':
                    derivative_data = np.zeros([num_data, self._env_info['action_size']],
                                               dtype=np.float)
                    derivative_data[:,0] = 5/(1 + np.exp(-5*action[:,0])) * \
                                        (1 - 1/(1 + np.exp(-5*action[:,0])))
                    derivative_data[:,1] = 5/12 * np.exp(5*action[:,1] - 2.5)

                elif target == 'state-state':
                    derivative_data = np.zeros(
                            [num_data, self._env_info['ob_size'], self._env_info['ob_size']],
                            dtype=np.float)
                    derivative_data[:,x_ob_pos,x_ob_pos] = 2
                    derivative_data[:,y_ob_pos,y_ob_pos] = 2
                    derivative_data[:,x_vel_pos,x_vel_pos] = 2
                    derivative_data[:,y_vel_pos,y_vel_pos] = 2
                    derivative_data[:,theta_pos,theta_pos] = 2

                elif target == 'action-state':
                    derivative_data = np.zeros(
                            [num_data, self._env_info['action_size'], self._env_info['ob_size']],
                            dtype=np.float)

                elif target == 'state-action':
                    derivative_data = np.zeros(
                            [num_data, self._env_info['ob_size'], self._env_info['action_size']],
                            dtype=np.float)

                elif target == 'action-action':
                    derivative_data = np.zeros(
                            [num_data, self._env_info['action_size'], self._env_info['action_size']],
                            dtype=np.float)

                else:
                    assert False, logger.error('Invalid target {}'.format(target))

                return derivative_data

            def continuous():
                if target == 'state':
                    derivative_data = np.zeros([num_data, self._env_info['ob_size']],
                                               dtype=np.float)
                    derivative_data[:,x_ob_pos] = 2*state[:,0]
                    derivative_data[:,y_ob_pos] = 2*state[:,1]
                    derivative_data[:,x_vel_pos] = 2*state[:,2]
                    derivative_data[:,y_vel_pos] = 2*state[:,3]
                    derivative_data[:,theta_pos] = 2*state[:,4]
                    derivative_data[:,contact_one_pos] = .1
                    derivative_data[:,contact_two_pos] = .1

                elif target == 'action':
                    derivative_data = np.zeros([num_data, self._env_info['action_size']],
                                               dtype=np.float)
                    derivative_data[:,0] = 5/(1 + np.exp(-5*action[:,0])) * \
                                        (1 - 1/(1 + np.exp(-5*action[:,0])))
                    derivative_data[:,1] = 5/12 * np.exp(5*action[:,1] - 2.5)

                elif target == 'state-state':
                    derivative_data = np.zeros(
                            [num_data, self._env_info['ob_size'], self._env_info['ob_size']],
                            dtype=np.float)
                    derivative_data[:,x_ob_pos,x_ob_pos] = 2
                    derivative_data[:,y_ob_pos,y_ob_pos] = 2
                    derivative_data[:,x_vel_pos,x_vel_pos] = 2
                    derivative_data[:,y_vel_pos,y_vel_pos] = 2
                    derivative_data[:,theta_pos,theta_pos] = 2

                elif target == 'action-state':
                    derivative_data = np.zeros(
                            [num_data, self._env_info['action_size'], self._env_info['ob_size']],
                            dtype=np.float)

                elif target == 'state-action':
                    derivative_data = np.zeros(
                            [num_data, self._env_info['ob_size'], self._env_info['action_size']],
                            dtype=np.float)

                elif target == 'action-action':
                    derivative_data = np.zeros(
                            [num_data, self._env_info['action_size'], self._env_info['action_size']],
                            dtype=np.float)
                    derivative_data[:,0,0] = 25/(1 + np.exp(-5*action[:,0])) * \
                                        (1 - 1/(1 + np.exp(-5*action[:,0])))**2 \
                                        + 5/(1 + np.exp(-5*action[:,0])) * \
                                        (1 - 5/(1 + np.exp(-5*action[:,0]) * \
                                        (1 - 1/(1 + np.exp(-5*action[:,0])))))
                    # function form of second derivative is sig(x)(1-sig(x)) ** 2
                    # + sig(x)(1-[sig(x)*(1-sig(x))])
                    derivative_data[:,1,1] = 25/12 * np.exp(5*action[:,1] - 2.5)


                else:
                    assert False, logger.error('Invalid target {}'.format(target))

                return derivative_data
            reward_derivative = {'gym_cartpole':discrete,
                      'gym_cartpole_continuous':continuous}[self._env_name]
            return reward_derivative()

        self.reward_derivative = reward_derivative

if __name__ == '__main__':

    test_env_name = ['gym_lunar_lander', 'gym_lunar_lander_continuous']
    for env_name in test_env_name:
        test_env = env(env_name, 1234, None)
        api_env = env(env_name, 1234, None)
        api_env.reset()
        ob, reward, _, _ = test_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, test_env._env.action_space.shape)
            new_ob, reward, _, _ = test_env.step(action)

            # test the reward api
            reward_from_api = \
                api_env.reward({'start_state': ob, 'action': action})
            reward_error = np.sum(np.abs(reward_from_api - reward))

            # test the dynamics api
            newob_from_api = \
                api_env.fdynamics({'start_state': ob, 'action': action})
            ob_error = np.sum(np.abs(newob_from_api - new_ob))

            ob = new_ob

            print('reward error: {}, dynamics error: {}'.format(
                reward_error, ob_error)
            )
