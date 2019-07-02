#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:05:28 2018

@author: matthewszhang
"""
import numpy as np
from Box2D.b2 import (fixtureDef, polygonShape, revoluteJointDef)
import gym
import gym.envs.box2d.lunar_lander as LunarLander
from mbbl.util.common import logger

class Box2D_Wrapper(object):
    __SEED = 0
    __SEED_CACHE = [__SEED]

    def __init__(self, gym_id):
        self.env = gym.make(gym_id)
        self.env.env.np_random = np.random.RandomState(self.__SEED)
        self.env.reset() # reset again with fixed seed

        self.action_space = self.env.action_space # debugging purposes
        self.observation_space = self.env.observation_space

    def set_fixed_seed(self, seed):
        self.__SEED = seed
        logger.log('Random seed changed to {}, invalidating previous dynamics \
                   data. Ensure seeds are backed up if you wish to reset \
                   dynamically.'.format(seed))
        self.__SEED_CACHE.append(seed)

    def get_fixed_seed(self):
        if len(self.__SEED_CACHE) > 1:
            logger.log('Seed has been changed before, ensure that \
                       the proper seed is used when resetting dynamics')
        return self.__SEED

    '''
    Recycled Box2D methods
    '''
    def __seed(self, *args, **kwargs):
        '''
        Private this method, gym.seed should not be called as the hashing cannot
        be recovered
        '''
        return self.env.env.seed(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)

    def reset(self, *args, **kwargs):
        '''
        ###### IMPORTANT ######
        RESET THE RANDOM SEED TO BE THE SAME EVERY TIME, OR ELSE CACHE TERRAIN
        '''
        self.env.env.np_random = np.random.RandomState(self.__SEED)

        return self.env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.env.env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.env.env.close(*args, **kwargs)

    def reset_dynamically(self, *args, **kwargs):
        raise NotImplementedError("Subclass must add dynamic reset")

class LunarLanderWrapper(Box2D_Wrapper):
    VIEWPORT_W = LunarLander.VIEWPORT_W
    VIEWPORT_H = LunarLander.VIEWPORT_H
    FPS = LunarLander.FPS
    LANDER_POLY = LunarLander.LANDER_POLY
    SCALE = LunarLander.SCALE
    LEG_DOWN = LunarLander.LEG_DOWN
    LEG_SPRING_TORQUE = LunarLander.LEG_SPRING_TORQUE
    LEG_H = LunarLander.LEG_H
    LEG_W = LunarLander.LEG_W
    LEG_AWAY = LunarLander.LEG_AWAY
    HELIPAD_Y = VIEWPORT_H/SCALE/4 # Fix this value

    def __init__(self, gym_id):
        super(LunarLanderWrapper, self).__init__(gym_id)

        import re
        remove_version = re.compile(r'-v(\d+)$') # version safety
        gym_id_base = remove_version.sub('', gym_id)

        self.continuous = {
                'LunarLander':False,
                'LunarLanderContinuous':True
                }[gym_id_base]


    def reset_dynamically(self, infos):
        self.env.env.np_random = np.random.RandomState(
                self.get_fixed_seed())
        self.env.reset() # Reset all features, ignore ground

        # Force destroy objects
        self.env.env.world.DestroyBody(self.env.env.lander)
        self.env.env.world.DestroyBody(self.env.env.legs[0])
        self.env.env.world.DestroyBody(self.env.env.legs[1])
        while self.env.env.particles:
            self.env.env.world.DestroyBody(self.env.env.particles.pop(0))

        # Reconstruct new lander based on infos
        self.env.env.lander = self.env.env.world.CreateDynamicBody(
            position = (infos['x_pos'], infos['y_pos']),
            angle=infos['theta'],
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/self.SCALE,y/self.SCALE)
                                    for x,y in self.LANDER_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )

        self.env.env.lander.linearVelocity = (infos['x_vel'], infos['y_vel'])
        self.env.env.lander.angularVelocity = (infos['theta_dot'])

        self.env.env.legs = []
        for i in [-1,+1]:
            leg = self.env.env.world.CreateDynamicBody(
                position = (infos['x_pos'] - i*self.LEG_AWAY/self.SCALE*np.cos(infos['theta']),
                            infos['y_pos'] - i*self.LEG_AWAY/self.SCALE*np.sin(infos['theta'])),
                angle = (infos['theta'] + i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(self.LEG_W/self.SCALE, self.LEG_H/self.SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.linearVelocity = \
                    ((infos['x_vel'] + i*self.LEG_AWAY/self.SCALE*
                             np.sin(infos['theta'])*infos['theta_dot'], # rotational term
                     infos['y_vel'] - i*self.LEG_AWAY/self.SCALE*
                             np.cos(infos['theta'])*infos['theta_dot']))
            leg.angularVelocity = 0
            leg.color1 = (0.5,0.4,0.9)
            leg.color2 = (0.3,0.3,0.5)
            rjd = revoluteJointDef(
                bodyA=self.env.env.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i*self.LEG_AWAY/self.SCALE, self.LEG_DOWN/self.SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.LEG_SPRING_TORQUE,
                motorSpeed=+0.3*i  # low enough not to jump back into the sky
                )
            if i==-1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.env.env.world.CreateJoint(rjd)
            self.env.env.legs.append(leg)

class WalkerWrapper(Box2D_Wrapper):
    pass

class RacerWrapper(Box2D_Wrapper):
    pass
