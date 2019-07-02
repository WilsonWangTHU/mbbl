# -----------------------------------------------------------------------------
#   @brief:
#       define the how to record the summary
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------
import os

import tensorflow as tf

from mbbl.config import init_path
from mbbl.util.common import logger


class summary_handler(object):
    '''
        @brief:
            Tell the handler where to record all the information.
            Normally, we want to record the prediction of value loss, and the
            average reward (maybe learning rate)
    '''

    def __init__(self, sess, summary_name, enable=True, summary_dir=None):
        # the interface we need
        self.summary = None
        self.sess = sess
        self.enable = enable
        if not self.enable:  # the summary handler is disabled
            return
        if summary_dir is None:
            self.path = os.path.join(
                init_path.get_base_dir(), 'summary'
            )
        else:
            self.path = os.path.join(summary_dir, 'summary')
        self.path = os.path.abspath(self.path)

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.path = os.path.join(self.path, summary_name)

        self.train_writer = tf.summary.FileWriter(self.path, self.sess.graph)

        logger.info(
            'summary write initialized, writing to {}'.format(self.path))

    def get_tf_summary(self):
        assert self.summary is not None, logger.error(
            'tf summary not defined, call the summary object separately')
        return self.summary


class gym_summary_handler(summary_handler):
    '''
        @brief:
            For the gym environment, we pass the stuff we want to record
    '''

    def __init__(self, sess, summary_name, enable=True,
                 scalar_var_list=dict(), summary_dir=None):
        super(self.__class__, self).__init__(sess, summary_name, enable=enable,
                                             summary_dir=summary_dir)
        if not self.enable:
            return
        assert type(scalar_var_list) == dict, logger.error(
            'We only take the dict where the name is given as the key')

        if len(scalar_var_list) > 0:
            self.summary_list = []
            for name, var in scalar_var_list.items():
                self.summary_list.append(tf.summary.scalar(name, var))
            self.summary = tf.summary.merge(self.summary_list)

    def manually_add_scalar_summary(self, summary_name, summary_value, x_axis):
        '''
            @brief:
                might be useful to record the average game_length, and average
                reward
            @input:
                x_axis could either be the episode number of step number
        '''
        if not self.enable:  # it happens when we are just debugging
            return

        if 'expert_traj' in summary_name:
            return

        summary = tf.Summary(
            value=[tf.Summary.Value(
                tag=summary_name, simple_value=summary_value
            ), ]
        )
        self.train_writer.add_summary(summary, x_axis)
