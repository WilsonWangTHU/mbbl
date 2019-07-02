# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

_ALLOW_KEY = ['state', 'diff_state', 'action']


def init_whitening_stats(key_list):
    whitening_stats = {}
    for key in key_list:
        whitening_stats[key] = {'mean': 0.0, 'variance': 1, 'step': 0.01,
                                'square_sum': 0.01, 'sum': 0.0, 'std': np.nan}
    return whitening_stats


def update_whitening_stats(whitening_stats, rollout_data, key):
    # collect the info
    new_sum, new_step_sum, new_sq_sum = 0.0, 0.0, 0.0

    for i_episode in rollout_data:
        if key == 'state':
            i_data = i_episode['obs']
        elif key == 'action':
            i_data = i_episode['actions']
        else:
            assert key == 'diff_state'
            i_data = i_episode['obs'][1:] - i_episode['obs'][:-1]

        new_sum += i_data.sum(axis=0)
        new_sq_sum += (np.square(i_data)).sum(axis=0)
        new_step_sum += i_data.shape[0]

    # update the whitening info
    whitening_stats[key]['step'] += new_step_sum
    whitening_stats[key]['sum'] += new_sum
    whitening_stats[key]['square_sum'] += new_sq_sum
    whitening_stats[key]['mean'] = \
        whitening_stats[key]['sum'] / whitening_stats[key]['step']
    whitening_stats[key]['variance'] = np.maximum(
        whitening_stats[key]['square_sum'] / whitening_stats[key]['step'] -
        np.square(whitening_stats[key]['mean']), 1e-2
    )
    whitening_stats[key]['std'] = \
        (whitening_stats[key]['variance'] + 1e-6) ** .5


def add_whitening_operator(whitening_operator, whitening_variable, name, size):

    with tf.variable_scope('whitening_' + name):
        whitening_operator[name + '_mean'] = tf.Variable(
            np.zeros([1, size], np.float32),
            name=name + "_mean", trainable=False
        )
        whitening_operator[name + '_std'] = tf.Variable(
            np.ones([1, size], np.float32),
            name=name + "_std", trainable=False
        )
        whitening_variable.append(whitening_operator[name + '_mean'])
        whitening_variable.append(whitening_operator[name + '_std'])

        # the reset placeholders
        whitening_operator[name + '_mean_ph'] = tf.placeholder(
            tf.float32, shape=(1, size), name=name + '_reset_mean_ph'
        )
        whitening_operator[name + '_std_ph'] = tf.placeholder(
            tf.float32, shape=(1, size), name=name + '_reset_std_ph'
        )

        # the tensorflow operators
        whitening_operator[name + '_mean_op'] = \
            whitening_operator[name + '_mean'].assign(
                whitening_operator[name + '_mean_ph']
        )

        whitening_operator[name + '_std_op'] = \
            whitening_operator[name + '_std'].assign(
                whitening_operator[name + '_std_ph']
        )


def set_whitening_var(session, whitening_operator, whitening_stats, key_list):

    for i_key in key_list:
        for i_item in ['mean', 'std']:
            session.run(
                whitening_operator[i_key + '_' + i_item + '_op'],
                feed_dict={whitening_operator[i_key + '_' + i_item + '_ph']:
                           np.reshape(whitening_stats[i_key][i_item], [1, -1])}
            )


def append_normalized_data_dict(data_dict, whitening_stats,
                                target=['start_state', 'diff_state',
                                        'end_state']):
    data_dict['n_start_state'] = \
        (data_dict['start_state'] - whitening_stats['state']['mean']) / \
        whitening_stats['state']['std']
    data_dict['n_end_state'] = \
        (data_dict['end_state'] - whitening_stats['state']['mean']) / \
        whitening_stats['state']['std']
    data_dict['n_diff_state'] = \
        (data_dict['end_state'] - data_dict['start_state'] -
         whitening_stats['diff_state']['mean']) / \
        whitening_stats['diff_state']['std']
    data_dict['diff_state'] = \
        data_dict['end_state'] - data_dict['start_state']
