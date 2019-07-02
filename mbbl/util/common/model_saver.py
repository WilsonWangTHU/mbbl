# -----------------------------------------------------------------------------
#   @brief: it is out-dated
#       in this model, we save the weights by our own function
#       written by Tingwu Wang
# -----------------------------------------------------------------------------
import os

import numpy as np

from mbbl.config import init_path
from mbbl.util.common import logger
init_path.bypass_frost_warning()


def save_tf_model(sess, model_path, tf_var_list=[]):
    '''
        @brief: save the tensorflow variables into a numpy npy file
    '''
    is_file_valid(model_path, save_file=True)
    logger.info('\tSAVING tensorflow variables')

    # get the tf weights one by one
    output_save_list = dict()
    for var in tf_var_list:
        weights = sess.run(var)
        output_save_list[var.name] = weights
        logger.info('\t\t[Checkpoint] saving tf parameter {}'.format(var.name))

    # save the model
    np.save(model_path, output_save_list)


def save_numpy_model(model_path, numpy_var_list=[]):
    '''
        @brief: save the numpy variables into a numpy npy file
    '''
    is_file_valid(model_path, save_file=True)

    logger.info('\tSAVING numpy variables')

    # get the numpy weights one by one
    output_save_list = dict()
    for key, var in numpy_var_list.items():
        output_save_list[key] = var
        logger.info('\t\t[Checkpoint] saving numpy parameter {}'.format(key))

    # save the model
    np.save(model_path, output_save_list)


def load_tf_model(sess, model_path, tf_var_list=[], ignore_prefix='INVALID'):
    '''
        @brief: load the tensorflow variables from a numpy npy files
    '''
    is_file_valid(model_path)
    logger.info('\tLOADING tensorflow variables')

    # load the parameters
    output_save_list = np.load(model_path, encoding='latin1').item()
    tf_name_list = [var.name for var in tf_var_list]

    # get the weights one by one
    for name, val in output_save_list.items():
        if name in tf_name_list:
            logger.info('\t\tloading TF pretrained parameters {}'.format(name))
            tf_name_list.remove(name)  # just for sanity check

            # pick up the variable that has the name
            var = [var for var in tf_var_list if var.name == name][0]

            assign_op = var.assign(val)
            sess.run(assign_op)  # or `assign_op.op.run()`
        else:
            logger.warning('\t\t**** Parameters Not Exist **** {}'.format(name))

    if len(tf_name_list) > 0:
        logger.warning(
            'Some parameters are not load from the checkpoint: {}'.format(
                tf_name_list
            )
        )


def load_numpy_model(model_path, numpy_var_list={}):
    '''
        @brief: load numpy variables from npy files. The variables could be
            from baseline or from ob_normalizer
        @output:
            It is worth mentioning that this function only returns the value,
            but won't load the value (while the tf variables will be loaded at
            the same time)
    '''
    is_file_valid(model_path)
    logger.info('LOADING numpy variables')

    output_save_list = np.load(model_path, encoding='latin1').item()
    numpy_name_list = [key for key, val in numpy_var_list.items()]

    # get the weights one by one
    for name, val in output_save_list.items():
        if name in numpy_name_list:
            logger.info(
                '\t\tloading numpy pretrained parameters {}'.format(name))
            numpy_name_list.remove(name)  # just for sanity check
            numpy_var_list[name] = val
        else:
            logger.warning('\t\t**** Parameters Not Exist **** {}'.format(name))

    if len(numpy_name_list) > 0:
        logger.warning(
            'Some parameters are not load from the checkpoint: {}'.format(
                numpy_name_list))
    return numpy_var_list


'''
    @brief:
        The following variables might be a little bit of out-dated
'''


def model_save_from_list(sess, model_path, tf_var_list=[], numpy_var_list={}):
    '''
        @brief:
            if the var list is given, we just save them
    '''
    if not model_path.endswith('.npy'):
        model_path = model_path + '.npy'

    logger.info('saving checkpoint to {}'.format(model_path))
    output_save_list = dict()

    # get the tf weights one by one
    for var in tf_var_list:
        weights = sess.run(var)
        output_save_list[var.name] = weights
        logger.info('[checkpoint] saving tf parameter {}'.format(var.name))

    # get the numpy weights one by one
    for key, var in numpy_var_list.items():
        output_save_list[key] = var
        logger.info('[checkpoint] saving numpy parameter {}'.format(key))

    # save the model
    np.save(model_path, output_save_list)

    return


def model_load_from_list(sess, model_path, tf_var_list=[], numpy_var_list={},
                         target_scope_switch='trpo_agent_policy'):
    '''
        @brief:
            if the var list is given, we just save them
        @input:
            @target_scope_switch:
    '''
    if not model_path.endswith('.npy'):
        model_path = model_path + '.npy'
        logger.warning('[checkpoint] adding the ".npy" to the path name')
    logger.info('[checkpoint] loading checkpoint from {}'.format(model_path))

    output_save_list = np.load(model_path, encoding='latin1').item()
    tf_name_list = [var.name for var in tf_var_list]
    numpy_name_list = [key for key, val in numpy_var_list.items()]

    # get the weights one by one
    for name, val in output_save_list.items():
        name = name.replace('trpo_agent_policy', target_scope_switch)
        if name not in tf_name_list and name not in numpy_var_list:
            logger.info('**** Parameters Not Exist **** {}'.format(name))
            continue
        elif name in tf_name_list:
            logger.info('loading TF pretrained parameters {}'.format(name))
            tf_name_list.remove(name)  # just for sanity check

            # pick up the variable that has the name
            var = [var for var in tf_var_list if var.name == name][0]
            assign_op = var.assign(val)
            sess.run(assign_op)  # or `assign_op.op.run()`
        else:
            logger.info('loading numpy pretrained parameters {}'.format(name))
            numpy_name_list.remove(name)  # just for sanity check

            # pick up the variable that has the name
            numpy_var_list[name] = val

    if len(tf_name_list) or len(numpy_name_list) > 0:
        logger.warning(
            'Some parameters are not load from the checkpoint: {}\n {}'.format(
                tf_name_list, numpy_name_list))
    return numpy_var_list


def is_file_valid(model_path, save_file=False):
    assert model_path.endswith('.npy'), logger.error(
        'Invalid file provided {}'.format(model_path))
    if not save_file:
        assert os.path.exists(model_path), logger.error(
            'file not found: {}'.format(model_path))
    logger.info('[LOAD/SAVE] checkpoint path is {}'.format(model_path))
