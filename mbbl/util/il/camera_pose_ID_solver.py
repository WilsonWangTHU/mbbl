"""
    @brief: Solve the joints (states) via inverse dynmaics from a list of 2d
        frames
    @input:
        expert_2d_poses
        dynamics_model (inverse_dynamics, forward_dynamics)
    @output:
        qpos, camera_state
    @author: Tingwu Wang

    @Date:  Jan 15, 2019
"""
# from pyquaternion import Quaternion
import numpy as np
from mbbl.util.il.expert_data_util import load_pose_data
from mbbl.env import env_register
from mbbl.util.il.camera_model import camera_matrix_from_state
from mbbl.util.il.camera_model import camera_state_from_info
from mbbl.util.il.camera_model import get_projected_2dpose
from mbbl.util.il.pose_visualization import visualize_sol_pose
from mbbl.util.il.il_util import interpolate_qvel_qacc
# from mbbl.util.il.camera_model import xyaxis2quaternion
from mbbl.util.common import logger
from scipy import optimize


class solver(object):

    def __init__(self, data_path, sol_qpos_freq, opt_var_list,
                 camera_info, camera_id, optimize_args,
                 imitation_length=1000, gt_camera_info={}, log_path=''):
        """ @brief:
                @data_path: the path to the expert data npy file
                @opt_var_list: the variable to optimize
                    # OLD: ['qpos', 'xyz_pos', 'quaternion', 'fov', 'image_size']
                    # NEW: ['qpos', 'xyz_pos', 'cam_view', 'fov', 'image_size']

                    cam_view is the two angels, that are used to describe the
                    direction of the camera

                    If the var is not in @opt_var_list, then the value is
                    assumed to be given and will be read from @camera_info

                @camera_info: the dict to record the camera info. A typical
                    camera_info looks like this.
                    {'mode': 'trackcom',
                     'tracker_type': 'com',
                     'tracker': 'torso',
                     'xyz_pos': np.array([0, -3, 0]),
                     'xyaxis': np.array([1, 0, 0, 0, 0, -1]),
                     'image_size': 400,
                     'fov': 45}
        """

        self._optimize_args = optimize_args
        self._log_path = log_path
        self._gt_camera_info = gt_camera_info[camera_id]
        self._camera_info = camera_info[camera_id]
        self._sol_qpos_freq = sol_qpos_freq
        self._imitation_length = imitation_length
        self._iteration = 0

        # parse the optimization vars
        self._all_var_list = \
            ['qpos', 'xyz_pos', 'cam_view', 'fov', 'image_size']
        self._opt_var_list = opt_var_list
        self._known_var_list = [key for key in self._all_var_list
                                if key not in self._opt_var_list]

        self._load_target_data(data_path, camera_id)

        self._init_env_engine()
        self._init_data()

    def _init_env_engine(self):
        self._env_physics_engine = env_physics_engine(self._env_name)
        self._len_qpos = self._env_physics_engine._env.get_qpos_size()

    def _load_target_data(self, data_path, camera_id):
        self._data_path = data_path

        # self._target_data should be kept from being accessed!
        self._target_data, self._target_2d_poses, self._env_name, \
            self._frame_dt = load_pose_data(self._data_path, camera_id,
                                            self._imitation_length)

        self._num_target_frames = len(self._target_2d_poses)

    def _init_data(self):
        """ @brief:
                The optimizer needs to optimize the following variables:
                @qpos: the states (qpos, aka joint angles) of all the frames
                    [self._num_frames // self._sol_qpos_freq, len(env.qpos)]

                @camera_state: the parameters needed for the

                @self._sol: qpos + camera_state

                1. self._sol_qpos_frameid: record the frameid of each qpos in
                    the self._sol
                    The actual qpos is interpolated from the sol_qpos using
                    catmul-rom algorithm

                see mbbl/util/il/camera_model.py
        """
        assert self._camera_info['mode'] in ['trackcom', 'static', 'free']
        assert (self._num_target_frames - 1) % self._sol_qpos_freq == 0

        # qpos of the frames in between two sol_qpos will be interpolated
        self._sol_qpos_frameid = \
            np.arange(0, self._num_target_frames, self._sol_qpos_freq)
        self._num_sol_qpos = len(self._sol_qpos_frameid)

        # pos(3), cam_view(2), fov(1), image_size(1)
        if self._camera_info['mode'] in ['static', 'trackcom']:
            self._camera_state_size = 7
            qpos_sol_size = self._num_sol_qpos * self._len_qpos
            # self._sol = np.zeros([qpos_sol_size + self._camera_state_size])
            self._bounds, self._sol = \
                self._env_physics_engine.get_bounds_and_init_sol(
                    self._camera_info['mode'], self._num_sol_qpos,
                    self._len_qpos, self._camera_state_size
                )

            self._var_to_sol_id = {
                'qpos': np.array(range(0, qpos_sol_size)),
                'camera_state': np.array(range(qpos_sol_size,
                                               qpos_sol_size + 7)),
                # details
                'xyz_pos': np.array(range(qpos_sol_size, qpos_sol_size + 3)),
                'cam_view': np.array(range(qpos_sol_size + 3,
                                           qpos_sol_size + 5)),
                'fov': np.array(range(qpos_sol_size + 5,
                                      qpos_sol_size + 6)),
                'image_size': np.array(range(qpos_sol_size + 6,
                                       qpos_sol_size + 7)),
            }

            # assign the known variables
            for key in self._known_var_list:
                if key == 'qpos':
                    res = (self._imitation_length - 1) % self._sol_qpos_freq
                    assert res == 0
                    num_sol_qpos = \
                        (self._imitation_length - 1) // self._sol_qpos_freq + 1
                    sol_qpos_id = [i_pos * self._sol_qpos_freq
                                   for i_pos in range(num_sol_qpos)]
                    self._sol[self._var_to_sol_id['qpos']] = np.reshape(
                        self._target_data['qpos'][np.array(sol_qpos_id)], [-1]
                    )
                    print(self._sol[self._var_to_sol_id['qpos']])
                else:
                    self._sol[self._var_to_sol_id[key]] = self._camera_info[key]
                    print(self._sol[self._var_to_sol_id[key]])
                # if key == 'quaternion':
                # #### xyaxis is equivalent to quaternion
                # #### self._sol[self._var_to_sol_id['quaternion']] = \
                # #### xyaxis2quaternion(self._camera_info['xyaxis'])
                # else:

        elif self._camera_state_type == 'free':
            self._sol = np.zeros(
                [self._num_sol_qpos * self._len_qpos +
                 self._num_sol_qpos * self._camera_state_size]
            )

            qpos_sol_size = self._num_sol_qpos * self._len_qpos
            self._var_to_sol_id = {
                'qpos': np.array(range(0, qpos_sol_size)),
                'xyz_pos': np.array(range(qpos_sol_size, qpos_sol_size + 3)),
                'cam_view': np.array(range(qpos_sol_size + 3,
                                           qpos_sol_size + 5)),
                'fov': np.array(range(qpos_sol_size + 5, qpos_sol_size + 6)),
                'image_size': np.array(range(qpos_sol_size + 6,
                                             qpos_sol_size + 7)),
            }

            # TODO
            raise NotImplementedError

        else:
            raise NotImplementedError

    def _get_fd_gradient(self, sol):
        """ @brief: use finite_difference to calculate the gradient. Due to the
            locality of the solution space, we can use some small trick to speed
            up the gradient process
        """

        gradient = np.zeros([1, len(sol)])
        epsilon = 1e-3  # used for finite difference

        # get the base values, and the base interpolation values:
        center_data_dict = {'physics_loss': None, 'projection_loss': None}
        center_loss = self._loss_function(sol, fetch_data_dict=center_data_dict)
        sol_qpos = np.reshape(sol[self._var_to_sol_id['qpos']],
                              [-1, self._len_qpos])
        camera_state = sol[self._var_to_sol_id['camera_state']]

        if 'qpos' in self._opt_var_list:

            logger.info('Calculating the gradient of qpos')
            # utilize the local connectivity of the qpos
            # locate the id of the qpos
            for i_derivative in range(self._num_sol_qpos * self._len_qpos):
                sol_id = i_derivative + self._var_to_sol_id['qpos'][0]
                center_sol_qpos_id = i_derivative // self._len_qpos
                start_sol_qpos_id = max(center_sol_qpos_id - 3, 0)
                end_sol_qpos_id = min(center_sol_qpos_id + 3,
                                      self._num_sol_qpos - 1)

                # get everything within the range of [start_sol_qpos_id,
                # end_sol_qpos_id], take the forward finite difference step
                forward_sol_qpos = np.array(
                    sol_qpos[start_sol_qpos_id: end_sol_qpos_id + 1], copy=True
                )
                forward_sol_qpos[center_sol_qpos_id - start_sol_qpos_id,
                                 i_derivative % self._len_qpos] += epsilon

                forward_loss, forward_data_dict = \
                    self._loss_from_sol_qpos_camera_state(
                        forward_sol_qpos, camera_state,
                        center_sol_qpos_id=center_sol_qpos_id
                    )

                center_physics_loss = center_data_dict['physics_loss'][
                    start_sol_qpos_id * self._sol_qpos_freq:
                    end_sol_qpos_id * self._sol_qpos_freq
                ]
                center_projection_loss = center_data_dict['projection_loss'][
                    start_sol_qpos_id * self._sol_qpos_freq:
                    end_sol_qpos_id * self._sol_qpos_freq + 1
                ]

                # make sure the ids are matched
                assert len(forward_data_dict['physics_loss']) == \
                    len(center_physics_loss) and \
                    len(forward_data_dict['projection_loss']) == \
                    len(center_projection_loss)

                difference_of_loss = forward_loss - \
                    np.mean(center_physics_loss) - \
                    np.mean(center_projection_loss)

                gradient[0, sol_id] = difference_of_loss

        for opt_var in ['xyz_pos', 'cam_view', 'fov', 'image_size']:
            if opt_var not in self._opt_var_list:
                continue
            logger.info('Calculating the gradient of {}'.format(opt_var))

            # TODO: for xyz_pos / fov / image_size, there is speed-up available
            for i_derivative in range(len(self._var_to_sol_id[opt_var])):

                sol_id = i_derivative + self._var_to_sol_id[opt_var][0]
                camera_state_id = sol_id - len(self._var_to_sol_id['qpos'])
                forward_camera_state = np.array(camera_state, copy=True)
                if opt_var == 'cam_view':
                    # for quaternion, take care of the length invariance
                    quat_id = self._var_to_sol_id['quaternion']
                    raise NotImplementedError
                    forward_camera_state[camera_state_id] += \
                        epsilon * np.linalg.norm(sol[quat_id])
                else:
                    forward_camera_state[camera_state_id] += epsilon

                forward_loss, _ = self._loss_from_sol_qpos_camera_state(
                    sol_qpos, forward_camera_state
                )
                gradient[0, sol_id] = forward_loss - center_loss

        if len(self._opt_var_list) == 0:
            raise ValueError('At least one of the var needs to be optimzied')
        logger.info('Gradient calculated')

        return gradient

    def _loss_function(self, sol, fetch_data_dict={}):
        """ @brief: the loss function to be used by the LBFGS optimizer

            @fetch_data_dict:
                We can fetch some intermediate variables (interpolated qpos /
                qvel / qacc)
        """

        if self._camera_info['mode'] in ['static', 'trackcom']:
            # only the qposes
            sol_qpos = sol[self._var_to_sol_id['qpos']]
            sol_qpos = sol_qpos.reshape([-1, self._len_qpos])
            camera_state = sol[self._var_to_sol_id['camera_state']]
            total_loss, _fetch_data_dict = \
                self._loss_from_sol_qpos_camera_state(sol_qpos, camera_state)

        else:
            raise NotImplementedError  # TODO for free

        # gather the data that can be reused
        for key in fetch_data_dict:
            fetch_data_dict[key] = _fetch_data_dict[key]
        logger.info("Current loss: {}".format(total_loss))
        logger.info("\tphysics loss: {}".format(
            np.mean(_fetch_data_dict['physics_loss']))
        )
        logger.info("\tproject loss: {}".format(
            np.mean(_fetch_data_dict['projection_loss']))
        )
        return total_loss

    def generate_pose_3d(self, sub_iter=0):
        data_dict = {'gt': {}, 'sol': {}}
        # the optimized qpos and matrix
        sol_qpos = np.reshape(self._sol[self._var_to_sol_id['qpos']],
                              [-1, self._len_qpos])

        data_dict['sol']['camera_state'] = \
            self._sol[self._var_to_sol_id['camera_state']]
        data_dict['sol']['mode'] = self._camera_info['mode']
        _, _, data_dict['sol']['qpos'] = interpolate_qvel_qacc(
            sol_qpos, self._len_qpos, self._sol_qpos_freq, self._frame_dt
        )

        # the groundtruth qpos and camera_matrix
        data_dict['gt']['qpos'] = \
            self._target_data['qpos'][:self._imitation_length]
        data_dict['gt']['mode'] = self._gt_camera_info['mode']

        data_dict['gt']['camera_state'] = \
            camera_state_from_info(self._gt_camera_info,
                                   consider_trackcom=False,
                                   use_quaternion=False)
        # visualize and save the results
        visualize_sol_pose(self._env_physics_engine, self._log_path,
                           data_dict, self._env_name, self._iteration, sub_iter)

    def solve(self):
        self._iteration += 1
        assert self._camera_info['mode'] in ['static', 'trackcom']
        bounds = np.transpose(np.array(self._bounds, copy=True), [1, 0])

        """
        optimized_results = optimize.minimize(
            self._loss_function, self._sol, method='L-BFGS-B',
            jac=self._get_fd_gradient, bounds=bounds,
        )
        """
        self.generate_pose_3d()  # visualize results
        for sub_iter in range(self._optimize_args['lbfgs_opt_iteration']):
            optimized_results = optimize.minimize(
                self._loss_function, self._sol, method='L-BFGS-B',
                bounds=bounds
            )
            self._sol = optimized_results['x']

            self.generate_pose_3d(sub_iter + 1)  # visualize results

        return self._sol

    def _loss_from_sol_qpos_camera_state(self, sol_qpos, camera_state,
                                         center_sol_qpos_id=-1):

        # get the qvel and qpos
        qvel, qacc, qpos = interpolate_qvel_qacc(
            sol_qpos, self._len_qpos, self._sol_qpos_freq, self._frame_dt
        )

        # calculate the physics loss (action loss)
        qfrc_inverse = \
            self._env_physics_engine.get_inverse_action(qpos, qvel, qacc)
        physics_loss = \
            self._env_physics_engine.get_physics_loss(qfrc_inverse)

        # the projection loss
        pose_3d, center_of_mass = self._env_physics_engine.get_pose3d(
            qpos,
            get_center_of_mass=(self._camera_info['mode'] == 'trackcom')
        )
        matrix = self._env_physics_engine.camera_matrix_from_state(
            camera_state, center_of_mass
        )
        projected_pose = \
            self._env_physics_engine.get_projected_2dpose(pose_3d, matrix)
        if center_sol_qpos_id < 0:
            target_2d_pose = self._target_2d_poses
        else:
            start_frame_id = (center_sol_qpos_id - 3) * self._sol_qpos_freq
            start_frame_id = max(start_frame_id, 0)
            target_2d_pose = self._target_2d_poses[
                start_frame_id: start_frame_id + len(projected_pose)
            ]
        projection_loss = np.square(projected_pose - target_2d_pose)
        var = np.var(target_2d_pose, axis=0)
        projection_loss /= (var[None, :, :] * 0.0 + 1.0)  # NOTE: TODO
        # projection_loss /= var[None, :, :]

        # TODO: how to normalize the loss?
        total_loss = np.mean(projection_loss) + \
            self._optimize_args['physics_loss_lambda'] * np.mean(physics_loss)
        fetch_data_dict = {
            'qvel': qvel, 'qacc': qacc, 'qpos': qpos,
            'qfrc_inverse': qfrc_inverse,
            'pose_2d': projected_pose, 'pose_3d': pose_3d,
            'physics_loss': physics_loss, 'projection_loss': projection_loss
        }

        return total_loss, fetch_data_dict


class env_physics_engine(object):

    def __init__(self, env_name):
        self._env_name = env_name

        self._env, self._env_info = env_register.make_env(self._env_name, 1234)
        self._joint_range = \
            np.array(self._env._controller_info['range']) / 180.0 * np.pi
        self._len_qpos = self._env.get_qpos_size()
        self._control_info = self._env.get_controller_info()

    def get_pose3d(self, qpos, get_center_of_mass=False):
        # loop over each qpos candidates
        assert qpos.shape[1] == self._len_qpos

        num_data = qpos.shape[0]
        pos_shape = self._env.get_pos_size()  # [num of pose, 3]
        poses_3d = np.zeros([num_data, pos_shape[0], pos_shape[1]])
        if get_center_of_mass:
            center_of_mass = np.zeros([num_data, 3])
        else:
            center_of_mass = None

        for i_data in range(num_data):

            # set the qpos
            with self._env._env.physics.reset_context():
                self._env._env.physics.data.qpos[:] = qpos[i_data]

            # get the poses3d
            poses_3d[i_data, :, :] = self._env.get_pos()

            if get_center_of_mass:
                center_of_mass[i_data, :] = np.array(
                    self._env._env.physics.named.data.subtree_com['torso'],
                    copy=True
                )

        return poses_3d, center_of_mass

    def get_inverse_action(self, qpos, qvel, qacc):
        num_data = qpos.shape[0]
        assert num_data == qvel.shape[0] and num_data == qacc.shape[0]
        # TODO: not necessary true
        assert qpos.shape[1] == qvel.shape[1] and qpos.shape[1] == qacc.shape[1]

        # qfrc_inverse = np.zeros([num_data, self._env_info['action_size']])
        qfrc_inverse = np.zeros([num_data - 1, self._control_info['dof']])

        for i_data in range(num_data - 1):

            # set the qpos TODO: TEST THIS FUNCTION
            inverse_output = self._env._env.physics.get_inverse_output(
                qpos[i_data], qvel[i_data], qacc[i_data]
            )

            # get the poses3d
            qfrc_inverse[i_data, :] = inverse_output

        return qfrc_inverse

    def get_physics_loss(self, qfrc_inverse):
        # loss = 0.0
        # the forces on the free joints, TODO: contact model?
        # loss += np.square(actions[:, self._control_info['unactuated_id']])
        # actions = qfrc_inverse[]

        # the forces applied on the joints
        action = np.abs(
            qfrc_inverse[:, self._control_info['actuated_id']] /
            self._control_info['gear'][None, :]
        )
        gear_violation = np.maximum(action - 1.0, 0.0)
        # gear_violation = -self._control_info['gear'][None, :] + \
        # np.abs(actions[:, self._control_info['actuated_id']])
        losses = np.square(gear_violation)

        if len(losses) == 0:
            losses = np.array([0])

        return losses

    def get_center_of_mass(self, qpos):
        # loop over each qpos candidates
        assert qpos.shape[1] == self._len_qpos

        num_data = qpos.shape[0]
        pos_shape = self._env.get_pos_size()  # [num of pose, 3]
        poses_3d = np.zeros([num_data, pos_shape[0], pos_shape[1]])

        for i_data in range(num_data):

            # set the qpos
            with self._env._env.physics.reset_context():
                self._env._env.physics.data.qpos[:] = qpos[i_data]

            # get the poses3d
            poses_3d[i_data, :, :] = self._env.get_pos()

        return poses_3d

    def camera_matrix_from_state(self, camera_state, center_of_mass=None):
        """ @brief: it is different from
                mbbl.util.il.camera_model.camera_matrix_from_state
                in that it operates on batched data
        """
        if center_of_mass is None:
            num_data = 1
            camera_state = camera_state.reshape([1, -1])
            center_of_mass = 0.0 * camera_state[:3]
        else:
            num_data = center_of_mass.shape[0]
            camera_state = np.tile(camera_state.reshape([1, -1]), [num_data, 1])

        batched_camera_matrix = np.zeros([num_data, 4, 4])
        com_offset = np.zeros(camera_state.shape[1])
        for i_data in range(num_data):
            # for static camera, this term is 0
            com_offset[:3] = center_of_mass[i_data]
            # import pdb; pdb.set_trace()
            i_state = camera_state[i_data] + com_offset
            batched_camera_matrix[i_data] = \
                camera_matrix_from_state(i_state, use_quaternion=False)

        return batched_camera_matrix

    def get_projected_2dpose(self, poses_3d, matrix):
        num_data = poses_3d.shape[0]
        assert matrix.shape[0] == num_data

        poses_2d = np.zeros([num_data, poses_3d.shape[1], 2])

        for i_data in range(num_data):
            i_matrix = matrix[i_data] if len(matrix) == num_data else matrix[0]
            poses_2d[i_data] = \
                get_projected_2dpose(poses_3d[i_data], i_matrix)
        return poses_2d

    def get_bounds_and_init_sol(self, camera_mode, num_sol_qpos, len_qpos,
                                camera_state_size):
        if camera_mode in ['static', 'trackcom']:
            qpos_sol_size = num_sol_qpos * len_qpos
            self._bounds = np.zeros([2, qpos_sol_size + camera_state_size])
            self._init_sol = np.zeros([qpos_sol_size + camera_state_size])
            self._env.reset()
            init_qpos = self._env._env.physics.data.qpos

            # the bounds and init value for the qpos
            self._bounds[:, :qpos_sol_size] = np.tile(
                np.transpose(self._joint_range, [1, 0]), [1, num_sol_qpos]
            )
            self._init_sol[:qpos_sol_size] = np.tile(init_qpos, [num_sol_qpos])

            # the bounds for camera_state
            assert camera_state_size == 7
            self._bounds[:, qpos_sol_size:] = np.transpose(
                np.array([[-5, 5], [-5, 5], [-5, 5],  # xyz_pos
                          [-31.4, 31.4], [-31.4, 31.4],  # is it ok?
                          # fov and image_size
                          [30, 60], [300, 500]])
            )
            self._init_sol[qpos_sol_size:] = \
                np.array([0, -2, 0, 1.57, 1.57, 45, 400])
        else:
            raise NotImplementedError

        return self._bounds, np.array(self._init_sol, copy=True)
