"""
    @brief: camera model utility functions
    @author: Tingwu Wang & Hang Chu

    @Date:  Jan 12, 2019
"""
from pyquaternion import Quaternion
import numpy as np

# the data structure for camera model
_DEFAULT_FOV = 45
_DEFAULT_IAMGE_SIZE = 400


def parse_camera_state(state, use_quaternion=False):
    """ @brief: This function maps the state vector
        @input:
            @state: a 1D numpy.array, 7 <= size <= 9

        @output:
            @fov: float
            @image_size: float
            @camera_pos: numpy.array of size (3,)

            @quaternion: numpy.array of size (4,)
            or
            @cam_view: numpy.array of size (2,)

    """

    if use_quaternion:
        assert state.shape[0] == 9
        camera_pos = state[0: 3]
        quaternion = state[3: 7]
        fov = state[7]
        image_size = state[8]
        return fov, image_size, camera_pos, quaternion
    else:
        assert state.shape[0] == 7
        camera_pos = state[0: 3]
        cam_view = state[3: 5]
        fov = state[5]
        image_size = state[6]
        return fov, image_size, camera_pos, cam_view


def camera_state_from_info(camera_info_dict, physics_engine=None,
                           consider_trackcom=True, use_quaternion=False):

    if use_quaternion:
        camera_state = np.zeros(9)
        camera_state[7] = camera_info_dict['fov']
        camera_state[8] = camera_info_dict['image_size']
        if 'xyaxis' in camera_info_dict:
            camera_state[3: 7] = xyaxis2quaternion(camera_info_dict['xyaxis'])
        else:
            assert 'quaternion' in camera_info_dict
            camera_state[3: 7] = camera_info_dict['quaternion']
    else:
        camera_state = np.zeros(7)
        camera_state[5] = camera_info_dict['fov']
        camera_state[6] = camera_info_dict['image_size']
        camera_state[3: 5] = camera_info_dict['cam_view']

    camera_type = camera_info_dict['mode']
    if camera_type == 'trackcom' and consider_trackcom:
        assert physics_engine is not None and 'tracker' in camera_info_dict
        tracker = camera_info_dict['tracker']

        reference_pos = physics_engine.named.data.subtree_com[tracker].copy()
        pos = reference_pos + camera_info_dict['xyz_pos']
        camera_state[:3] = pos
    else:
        camera_state[:3] = camera_info_dict['xyz_pos']

    return camera_state


def camera_matrix_from_state(state, use_quaternion=False):
    if use_quaternion:
        fov, image_size, camera_pos, quaternion = \
            parse_camera_state(state, use_quaternion)

        # camera rotation matrix
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = Quaternion(
            quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        ).rotation_matrix.T

    else:
        fov, image_size, camera_pos, cam_view = \
            parse_camera_state(state, use_quaternion)

        # camera rotation matrix
        x_axis = np.array([np.cos(cam_view[0]), np.sin(cam_view[0]), 0])
        y_axis = np.array([-np.sin(cam_view[1]), -np.cos(cam_view[1]), 0])
        rotation_matrix = xyaxis2rotation(x_axis, y_axis)

    # camera shift matrix
    cam_pos_matrix = np.eye(4)
    cam_pos_matrix[:3, 3] = -camera_pos

    # camera projection matrix
    projection_matrix = np.eye(4)
    f = 1.0 / np.tan(fov / 2.0 * np.pi / 180)
    projection_matrix[0, 0] = f
    projection_matrix[1, 1] = f
    projection_matrix[3, 2] = 1.0
    projection_matrix[3, 3] = 0.0

    # viewport matrix
    viewport_matrix = np.eye(4)
    half_size = image_size / 2.0
    viewport_matrix[0, 0] = half_size  # x direction
    viewport_matrix[0, 3] = half_size
    viewport_matrix[1, 1] = half_size  # y direction
    viewport_matrix[1, 3] = half_size

    # assemble the camera matrix
    camera_matrix = cam_pos_matrix
    camera_matrix = rotation_matrix.dot(camera_matrix)
    camera_matrix = projection_matrix.dot(camera_matrix)
    camera_matrix = viewport_matrix.dot(camera_matrix)

    return camera_matrix


def get_projected_2dpose(pose, camera_matrix):
    """ @input:
            @camera_matrix: [4 x 4] np.array
            @pose: [num_pos x 3] np.array
    """
    pose_shape = pose.shape
    assert len(pose_shape) == 2

    homogeneous_pose = np.ones([pose_shape[0], 4])
    homogeneous_pose[:, :3] = pose

    projected_pose = homogeneous_pose.dot(camera_matrix.transpose())
    projected_2d_pose = projected_pose[:, :2] / projected_pose[:, [3]]

    return projected_2d_pose


def xyaxis2rotation(x_axis, y_axis):
    rotation = np.zeros([4, 4])
    rotation[3, 3] = 1.0
    xaxis = x_axis / np.linalg.norm(x_axis)
    yaxis = y_axis / np.linalg.norm(y_axis)
    zaxis = np.cross(xaxis, yaxis)
    rotation[:3, 0] = xaxis
    rotation[:3, 1] = yaxis
    rotation[:3, 2] = zaxis
    return rotation


def xyaxis2quaternion(xyaxis, to_np_array=True):
    rotation = np.zeros([3, 3])
    xaxis = xyaxis[:3] / np.linalg.norm(xyaxis[:3])
    yaxis = xyaxis[3:] / np.linalg.norm(xyaxis[3:])
    zaxis = np.cross(xaxis, yaxis)
    rotation[:3, 0] = xaxis
    rotation[:3, 1] = yaxis
    rotation[:3, 2] = zaxis

    if to_np_array:
        quat = Quaternion(matrix=rotation)
        return np.array([quat[0], quat[1], quat[2], quat[3]])
    else:
        return Quaternion(matrix=rotation)


if __name__ == '__main__':

    test_pose = np.array([[0., 0., 0.]])

    xyaxis = np.array([1.0, -1.0, 0., 0., 0., -1])
    quat = xyaxis2quaternion(xyaxis)
    state = np.zeros([7])
    state[:3] = np.array([0, -3, 0])
    state[3:] = quat

    camera_matrix = camera_matrix_from_state(state)

    pose_2d = get_projected_2dpose(test_pose, camera_matrix)
    print(pose_2d)
