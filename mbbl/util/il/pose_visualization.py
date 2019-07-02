"""
    @brief: visualize the results
    @author: Tingwu Wang

    @Date:  Jan 12, 2019
"""
import matplotlib
import matplotlib.pyplot as plt
from mbbl.util.il.expert_data_util import load_pose_data
from mbbl.env.dm_env.pos_dm_env import POS_CONNECTION
from mbbl.env.env_register import make_env
from mbbl.util.common import logger
from mbbl.config import init_path
import numpy as np
import cv2
import os
from skimage import draw

# the data structure for camera model
_DEFAULT_FOV = 45
_DEFAULT_IAMGE_SIZE = 400


def visualize_sol_pose(physics_engine, output_dir, data_dict, env_name,
                       iteration, sub_iter):
    """ @brief: visualize the following four images

        0. the rendered image from dm_control
        1. the image using the qpos + camera_state (trained)
        2. the image using the qpos + gt_camera_state
        3. the image using the (gt_qpos + gt_camera_state)
    """
    logger.info("generating the visualization")
    image_size = int(data_dict['gt']['camera_state'][-1])
    assert image_size == int(data_dict['gt']['camera_state'][-1])  # TODO

    # from camera and qposes to the 2d poses
    for qpos_key, camera_state_key in \
            [['gt', 'gt'], ['sol', 'sol'], ['sol', 'gt']]:
        pose_2d_key = qpos_key + "-" + camera_state_key
        data_dict[pose_2d_key] = {}  # save pose_2d

        is_trackcom = data_dict[camera_state_key]['mode'] == 'trackcom'
        pose_3d, center_of_mass = physics_engine.get_pose3d(
            data_dict[qpos_key]['qpos'],
            get_center_of_mass=is_trackcom
        )
        matrix = physics_engine.camera_matrix_from_state(
            data_dict[camera_state_key]['camera_state'], center_of_mass
        )
        data_dict[pose_2d_key]['pose_2d'] = \
            physics_engine.get_projected_2dpose(pose_3d, matrix)
        data_dict[pose_2d_key]['image_size'] = \
            data_dict[camera_state_key]['camera_state'][-1]

    pos_connection = POS_CONNECTION[env_name]

    # the output directory
    directory = os.path.join(output_dir, "video")
    if not os.path.exists(directory):
        os.mkdir(directory)
    output_dir = os.path.join(directory, "pos_Iter_" + str(iteration) +
                              '_sub_' + str(sub_iter) + '.mp4')
    video = cv2.VideoWriter(
        os.path.join(init_path.get_abs_base_dir(), output_dir),
        cv2.VideoWriter_fourcc(*'mp4v'), 40, (image_size * 4, image_size)
    )

    for i_pos_id in range(len(data_dict['gt']['qpos'])):
        # render the image using the default renderer
        render_image = physics_engine._env.render(
            camera_id=0, qpos=data_dict['gt']['qpos'][i_pos_id]
        )

        # the sol_qpos + sol_camera_state
        sol_sol_image = draw_pose3d(render_image * 0.0,
                                    data_dict['sol-sol']['pose_2d'][i_pos_id],
                                    pos_connection)
        # the sol_qpos + gt_camera_state
        sol_gt_image = draw_pose3d(render_image * 0.0,
                                   data_dict['sol-gt']['pose_2d'][i_pos_id],
                                   pos_connection)
        # the gt_qpos + gt_camera_state
        gt_gt_image = draw_pose3d(render_image * 0.0,
                                  data_dict['gt-gt']['pose_2d'][i_pos_id],
                                  pos_connection)

        image = \
            np.hstack([render_image, sol_sol_image, sol_gt_image, gt_gt_image])
        # import pdb; pdb.set_trace()

        print('Processing %d out of %d' %
              (i_pos_id, len(data_dict['gt']['qpos'])))
        video.write(np.array(image[:, :, [2, 1, 0]], dtype=np.uint8))

    video.release()


def draw_pose3d(image, pose_2d, pos_connection):

    pose_cmap = matplotlib.cm.get_cmap('Spectral')
    line_cmap = matplotlib.cm.get_cmap('Spectral')
    num_pos = len(pose_2d)

    # the circles at the pose
    for i_pos in range(num_pos):
        rr, cc = draw.circle(pose_2d[i_pos, 0], pose_2d[i_pos, 1], 5)
        image[rr, cc] = list(pose_cmap(float(i_pos) / num_pos))[:3]

    i_connection = 0
    for start, end in pos_connection:
        i_connection += 1
        rr, cc, var = draw.line_aa(
            int(pose_2d[start, 0]), int(pose_2d[start, 1]),
            int(pose_2d[end, 0]), int(pose_2d[end, 1])
        )
        color = np.array(line_cmap(float(i_connection) / len(pos_connection)))
        image[rr, cc] = np.reshape(color[:3], [1, 3]) * np.reshape(var, [-1, 1])
        # plt.plot(pose_2d[segements, 0], pose_2d[segements, 1])
    return np.transpose(image * 255.0, [1, 0, 2])


def visualize_pose_from_expert_data(data_file, camera_id):
    expert_traj, pos_data, env_name, dt = load_pose_data(data_file, camera_id)
    pos_connection = POS_CONNECTION[env_name]

    # import pdb; pdb.set_trace()
    image_size = expert_traj[camera_id]['camera_info'][camera_id]['image_size']
    image = np.zeros([image_size, image_size, 3], dtype=np.uint8)
    env, _ = make_env(env_name, 1234, {})

    fig = plt.figure()
    for i_pos_id in range(100):
        i_pos_data = pos_data[i_pos_id]

        # render the image
        image = env.render(camera_id=camera_id,
                           qpos=expert_traj[0]['qpos'][i_pos_id])

        fig = plt.figure()
        visualize_pose(image, i_pos_data, pos_connection, show=False)
        fig.canvas.draw()
        plt_results = np.array(fig.canvas.renderer._renderer)
        print('Processing %d out of %d' % (i_pos_id, 100))
        if i_pos_id == 0:
            width, height, _ = plt_results.shape
            output_dir = \
                data_file.replace('.npy', '_' + str(camera_id) + '.mp4')
            video = cv2.VideoWriter(
                os.path.join(init_path.get_abs_base_dir(), output_dir),
                cv2.VideoWriter_fourcc(*'mp4v'), 40, (height, width)
            )
        plt.imshow(plt_results)
        video.write(plt_results[:, :, [2, 1, 0]])
        plt.close()

    video.release()


def visualize_pose(image, pose_2d, pos_connection, show=True):

    plt.subplot(121)
    plt.imshow(image)

    plt.subplot(122)
    plt.imshow(image * 0.0)
    plt.plot(pose_2d[:, 0], pose_2d[:, 1], 'o', markersize=4)

    for segements in pos_connection:
        plt.plot(pose_2d[segements, 0], pose_2d[segements, 1])

    if show:
        plt.show()


if __name__ == '__main__':
    visualize_pose_from_expert_data(
        'data/cheetah-run-pos_2019_01_21-15:08:27.npy', 0
    )
