# -----------------------------------------------------------------------------
#   @brief:
# -----------------------------------------------------------------------------
from mbbl.config import init_path

_BASE_DIR = init_path.get_abs_base_dir()


def update_damping_lambda(traj_data, increase, damping_args):
    if increase:
        traj_data['lambda_multiplier'] = max(
            traj_data['lambda_multiplier'] * damping_args['factor'],
            damping_args['factor']
        )
        traj_data['damping_lambda'] = max(
            traj_data['damping_lambda'] * traj_data['lambda_multiplier'],
            damping_args['min_damping']
        )
    else:  # decrease
        traj_data['lambda_multiplier'] = min(
            traj_data['lambda_multiplier'] / damping_args['factor'],
            1.0 / damping_args['factor']
        )
        traj_data['damping_lambda'] = \
            traj_data['damping_lambda'] * traj_data['lambda_multiplier'] * \
            (traj_data['damping_lambda'] > damping_args['min_damping'])

    traj_data['damping_lambda'] = \
        max(traj_data['damping_lambda'], damping_args['min_damping'])
    if traj_data['damping_lambda'] > damping_args['max_damping']:
        traj_data['active'] = False
