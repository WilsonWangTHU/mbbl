# -----------------------------------------------------------------------------
#   @author:
#       Matthew Zhang
#   @brief:
#       Alternate code-up of the box2d lunar lander environment from gym
# -----------------------------------------------------------------------------
from mbbl.env import base_env_wrapper as bew
import mbbl.env.gym_env.box2d.wrappers

Racer = mbbl.env.gym_env.box2d.wrappers.RacerWrapper

class env(bew.base_env):
    pass
