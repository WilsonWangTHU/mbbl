from gym.envs.registration import register


register(
    id='MBRLCartpole-v0',
    entry_point='mbbl.env.gym_env.pets_env.cartpole:CartpoleEnv'
)


register(
    id='MBRLReacher3D-v0',
    entry_point='mbbl.env.gym_env.pets_env.reacher:Reacher3DEnv'
)


register(
    id='MBRLPusher-v0',
    entry_point='mbbl.env.gym_env.pets_env.pusher:PusherEnv'
)


register(
    id='MBRLHalfCheetah-v0',
    entry_point='mbbl.env.gym_env.pets_env.half_cheetah:HalfCheetahEnv'
)
