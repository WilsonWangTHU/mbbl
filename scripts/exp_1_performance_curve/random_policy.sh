# see how the ppo works for the environments in terms of the performance:
# batch size
# max_timesteps 1e8
# for env_type in gym_reacher gym_cheetah gym_walker2d gym_hopper gym_swimmer gym_ant; do


# for env_type in gym_reacher gym_cheetah gym_walker2d gym_hopper gym_swimmer gym_ant; do
for env_type in gym_reacher gym_cheetah gym_walker2d gym_hopper gym_swimmer gym_ant gym_pendulum gym_invertedPendulum gym_acrobot gym_mountain gym_cartpole; do
    python main/random_main.py --exp_id random_${env_type} \
        --timesteps_per_batch 1 --task $env_type \
        --num_workers 1 --max_timesteps 40000
done
