# see how the ppo works for the environments in terms of the performance:
# batch size
# max_timesteps 1e8

# bash gym_reacher 2000
# bash gym_cheetah 20000
# bash gym_walker2d 20000
# bash gym_hopper 20000
# bash gym_swimmer 20000
# bash gym_ant 20000
# bash gym_pendulum 5000
# bash gym_invertedPendulum 2500
# bash gym_acrobot 5000
# bash gym_mountain 5000
# bash gym_cartpole 5000


# bash gym_reacher 2000 30
# bash gym_cheetah 20000 100
# bash gym_walker2d 20000 50
# bash gym_hopper 20000 100
# bash gym_swimmer 20000 50
# bash gym_ant 20000 30
# bash gym_pendulum 5000 30
# bash gym_invertedPendulum 2500 30
# bash gym_acrobot 5000 30
# bash gym_mountain 5000 30
# bash gym_cartpole 5000 30

# for env_type in gym_reacher gym_cheetah gym_walker2d gym_hopper gym_swimmer gym_ant; do
# for env_type in gym_pendulum gym_invertedPendulum gym_acrobot gym_mountain gym_cartpole; do
for env_type in $1; do
    python main/pets_main.py --exp_id rs_${env_type}\
        --task $env_type \
        --num_planning_traj 500 --planning_depth $2 --random_timesteps 0 \
        --timesteps_per_batch 1 --num_workers 10 --max_timesteps 20000 \
        --gt_dynamics 1
done

# python2 main/pets_main.py --timesteps_per_batch 2000 --task gym_cheetah --num_workers 5 --planning_depth 15 --num_planning_traj 500
