# see how the ppo works for the environments in terms of the performance:
# batch size
# max_timesteps 1e8


for batch_size in 2000 5000; do
    # for env_type in gym_reacher gym_cheetah gym_walker2d gym_hopper gym_swimmer gym_ant; do
    for env_type in gym_pendulum gym_invertedPendulum gym_acrobot gym_mountain gym_cartpole; do
        for seed in 1234 2341 3412 4123; do
            python main/mf_main.py --exp_id trpo_${env_type}_batch_${batch_size}_seed_${seed} \
                --timesteps_per_batch $batch_size --task $env_type \
                --num_workers 5 --trust_region_method trpo --max_timesteps 1000000
        done
    done
done
