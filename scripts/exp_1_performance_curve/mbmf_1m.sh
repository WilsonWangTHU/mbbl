# see how the ppo works for the environments in terms of the performance:
# batch size
# max_timesteps 1e8
# gym_reacher gym_cheetah gym_walker2d gym_hopper gym_swimmer gym_ant

for batch_size in 2000; do
    for env_type in $1; do
        for seed in 1234 2341 3412 4123; do
            python main/mbmf_main.py --exp_id mbmf_${env_type}_seed_${seed}_1m \
                --task $env_type \
                --num_planning_traj 5000 --planning_depth 20 --random_timesteps 10000 \
                --timesteps_per_batch $batch_size --dynamics_epochs 30 --num_workers 10 \
                --mb_timesteps 70000 --dagger_epoch 300 --dagger_timesteps_per_iter 1750 \
                --trust_region_method ppo \
                --max_timesteps 1000000 --seed $seed --dynamics_batch_size 500
        done
    done
done
