# see how the ppo works for the environments in terms of the performance:
# batch size
# max_timesteps 1e8

for seed in 1234 2341 3412 4123; do
    for env_type in gym_reacher gym_cheetah gym_walker2d gym_hopper gym_swimmer gym_ant; do
        python main/mbmf_main.py --exp_id mbmf_${env_type}_seed_${seed}\
	    --task $env_type \
            --num_planning_traj 5000 --planning_depth 20 --random_timesteps 10000 --timesteps_per_batch 3000 --dynamics_epochs 30 --num_workers 24 --mb_timesteps 70000 --dagger_epoch 300 --dagger_timesteps_per_iter 1750 --max_timesteps 10000000 --seed $seed
    done
done
