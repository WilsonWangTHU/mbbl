# see how the ppo works for the environments in terms of the performance:
# batch size
# max_timesteps 1e8
# gym_reacher gym_cheetah gym_walker2d gym_hopper gym_swimmer gym_ant
# gym_pendulum gym_invertedPendulum gym_acrobot gym_mountain gym_cartpole

for trust_region_method in ppo; do
    for batch_size in 1000; do
        for env_type in $1; do
            for seed in 1234 2341 3412 4123; do
                python main/mbmf_main.py --exp_id mbmf_${env_type}_${trust_region_method}_seed_${seed}\
                    --task $env_type \
                    --trust_region_method ${trust_region_method} \
                    --num_planning_traj 5000 --planning_depth 20 --random_timesteps 1000 \
                    --timesteps_per_batch $batch_size --dynamics_epochs 30 \
                    --num_workers 20 --mb_timesteps 7000 --dagger_epoch 300 \
                    --dagger_timesteps_per_iter 1750 --max_timesteps 200000 \
                    --seed $seed --dynamics_batch_size 500
            done
        done
    done
done
