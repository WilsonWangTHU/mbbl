# see how the ppo works for the environments in terms of the performance:
# batch size
# max_timesteps 1e8


for seed in 1234 2341 3412 4123; do
    for env_type in gym_hopper gym_swimmer gym_ant; do
        python main/rs_main.py --exp_id rs_${env_type}_seed_${seed}\
            --task $env_type \
            --num_planning_traj 1000 --planning_depth 10 --random_timesteps 10000 \
            --timesteps_per_batch 3000 --num_workers 20 --max_timesteps 300000 --seed $seed
    done
done
