
for env_type in gym_cheetah gym_ant; do
    python main/rs_main.py --exp_id rs_${env_type}_depth_$1 \
        --task $env_type \
        --num_planning_traj 1000 --planning_depth $1 --random_timesteps 0 \
        --timesteps_per_batch 1 --num_workers 20 --max_timesteps 10000 \
        --gt_dynamics 1 --check_done 1
    
    python main/pets_main.py --exp_id rs_${env_type}_depth_$1 \
        --task $env_type \
        --num_planning_traj 500 --planning_depth $1 --random_timesteps 0 \
        --timesteps_per_batch 1 --num_workers 20 --max_timesteps 10000 \
        --gt_dynamics 1 --check_done 1
done
