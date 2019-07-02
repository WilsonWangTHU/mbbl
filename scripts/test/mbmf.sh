seeds=($(shuf -i 0-1000 -n 3))
for seed in "${seeds[@]}"; do
    echo "$seed"
    # Swimmer
    python main/mbmf_main.py --task gym_swimmer --num_planning_traj 5000 --planning_depth 20 --random_timesteps 10000 --timesteps_per_batch 3000 --dynamics_epochs 30 --output_dir mbmf --num_workers 24 --mb_timesteps 70000 --dagger_epoch 300 --dagger_timesteps_per_iter 1750 --max_timesteps 10000000 --seed $seed
    # Half cheetah
    python main/mbmf_main.py --task gym_cheetah --num_planning_traj 1000 --planning_depth 20 --random_timesteps 10000 --timesteps_per_batch 9000 --dynamics_epochs 60 --output_dir mbmf --num_workers 24 --mb_timesteps 80000 --dagger_epoch 300 --dagger_timesteps_per_iter 2000 --max_timesteps 100000000 --seed $seed
    # Hopper
    python main/mbmf_main.py --task gym_hopper --num_planning_traj 1000 --planning_depth 40 --random_timesteps 4000 --timesteps_per_batch 10000 --dynamics_epochs 40 --output_dir mbmf --num_workers 24 --mb_timesteps 50000 --dagger_epoch 200 --dagger_timesteps_per_iter 5000 --max_timesteps 100000000 --seed $seed --dagger_saved_rollout 60 --dagger_iter 5
    #python main/mf_main.py --task gym_hopper --timesteps_per_batch 50000 --policy_batch_size 50000 --output_dir mbmf --num_workers 24 --seed $seed --max_timestep 100000000
done
