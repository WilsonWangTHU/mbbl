# see how the humanoid works in terms of the performance:
# trpo / ppo
# batch size
# max_timesteps 1e8


for batch_size in 5000 20000 50000; do
    for tr_method in ppo trpo; do
        for env_type in dm-humanoid dm-humanoid-noise; do
            python main/mf_main.py --exp_id ${env_type}_${tr_method}_${batch_size} \
                --timesteps_per_batch $batch_size --task $env_type \
                --num_workers 5 --trust_region_method $tr_method --max_timesteps 50000000 \
                --num_expert_episode_to_save 5
        done
    done
done
