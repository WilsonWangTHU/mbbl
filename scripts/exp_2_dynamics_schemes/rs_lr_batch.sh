# see how the ppo works for the environments in terms of the performance:
# batch size
# max_timesteps 1e8

# Tanh + None (done)
# Relu + None
# Leaky-Relu + None
# Tanh + layer-norm
# Tanh + batch-norm
# Relu + layer-norm
# Relu + batch-norm
# Leaky-relu + layernorm
# Leaky-relu + batch-norm
# Tanh + None (done)
# Relu + None
# Leaky-Relu + None
# Tanh + layer-norm
# Tanh + batch-norm
# Relu + layer-norm
# Relu + batch-norm
# Leaky-relu + layernorm
# Leaky-relu + batch-norm


for seed in 1234 2341 3412 4123; do
    for env_type in gym_cheetah; do
        for dynamics_lr in 0.0003 0.001 0.003; do
            for dynamics_batch_size in 256 512 1024; do
                for dynamics_epochs in 10 30 50; do

                    python main/rs_main.py \
                        --exp_id rs_${env_type}_lr_${dynamics_lr}_batchsize_${dynamics_batch_size}_epochs_${dynamics_epochs}_seed_${seed} \
                        --task $env_type \
                        --dynamics_batch_size 512 \
                        --num_planning_traj 1000 --planning_depth 10 --random_timesteps 10000 \
                        --timesteps_per_batch 3000 --num_workers 20 --max_timesteps 200000 \
                        --seed $seed \
                        --dynamics_lr $dynamics_lr \
			--dynamics_batch_size $dynamics_batch_size \
                        --dynamics_epochs $dynamics_epochs
                done

            done
        done
    done
done
