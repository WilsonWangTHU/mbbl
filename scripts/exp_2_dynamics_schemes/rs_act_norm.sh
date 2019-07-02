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
    for env_type in gym_reacher gym_cheetah gym_ant; do
        for dynamics_activation_type in 'leaky_relu' 'tanh' 'relu' 'swish' 'none'; do
            for dynamics_normalizer_type in 'layer_norm' 'batch_norm' 'none'; do

                python main/rs_main.py \
                    --exp_id rs_${env_type}_act_${dynamics_activation_type}_norm_${dynamics_normalizer_type}_seed_${seed} \
                    --task $env_type \
                    --dynamics_batch_size 512 \
                    --num_planning_traj 1000 --planning_depth 10 --random_timesteps 10000 \
                    --timesteps_per_batch 3000 --num_workers 20 --max_timesteps 200000 \
                    --seed $seed \
                    --dynamics_activation_type $dynamics_activation_type \
                    --dynamics_normalizer_type $dynamics_normalizer_type

            done
        done
    done
done
