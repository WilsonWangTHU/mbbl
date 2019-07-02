
# $1 is the environment
# ilqr_depth: [10, 20, 50, 100]
for ilqr_depth in 100 80 60 40 20 10; do
    python main/ilqr_main.py --max_timesteps 10000 --task $1-1000 --timesteps_per_batch 1000 \
        --ilqr_iteration 5 --ilqr_depth $ilqr_depth --max_ilqr_linesearch_backtrack 10 \
        --exp_id $1-depth-$ilqr_depth --num_workers 2 --gt_dynamics 1
done
