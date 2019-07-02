
# $1 is the environment
# ilqr_depth: [10, 20, 50, 100]
for ilqr_depth in 10 20 50 100; do
    python main/ilqr_main.py --max_timesteps 20000 --task $1-1000 --timesteps_per_batch 1000 --ilqr_iteration 10 --ilqr_depth $ilqr_depth --max_ilqr_linesearch_backtrack 10 --exp_id $1-depth-$ilqr_depth --num_workers 2
done
