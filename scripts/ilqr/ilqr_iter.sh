# ilqr_iteration: [5, 10, 15, 20]
for ilqr_iteration in 5 10 15 20; do
    python main/ilqr_main.py --max_timesteps 20000 --task $1-1000 --timesteps_per_batch 20000 --ilqr_iteration $ilqr_iteration --ilqr_depth 10 --max_ilqr_linesearch_backtrack 10 --exp_id $1-iteration_$ilqr_iteration --num_workers 2
done
