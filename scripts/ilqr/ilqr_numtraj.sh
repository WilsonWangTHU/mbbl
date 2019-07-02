# num_ilqr_traj: [2, 5, 10]
for num_ilqr_traj in 2 5 10; do
    python main/ilqr_main.py --max_timesteps 20000 --task $1-1000 --timesteps_per_batch 1000 --ilqr_iteration 10 --ilqr_depth 10 --max_ilqr_linesearch_backtrack 10 --num_ilqr_traj $num_ilqr_traj  --exp_id $1-traj-$num_ilqr_traj --num_workers 2

done
