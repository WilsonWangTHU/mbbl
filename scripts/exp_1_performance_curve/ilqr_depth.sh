
# $1 is the environment
# ilqr_depth: [10, 20, 50, 100]
for ilqr_depth in 10 20 30 50 ; do
    python main/ilqr_main.py --max_timesteps 2000 --task $1 --timesteps_per_batch 50 --ilqr_iteration 10 --ilqr_depth $ilqr_depth --max_ilqr_linesearch_backtrack 10 --exp_id $1-depth-$ilqr_depth --num_workers 2 --gt_dynamics 1
done

for num_ilqr_traj in 2 5 10; do
    python main/ilqr_main.py --max_timesteps 2000 --task $1 --timesteps_per_batch 50 --ilqr_iteration 10 --ilqr_depth 30 --max_ilqr_linesearch_backtrack 10 --num_ilqr_traj $num_ilqr_traj  --exp_id $1-traj-$num_ilqr_traj --num_workers 2  --gt_dynamics 1
done
