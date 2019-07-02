
# $1 is the environment
# ilqr_depth: [10, 20, 50, 100]
for ilqr_depth in 10 20 50 100; do
    python main/ilqr_main.py --max_timesteps 20000 --task $1-1000 --timesteps_per_batch 1000 --ilqr_iteration 10 --ilqr_depth $ilqr_depth --max_ilqr_linesearch_backtrack 10 --exp_id cheetah_depth_$ilqr_depth --num_workers 2
done

# ilqr_iteration: [5, 10, 15, 20]
for ilqr_iteration in 5 10 15 20; do
    python main/ilqr_main.py --max_timesteps 20000 --task $1-1000 --timesteps_per_batch 20000 --ilqr_iteration $ilqr_iteration --ilqr_depth 10 --max_ilqr_linesearch_backtrack 10 --exp_id cheetah_ilqr_iteration_$ilqr_iteration --num_workers 2
done

# num_ilqr_traj: [2, 5, 10]
for num_ilqr_traj in 2 5 10; do
    python main/ilqr_main.py --max_timesteps 20000 --task $1-1000 --timesteps_per_batch 1000 --ilqr_iteration 10 --ilqr_depth 10 --max_ilqr_linesearch_backtrack 10 --num_ilqr_traj $num_ilqr_traj  --exp_id cheetah_traj_$num_ilqr_traj --num_workers 2

done
