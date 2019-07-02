# the first batch:
# gym_reacher, gym_cheetah, gym_walker2d, gym_hopper, gym_swimmer, gym_ant

# the second batch:
# gym_pendulum, gym_invertedPendulum, gym_acrobot, gym_mountain, gym_cartpole
# 200         , 100                 , 200        , 200         , 200 

# bash ilqr.sh gym_pendulum 4000 200
# bash ilqr.sh gym_invertedPendulum 2000 100
# bash ilqr.sh gym_acrobot 4000 200
# bash ilqr.sh gym_mountain 4000 200
# bash ilqr.sh gym_cartpole 4000 200

# $1 is the environment
# ilqr_depth: [10, 20, 50, 100]
for ilqr_depth in 10 20 30 50 ; do
    python main/ilqr_main.py --max_timesteps $2 --task $1 --timesteps_per_batch $3 --ilqr_iteration 10 --ilqr_depth $ilqr_depth --max_ilqr_linesearch_backtrack 10 --exp_id $1-depth-$ilqr_depth --num_workers 2 --gt_dynamics 1
done

for num_ilqr_traj in 2 5 10; do
    python main/ilqr_main.py --max_timesteps $2 --task $1 --timesteps_per_batch $3 --ilqr_iteration 10 --ilqr_depth 30 --max_ilqr_linesearch_backtrack 10 --num_ilqr_traj $num_ilqr_traj  --exp_id $1-traj-$num_ilqr_traj --num_workers 2  --gt_dynamics 1
done
