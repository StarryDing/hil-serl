export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_consistency_bc.py "$@" \
    --exp_name=franka_pick_block \
    --bc_checkpoint_path=consistency_bc_chek \
    --eval_n_trajs=10 \
    --save_video
