export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=franka_pick_block \
    --checkpoint_path=first_run \
    --eval_n_trajs=10 \
    --eval_checkpoint_step=490000 \
    --save_video \
    --actor \