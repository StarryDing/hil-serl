export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=franka_pick_block \
    --checkpoint_path=first_run \
    --demo_path=/home/starry/Projects/hil-serl/examples/experiments/franka_pick_block/demo_data/franka_lift_cube_image_20_trajs.pkl \
    --learner \