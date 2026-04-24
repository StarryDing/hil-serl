#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import pickle as pkl
import imageio
from gymnasium.wrappers import RecordEpisodeStatistics

from serl_launcher.agents.continuous.consistency_bc import ConsistencyBCAgent

from serl_launcher.utils.launcher import (
    make_consistency_bc_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING
from experiments.config import DefaultTrainingConfig
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 20000, "Number of pretraining steps.")
flags.DEFINE_integer("checkpoint_period", 5000, "Save checkpoint every N steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")


flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


##############################################################################

def eval(
    env,
    bc_agent: ConsistencyBCAgent,
    sampling_rng,
    max_episode_steps: int = 100,
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    success_counter = 0
    time_list = []

    if FLAGS.save_video:
        video_dir = os.path.join(os.path.abspath(FLAGS.bc_checkpoint_path), "eval_videos")
        os.makedirs(video_dir, exist_ok=True)

    ep_pbar = tqdm.tqdm(range(FLAGS.eval_n_trajs), desc="eval episodes", dynamic_ncols=True)
    for episode in ep_pbar:
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        frames = []
        step = 0
        step_pbar = tqdm.tqdm(total=max_episode_steps, desc=f"ep {episode+1}/{FLAGS.eval_n_trajs}", dynamic_ncols=True, leave=False)
        while not done:
            sampling_rng, key = jax.random.split(sampling_rng)

            actions = bc_agent.sample_actions(observations=obs, seed=key)
            actions = np.asarray(jax.device_get(actions))
            next_obs, reward, terminated, truncated, info = env.step(actions)
            step += 1
            step_pbar.update(1)

            if FLAGS.save_video:
                base_env = env.unwrapped
                rendered = base_env.render()
                if rendered is not None:
                    if isinstance(rendered, list):
                        frame = np.concatenate(rendered, axis=1)
                    else:
                        frame = rendered
                    frames.append(frame)

            obs = next_obs
            done = terminated or truncated
            if done:
                step_pbar.close()
                dt = time.time() - start_time
                is_success = terminated
                result = "success" if is_success else "fail"
                if is_success:
                    time_list.append(dt)
                    success_counter += 1
                ep_pbar.set_postfix(success=f"{success_counter}/{episode+1}", last=result, steps=step, time=f"{dt:.2f}s")

                if FLAGS.save_video and frames:
                    video_path = os.path.join(
                        video_dir, f"eval_ep{episode}_{result}.mp4"
                    )
                    imageio.mimsave(video_path, frames, fps=20)
                    print(f"Saved video to {video_path}")

    print(f"\nsuccess rate: {success_counter / FLAGS.eval_n_trajs}")
    print(f"average time: {np.mean(time_list) if time_list else 0:.3f}s")


##############################################################################


def train(
    bc_agent: ConsistencyBCAgent,
    bc_replay_buffer,
    config: DefaultTrainingConfig,
    wandb_logger=None,
):

    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )
    
    # Pretrain BC policy to get started
    for step in tqdm.tqdm(
        range(FLAGS.train_steps),
        dynamic_ncols=True,
        desc="bc_pretraining",
    ):
        batch = next(bc_replay_iterator)
        bc_agent, bc_update_info = bc_agent.update(batch)
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log({"bc": bc_update_info}, step=step)
        if (step + 1) % FLAGS.checkpoint_period == 0 or step == FLAGS.train_steps - 1:
            checkpoints.save_checkpoint(
                bc_checkpoint_path, bc_agent.state, step=step, keep=5
            )
    print_green("bc pretraining done and saved checkpoint")


##############################################################################


def main(_):
    config: DefaultTrainingConfig = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    eval_mode = FLAGS.eval_n_trajs > 0
    env = config.get_environment(
        fake_env=not eval_mode,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    bc_agent: ConsistencyBCAgent = make_consistency_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    bc_agent: ConsistencyBCAgent = jax.device_put(
        jax.tree.map(jnp.array, bc_agent), sharding.replicate()
    )

    bc_checkpoint_path = os.path.abspath(FLAGS.bc_checkpoint_path)

    if not eval_mode:
        assert not os.path.isdir(
            os.path.join(bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")
        )

        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="consistency-bc",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        demo_path = glob.glob(os.path.join(os.getcwd(), "demo_data", "*.pkl"))
        
        assert demo_path is not []

        for path in demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if np.linalg.norm(transition['actions']) > 0.0:
                        bc_replay_buffer.insert(transition)
        print(f"bc replay buffer size: {len(bc_replay_buffer)}")

        # learner loop
        print_green("starting learner loop")
        train(
            bc_agent=bc_agent,
            bc_replay_buffer=bc_replay_buffer,
            wandb_logger=wandb_logger,
            config=config,
        )

    else:
        rng = jax.random.PRNGKey(FLAGS.seed)
        sampling_rng = jax.device_put(rng, sharding.replicate())

        bc_ckpt = checkpoints.restore_checkpoint(
            bc_checkpoint_path,
            bc_agent.state,
        )
        bc_agent = bc_agent.replace(state=bc_ckpt)

        print_green("starting actor loop")
        eval(
            env=env,
            bc_agent=bc_agent,
            sampling_rng=sampling_rng,
            max_episode_steps=config.max_traj_length,
        )


if __name__ == "__main__":
    app.run(main)
