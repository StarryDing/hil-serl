from gymnasium.wrappers import TimeLimit
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from franka_sim import envs
from experiments.franka_pick_block.wrapper import GripperPenaltyWrapper


class TrainConfig(DefaultTrainingConfig):
    image_keys = ['front', "wrist"]
    proprio_keys = ['panda/tcp_pos', 'panda/tcp_vel', 'panda/gripper_pos']
    discount = 0.97
    buffer_period = 1000
    checkpoint_period = 5000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-gripper"

    cta_ratio = 4
    
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # obs_image_size defaults to (128, 128) for efficient replay buffer storage
        # render_spec defaults to (1280, 1280) for human viewing
        # save_video=True: disable GUI, only capture images for video saving
        render_mode = "rgb_array" if save_video else "human"
        env = envs.PandaPickCubeGymEnv(
            action_scale=(0.1, 1), 
            render_mode="rgb_array", 
            image_obs=True,
            time_limit=float("inf"),  # 禁用内部时间限制，由外部 wrapper 控制
        )
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        env = TimeLimit(env, max_episode_steps=self.max_traj_length)
        # env = GripperPenaltyWrapper(env, penalty=-0.05)
 
        return env