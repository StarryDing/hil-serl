'''
 * @file consistency_bc.py
 * @author your dingyun (dingyun@psirobot.ai)
 * @brief 
 * @version 0.1
 * @date 2026-04-20
 * 
 * @copyright Copyright (c) 2026
'''

from typing import Any, Iterable, Optional

import flax
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.utils.train_utils import _unpack


class ConsistencyBCAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        '''
        consistency_bc agent 更新
        流程：
            - 处理 batch
            - 数据增强
            - 定义 loss function
                - 采样 sigma
                - 采样 noisy actions
                - apply_fn 计算预测的 clean actions
                - 计算 loss
                - 计算 gradient
                - 更新模型
            - 返回 agent state
        '''
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        
        # 数据增强
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        def loss_fn(params, rng):
            rng, key = jax.random.split(rng)

    @jax.jit
    def sample_actions(self, observations: np.ndarray, seed: Optional[PRNGKey] = None):
        '''
        consistency_bc agent 采样动作
        流程：
            - 传入 observations
            - 计算最大噪声
            - apply_fn 根据 max noises、max steps、observations 计算预测的 clean actions
            - 返回动作
        '''
        pass

    @classmethod
    def create(cls,
        rng: PRNGKey,
        observations: np.ndarray,
        actions: jnp.ndarray,
        encoder_type: str = "resnet-pretrained",
        image_keys: Iterable[str] = ("image",),
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        augmentation_function: Optional[callable] = None,
        learning_rate: float = 3e-4,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        steps: int = 40,
        sigma_data: float = 0.5,
    ):
        '''
        创建consistency_bc agent
        流程：
            - 创建 encoder
            - 创建 actor
            - 创建 optimizer
            - 初始化模型 params
            - 创建 agent state
            - 创建 agent config
            - 创建 agent
            - 根据 encoder type 加载 encoder 权重
        '''
        pass
