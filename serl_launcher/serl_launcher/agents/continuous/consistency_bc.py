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
import optax

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.utils.train_utils import _unpack
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.networks.consistency_policy import ConsistencyPolicy
from serl_launcher.networks.mlp import MLP
from serl_launcher.diffusion.noise_process import sample_sigmas, add_ve_noise
from serl_launcher.diffusion.schedules import karras_sigmas
from serl_launcher.diffusion.losses import get_snr, get_ve_weightings



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
        batch_size = batch["actions"].shape[0]

        # 数据增强
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        def loss_fn(params, rng):
            rng, sigma_rng = jax.random.split(rng)
            rng, noise_rng = jax.random.split(rng)
            batch_actions = batch["actions"]
            batch_sigmas = sample_sigmas(sigma_rng, self.config["sigma_schedule"], batch_size)
            batch_noisy_actions, _ = add_ve_noise(batch_actions, batch_sigmas, noise_rng)
            denoised_actions = self.state.apply_fn(
                {"params": params},
                batch["observations"],
                noisy_actions=batch_noisy_actions,
                sigmas=batch_sigmas,
                train=True,
                name="actor",
            )
            recon_diffs = (denoised_actions - batch_actions) ** 2   # (B, action_dim)
            mse = jnp.mean(recon_diffs, axis=tuple(range(1, recon_diffs.ndim)))                             # (B,)
            ve_loss_weight = get_ve_weightings(
                        self.config["weight_schedule"], 
                        get_snr(batch_sigmas), 
                        self.config["sigma_data"]
                    )                                               # (B,)
            loss = jnp.mean(ve_loss_weight * mse)
            return loss, {"loss": loss}

        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        return self.replace(state=new_state), info

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
        actions = jnp.zeros((1, self.config["action_dim"]), dtype=jnp.float32)
        sigmas = jnp.ones((1, ), dtype=jnp.float32) * self.config["sigma_schedule"][0]
        noisy_actions, _ = add_ve_noise(actions, sigmas, seed)
        denoised_actions = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            noisy_actions=noisy_actions,
            sigmas=sigmas,
            train=False,
            name="actor",
        )
        return jnp.squeeze(denoised_actions, axis=0)

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
        sigma_emb_dim: int = 16,
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
        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        network_kwargs["activate_final"] = True
        networks = {
            "actor": ConsistencyPolicy(
                encoder_def,
                MLP(**network_kwargs),
                action_dim=actions.shape[-1],
                sigma_emb_dim=sigma_emb_dim,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                sigma_data=sigma_data,
                max_t=steps,
            )
        }
        model_def = ModuleDict(networks)

        tx = optax.adam(learning_rate)

        rng, init_sigma_rng = jax.random.split(rng)
        sigmas = sample_sigmas(init_sigma_rng, karras_sigmas(sigma_min, sigma_max, rho, steps), actions.shape[0])
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, actor=[observations, actions, sigmas])["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        config = dict(
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            sigma_schedule=karras_sigmas(sigma_min, sigma_max, rho, steps),
            sigma_data=sigma_data,
            weight_schedule="karras",
            action_dim=actions.shape[-1]
        )
        
        agent = cls(state, config)
        if encoder_type == "resnet-pretrained":
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent