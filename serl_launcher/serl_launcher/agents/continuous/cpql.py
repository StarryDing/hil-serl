from serl_launcher.common.common import JaxRLTrainState, nonpytree_field
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Iterable
from functools import partial
from typing import Optional, callable

from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.typing import PRNGKey, Batch, Data
from serl_launcher.networks.consistency_policy import ConsistencyPolicy
from serl_launcher.networks.mlp import MLP
from serl_launcher.networks.time_embedding import TimeEmbedding
from serl_launcher.networks.actor_critic_nets import Critic, ensemblize
from serl_launcher.common.common import ModuleDict
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.diffusion.noise_process import sample_sigmas
from serl_launcher.diffusion.schedules import karras_sigmas
from serl_launcher.diffusion.noise_process import make_init_noisy_action_ve


class CPQLAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jnp.ndarray,
    ):
        pass

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
    ):
        pass

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
        noisy_actions, sigmas = make_init_noisy_action_ve(seed, self.config["sigma_schedule"][0], self.config["action_dim"], 1)
        batched_obs = jax.tree.map(lambda x: x[jnp.newaxis], observations)
        denoised_actions = self.state.apply_fn(
            {"params": self.state.params},
            batched_obs,
            noisy_actions=noisy_actions,
            sigmas=sigmas,
            train=False,
            name="actor",
        )
        return jnp.squeeze(denoised_actions, axis=0)
    
    def update(self, batch: jax.Array):
        pass

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        sigmas: jnp.ndarray,
        actor_def: nn.Module,
        critic_def: nn.Module,
        sigma_schedule: jnp.ndarray,
        sigma_data: float,
        actor_optimizer_kwargs: dict ={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs: dict ={
            "learning_rate": 3e-4,
        },
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
        }
        model_def = ModuleDict(networks)

        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng, 
            actor=[observations, actions, sigmas], 
            critic=[observations, actions])["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = dict(
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            sigma_schedule=sigma_schedule,
            sigma_data=sigma_data,
            weight_schedule="karras",
            action_dim=actions.shape[-1]
        )
        return cls(
            state=state,
            config=config,
        )
    
    @classmethod
    def create_pixels(cls,
        rng: PRNGKey,
        observations: jnp.ndarray,
        image_keys: Iterable[str],
        actions: jnp.ndarray,
        # network definitions
        encoder_type: str = "resnet-pretrained",
        train_encoder: bool = False,
        use_proprio: bool = True,
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        t_network_kwargs: dict = {
            "t_dim": 16,
        },
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        steps: int = 40,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        critic_ensemble_size: int = 2,
        augmentation_function: Optional[callable] = None,
    ):
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
            from serl_launcher.vision.resnet_v1 import PreTrainedResNetEncoder, resnetv1_configs
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

        encoders = {
            "actor": encoder_def,
            "critic": encoder_def,
        }

        policy_network_kwargs["activate_final"] = True
        actor_def = ConsistencyPolicy(
                encoder=encoders["actor"],
                train_encoder=train_encoder,
                network=MLP(**policy_network_kwargs),
                t_network=TimeEmbedding(**t_network_kwargs),
                action_dim=actions.shape[-1],
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                sigma_data=sigma_data,
                clip_denoised=True,
                name="actor",
            )

        critic_network_kwargs["activate_final"] = True
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        init_observations = jax.tree.map(lambda x: x[jnp.newaxis], observations)
        init_actions = actions[jnp.newaxis]
        rng, init_sigma_rng = jax.random.split(rng)
        init_sigmas = sample_sigmas(init_sigma_rng, karras_sigmas(sigma_min, sigma_max, rho, steps), 1)
        

        agent = cls.create(
            rng,
            observations=init_observations,
            actions=init_actions,
            sigmas=init_sigmas,
            actor_def=actor_def,
            critic_def=critic_def,
            sigma_schedule=karras_sigmas(sigma_min, sigma_max, rho, steps),
            sigma_data=sigma_data,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
        )

        if "pretrained" in encoder_type:  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
        