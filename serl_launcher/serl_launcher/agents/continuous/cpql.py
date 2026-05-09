from serl_launcher.common.common import JaxRLTrainState, nonpytree_field
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Iterable
from functools import partial
from typing import Optional

from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.typing import PRNGKey, Batch, Data, Params
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
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ):
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            rngs={"dropout": rng} if train else {},
            train=train,
            name="critic"
        )

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
    ):
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations,
            actions,
            grad_params=self.state.target_params,
            train=False,
        )

    def forward_next_actions(self, observations: Data, rng: PRNGKey, batch_size: int):
        rng, policy_rng = jax.random.split(rng)
        noisy_actions, sigmas = make_init_noisy_action_ve(
            policy_rng, 
            self.config["sigma_schedule"][0], 
            self.config["action_dim"],
            batch_size)
        return self.state.apply_fn(
            {"params": self.state.params},
            observations,
            noisy_actions=noisy_actions,
            sigmas=sigmas,
            train=False,
            name="actor",
        )

    @jax.jit
    def sample_actions(self, observations: Data, seed: Optional[PRNGKey] = None):
        '''
        agent 采样动作
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

    def critic_loss_fn(self, batch: jax.Array, params: Params, rng: PRNGKey):
        """
        td loss 计算
        公式: (Q(s, a) - (r + gamma * min(Q(s', a')))) ^ 2
        """
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions = self.forward_next_actions(batch["next_observations"], next_action_sample_key, batch["next_observations"].shape[0])
        next_q = self.forward_target_critic(batch["next_observations"], next_actions)
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            next_q = next_q[subsample_idcs]
        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q.min(axis=0)
        # 扩充 target q 的维度到 critic_ensemble_size，更新所有 critic
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        # 显示阻值 target_qs 的梯度回传
        target_qs = jax.lax.stop_gradient(target_qs)
        rng, critic_rng = jax.random.split(rng)
        q_pred = self.forward_critic(batch["observations"], batch["actions"], critic_rng, grad_params=params)
        critic_loss = jnp.mean((q_pred - target_qs) ** 2)
        info = {
            "critic_loss": critic_loss,
            "target_q": jnp.mean(target_q),
            "q_pred": jnp.mean(q_pred),
            "rewards": batch["rewards"].mean(),
        }
        return critic_loss, info


    def  _get_cql_q_diff(self, batch: jax.Array, params: Params, rng: PRNGKey):
        """
        计算 Q(s,a_ood) - Q(s,a_data), 
        反向传播时用 min(Q(s,a_ood)) - Q(s,a_data) 计算
        min(Q(s,a_ood)) 用来压低OOD数据的Q值, min(-Q(s,a_data)) 用来防止数据内的Q被压的过低
        """
        rng, policy_action_rng = jax.random.split(rng)
        rng, random_action_rng = jax.random.split(rng)
        B = batch["rewards"].shape[0]
        n = self.config["cql_n_actions"]
        # repeat obs to cql_n_actions times
        obs_rep = jax.tree.map(lambda x: jnp.repeat(x, n, axis=0), batch["observations"])
        # sample actions from policy and random actions
        policy_actions = self.forward_next_actions(obs_rep, policy_action_rng, B*n)
        random_actions = jax.random.uniform(random_action_rng, 
                                shape=(B*n, self.config["action_dim"]), 
                                minval=-1.0, maxval=1.0)
        # forward critic to get ood q values => [N, B, n]
        rng, q_policy_rng, q_random_rng = jax.random.split(rng, 3)
        q_policy = self.forward_critic(obs_rep, policy_actions, rng=q_policy_rng, grad_params=params)
        q_random = self.forward_critic(obs_rep, random_actions, rng=q_random_rng, grad_params=params)
        q_policy = q_policy.reshape(self.config["critic_ensemble_size"], B, n)
        q_random = q_random.reshape(self.config["critic_ensemble_size"], B, n)

        # clip ood q values by mc_returns
        mc_returns = batch["mc_returns"]
        # repeat mc_returns to [N, B, 1]
        mc_lower_bound = mc_returns.reshape(1, B, 1).repeat(self.config["critic_ensemble_size"], axis=0)
        # clip ood q values by mc_returns => [N, B, n]
        q_policy = jnp.maximum(q_policy, mc_lower_bound)
        q_random = jnp.maximum(q_random, mc_lower_bound)
        # logsumexp of ood q values => [N, B]
        q_ood = jnp.concatenate([q_random, q_policy], axis=-1)
        q_ood = jnp.logsumexp(q_ood , axis=-1)

        # critic forward to data actions => [N, B]
        rng, q_data_rng = jax.random.split(rng)
        q_data = self.forward_critic(batch["observations"], batch["actions"], rng=q_data_rng, grad_params=params)

        # calculate cql q diff
        cql_q_diff = q_ood - q_data

        info = {}
        info.update({
            "q_data": q_data.mean(),
            "q_random": q_random.mean(),
            "q_policy": q_policy.mean(),
            "q_ood": q_ood.mean(),
            "cql_q_diff": cql_q_diff.mean(),
        })
        return cql_q_diff, info
        

    def calql_critic_loss_fn(self, batch: jax.Array, params: Params, rng: PRNGKey):

        rng, td_loss_rng = jax.random.split(rng)
        td_loss, td_loss_info = self.critic_loss_fn(batch, params, td_loss_rng)
        cql_q_diff, cql_intermediate_results = self._get_cql_q_diff(batch, params, rng)
        cql_loss = jnp.clip(cql_q_diff, self.config["cql_clip_diff_min"], self.config["cql_clip_diff_max"]).mean()
        critic_loss = td_loss + self.config["cql_alpha"] * cql_loss
        info = {
            **td_loss_info,
            "critic_loss": critic_loss,
        }

    def policy_loss_fn(self, batch: jax.Array, params: Params, rng: PRNGKey):
        pass

    def update(self, batch: jax.Array):
        pass

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        sigmas: jnp.ndarray,
        # actor定义
        actor_def: nn.Module,
        # critic定义
        critic_def: nn.Module,
        # 噪声调度器
        sigma_schedule: jnp.ndarray,
        # 数据噪声(均值)
        sigma_data: float,
        # 优化器
        actor_optimizer_kwargs: dict ={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs: dict ={
            "learning_rate": 3e-4,
        },
        # encoder 图像键
        image_keys: Iterable[str] = None,
        # 数据增强函数
        augmentation_function: Optional[callable] = None,
        # 折扣因子
        discount: float = 0.95,
        # terget critic 更新率
        soft_target_update_rate: float = 0.005,
        # 独立 critic 个数
        critic_ensemble_size: int = 2,
        # 独立 critic 子采样大小
        critic_subsample_size: Optional[int] = None,
        # cql alpha
        cql_alpha: float = 0.1,
        # cql 采样 action 个数
        cql_n_actions: int = 10,
        # cql 采样action方法
        cql_action_sample_method: str = "uniform",
        # cql temp
        cql_temp: float = 1.0,
        # bc 权重
        bc_weight: float = 1.0,
        # bc 权重衰减方法
        bc_weight_decay_method: str = "exponential",
        # bc 权重最小值
        bc_weight_min: float = 0.05,
        # bc 权重最大值
        bc_weight_max: float = 1.0,
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
            action_dim=actions.shape[-1],
            # consistency policy 参数
            sigma_schedule=sigma_schedule,
            sigma_data=sigma_data,
            weight_schedule="karras",
            # RL 参数
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            # cpql 参数
            cql_alpha=cql_alpha,
            cql_n_actions=cql_n_actions,
            cql_action_sample_method=cql_action_sample_method,
            cql_temp=cql_temp
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
        critic_subsample_size: Optional[int] = None,
        augmentation_function: Optional[callable] = None,
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        
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
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            discount=discount,
            soft_target_update_rate=soft_target_update_rate,

        )

        if "pretrained" in encoder_type:  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
        