# Consistency Policy 代码审查

对比 hil-serl 中的 `ConsistencyBCAgent` 与 dexaoyi 中能跑通的 `CALQLAgent (ConsistencyPolicy)` 参考实现，发现并修复了多个 bug。

## 第一轮：严重 Bug（运行时直接崩溃）

### 1. `time_embedding.py` — PyTorch 语法混用

`sinusoidal_embedding` 使用了 `jnp.cat` 和 `dim=`，这是 PyTorch API，JAX 中不存在。

**修复**：`jnp.cat(..., dim=-1)` → `jnp.concatenate([...], axis=-1)`

### 2. `consistency_policy.py` — 属性名拼写错误

引用了 `self.t_emb_dim`，但 Module 定义的字段名是 `sigma_emb_dim`。

**修复**：`self.t_emb_dim` → `self.sigma_emb_dim`

### 3. `noise_process.py` — `jnp.random` 不存在

JAX 的随机数函数在 `jax.random` 下，不在 `jnp.random` 下。三处调用都会崩溃。

**修复**：所有 `jnp.random.xxx` → `jax.random.xxx`，文件头加 `import jax`。

## 第一轮：逻辑 Bug（能跑但训练结果错误）

### 4. `ConsistencyPolicy` — 缺少 `nn.Dense(action_dim)` 输出层

MLP 设置了 `activate_final=True`，输出维度是 `hidden_dims[-1]`（如 256），不是 `action_dim`。参考实现中有一个额外的 Dense 层做投影。

**修复**：在 MLP 输出后加 `nn.Dense(self.action_dim, kernel_init=default_init())`。

### 5. Loss 中的权重乘法逻辑错误

`mse = jnp.mean(recon_diffs)` 对所有维度求均值得到标量，导致 `ve_loss_weight (B,) * mse (scalar)` 的 per-sample 权重区分完全失效。

正确做法：先对 action_dim 求均值保留 batch 维度，再逐样本乘权重。

**修复**：`jnp.mean(recon_diffs)` → `jnp.mean(recon_diffs, axis=-1)` 或 `jnp.mean(recon_diffs, axis=tuple(range(1, recon_diffs.ndim)))`

### 6. `sample_sigmas` 可能采样到 sigma=0

`karras_sigmas` 带 `append_zero=True` 会在末尾加 0，而 `sample_sigmas` 的采样范围包含该索引。`sigma=0` 会导致 `get_snr` 返回 inf 和除以 0。

**修复**：将 `karras_sigmas` 的 `append_zero` 默认值改为 `False`。

## 第一轮：功能缺失

### 7. `sample_actions` 未实现

Consistency model 的推理逻辑：从纯噪声 `sigma_max * N(0,1)` 出发，单步前向得到去噪结果。

**修复**：实现 `sample_actions`，构造 `(1, action_dim)` 的零向量 + `sigma_schedule[0]`（即 `sigma_max`）加噪后单步去噪。

### 8. 预训练 ResNet 权重未加载

`PreTrainedResNetEncoder` 只定义了模型结构，`model_def.init` 只是随机初始化。需要额外调用 `load_resnet10_params` 用 ImageNet 预训练权重替换。

**修复**：在 `create` 方法末尾，先创建 agent 再调用 `load_resnet10_params(agent, image_keys)`。

## 第二轮：补充问题

### 9. `ConsistencyPolicy.__call__` 和 `TimeEmbedding.__call__` 缺少 `@nn.compact`

两个 Module 的 `__call__` 中都有内联创建子模块的写法（`nn.Dense(...)(x)`、`TimeEmbedding(...)(x)` 等），在 Flax 中必须加 `@nn.compact` 装饰器，否则无法注册子模块参数。

**修复**：两个 `__call__` 方法都加上 `@nn.compact`。

### 10. `init_rng` 被复用

`consistency_bc.py` 的 `create` 中，同一个 `init_rng` 同时用于 `sample_sigmas` 和 `model_def.init`，违反 JAX 的 PRNG 使用规范。

**修复**：`jax.random.split(rng, 3)` 拆分为 `sigma_rng` 和 `init_rng`。

### 11. `sample_actions` 返回值多一维

返回 shape 为 `(1, action_dim)` 而非 `(action_dim,)`，可能影响下游使用。

**修复**：返回前加 `jnp.squeeze(denoised_actions, axis=0)`。

## 问题汇总（按严重程度排序）

| 优先级 | 问题 | 文件 |
|--------|------|------|
| P0 崩溃 | `jnp.cat` → `jnp.concatenate` | `time_embedding.py` |
| P0 崩溃 | `jnp.random` → `jax.random`（三处） | `noise_process.py` |
| P0 崩溃 | `self.t_emb_dim` → `self.sigma_emb_dim` | `consistency_policy.py` |
| P0 崩溃 | 缺少 `@nn.compact` 装饰器 | `consistency_policy.py`, `time_embedding.py` |
| P0 崩溃 | `default_init` 未 import | `consistency_policy.py` |
| P1 训练错误 | 缺少 `nn.Dense(action_dim)` 输出投影 | `consistency_policy.py` |
| P1 训练错误 | Loss 权重乘法逻辑失效 | `consistency_bc.py` |
| P1 NaN 风险 | `sample_sigmas` 可能采样到 sigma=0 | `noise_process.py` |
| P2 功能缺失 | `sample_actions` 未实现 | `consistency_bc.py` |
| P2 功能缺失 | 未加载预训练 ResNet 权重 | `consistency_bc.py` |
| P2 不规范 | `init_rng` 被复用 | `consistency_bc.py` |
| P2 下游影响 | `sample_actions` 返回值多一维 | `consistency_bc.py` |

## 知识点笔记

### `axis` 参数
- `jnp.mean(x)` — 对所有维度求均值，返回标量
- `jnp.mean(x, axis=-1)` — 对最后一个维度求均值
- `jnp.mean(x, axis=tuple(range(1, x.ndim)))` — 对除 batch 外所有维度求均值（通用写法）

### `@nn.compact` vs `setup()`
在 Flax 中，如果在 `__call__` 里内联创建子模块（如 `nn.Dense(...)(x)`），必须加 `@nn.compact`。否则需要在 `setup()` 中预先定义所有子模块。

### `kernel_init`
`nn.Dense` 的 `kernel_init` 指定权重矩阵初始化方式，仅影响 `model_def.init()` 时的初始值。`default_init = nn.initializers.xavier_uniform`。

### 预训练 ResNet 加载
`PreTrainedResNetEncoder` / `resnetv1-10-frozen` 只定义模型结构（哪些层 frozen）。`model_def.init` 只是随机初始化。必须额外调用 `load_resnet10_params` 才能加载真正的 ImageNet 权重。

### Consistency Model 推理
一步去噪：从 `x_T = N(0, I) * sigma_max` 出发，传入 `sigma = sigma_max`，网络一次前向输出 `x_0`。

### `repeat` 参数（参考实现中的 CAL-QL 特有功能）
用于 CQL 中对每个 observation 采样多个候选动作（如 `cql_n_actions=10`），纯 BC 不需要。
