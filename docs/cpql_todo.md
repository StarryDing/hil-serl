# CPQL Agent 实现代办事项表

## 0. 当前基础确认

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 确认 Consistency Policy BC 可以独立训练 | BC loss 正常下降，动作输出范围正常 | 高 |
| ⬜ | 确认 demo action 已归一化 | 建议统一到 `[-1, 1]`，critic 和 actor 都使用同一尺度 | 高 |
| ⬜ | 确认 actor 推理采样逻辑 | 从高斯噪声 `sigma_max * eps` 输入，输出去噪动作 | 高 |
| ⬜ | 记录当前 BC-only policy 表现 | 作为后续 CPQL 改进的 baseline | 中 |

---

## 1. Agent 网络结构

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 在 agent module 中加入 critic | 输入 `(obs, action)`，输出 `Q(s,a)` | 高 |
| ⬜ | 使用 double Q 或 critic ensemble | 最少两个 critic：`Q1, Q2` | 高 |
| ⬜ | 加入 target critic | 用于 TD target，避免训练震荡 | 高 |
| ⬜ | 初始化 target critic 参数 | 初始时与 critic 参数完全一致 | 高 |
| ⬜ | 实现 Polyak soft update | `target = tau * critic + (1 - tau) * target` | 高 |
| ⬜ | 可选：加入 actor EMA | 如果你的 consistency policy 本身依赖 EMA，可保留 | 中 |

---

## 2. Replay Buffer / Demo Buffer 数据结构

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 确认 buffer 包含 `obs` | 当前状态 | 高 |
| ⬜ | 确认 buffer 包含 `action` | 数据集动作 / 交互动作 | 高 |
| ⬜ | 确认 buffer 包含 `reward` | 用于 TD target | 高 |
| ⬜ | 确认 buffer 包含 `next_obs` | 下一状态 | 高 |
| ⬜ | 确认 buffer 包含 `done` 或 `mask` | `mask = 1 - done` | 高 |
| ⬜ | 为 demo buffer 增加 `mc_return` | Cal-QL 需要 | 高 |
| ⬜ | 可选：增加 `is_demo` 标记 | 区分 demo 数据和 online 数据 | 中 |
| ⬜ | 实现 demo buffer 与 online replay buffer 混合采样 | 适配 HIL-SERL / RLPD 风格 | 高 |

---

## 3. MC Return 预计算

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 按 trajectory 计算 MC return | `G_t = r_t + gamma * G_{t+1}` | 高 |
| ⬜ | 将 `mc_return` 存入 demo transition | 每个 transition 都要有对应 return | 高 |
| ⬜ | 检查 reward scale | 确认 MC return 与 Q 值尺度一致 | 高 |
| ⬜ | 对 sparse reward 场景单独检查 return 分布 | 成功轨迹与失败轨迹的 return 差异是否合理 | 中 |
| ⬜ | 记录 `mc_return` 的均值、最大值、最小值 | 用于后续 debug Q 值 | 中 |

---

## 4. Critic TD Loss

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 实现 next action 采样 | `next_action = actor.sample(next_obs)` | 高 |
| ⬜ | 使用 target critic 计算 target Q | `target_q = min(target_Q1, target_Q2)` | 高 |
| ⬜ | 实现 TD target | `y = r + gamma * mask * target_q` | 高 |
| ⬜ | 实现 TD loss | `MSE(Q(s,a), y)` | 高 |
| ⬜ | 先只训练 TD critic | 不加 CQL / Cal-QL，先验证 critic 是否正常 | 高 |
| ⬜ | 监控 TD loss | 看是否稳定下降，有无 NaN | 高 |
| ⬜ | 监控 Q 值范围 | 防止 Q 值爆炸或塌缩 | 高 |

---

## 5. CQL / Conservative Loss

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 实现 random action 采样 | 从动作空间均匀采样，例如 `Uniform(-1, 1)` | 高 |
| ⬜ | 实现当前 actor action 采样 | `a_pi = actor.sample(obs)` | 高 |
| ⬜ | 实现 next actor action 采样 | `a_next_pi = actor.sample(next_obs)` | 中 |
| ⬜ | 计算 `Q(s, a_data)` | 数据动作的 Q 值 | 高 |
| ⬜ | 计算 `Q(s, a_rand)` | 随机动作的 Q 值 | 高 |
| ⬜ | 计算 `Q(s, a_pi)` | 当前策略动作的 Q 值 | 高 |
| ⬜ | 实现 logsumexp conservative loss | `logsumexp(Q_ood) - Q_data` | 高 |
| ⬜ | 加入 `cql_alpha` 权重 | 控制保守正则强度 | 高 |
| ⬜ | 监控 `Q_data > Q_rand / Q_pi` 是否成立 | 判断保守项是否有效 | 高 |

---

## 6. Cal-QL 校准项

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 读取 batch 中的 `mc_return` | demo 数据必须有 | 高 |
| ⬜ | 实现 Cal-QL 下界约束 | `relu(mc_return - Q_data)^2` | 高 |
| ⬜ | 加入 `cal_alpha` 权重 | 控制 MC return 校准强度 | 高 |
| ⬜ | 检查 demo action 的 Q 是否被压得过低 | Cal 项应防止 CQL 过度保守 | 高 |
| ⬜ | 监控 `Q_data` 与 `mc_return` 的差距 | 观察校准是否生效 | 高 |
| ⬜ | 可选：只对 demo 数据使用 Cal loss | online 数据没有可靠 MC return 时建议如此 | 中 |

---

## 7. Critic 总 Loss

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 组合 critic loss | `td_loss + cql_alpha * cql_loss + cal_alpha * cal_loss` | 高 |
| ⬜ | 支持开关 CQL loss | 方便 ablation 和 debug | 中 |
| ⬜ | 支持开关 Cal loss | 方便验证 Cal-QL 是否有效 | 中 |
| ⬜ | 支持调节 `cql_alpha` | 过大容易压低所有 Q | 高 |
| ⬜ | 支持调节 `cal_alpha` | 过大可能让 Q 过度贴合 MC return | 高 |
| ⬜ | 加入 gradient clipping | 防止 critic 梯度爆炸 | 中 |

---

## 8. Actor Consistency / BC Loss

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 保留当前 consistency BC loss | 已有实现继续复用 | 高 |
| ⬜ | 确认噪声采样 schedule | 例如 Karras sigma 或 log sigma | 高 |
| ⬜ | 确认训练输入为 `a + sigma * eps` | 对 demo action 加噪 | 高 |
| ⬜ | 确认 actor 输出目标为 clean action | 即还原 `a_data` | 高 |
| ⬜ | 记录不同 sigma 下的 denoising error | 判断低噪声/高噪声是否都学到 | 中 |

---

## 9. Actor Q Loss

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 实现 actor 当前动作采样 | `a_theta = actor.sample(obs)` | 高 |
| ⬜ | 用 critic 计算 `Q(s, a_theta)` | 建议使用 `min(Q1, Q2)` | 高 |
| ⬜ | 实现 actor Q loss | `actor_q_loss = -Q(s, a_theta).mean()` | 高 |
| ⬜ | 确保 actor Q loss 对 actor 可导 | 这里不能 `detach(a_theta)` | 高 |
| ⬜ | 确保 critic 参数不被 actor optimizer 更新 | 只让梯度通过 Q 回传到 actor | 高 |
| ⬜ | 可选：对 Q loss 做尺度归一化 | 防止 Q 值尺度压过 BC loss | 中 |

---

## 10. Actor 总 Loss

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 组合 actor loss | `bc_loss + q_weight * actor_q_loss` | 高 |
| ⬜ | 加入 BC warmup | 前期只训练 BC，不加 Q loss | 高 |
| ⬜ | 加入 Q weight ramp-up | 逐渐增大 Q loss 权重 | 高 |
| ⬜ | 支持调节 `q_weight` | CPQL 稳定性的关键超参数 | 高 |
| ⬜ | 监控 `bc_loss` 与 `actor_q_loss` 的相对尺度 | 防止其中一项完全主导 | 高 |
| ⬜ | 可选：actor update interval | critic 更新多次后再更新 actor | 中 |

---

## 11. HIL-SERL / RLPD 训练流程集成

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 接入现有 learner update 框架 | 保持 HIL-SERL 原有结构 | 高 |
| ⬜ | 支持 UTD ratio | critic 可以多次更新 | 高 |
| ⬜ | 设置 actor 更新频率 | 例如每 N 次 critic update 更新一次 actor | 中 |
| ⬜ | 支持 demo / replay 混合 batch | 类似 RLPD 数据混合 | 高 |
| ⬜ | 支持在线数据持续写入 replay buffer | HIL 阶段需要 | 高 |
| ⬜ | 支持 human intervention 数据加入 demo-like buffer | 可提高人工接管样本权重 | 中 |

---

## 12. 动作执行与归一化

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 统一 actor 输出动作范围 | 建议训练空间为 `[-1, 1]` | 高 |
| ⬜ | 统一 critic 输入动作范围 | critic 也看归一化动作 | 高 |
| ⬜ | 环境执行前进行 unnormalize | 转成真实机器人动作范围 | 高 |
| ⬜ | 对 actor 输出进行 clamp | 防止越界动作 | 高 |
| ⬜ | 检查 noise scale 与 action scale 是否匹配 | `sigma_max` 不应远超动作范围太多，除非网络设计支持 | 高 |

---

## 13. Target / EMA 更新

| 状态 | 任务 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | 每次 critic 更新后 soft update target critic | `tau` 通常可从 `0.005` 开始 | 高 |
| ⬜ | 检查 target critic 不参与梯度更新 | 只通过 Polyak update 更新 | 高 |
| ⬜ | 可选：维护 actor EMA | 如果 consistency policy 需要 teacher/EMA | 中 |
| ⬜ | 保存 checkpoint 时同时保存 target critic | 否则恢复训练会不一致 | 中 |

---

## 14. Logging / Debug 指标

| 状态 | 指标 | 说明 | 优先级 |
|---|---|---|---|
| ⬜ | `critic_loss` | critic 总 loss | 高 |
| ⬜ | `td_loss` | TD error | 高 |
| ⬜ | `cql_loss` | 保守正则项 | 高 |
| ⬜ | `cal_loss` | Cal-QL 校准项 | 高 |
| ⬜ | `actor_loss` | actor 总 loss | 高 |
| ⬜ | `bc_loss` | consistency BC loss | 高 |
| ⬜ | `actor_q_loss` | Q-guided loss | 高 |
| ⬜ | `q_data` | 数据动作 Q 值 | 高 |
| ⬜ | `q_pi` | actor 动作 Q 值 | 高 |
| ⬜ | `q_rand` | 随机动作 Q 值 | 高 |
| ⬜ | `mc_return_mean` | MC return 均值 | 中 |
| ⬜ | `action_mean/std/max/min` | actor 输出动作分布 | 高 |
| ⬜ | `grad_norm_actor` | actor 梯度范数 | 中 |
| ⬜ | `grad_norm_critic` | critic 梯度范数 | 中 |

---

## 15. 推荐实现顺序

| 阶段 | 目标 | 完成标准 |
|---|---|---|
| Step 1 | 验证 BC-only consistency policy | 能稳定训练并在环境中执行 |
| Step 2 | 加入 critic + target critic | TD loss 正常下降，Q 值不爆炸 |
| Step 3 | 加入 actor Q loss | 小权重下 actor 不崩 |
| Step 4 | 加入 CQL conservative loss | OOD 动作 Q 被压低 |
| Step 5 | 加入 MC return + Cal loss | demo action Q 不被过度压低 |
| Step 6 | 接入 demo / replay 混合采样 | 支持 HIL-SERL/RLPD 训练流程 |
| Step 7 | 加入完整 logging 和 ablation 开关 | 可以定位问题来源 |
| Step 8 | 上真机前做离线回放与仿真验证 | 动作范围、安全边界、Q 值稳定 |

---

## 16. 第一版最小可运行配置

建议第一版不要直接上完整复杂版本，可以先做：

```python
critic_loss = td_loss
actor_loss = bc_loss + q_weight * actor_q_loss