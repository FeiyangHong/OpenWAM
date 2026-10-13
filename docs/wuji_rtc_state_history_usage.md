# Wuji TrainingRTC：完整 state 时序增强对照

已做好手动启动准备，未启动本轮长训。旧 A/B 及其调度器已按要求停止，已有 checkpoint 保留；不要恢复旧调度器。

## 设置

首轮选择：从原同步 `step_18000` 权重训练 TrainingRTC 10 epoch，开启 **30% 非空前缀样本、过去 1～3 帧完整 state、不设额外幅度上限**的增强。原 RTC `step_9000` 作为已有基线，**不要求重训 clean**。脚本保留可选 `clean` 入口，仅在需要控制 8 卡执行差异时使用。

| 项目 | 设置 |
|---|---|
| 初始化 | `outputs/grasp_anything_base/2026-09-16_21-50-22/checkpoint_step_18000.safetensors` |
| 数据/表示 | 原 Wuji 数据，绝对 EEF、column ROT6D、双手绝对关节角 |
| 卡数 / 每卡 batch / 累积 | **8 / 32 / 2**，有效 batch **512** |
| 训练量 | 10 epoch，约 **4500 微步 / 2250 次优化器更新** |
| LR | **1e-5**，warmup 5% + 原 cosine；不随卡数翻倍 |
| 其余 | seed 42、bf16、ZeRO-2、原冻结策略、原 loss 权重 |
| RTC | 原 1～12 步非空前缀，空前缀概率 25% |
| 保存 | 每 **500 微步**，含完整恢复状态；保留最近 8 份 |

对齐原来生成 RTC step_9000 的设置：同步 step_18000 初始化、10 epoch、LR 1e-5、有效 batch 512。旧 4×32×4 改为新 8×32×2，约每 epoch 225 次更新；保存周期由 1000 改为 500 微步，同为约 250 次更新。**新 step_4500 对应旧 step_9000 的训练量**，不是少训一半。卡数变化会改变随机数分配、浮点归约及 epoch 尾部补齐，现有 LR 调度的 world-size 缩放也不保证逐微步数值完全相同。

这里 `finetune_ckpt_path` 只加载模型权重，AdamW 动量/方差与 LR 进度重新初始化；这是原来从同步模型切换到 RTC 的 finetune 方式，没有更换优化算法。`resume_ckpt_path=null`，不使用 4 卡的优化器恢复状态。与刚停止的 A/B（20 epoch、LR 5e-5）是不同实验。

## 手动启动

启动增强长训；命令前台运行，建议在已有 tmux 会话内执行：

```bash
cd /gaozt-test1/fyhong/OpenWAM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=29652 \
  bash scripts/train_wuji_rtc_history.sh history
```

可选：如果之后需要严格排除卡数/随机数差异，另跑同设置的 8 卡干净对照。**本轮不必先执行它**，两者不能同时占用同一组卡：

```bash
cd /gaozt-test1/fyhong/OpenWAM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=29651 \
  bash scripts/train_wuji_rtc_history.sh clean
```

只展开配置，不加载模型、不训练：

```bash
cd /gaozt-test1/fyhong/OpenWAM
bash scripts/train_wuji_rtc_history.sh clean --print-config
bash scripts/train_wuji_rtc_history.sh history --print-config
```

输出：`outputs/wuji_rtc_history/{clean,history}/<运行时间>/`。日志、实际配置、初始化权重及源码 SHA256 在对应 `launches/`。脚本加锁防止两组同时争用 8 卡；不按显卡利用率阻止启动，共享服务器仍需有足够显存。

W&B 默认 online，复用原 `wandb/.netrc` 和项目设置，不打印密钥。增强组增加 `rtc_history/*` 指标，均为全局 batch 的占比：`prefix_fraction`、`requested_fraction`、`selected_fraction`、`changed_fraction`、`fallback_fraction`、`k1_fraction`～`k3_fraction`。

## 增强规则

配置：`configs/train_wuji_rtc_history.yaml` 中 `training.rtc_state_history`，默认关闭，`history` 启动入口显式打开。原训练入口和旧 XYZ 增强行为保留；新旧两种增强不允许同时开启。

- 仅对非空前缀样本，以 30% 概率申请增强；空前缀保持原样。
- 等概率申请过去 1、2、3 帧的完整 state；同一个历史时刻覆盖双臂 XYZ/旋转与双手，无左手应急逻辑。
- 本轮 `max_translation_m`、`max_rotation_deg`、`max_joint_rad` 均为 `null`：取消先前 25 mm / 3° / 0.05 rad 的额外筛选，不制造随机姿态，不外推历史 state。
- 保留有限数值和有效旋转检查；历史帧不存在、越过任务边界或无效时，向更近帧回退，无合法候选则不增强。可选幅度限制功能仍支持显式数值，便于复现旧版。
- 不跨 episode、有效片段起点或任务指令边界。图像、真实前缀、监督 action、原始数据文件保持原样，沿用原归一化及 58→80 映射。
- 增强采样不推进原训练随机数流；关闭时不读历史，不增加模型参数。

## 已做验证与局限

训练/RTC 回归测试包括关闭时 loss、梯度、随机数状态及 checkpoint 键保持一致，空前缀不增强，真实数据读取/归一化、完整 state 传入模型、无上限与可选上限、无效旋转及边界回退。最终复核 **49 项测试通过**，记录在 `outputs/wuji_rtc_history/final_preflight/tests.log`。另以真实数值数据、小型 Wan/ActionDiT 和合成图像/文本特征，跑通 **8 卡、每卡 batch 32、累积 2、bf16、ZeRO-2** 的 3 次优化器更新；验证梯度、LR 步进、增强指标归约、权重/完整状态保存，以及恢复后重放的 loss、增强采样和权重逐项一致。结果在 `final_preflight/distributed_final/summary.json`，工具为 `tools/audit_rtc_history_distributed.py`。这不等同于完整模型显存/吞吐测试；未启动本轮完整模型长训，W&B 实际在线写入仍在手动启动时验证。

不设幅度上限后，理论约 30% 非空前缀、22.5% 全部样本申请增强，仅少量历史边界/无效数据回退。实际训练比例以 W&B 为准。当前 114 个 episode / 115176 个窗口的采样审计在 `outputs/wuji_rtc_history/preflight_uncapped/input_audit.json`；此前有上限结果保留在 `preflight/input_audit.json`，不要混用。

111528 个完整窗口、与真机命令速度/手部活动程度匹配后的无上限增强候选位置差距 P90 约 20.0～20.6 mm，与 9/29 较接近，仍未充分覆盖当天约 33～35 mm 的较大偏差。取消上限是为了让完整历史 state 的采样更直接，不代表扩大了很多覆盖范围。首轮检验时序增强收益，不再同时加空间噪声或自生成前缀；分别验证干净、小偏差与较大偏差场景，避免只适应某一天。

复核输入分布（CPU，不训练）：

```bash
cd /gaozt-test1/fyhong/OpenWAM
/gaozt-test1/fyhong/envs/statewam/bin/python tools/audit_rtc_state_history.py \
  --output outputs/wuji_rtc_history/preflight_uncapped/input_audit.json
```

先在新 step_1000 / 2000 / 4500 检查收益趋势，并把新 step_4500 与旧 RTC step_9000 做相同离线输入、噪声种子的续写/偏离压力测试，比较双臂与双手接缝反向、动作误差和干净输入能力，再做真机任务验证；不能只看 loss。若需要严格归因增强效果，再执行可选 clean。

## Git 保存

最初快照：`codex/rtc-state-history-before-20261008_223816`（`0f3698e7`）；取消上限前快照：`codex/rtc-history-before-uncapped`（`ddb633d8`）。当前准备版本：`codex/rtc-state-history`。当前工作分支与原暂存区保留，没有切换项目或清除其他未提交工作。

```bash
cd /gaozt-test1/fyhong/OpenWAM
git show --stat codex/rtc-state-history
git diff codex/rtc-state-history-before-20261008_223816 codex/rtc-state-history
```
