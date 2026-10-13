# Wuji TrainingRTC：前缀重点采样长训

功能已应用到 H20 正式仓库，通过104项CPU回归测试及配置解析；下一轮训练尚未启动。不再需要运行待应用包的 `activate.py`。

## 训练设置

| 项目 | 当前方案 |
|---|---|
| 初始化 | 原同步 `grasp_anything_base/2026-09-16_21-50-22/checkpoint_step_18000.safetensors` |
| 预算 | 10 epoch，4卡预计约9000 micro-step／2250次优化器更新 |
| GPU / batch / 累积 | 4卡 / 每卡32 / 4，有效batch512 |
| 学习率 | 1e-5，cosine，warmup 5%，最低1e-7 |
| 优化器 | 新AdamW，betas=(0.9,0.95)，weight decay=0.01，梯度裁剪1 |
| 精度 | bf16、ZeRO-2、梯度检查点，无CPU offload |
| state增强 | 非空前缀样本中30%替换为过去1～3帧完整state，不限幅 |
| 接缝加权 | 非空前缀后首3步权重2，其余1，按有效权重总和归一化 |
| 动作 / 数据 | 原绝对EEF＋双手关节，column ROT6D，原数据及多视角，seed42 |
| 保存 | 4卡每1000 micro-step保存，保留12个checkpoint |

| 前缀步数 | 0 | 1～7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 概率 | 10% | 合计10%，均分 | 15% | 15% | 35% | 10% | 5% |

本轮算法改动为前缀采样分布；保留history和首3步加权。不是接着刚完成的step4500追加训练，而是从相同同步18000起点重新训练，便于与刚完成的seam实验比较。

PMF表包括0步，启用时覆盖原 `zero_prefix_probability: 0.25`，不会叠加旧概率。短有效窗口仍截短前缀并保留有效后缀，所以实际采样频率可能偏离表中值。history条件概率仍为30%；因非空前缀占比提高，全样本中的名义触发比例由22.5%变为27%。

## 查看配置（不训练）

```bash
cd /gaozt-test1/fyhong/OpenWAM
CUDA_VISIBLE_DEVICES=3,4,5,6 bash scripts/train_wuji_rtc_prefix_pmf.sh seam --print-config
```

确认 `rtc_prefix_sampling.enabled=true`、`rtc_state_history.enabled=true`、`rtc_seam_loss.enabled=true`，初始化为同步18000，`num_epochs=10`、`batch_size=32`、`gradient_accumulation_steps=4`、`save_steps=1000`。

## 手动长训

建议在tmux中执行。自行确认所选4卡显存够用；脚本不停止其他进程。

```bash
cd /gaozt-test1/fyhong/OpenWAM
CUDA_VISIBLE_DEVICES=3,4,5,6 WANDB_MODE=online \
  RTC_INIT_CKPT=/gaozt-test1/fyhong/OpenWAM/outputs/grasp_anything_base/2026-09-16_21-50-22/checkpoint_step_18000.safetensors \
  bash scripts/train_wuji_rtc_prefix_pmf.sh seam
```

脚本按所选GPU数量自动配置：4卡为accu4／保存间隔1000，8卡为accu2／保存间隔500；两者有效batch均为512、训练均为10 epoch。若恢复8卡，将命令的GPU列表改为8张可用卡即可；未设置GPU列表时仍沿用原8卡默认值。

模型输出：`outputs/wuji_rtc_prefix_pmf/seam/日期时间/`。
启动配置、源码归档、初始化校验和、终端日志：该组 `launches/`。
W&B沿用 `openwam` 项目，run名带 `prefix_pmf`；保留原有loss、LR和history指标。

重复该命令会从同步18000新开实验，不是断点恢复。完整恢复须使用本次训练状态并保持PMF等配置不变；单个safetensors只含权重。

## 对照基准

刚完成的均匀前缀＋history＋首3步加权实验：

`outputs/wuji_rtc_seam/seam/2026-10-12_11-47-50/checkpoint_step_4500.safetensors`

下一轮优先比较对应进度的checkpoint，如本轮4卡新step9000对上述8卡step4500（以实际完成10 epoch为准）；不要直接按相同micro-step编号比较。4卡与8卡的批次划分和随机数流不同，不保证逐步数值一致。固定离线输入、噪声种子、SDK与滤波参数，检查首3步回拉、接缝速度、整段动作和双手表现。

若另做“不加接缝权重”的实验，将组名改为 `baseline`。注意这**只关闭接缝加权，history仍开启**；当前命令不是“关闭history、只改前缀分布”的实验。

未指定 `training.rtc_prefix_sampling` 或设为 `enabled=false` 时，原采样逻辑与RNG调用不变。原同步、RTC部署入口和checkpoint结构不变。
