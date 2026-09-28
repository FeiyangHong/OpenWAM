# TrainingRTC 训练用法

在 h20 的 OpenWAM 环境中执行。先改数据集、同步 checkpoint 和显卡编号，再跑 smoke；通过后跑正式训练。

## 1. 数据与起始权重

```bash
cd /gaozt-test1/fyhong/OpenWAM

# 抓取任务数据；喷壶任务使用下方列出的路径。
export DATASET_DIR=/gaozt-test1/fyhong/OpenWAM/data/grasp_anything/grasp_anything_eef_rot6d_col

# 改成你确认效果好的同步 run 目录，加载其中最新 checkpoint。
export SYNC_RUN=/gaozt-test1/fyhong/OpenWAM/outputs/grasp_anything_base/2026-09-16_21-50-22
export RTC_OUTPUT=/gaozt-test1/fyhong/OpenWAM/outputs/grasp_anything_base_rtc

# 从同步权重开始新的 RTC 训练，不恢复旧优化器。
unset RESUME_CKPT_PATH

# 使用与同步训练相同的仓库内 W&B 登录配置。
export NETRC="$PWD/wandb/.netrc"
export WANDB_CONFIG_DIR="$PWD/wandb/config"
```

喷壶数据：`/gaozt-test1/fyhong/OpenWAM/data/spray_water/spray_water_eef_rot6d_col`。
切换任务时，同时修改 `DATASET_DIR`、`SYNC_RUN`、`RTC_OUTPUT`。
`SYNC_RUN` 填包含 `config.yaml`、`normalization_stats.npy` 和权重的目录，不填单个权重文件。

## 2. Smoke：单卡跑 20 步

显卡 `0` 是示例，改成你要使用的空闲卡。单卡启用 optimizer CPU offload。

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
MASTER_PORT=29611 \
FINETUNE_CKPT_PATH="$SYNC_RUN" \
OUTPUT_PATH="${RTC_OUTPUT}_smoke" \
BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
NUM_EPOCHS=null MAX_STEPS=20 \
LEARNING_RATE=1e-5 \
MIXED_PRECISION=bf16 ZERO_STAGE=2 \
USE_GRADIENT_CHECKPOINTING=true \
OFFLOAD_OPTIMIZER_DEVICE=cpu \
DATASET_NUM_WORKERS=0 \
SAVE_STEPS=10 KEEP_LAST_K_CKPTS=2 \
SAVE_FULL_STATES_FOR_RESUME=false \
WANDB_MODE=online DEBUG=false \
bash scripts/train_grasp_anything_wuji.sh \
  dataloader.dataset_dir="$DATASET_DIR" \
  dataloader.num_frames=33 \
  training.rtc.enabled=true \
  training.rtc.max_prefix_steps=12 \
  training.rtc.zero_prefix_probability=0.25
```

通过标准：20 个优化器步完成、loss 有限、无 OOM；输出 run 目录包含配置、归一化统计和权重，W&B 上能看到训练曲线。这里使用 `DEBUG=false`，因为 debug 模式会跳过 W&B 初始化。

## 3. 正式训练：4 卡示例

将显卡列表改成实际使用的卡，`NPROC_PER_NODE` 与卡数一致。batch、梯度累积、workers 和保存间隔与同步训练的 4 卡命令一致；保留你设置的学习率 `1e-5`、10 epochs 和 20000 步上限，先达到哪个就结束。正式训练仍从 `SYNC_RUN` 开始。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MASTER_PORT=29612 \
FINETUNE_CKPT_PATH="$SYNC_RUN" \
OUTPUT_PATH="$RTC_OUTPUT" \
BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS=4 \
NUM_EPOCHS=10 \
MAX_STEPS=20000 \
LEARNING_RATE=1e-5 \
MIXED_PRECISION=bf16 ZERO_STAGE=2 \
USE_GRADIENT_CHECKPOINTING=true \
OFFLOAD_OPTIMIZER_DEVICE=none \
DATASET_NUM_WORKERS=8 \
SAVE_STEPS=1000 \
KEEP_LAST_K_CKPTS=4 \
SAVE_FULL_STATES_FOR_RESUME=true \
WANDB_MODE=online DEBUG=false \
bash scripts/train_grasp_anything_wuji.sh \
  dataloader.dataset_dir="$DATASET_DIR" \
  dataloader.num_frames=33 \
  training.rtc.enabled=true \
  training.rtc.max_prefix_steps=12 \
  training.rtc.zero_prefix_probability=0.25
```

- `BATCH_SIZE` 是每卡 batch；上例有效 batch 为 `32 × 4 × 4 = 512`，与同步训练一致。
- `MAX_STEPS` 按梯度累积前的微批步计数；上例 20000 步对应约 5000 次优化器更新。
- `num_frames=33` 对应 32 步动作；RTC 最大前缀 12 步，25% 样本使用零前缀。
- 显存不足时先降低 `BATCH_SIZE`；改单卡时使用 `OFFLOAD_OPTIMIZER_DEVICE=cpu`。
- `MASTER_PORT` 选未占用端口；smoke 和正式训练均在线记录到原有 W&B 项目 `-harbin-institute-of-technology/openwam`。
