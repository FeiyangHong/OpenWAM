# Grasp Anything Wuji 训练用法

仓库根目录为 `/gaozt-test1/fyhong/OpenWAM`。

## 转换数据

```bash
cd /gaozt-test1/fyhong/OpenWAM
conda activate openwam

python scripts/prepare_grasp_anything_openwam.py \
  --source data/grasp_anything/grasp_anything_eef_rot6d \
  --destination data/grasp_anything/grasp_anything_eef_rot6d_col
```

默认创建绝对视频软链接，不会修改原始数据。目标目录非空时需显式加 `--overwrite`；需要复制视频时加 `--copy-videos`。

### 转换新的同格式 v2 数据集

新数据集必须与原始 Wuji/Grasp Anything 数据契约一致：`action` 和
`observation.state` 均为 58D，原始排列是
`[L EEF9, R EEF9, L hand20, R hand20]`，两个 EEF 的 rot6d 均为旋转矩阵前两行，
并包含 head、left wrist、right wrist 三路视频。
源数据可以没有 `meta/stats.json`；转换器会从转换后的 action/state 重新计算并生成目标
`meta/stats.json` 和 `meta/normalization_stats.npy`。

在仓库根目录执行；只需将 `WUJI_V2_SOURCE` 和 `WUJI_OPENWAM_TARGET` 改成新数据的实际路径：

```bash
cd /gaozt-test1/fyhong/OpenWAM
conda activate openwam

WUJI_V2_SOURCE=/gaozt-test1/fyhong/OpenWAM/data/spray_water/spray_water_eef_rot6d
WUJI_OPENWAM_TARGET=/gaozt-test1/fyhong/OpenWAM/data/spray_water/spray_water_eef_rot6d_col

python scripts/prepare_grasp_anything_openwam.py \
  --source "$WUJI_V2_SOURCE" \
  --destination "$WUJI_OPENWAM_TARGET"
```

该命令不修改源数据，默认让目标数据的视频指向源数据的绝对软链接。如果转换结果之后需要
单独传到其他机器，首次转换时复制视频：

```bash
python scripts/prepare_grasp_anything_openwam.py \
  --source "$WUJI_V2_SOURCE" \
  --destination "$WUJI_OPENWAM_TARGET" \
  --copy-videos
```

目标目录必须与源目录不同。目标目录已经存在且非空时脚本会拒绝覆盖；确认需要重新生成后
才添加 `--overwrite`。转换完成后可直接用该目录训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
FINETUNE_CKPT_PATH=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
OUTPUT_PATH=outputs/新数据集_wuji \
BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS=2 \
NUM_EPOCHS=20 \
LEARNING_RATE=5e-5 \
MIXED_PRECISION=bf16 \
ZERO_STAGE=2 \
USE_GRADIENT_CHECKPOINTING=true \
DATASET_NUM_WORKERS=8 \
SAVE_STEPS=1000 \
SAVE_FULL_STATES_FOR_RESUME=true \
KEEP_LAST_K_CKPTS=5 \
bash scripts/train_grasp_anything_wuji.sh \
  dataloader.dataset_dir="$WUJI_OPENWAM_TARGET"
```

训练前检查转换结果；数量应与新数据集自身的 episode 和三路视频数量一致：

```bash
find "$WUJI_OPENWAM_TARGET/data" -name '*.parquet' | wc -l
find "$WUJI_OPENWAM_TARGET/videos" -name '*.mp4' | wc -l
find -L "$WUJI_OPENWAM_TARGET/videos" -type l -print
```

最后一条命令应无输出；有输出表示存在失效视频软链接。

## 验证

```bash
find data/grasp_anything/grasp_anything_eef_rot6d_col/data -name '*.parquet' | wc -l
find data/grasp_anything/grasp_anything_eef_rot6d_col/videos -type l | wc -l
python scripts/train.py dataloader=wuji_real_task --cfg job
```

预期为 75 个 parquet、225 个视频链接，模型 `action_dim` 和 `state_dim` 均为 80。

## 单卡 20-step debug

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
FINETUNE_CKPT_PATH=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Real-Dexterous-Hand-Wuji \
OUTPUT_PATH=outputs/grasp_anything_wuji_debug \
BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
DATASET_NUM_WORKERS=0 \
OFFLOAD_OPTIMIZER_DEVICE=cpu \
DEBUG=true \
bash scripts/train_grasp_anything_wuji.sh \
  dataloader.dataset_dir=/gaozt-test1/fyhong/OpenWAM/data/grasp_anything/grasp_anything_eef_rot6d_col
```

单张约 96 GB H20 使用 ZeRO-2 时需要将 optimizer offload 到 CPU，否则首次分配 Adam 状态会 OOM。确认 loss 有限、三视角视频可读，并检查输出目录含 `config.yaml`、`normalization_stats.npy` 和 safetensors。

## 正式训练

训练前在同一个 shell 会话中启用仓库内的 W&B 凭据和配置。这样训练使用
`-harbin-institute-of-technology/openwam`，不会依赖其他目录的 W&B 登录状态：

```bash
export NETRC="$PWD/wandb/.netrc"
export WANDB_CONFIG_DIR="$PWD/wandb/config"
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
FINETUNE_CKPT_PATH=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
OUTPUT_PATH=outputs/grasp_anything_wuji \
BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS=2 \
NUM_EPOCHS=20 \
LEARNING_RATE=5e-5 \
MIXED_PRECISION=bf16 \
ZERO_STAGE=2 \
USE_GRADIENT_CHECKPOINTING=true \
DATASET_NUM_WORKERS=8 \
SAVE_STEPS=1000 \
SAVE_FULL_STATES_FOR_RESUME=true \
KEEP_LAST_K_CKPTS=5 \
bash scripts/train_grasp_anything_wuji.sh \
  dataloader.dataset_dir=/gaozt-test1/fyhong/OpenWAM/data/grasp_anything/grasp_anything_eef_rot6d_col
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
FINETUNE_CKPT_PATH=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
OUTPUT_PATH=outputs/grasp_anything_wuji \
BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS=2 \
NUM_EPOCHS=20 \
LEARNING_RATE=5e-5 \
MIXED_PRECISION=bf16 \
ZERO_STAGE=2 \
USE_GRADIENT_CHECKPOINTING=true \
DATASET_NUM_WORKERS=8 \
SAVE_STEPS=1000 \
SAVE_FULL_STATES_FOR_RESUME=true \
KEEP_LAST_K_CKPTS=5 \
bash scripts/train_grasp_anything_wuji.sh \
  dataloader.dataset_dir=/gaozt-test1/fyhong/OpenWAM/data/spray_water/spray_water_eef_rot6d_col
```

FINETUNE_CKPT_PATH=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
FINETUNE_CKPT_PATH=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Real-Dexterous-Hand-Wuji \

/gaozt-test1/fyhong/OpenWAM/data/spray_water/spray_water_eef_rot6d_col

有效 global batch 为 `BATCH_SIZE * GPU 数 * GRADIENT_ACCUMULATION_STEPS`。恢复中断训练时用 `RESUME_CKPT_PATH=<run目录>` 替换 `FINETUNE_CKPT_PATH`，二者不能同时设置。

部署端输入输出是转换后的 raw 58D：`[L EEF9, L hand20, R EEF9, R hand20]`；rot6d 为旋转矩阵前两列，不要再次转换。
