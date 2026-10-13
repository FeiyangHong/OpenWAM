# H20 TrainingRTC 对照训练

绝对 EEF / column rot6d。A/B 先跑；C/D 等选出 checkpoint 后再跑。脚本不会自动启动另一组。

本轮已交给 `outputs/eval_then_rtc_ab_20260930/` 调度器：评估最多8卡；评估完整审计通过后，A/B每组4卡、合计最多8卡，显存充足则并行，不足则A优先排队串行。两组动态选择互不重叠的显卡，允许与其他人的作业共享。**不要再手动重复启动 A/B。** 资源与交接规则见该目录 `HANDOFF.md`。

| 组 | 起点 | epoch | LR | state 增强 |
|---|---|---:|---:|---|
| A | 基模 step_154000 | 20 | 5e-5 | 关 |
| B | 原 20 epoch 同步模型 step_18000 | 20 | 5e-5 | 关 |
| C | 同一个获选 checkpoint | 5 | 1e-5 | 关 |
| D | 与 C 完全相同 | 5 | 1e-5 | 开 |

统一：4 卡、每卡 batch 32、累积 4（有效 batch 512）、seed 42、bf16、ZeRO-2、warmup 5% + cosine；RTC 前缀 0–12 步、无前缀概率 25%。每组新建优化器，不恢复旧优化器。A/B 匹配 RTC 阶段预算，不匹配包含同步预训练在内的总预算。

## 先启动 A/B

以下仅为脱离调度器时的手动串行示例：A 成功退出后再用同一组卡启动 B。手动并行时需指定两套互不重叠且显存充足的4卡，每组4卡、合计最多8卡。当前自动队列运行期间不要执行。

```bash
cd /gaozt-test1/fyhong/OpenWAM
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29641 \
  bash scripts/train_wuji_rtc_ablation.sh A
```

```bash
cd /gaozt-test1/fyhong/OpenWAM
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29642 \
  bash scripts/train_wuji_rtc_ablation.sh B
```

只看完整配置，不训练：

```bash
cd /gaozt-test1/fyhong/OpenWAM
bash scripts/train_wuji_rtc_ablation.sh A --print-config
bash scripts/train_wuji_rtc_ablation.sh B --print-config
```

输出：`outputs/wuji_rtc_ablation/{A_foundation,B_sync}/<run时间>/`；启动日志、展开配置和初始权重 SHA256 在各组的 `launches/`。每 1000 微批步保存，保留最近 21 份；累积 4 时约每 250 次优化器更新保存。训练结束按原 trainer 清理恢复状态、保留权重。W&B 使用原仓库登录配置。

约 5/10/15/20 epoch 比较干净续写、state 偏差压力测试、连续 chunk、轨迹误差。真机仍用固定 10 步调度；不要只按 loss 选模型。

## 后续 C/D

先把下面 `RTC_INIT_CKPT` 改成评估选出的**同一个完整权重文件路径**，两个终端填完全相同的值。这里不能预先指定尚未生成的 checkpoint。

```bash
cd /gaozt-test1/fyhong/OpenWAM
export RTC_INIT_CKPT=/绝对路径/获选run/checkpoint_step_XXXX.safetensors
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29643 \
  bash scripts/train_wuji_rtc_ablation.sh C
```

```bash
cd /gaozt-test1/fyhong/OpenWAM
export RTC_INIT_CKPT=/绝对路径/获选run/checkpoint_step_XXXX.safetensors
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29644 \
  bash scripts/train_wuji_rtc_ablation.sh D
```

输出：`outputs/wuji_rtc_ablation/{C_clean,D_state_aug}/<run时间>/`。C/D 只改变增强开关。

## 增强开关

配置文件：`configs/train_wuji_rtc_ab.yaml`。

```yaml
training:
  rtc_state_augmentation:
    enabled: false
    probability: 0.10
    max_translation_m: 0.005
```

D 对有前缀样本以 10% 概率向双臂 state 的 XYZ 加**同一个三维平移**，位移模长均匀采样 0–5 mm、方向各向同性；按各自统计量归一化并截断。无前缀样本不增强。旋转、手指、图像、前缀和监督 action 不变；没有左右侧专属参数，也没有左手损坏/应急逻辑。不修改原始数据文件。

这是位置观测偏差增强，不是机械延迟模拟。增强关闭时不增加随机数消耗；增强只在训练 loss 路径执行，推理和同步控制不受影响。是否有效由 C/D 对照决定。
