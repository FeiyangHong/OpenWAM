# Wuji TrainingRTC 接缝加权训练

当前方案：**只启动 B，从原同步 step18000 训练 10 epoch**，以已有 history4500 作历史基准，暂不重训 A。不自动启动训练。

## 配置

| 项目 | B：seam |
|---|---|
| 初始化 | `outputs/grasp_anything_base/2026-09-16_21-50-22/checkpoint_step_18000.safetensors` |
| 加权 | 非空前缀后第 1～3 步权重 2，其余 1 |
| 预算 | 10 epoch |
| GPU / 每卡 batch / 累积 | 8 / 32 / 2，有效 batch 512 |
| 学习率 | 1e-5，cosine，warmup 5%，最低 1e-7 |
| 优化器 | 新 AdamW，betas=(0.9,0.95)，weight decay=0.01，梯度裁剪 1 |
| 精度 / 内存 | bf16、ZeRO-2、梯度检查点，无 CPU offload |
| RTC | 最大前缀 12；空前缀概率 25% |
| 历史 state | 非空前缀样本中 30% 使用过去 1～3 帧完整 state，不限幅 |
| 数据 | 原 grasp_anything_eef_rot6d_col，绝对 EEF、column ROT6D、quantile、多视角 |
| 种子 / 保存 | 42 / 每 500 micro-step，保留最近 12 个 |

与旧 history 实验使用同一同步初始化、数据、增强、预算和超参数，主要训练变化为接缝加权。历史基准无法排除不同运行环境和非确定性；若结果含混，再补同版本 baseline。

只加载同步18000权重，不恢复旧优化器、scheduler 和步数，与此前 history 实验方式相同。新 run 从 step0 计数，不是从 history4500 再训 10 epoch。

加权按实际前缀长度定位，覆盖双臂 XYZ、旋转和双手所有有效维度；前缀和 padding 不计 loss，空前缀不加权；每个样本按有效权重总和归一化。原训练命令默认不开启，动作标签、视频 loss、模型结构和推理硬前缀不变。

## 查看配置（不开始训练）

```bash
cd /gaozt-test1/fyhong/OpenWAM
bash scripts/train_wuji_rtc_seam.sh seam --print-config
```

确认初始化为同步 `checkpoint_step_18000.safetensors`，`rtc_seam_loss.enabled: true`，`rtc_state_history.enabled: true`。

## B 长训命令

在 tmux 中手动执行，确认显存可用即可；脚本不结束其他进程，沿用已有 history 训练互斥锁。

```bash
cd /gaozt-test1/fyhong/OpenWAM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=online \
  RTC_INIT_CKPT=/gaozt-test1/fyhong/OpenWAM/outputs/grasp_anything_base/2026-09-16_21-50-22/checkpoint_step_18000.safetensors \
  bash scripts/train_wuji_rtc_seam.sh seam
```

输出：`outputs/wuji_rtc_seam/seam/日期时间/`；启动配置、源码快照、初始化 SHA256 和日志在该组 `launches/`。W&B 沿用 `openwam` 项目，名称包含组名和初始化 step。B 的加权 loss 与旧均匀 loss 定义不同，不能直接按训练 loss 高低判断收益。

## 比较

优先比较新 step4500 与旧 history4500。早期 step500 / 1000 / 1500 可与原 history 目录对应权重比较，不要求它们胜过已训完整预算的旧4500。固定离线窗口、图像、state、前缀和噪声种子，检查首 1～3 步回拉、接缝速度比、整段误差、双手及空前缀。之后在相同 SDK/滤波参数下比较真机任务和停顿；加权收益需实测。

checkpoint step 是 micro-step，累积 2 次才更新一次优化器。中断恢复需要该次输出的完整训练状态及相同配置；重复上述命令会从同步18000新开训练。

## 可选：补跑 A（目前不必执行）

同样从同步18000训练，保留 history 增强，仅关闭接缝加权。

```bash
cd /gaozt-test1/fyhong/OpenWAM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=online \
  RTC_INIT_CKPT=/gaozt-test1/fyhong/OpenWAM/outputs/grasp_anything_base/2026-09-16_21-50-22/checkpoint_step_18000.safetensors \
  bash scripts/train_wuji_rtc_seam.sh baseline
```

## 验证

接缝实现及原 RTC/history/共享 loss/trainer 共 83 项 CPU 测试通过；关闭或 weight=1 保持原 loss/RNG，开启不改变前向输入、视频 loss 和随机序列。配置与初始化路径另经核对；记录在 `outputs/rtc_seam_prepare_20261012/`。未启动完整 8 卡训练，未验证 W&B 在线连接。
