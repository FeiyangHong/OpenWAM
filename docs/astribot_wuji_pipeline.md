# Astribot + Wuji 真机数据适配：当前流程核对

核对日期：2026-09-21。基于 h20 原仓库 HEAD `0fb80b3951db9d0dca3f80c612cb238bba7e1bfb`
及其未提交工作区，不将当前源码版本等同于历史训练当时使用的版本。

## 当前数据与实际任务

两类数据都位于 `/gaozt-test1/fyhong/OpenWAM/data/`，每类分别保留原始 `*_eef_rot6d`
和转换后 `*_eef_rot6d_col` 两个目录。

| 数据 | episode | frame | 三路视频数 | metadata FPS | task 文本 |
| --- | ---: | ---: | ---: | ---: | --- |
| grasp_anything | 114 | 115,176 | 342 | 30 | Pick up the items from the table and place them in the trash bin. |
| spray_water | 234 | 172,043 | 702 | 30 | Pick up the spray bottle, pump it, then spray water on the flowers. |

转换后视频均为指向源数据的有效软链接。历史迁移文档中的 75 episodes / 63,725 frames 是早期样例，
不能用作当前数据集的数量验收。FPS 是数据元信息，不据此推断真实控制器频率。

## 从采集数据到训练样本

```text
Astribot 双臂 + Wuji 双手，LeRobot v2 parquet + 三路视频 + 任务文本
    ↓ prepare_grasp_anything_openwam.py（输出新 bucket）
原始 raw58： [左 EEF9, 右 EEF9, 左手20, 右手20]，row rot6d
目标 raw58： [左 EEF9, 左手20, 右 EEF9, 右手20]，column rot6d
    ↓ WujiRealTaskDataset
quantile 归一化 → scatter 到 canonical 80D，生成有效维度 mask
    ↓ scripts/train_grasp_anything_wuji.sh → scripts/train.sh → scripts/train.py
Foundation 或 Wuji checkpoint 初始化 → 新 run 的权重、配置和统计
```

1. **动作语义是绝对量。** EEF9 = xyz3 + rot6d6；左右手各 20 个关节。action 和 observation.state
   都按此契约转换，不计算 delta，不额外做 base/world 坐标变换。这里只确认进入本仓库的契约；
   原始传感器到 EEF 数据的生成、标定和单位来源未在本次检查中重做。
2. **rot6d 行转列是旋转表示转换。** 原始前两行先通过 Gram–Schmidt 恢复完整旋转矩阵，再取前两列：
   `[R00,R10,R20,R01,R11,R21]`。不是仅交换现有六个数，也不改变物理坐标系。
3. **离线转换同时处理 action/state。** parquet 使用 zstd；默认只链接视频，原始数据保持不变。
   生成 info、modality、episodes、stats、normalization_stats 和 adapter_manifest。
4. **两类统计用途不同。** `stats.json` 分别描述 action/state；部署和训练共用的
   `normalization_stats.npy['eef']` 从二者合并的所有行计算。quantile 归一化会把 q01/q99 外的值裁剪到边界。
   目标 raw58 中的 rot6d 维度 `3:9`、`32:38` 固定 identity 统计，保持旋转表示不缩放。
5. **reader 虽继承 LeRobotV3Reader，但读的是适配后的 v2 episode 文件布局。** 它覆写 episode 索引，
   从 tasks.jsonl 取 prompt，从首帧 state 取 proprio，从 action 列取动作窗口。
   正式设置 `rot6d_convention=column`，已经离线转换的数据不再次转换。

canonical 映射如下，表内区间用 Python 左闭右开形式；YAML 用闭区间字符串。

| 块 | 目标 raw58 | 模型80D | YAML |
| --- | --- | --- | --- |
| 左 EEF | `0:9` | `0:9` | `0-8` |
| 左手 | `9:29` | `10:30` | `10-29` |
| 右 EEF | `29:38` | `34:43` | `34-42` |
| 右手 | `38:58` | `44:64` | `44-63` |

**模型 head 保持 80D，只有 58 个槽有效。** 未使用槽位和对应监督 mask 都为零。
不能改为 raw58 的 identity mapping，也不能套用 RoboTwin 的 raw20 或 joint14 契约。

默认 `num_frames=33`、`video_stride=4`、`window_stride=1`：

- 动作取窗口最前面的 32 个原始采样步，输出 `[32,80]`；完整窗口 mask 有 `32×58=1856` 个有效值。
- proprio 取第一帧，输出 `[1,80]`，mask 有 58 个有效值。
- 视频按 0、4、…、32 取 9 个时间点；每个时间点组装 head、left wrist、right wrist 三个视角。
  reader 最终输出 9 张宽 320、高 384 的组合图像，不是 27 个独立时间帧。
- 视频步长只下采样视频；不会把 action 也每四步取一次。episode 尾部不足的窗口依靠 padding/mask 处理。

## 训练与已有结果目录

训练入口为 `scripts/train_grasp_anything_wuji.sh`，选择 `dataloader=wuji_real_task`，把环境变量转为 Hydra overrides。
喷水任务复用同一 reader，通过 `dataloader.dataset_dir=.../spray_water_eef_rot6d_col` 更换 bucket。

已读取以下实际输出目录的 config 和文件清单：

| 原仓库 outputs 下的 run | 初始化来源 | 本次看到的 safetensors step |
| --- | --- | --- |
| `grasp_anything_wuji/2026-09-11_22-22-59` | Real-Dexterous-Hand-Wuji | 4980 |
| `grasp_anything_base/2026-09-16_21-50-22` | Pretrain-Foundation-Model | 14000、15000、16000、17000、18000 |
| `spray_water_wuji/2026-09-11_23-14-41` | **Pretrain-Foundation-Model** | 13460 |

目录名字里的 “wuji” 不保证初始化来自 Wuji checkpoint；应读 `training.finetune_ckpt_path`。
三个 run 的实际配置均使用 column rot6d、上述 canonical mapping，以及 80D action/state head，
并带 normalization_stats.npy。权重文件约 24.8 GB；本次只查存在和文件大小，没有加载大模型或验证权重全部内容。

`FINETUNE_CKPT_PATH` 为加载权重后开始新 run；`RESUME_CKPT_PATH` 为恢复完整训练状态，两者互斥。
当前 finetune loader 将 checkpoint 的模型配置作为基底，叠加当前模型配置；当前数据配置仍来自这次运行。
trainer 将数据集统计复制到新 checkpoint。`global_step` 每个 micro-batch 增加一次，
因此 `SAVE_STEPS` 不是梯度累积后的 optimizer step 数。

既有文档记录了两种初始化都取得良好真机效果；本次没有重新测量成功率，也不据此判断谁更好。
当前 converter 把整个 bucket 设为 train，并用全 bucket 拟合统计；新的持出评估需另外冻结 split，
用训练部分拟合统计后让验证/测试复用，不能仅修改 split 标签就视作独立测试。

## 部署边界

`openwam/deploy/model_loader.py` 的 `_UnifyAwareNormalizer` 已实现：

```text
输入 raw58 state（转换后顺序、column rot6d）
    → 在58D归一化 → scatter 80D → 模型
模型输出80D → gather 58D → 反归一化
    → 输出转换后顺序的 raw58 绝对动作
```

部署时必须使用该训练 run 自带的配置、unify_action_map 和 normalization_stats.npy。
控制器若仍接收 `[左 EEF, 右 EEF, 左手, 右手]` 或 row rot6d，需要在机器人客户端明确执行逆变换；
不能把服务返回的 58D 直接当成原始采集布局。当前 OpenWAM 侧契约已核对，具体机器人驱动/客户端边界、
控制频率和标定没有在本次执行中验证，也没有向机器人发送动作。

## 本次实际验证与复现

在 H20 openwam 环境、CPU 模式完成：

- 两个 bucket 的全量文件数量和所有视频链接检查。
- 每个 bucket 首、中、末 episode 的 action 和 state 转换对比：最大误差 **0**。
- 上述样本的旋转正交误差最多 `5.96e-7`，det(R) 与 1 的误差最多 `4.77e-7`。
- 训练/部署归一化一致性误差 **0**；q01/q99 裁剪域内的归一化/scatter/gather/反归一化回环误差最多 `1.20e-7`。
  裁剪范围外的原值不能通过量化分位归一化无损恢复。
- 两个 reader 各抽首个真实窗口，32×80 action、1×80 proprio、有效 mask、9 帧三视角图像全部通过，未发生重试替代样本。
- `scripts/train.py dataloader=wuji_real_task --cfg job` 成功。
- 已有 rot6d/unify 与部署归一化测试 **43 项通过**；在现有 statewam 环境运行，未改动共享环境。

证据目录（现均保留于 `/gaozt-test1/fyhong/OpenWAM/.h20-local/runs/`；历史 manifest 保留当时执行路径）：

- `20260921T070636Z-wuji-reader-669df0/`：两份 reader verification.json 及首帧图。
- `20260921T070844Z-wuji-audit-89734d/`：audit.json 和实际 Hydra 配置。
- `20260921T070844Z-foreground-3a3d6d/`：43 项既有测试日志。
- `20260921T070844Z-foreground-863823/`：早期独立快照方案的 6 项测试日志，仅作为历史记录；当前已改为原仓库同步。

报告 JSON 另存于 Windows `.local-h20/reports/`。复现入口为：

```powershell
python scripts/dev/h20.py sync
python scripts/dev/h20.py run --background wuji-audit -- env CUDA_VISIBLE_DEVICES= bash -c 'python scripts/dev/audit_wuji_data.py --output "$OPENWAM_RUN_DIR/audit.json"'
```

这是数据适配与远程开发验证，不包含新训练、模型质量评估或真实机器人闭环验收。
