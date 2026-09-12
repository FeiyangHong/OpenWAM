# R1 RoboTwin 工程验收报告

Goal ID：`r1/robotwin-smoke`。日期：2026-09-12。状态：工程与仿真 smoke 验收通过。

范围以 [GOAL.md](GOAL.md) 为准，运行入口见 [USAGE.md](USAGE.md)。已完成 4 卡 200 次 optimizer updates profiling、恢复到第 201 步、真实权重生成/重载与发布基线和 StateWAM 各自的 clean/randomized rollout。仅为工程验收，不能称为完整成绩复现或控制收益验证。

## 环境与来源

- OpenWAM 分支 `feat/statewam-r1-robotwin`，基点 `5be50682eb45ef5d2dace77cce88c3baa0c48446`。实验执行时实现尚未提交，因此除 HEAD 外记录工作区文件哈希；验收后按用户要求将 R1 实现、测试和文档固化为独立 Git 提交，以 Git 历史定位。
- 实际设备为 H20 96 GB，仅使用空闲的 GPU 0–3；GPU 4–7 属于其他任务，不使用。不是设计文档里预估的 A100。
- 模型环境：`/gaozt-test1/fyhong/envs/statewam`，Python 3.10、torch 2.7.1+cu128；借用既有 OpenWAM 环境的系统包，新增包安装在独立 venv。
- 仿真环境：`/gaozt-test1/fyhong/envs/robotwin`，Python 3.10、torch 2.4.1+cu121、SAPIEN 3.0.0b1、mplib 0.2.1、warp 1.13.0。
- 保留用户的 `/gaozt-test1/fyhong/RoboTwin`，另建 detached worktree `/gaozt-test1/fyhong/RoboTwin-statewam-r1`，固定 OpenWAM 已核对的 `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`。
- cuRobo 使用 v0.7.8 源码归档构建。归档不带 Git 元数据，安装包版本显示 `0.0.0`，不能以此字段替代源码版本；需结合归档校验值识别。使用 CUDA 12.8 编译器、`TORCH_CUDA_ARCH_LIST=9.0`。
- 执行官方安装脚本中的 SAPIEN UTF-8 和 mplib screw 检查补丁；备份原依赖文件并记录哈希。SRDF 后缀保留实际脚本行为，不照抄其不一致的注释。
- 通过官方 `script/update_embodiment_config_path.py` 生成资产的 cuRobo 路径配置。RGB 路径不使用 pytorch3d；未将可选点云能力标为通过。
- 用户原 RoboTwin checkout 位于 `96c1feab536306b50c26af200044fcdf126e8904`，验收时工作区干净；专用 worktree 的受版本控制文件也无改动。没有创建 submodule 或覆盖原 checkout。R1 固化提交仅发生在 OpenWAM，不提交 RoboTwin 改动，也不推送远端。
- 两个策略服务及所有本 Goal 的训练/检查进程已结束，不保留后台 GPU 服务。

所有下述证据路径以 `OpenWAM/statewam_artifacts/robotwin/` 为根，均不进入 Git。

### 环境重建要点

本机已配置好环境，不需要在每次 smoke 时重装。完整包版本在两个环境 manifest 中；模型 venv 通过 `--system-site-packages` 复用 `/gaozt-test1/zhanxch/miniconda3/envs/openwam`，并非完全独立于此基础环境。仿真 venv 不复用模型环境。

重建时先按[仓库 RoboTwin 安装说明](../../../../../benchmarks/robotwin/README.md)安装上述固定版本，安装 `setuptools==80.10.2` 以保留 SAPIEN 所需的 `pkg_resources`。cuRobo v0.7.8 解压至专用 worktree 的 `envs/curobo` 后，用仿真 Python 执行 `pip install --no-build-isolation -e <worktree>/envs/curobo`；本机编译参数为 CUDA 12.8、`TORCH_CUDA_ARCH_LIST=9.0`、`MAX_JOBS=8`，其他型号 GPU 不能照搬架构值。再运行本仓库 `prepare_robotwin_env.py` 和上游路径生成脚本。模型环境额外需要 pytest；ruff 仅用于静态检查，py-spy 仅用于诊断。

资产实际目录为 `/gaozt-test1/datasets/robotwin/sim-assets`，专用 worktree 的 `assets` 链接到该目录。SAPIEN 导入会影响 locale，客户端统一设置 `PYTHONUTF8=1`；无系统 EGL ICD 时使用 SAPIEN 自带 ICD，见 usage。RGB、Vulkan 和 cuRobo 已用真实环境验证，不使用 planner fallback。

## 数据与权重

- 发布权重：`OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random`，revision `9bb966823547bc959800b45e9bb4005e9c2a3db3`，`checkpoint_step_10740.safetensors`。
- 源权重 SHA256：`2cab63b550ff45a52fbde4adc40ee2bdb7d4bafe5fa0dd6879e41e22ca2ff95f`。
- 受控转换：`checkpoints/clean2random-state-init/initialization.json`，新增 330 个 State 张量；无未解释 missing/unexpected keys，无 meta 张量，gate 为零。
- 数据来自 `TianxingChen/RoboTwin2.0`，revision `785feb15aa4a4f532395ad2b1d2be5f28cb561ad`。下载与派生数据均在 `/gaozt-test1/datasets/robotwin/`。
- profiling 使用 `adjust_bottle/aloha-agilex_clean_50` 的 50 条训练 episodes，共 7188 个窗口；复用发布模型归一化统计，不重新拟合、不混入 randomized 评测数据。
- 这是工程 profiling 训练子集，不是持出效果测试。不能根据该子集的 loss 或 smoke 结果报告泛化收益。
- 模型和仿真环境清单：`manifests/model-environment.json`、`manifests/simulator-environment.json`；前者包含本轮训练数据文件哈希。
- 下载通过公开匿名访问及 HF 镜像完成；代码、文档和 manifest 不保存 HF token。
- Foundation 与 Wuji 本地源也已分别完成受控转换：`checkpoints/foundation-state-init/initialization.json`、`checkpoints/wuji-state-init/initialization.json`。两者均新增 330 个张量，无未解释缺失项、额外键或 meta tensor；本轮没有在这两条来源上追加训练或运行真机。
- 用户历史 Foundation/Wuji 真机成功 run 的训练曝光、输出权重与实际控制频率未重新审计；不以源权重转换通过替代这些历史事实。

交付的 StateWAM 权重：`runs/r1-profile-200-001/2026-09-12_18-50-22/checkpoint_step_201.safetensors`，SHA256 为 `909b54a5b591224c5b42f0e9878d20cb2d9f3e08e35c3b69d2d6855d5bc4da80`。同目录保留 step 200 和 201 及 full state；目录式 loader 选择最新的 step 201。配置、统计和权重的校验值见 `manifests/delivery-code-and-weights.json`，不得只拷贝 safetensors 后省略 sidecar。

## 已核验实现

- 新增 experimental StateWAM residual 架构，K=16、D=512、6 个读写层 `[4,9,14,19,24,29]`。原 world/action 30 层配对不变。
- 来源模式 `off/current/future/future_action`；读取 residual world tokens、带 mask 的语言/proprio context，以及按配置开启的动作 hidden tokens。current 模式不读已经融合未来的动作流。
- 独立零 gate、非零 reader/writer 初始化；不缓存跨调用或跨 episode State；保留当前帧因果保护。
- 参数冻结与 no_grad 分离。当前短训练冻结 world DiT 权重，训练 action、proprio 和 State；这不是重训发布基线的配方。
- 原训练器与部署 loader 仅在配置明确允许时构建 experimental 架构；原架构默认路径保留。

## 验收证据

| 检查 | 状态 | 证据 |
|---|---|---|
| 小模型、共享接口、checkpoint、RoboTwin 与 Wuji adapter 回归 | 通过，242 tests，0 failures / skips | `reports/tests/r1-delivery.xml`；早期 163/236 项记录仍保留 |
| 真实权重 gate=0 / off 等价 | 通过，四种模式最大绝对误差均为 0 | `reports/real-weight-smoke/verification.json` |
| 真实梯度 | 通过，gate 首步、reader/writer 后续非零有限梯度 | 同上 |
| 4 卡训练启动、权重和全状态保存 | 通过，2 optimizer updates | `runs/r1-smoke-train-002/profile/` |
| 200 optimizer updates profiling | 通过，四 rank 步数、有限梯度与曝光量已审计 | `runs/r1-profile-200-001/profile/`、`reports/profile-resume-audit.json` |
| 优化器恢复 | 通过，四 rank 的 Adam step/exp_avg/exp_avg_sq 哈希完全相同，继续到 201 | `runs/r1-resume-201-001/profile/`、同上审计 |
| 梯度累积与 checkpoint 重计算 | 通过，小型真实 Wan/ActionDiT 的合并 batch 与 micro-batch 梯度对照 | `tests/statewam/test_statewam.py`；非另一轮 4 卡累积训练 |
| 已有 Wuji 真机数据 batch 契约 | 通过，raw 58D/unified 80D 回环与 mask、三路图像 | 下节独立 engineering 证据，不计为 RoboTwin rollout |
| 完整 generate、compile/cache 检查 | 通过；近似优化误差单列，不声称精确等价 | 下节三个 generation 目录 |
| 真实训练权重保存/重载生成 | 通过，完整 32×20 动作逐元素一致 | `reports/generate-state-roundtrip-001/generation.json` |
| RoboTwin clean 环境与专家规划 | 通过 | `runs/r1-env-clean-004/environment_smoke.json` 和相机 PNG |
| 发布基线 clean/randomized policy rollout | 通过，各 1 episode | 下节逐 episode 表 |
| StateWAM clean/randomized policy rollout | 通过，各 1 episode，使用 step 201 | 下节逐 episode 表 |
| 四段视频解码与内容 | 通过，320×240，帧数符合上游规则，首末画面非空且变化 | `reports/rollout-video-audit.json`，含 SHA256 |

模块测试覆盖 State 来源隔离、当前视觉因果保护、padding/all-invalid/no-action、注册和优化器参数组、参数冻结下的梯度传播；不将小张量检查称为真实数据上完整训练。真实训练梯度另由四 rank 的逐步 JSONL 支撑。

### 已有真机数据只读检查

额外读取已有 `data/grasp_anything/grasp_anything_eef_rot6d_col` 的 episode 0、window 0，不下载、搬移或修改该数据。使用原 Wuji reader 分别读取 normalized raw 和 unified 表示：action 为 32×58 / 32×80，proprio 为 1×58 / 1×80，gather 回 raw 后逐元素一致。每个时间点 58 个有效槽，其余槽数值和监督 mask 均为零；原三路视频合成为 9 帧 320×384 图像，raw/unified 两次读取画面一致且非空。复用既有 quantile 统计，不重新拟合；metadata fps=30 仅表示该数据采样元信息，不据此推断真实机器人控制频率。

这项证据不属于 RoboTwin 仿真结果，单独保存在仓库根下 `statewam_artifacts/engineering/r1-robotwin-smoke/wuji-data-001/`，包含 `verification.json`、首帧图像、实际 parquet/三路视频与元数据哈希。只检查一个真实窗口的数据契约，没有在此 batch 上训练 StateWAM 或执行机器人动作。可用模型环境重验，输出需使用新目录：

```bash
python scripts/statewam/verify_wuji_data.py \
  --dataset data/grasp_anything/grasp_anything_eef_rot6d_col \
  --output statewam_artifacts/engineering/r1-robotwin-smoke/my-wuji-data
```

## 当前性能记录

训练启动 run 使用 4 卡、per-GPU batch=4、accumulation=1、global batch=16，bf16、ZeRO-2、gradient checkpointing。保持 384×320 拼接图像、33 个原始时间点、video stride=4，即 9 个视觉帧和 32 个动作。

正式 200 步 profiling 排除前 10 个 microsteps。rank 0 稳态平均计算时间 1.2503 秒，含数据等待/额外处理的吞吐 12.7842 samples/s；预处理约 0.5023 秒，占该计算区间约 40.2%。四 rank 最大峰值分配显存为 31,388,785,664 bytes（29.23 GiB）；rank 0 分配 28.81 GiB、预留 33,965,473,792 bytes。使用发布 checkpoint 记录的 per-GPU batch=4，不为填满显存改变输入几何；这不是吞吐最优 batch 搜索或全量微调显存结论。

200 步共曝光 3200 个样本；训练及保存墙钟 332.61 秒，对应 0.3696 GPU-hours，不含启动和其他审计。启动阶段另计 112.27 秒。权重保存约 22.11 秒，全状态保存约 53.22 秒。恢复耗时为 152.45–153.65 秒，恢复后额外 1 次 optimizer update、16 个样本曝光单独记在 resume run。`wait_and_bookkeeping_seconds` 不是纯磁盘 I/O，不作此解读；四 rank 的原始 JSONL 和汇总均保留。

短训练冻结 world DiT，训练 action、proprio 与 State；bf16、原 trainer/Accelerate DeepSpeed ZeRO-2、gradient checkpointing。State/base LR=1e-4，ActionDiT LR=1e-5，不使用 LR scheduler；训练/数据种子均为 42。六个训练后 gate（层 4/9/14/19/24/29）为 `[0.0005226135, 0.0006713867, 0.0002918243, -0.0004405975, -0.0002765656, 0.0000996590]`，均非零且有限；这证明参数发生更新，不证明其具有控制收益。

## 生成与部署协议

真实同输入生成记录分别为 `reports/generate-baseline-001/`、`reports/generate-state-init-001/`、`reports/generate-state-trained-001/`。三者都保存 eager/repeat、compiled/repeat、compiled_cached 的 32×20 原始动作数组及 `generation.json`。使用同一训练样本、prompt 与 seed=42，关闭数据颜色增强；原模型与零 gate State 的完整 eager 输出逐元素一致，三者的 compiled 重复输出也逐元素一致。

bf16 下，三个模型 compiled 相对 eager 的最大原始动作分量差均为 0.0078125，cached 相对 eager 均为 0.025390625；不同分量单位不同，这些值不是统一物理误差。cache 在 10 步内跳过 6 次，threshold=0.99、max_skips=3。仅证明编译及缓存路径可运行并记录误差，没有认定缓存优化在 StateWAM 上无损。

State step 201 的单样本重复生成为 eager 0.771 秒、compiled 0.585 秒、cached 0.285 秒；首次 compiled 36.87 秒。不是 p50/p95 性能 benchmark，不包含客户端传输和控制等待。真实保存/重载审计另外耗时约 22.43/169.29 秒，重载 eager 最大误差为 0。最初三次 generation 记录早于 `compile_stats` 字段加入，但当时已检查 compiled handle 未回退，保存数组在交付时重新核对。

闭环配置保持 EEF 路径：三路 RGB → raw proprio 20D → unified 80D → min-max normalization → generate → raw action 20D → 双臂 EEF quaternion 16D → RoboTwin `take_action(action_type='ee')`。不将其套用到 joint 14D 或 Wuji raw 58D。

两个服务均使用 10 步 sync 去噪、同步 buffer executor，完整生成/执行 32-action chunk（episode 成功时提前停止），policy seed 默认 42。基线按发布设置开启 compile 和 velocity cache；State step 201 使用 eager、无 velocity cache。原配置、归一化和模型身份见权重 manifest；服务日志为 `runs/r1-baseline-server-001/console.log` 和 `runs/r1-state-server-001/console.log`。

物理 timestep 和 cuRobo 插值间隔均为 1/250 秒。每个 EEF 目标先规划再执行可变长度轨迹，所以不存在从这些记录可推出的固定 policy Hz；视频编码 10 FPS 不等于控制频率。保留原控制路径、原生规划失败行为与每任务步数上限，没有开启额外 mplib fallback。

## RoboTwin 双条件 Smoke

固定任务 `adjust_bottle`、embodiment `aloha-agilex`，每模型/条件 1 episode，上限 400 个策略动作。采用上游环境 seed 起点 100000、稳定性与专家可达性筛选；clean 接受 100001，randomized 接受 100000。成功定义直接使用固定版本 `envs/adjust_bottle.py::check_success`，不另改宽容度。

| 模型 / 条件 | run id | seed | 动作数 | 视频帧数 | 策略执行墙钟 | 任务结果 |
|---|---|---|---|---|---|---|
| 发布基线 / clean | `r1-baseline-clean-002` | 100001 | 118 | 119 | 78.93 s | 成功 |
| 发布基线 / randomized | `r1-baseline-randomized-001` | 100000 | 123 | 124 | 81.33 s | 成功 |
| State step 201 / clean | `r1-state-clean-001` | 100001 | 121 | 122 | 72.43 s | 成功 |
| State step 201 / randomized | `r1-state-randomized-001` | 100000 | 124 | 125 | 83.77 s | 成功 |

各目录 `runs/<run id>/` 内均保存 `smoke_result.json`、`episodes.json`、`resets.json`、`policy_config.yml`、`task_config.yml`、`console.log` 及 `runtime/eval_result/.../episode0.mp4`。专家筛选不计作策略 episode。耗时从首次策略调用至环境关闭前计，不包括专家筛选与服务启动。

视频审计完整解码每一帧，按上游“一动作一帧，成功额外一帧”核对数量；首末画面的标准差均大于 27，平均绝对画面变化为 12.44–31.44。四条样本成功不是成绩复现，不估计总体成功率；同一条件的实际 instruction 文本也不同，保存在 episode 记录中，因此这些不是严格配对结构比较或确定性复现证明。

## 产物与复核入口

- `manifests/model-environment.json`：模型包版本、训练数据逐文件哈希、数据 revision 和 split；`manifests/simulator-environment.json`：仿真包版本。
- `manifests/downloads.json`：四个数据/资产归档和 cuRobo 源码归档的大小与 SHA256。只下载所需训练子集与仿真资产。
- `manifests/dependency-patches.json`：官方依赖补丁、备份路径和哈希。
- `manifests/delivery-code-and-weights.json`：step 201 权重、配置、统计和固定 RoboTwin 源码身份；`manifests/final-code.json`：验收时审计脚本/测试/实现与文档快照，优先于较早代码清单。该快照保留验收时的文档措辞，后续提交说明和文档修订以 Git 历史为准，不覆盖历史 manifest。
- `reports/profile-resume-audit.json`、`reports/rollout-video-audit.json`：可分别用 `scripts/statewam/audit_profile.py`、`scripts/statewam/audit_rollouts.py` 对保留的原始产物重验。

最终回归命令（在 OpenWAM 根目录、模型环境执行）：

```bash
python -m pytest tests/statewam tests/test_mot_driver.py tests/test_proprioceptive.py \
  tests/test_finetune_config_override.py tests/test_optimizer_groups.py \
  tests/test_architecture_variants.py tests/test_wam_architecture.py tests/test_checkpointing.py \
  tests/test_robotwin_dataloader.py tests/test_robotwin_interface_proprio.py \
  tests/dataloader/test_wuji_real_task.py -q
```

242 项通过，仅有既有 SWIG 类型的弃用警告；新脚本/架构/测试的 ruff 检查及 `git diff --check` 通过。此列表是针对改动面的回归，不声称运行了仓库全部模型和 benchmark 测试。

## 问题记录

1. `r1-smoke-train-001`：未启动 Hydra 时 `${now:...}` 无法解析。已在专用 profiling 脚本中设定明确输出目录并提前解析配置，后续 2 步启动通过。
2. `r1-env-clean-001`：资产下载后尚未生成 cuRobo YAML。运行官方路径生成脚本后解除。
3. `r1-env-clean-002`：seed=100000 的物体不稳定，被上游 `UnStableError` 拒绝；未绕过稳定性检查。
4. `r1-env-clean-003`：实际还返回 front_camera，专用检查脚本误要求相机总数恰好为 3。改为验证三路策略必需视角均存在；seed=100001 的后续完整专家 smoke 通过。
5. 上游 `_result.txt` 按硬编码 100 episodes 计算，不能用于 capped smoke 成功率。专用 smoke 脚本将逐 episode 写入 `episodes.json`，以 `smoke_result.json` 的实际 count 为准，不据此作 benchmark 成绩比较。
6. `r1-baseline-clean-001`：SAPIEN 导入后默认 locale 导致 UTF-8 配置读取失败，未计入任何策略 episode。显式 UTF-8 读取并设置 `PYTHONUTF8=1` 后，`r1-baseline-clean-002` 通过。
7. 视频审计初版误要求视频帧数等于动作数。核对固定上游成功分支后，改为动作数加成功末帧，并用成功/失败、缺帧及非法步数测试固定该契约。没有修改或重编码原视频。

## 暂缓与结论边界

本 Goal 的适用工程检查、4 卡 profiling、基线与 StateWAM 双条件 smoke 均已通过，没有遗留的阻塞验收项。完整训练、50-task 全量评测、R2 E0–E3 匹配预算对照、真机 rollout、语义监督及蒸馏均未执行。未做 4 卡 accumulation>1 的独立训练、跨硬件/全任务推广或完整延迟分解；已有小模型梯度累积与单输入性能证据不得外推。后续实验必须新建对应 Goal 文档包，不追加到本报告冒充本轮结果。
