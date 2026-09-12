# StateWAM R2 RoboTwin 公平结构比较 Pilot Goal

日期：2026-09-12。阶段：R2。Goal ID：`r2/robotwin-pilot`。

状态：用户已正式下达 Goal，2026-09-12 开始执行；P0 通过，P1 执行中。协议准备通过后，直接进入训练、评测和交付，不再等待第二次预算或阶段确认；当前事实、未通过项与证据见 [REPORT](REPORT.md)，已验证命令见 [USAGE](USAGE.md)。

变更记录：2026-09-12，按用户要求取消准备与正式训练之间的人工预算确认门槛；保留工程和公平性检查，增加资源利用优化与全过程性能消耗记录。4 卡是同时占用上限，用户未另设累计时长或 GPU-hours 上限；执行者仍须制定有限、可解释的训练与评测计划，不据此进行无限训练或搜索。

变更记录：2026-09-12，补充对话中断下的独立任务运行、持久监控、训练检查点与逐 episode 恢复、显式停止与释放资源要求，并加入中断/恢复验收。本次仅更新计划，不启动任务；这些能力须在执行中实现和验证，不能视为已具备。

执行记录：2026-09-12，P0 数据与来源门槛通过。冻结数据版本 `clean50-785feb15-s20260912-v1`，每任务 45/5 episodes；train 103010 windows、val 9794 windows。容量候选采用 E1/E2 的无参数当前帧池化，驱动与 E3 同款的 `action_proj/read_action`；小模型隔离与梯度测试通过，正式容量方案仍须实模型审计后冻结。N、有效 batch 和训练配方尚未冻结。

前置工程版本：OpenWAM R1 提交 `98089d1fa612b7d1f0349ae68b3b8b0ef690aef7`，见 [R1 验收报告](../../r1/robotwin-smoke/REPORT.md)。长期依据：[研究计划](../../../design/02_Research_Plan.md)、[仿真协议](../../../design/04_Simulation_Protocol.md)、[文档管理规范](../../../README.md)。

## 1. 可直接下达的 Goal

> 请按 `docs/statewam/stages/r2/robotwin-pilot/GOAL.md` 连续完成 R2 单种子 pilot：基于已提交的 R1，实现可审计的 E0–E3 公平比较配置、train/validation 划分、训练统计、配对环境与固定指令评测、随机性控制和结果汇总。项目同时最多使用 4 张空闲 GPU，合理提高显存利用率与吞吐，并记录全过程资源与性能消耗。长时间训练、评测及资源监控须独立于对话会话运行，验证中断后的状态核验与恢复；interrupt 不等于停止后台任务，明确要求停止时按第 2 节安全退出并释放项目资源。先完成必要工程检查和 profiling，由你根据实测、数据规模与可核实配方自主确定并记录组间一致的训练预算及容量控制方案，检查通过后直接执行正式训练和配对评测，无需再次等待我确认。正式实验统一从同一 Foundation checkpoint 开始，只用 clean 数据训练，在固定任务集的 clean/randomized 条件下比较。按本 Goal 独立归档，交付 REPORT 与简洁 USAGE；不将单种子 pilot 当作全量 benchmark 成绩或最终机制结论。

本 Goal 目标是从“实现可用”进入“预算和协议受控的初步结构比较”，不是保证 StateWAM 获胜。完整 R2 的多训练种子复核另立 Goal，不自动进入 R3、真机、语义监督或蒸馏。

## 2. 版本、目录与资源

- R1 已提交，不改写该提交。已从计划提交 `368654c874cbbadc19fa18ec955e17982c7a0081` 新建 `feat/statewam-r2-robotwin-pilot` 分支；R1 基点仍为本文所列 `98089d1`，实现提交与运行源码身份在 REPORT 和产物中记录。
- 复用已验证的 `RoboTwin-statewam-r1` worktree 和两套 Python 环境，不因为进入 R2 就再 clone 一份仓库。名称中的 r1 不是版本约束，真实版本为 `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`。保留用户原 `RoboTwin/`；若必须修改上游源码，先说明并隔离补丁，不能静默改变对照环境。
- 固定 SAPIEN、cuRobo、mplib、warp、CUDA/驱动、资产、embodiment、物理参数与渲染后端，复核 R1 manifest 中的实际身份。环境/源码发生影响行为的变化时，所有比较组必须使用相同版本并重验。
- 项目同时占用最多 4 张空闲 GPU；训练用 4 卡，评测服务与渲染也计入总数。不要固定认为 0–3 永远空闲，不终止其他任务。先复用推荐 batch；没有特定配方时用实测吞吐与峰值余量选择，不改输入几何来填显存。
- 在上述上限内自主安排训练、服务和渲染，避免 GPU 空闲或 CPU/数据读取长期阻塞；可调 micro-batch、梯度累积、worker、预取和数据缓存。调优必须遵守第 5 节的公平性与监测要求，不为占满显存增加无效分配、改变实验语义或同时占用第五张卡。
- 新下载数据和派生 split/统计放在 `/gaozt-test1/datasets/robotwin/r2-robotwin-pilot/<data-version>/`；可链接已核验的上游数据，不覆盖 R1 数据和统计。
- 运行产物放在 `OpenWAM/statewam_artifacts/robotwin/r2-robotwin-pilot/`，按 `manifests/`、`checkpoints/`、`runs/<run-id>/`、`reports/` 管理。manifest 均记录 `goal_id`、`protocol_id` 和 `run_id`，checkpoint 再记录模型组、初始化哈希、train seed、optimizer update 与源 run。
- 本目录按 GOAL、REPORT、USAGE 管理执行文档；报告只记录已有证据，不提前创建通过结论。跨阶段通过相对链接引用，不把 R2 结果追加进 R1 报告。
- 不保存或输出 HF token；优先公开下载，必要认证由运行环境提供。固定 revision、文件清单与校验值，镜像不能改变内容身份。

### 对话中断、后台运行与恢复

1. 正式长任务使用已验证的独立运行机制，例如 `tmux` 或可用的作业调度器，不能仅依赖对话执行工具的前台会话或一个未经验证的 `&`。训练、策略服务、模拟器和资源采样均写入持久日志，不依赖助手持续轮询才能推进或记账。独立运行不能保证抵御主机重启、OOM 或设备故障；具体客户端的 interrupt 行为未经验证时明确标注，不承诺绝对不中断。
2. 在本 Goal 产物目录保存任务登记表，并为各 run 记录 session/job id、PID/进程组与启动时间、GPU UUID、工作目录、脱敏命令、代码/配置/协议哈希、依赖、日志位置和退出码。状态至少区分 pending/running/stopping/stopped/failed/completed；记录心跳、最近持久进度和检查点。启动/恢复入口通过锁或等效机制防止同一 run 重复运行；重连时结合进程身份、心跳、日志和实际 GPU 占用核验，不能仅信任旧 PID 或状态文件。
3. 用户点击 interrupt 只表示中断当前助手操作，不默认解释为停止训练或释放 GPU。已独立启动的任务可能继续运行并产生费用；助手被中断后不承诺继续主动监控、修复或启动后续阶段。只有预先实现、记录并验证的自动编排才会继续推进阶段，且必须遵守原公平性门槛、有限预算和最多 4 卡限制，遇到错误或门槛未通过即停止推进，不做无限重试。第 5 节“不等二次确认”不代表对话中断期间存在未实现的自动执行能力。
4. 训练在安全的 optimizer-update 边界定期保存完整检查点。P2 根据实测保存开销冻结按更新数或时间的保存频率，并报告可能损失的进度窗口；保存模型、所有 rank 的优化器分片、scheduler、使用时的 scaler、更新/曝光计数、各 RNG 流、sampler/数据游标及增强/worker 的可恢复状态或确定性重放依据。使用临时位置和原子发布/完成标记，只有全部分片齐全且验证通过才更新 latest-complete 指针；不覆盖唯一可恢复检查点，不把只有权重的导出当完整续训状态。
5. 恢复时先核验仍在运行的任务并接管监控，不重复启动；若任务已退出，核对源 checkpoint、协议和数据身份，从最近完整检查点恢复。不得重新从样本零点继续或静默加载不完整分片；未保存的更新可能重算，主训练预算仍以提交的 N 次更新及逻辑曝光为准，重算的实际计算/曝光和 GPU-hours 另行记账。无法恢复同一采样/噪声序列时，先修复或按明确记录重跑该组，不能声称精确恢复或偷偷改变比较条件。
6. 评测按 group/checkpoint/task/condition/scene/policy seed/protocol 的稳定键登记，每个 attempt 独立保存开始、结果、视频与结束状态，完整结果原子发布；恢复前核验身份和产物完整性，只跳过已完整落盘的 episode。已完成的策略失败也必须跳过，不能为得到成功而重跑。中断中的 episode 默认从相同初态、scene 和 seed 重新开始，不声称支持未实现的 episode 中途物理恢复；保留原 attempt，按第 6 节 invalid 分类与最多一次重试规则处理，无法判定为基础设施/外部中断的策略错误不得改记 invalid。重试仍不完整时如实保留配对缺口，不扩大重试额度。
7. 用户明确要求“停止训练/评测并释放 GPU”时，先阻止该范围内的新任务和自动重启，再通知对应任务在安全边界保存并退出；预先设置并记录有界退出等待，超时只针对已核验属于本 Goal 的进程组升级终止，不杀其他任务。退出后检查训练 rank、策略服务、模拟器与渲染子进程的实际状态及显存释放，最后收尾监测并汇报最近可恢复位置、未保存进度和仍未释放的资源。助手本身已被 interrupt 时不能保证执行这些操作，因此 USAGE 必须提供用户可独立执行的状态、停止、恢复和资源核验命令。
8. P2 利用小规模工程 run 验证独立会话断开后任务及采样继续、停止后重新接管不重复启动、四卡检查点恢复与不间断短参照的更新/采样/噪声一致性，以及评测在 episode 之间和 episode 内中断后的去重与有限重试。测试检查点写入中断时能拒绝不完整版本，验证显式停止可以释放本项目资源。记录实际测试的中断方式、容差、结果和成本，不将进程断连测试冒称客户端 interrupt 按钮实测；未通过时先修复，不开始无人值守正式长任务。

## 3. 模型矩阵与初始化

| 组别 | 拓扑 / 来源 | 主要比较作用 |
|---|---|---|
| E0 | 原 OpenWAM，无 State；同预算训练 | 排除额外微调本身的收益 |
| E1 | Residual-State，`current` | 当前视觉、语言和有效 proprio 的附加通路对照 |
| E2 | Residual-State，`future` | 在当前条件之外读取生成中的未来 world；无直接 action→State 边 |
| E3 | Residual-State，`future_action` | 在 E2 之外读取动作 hidden tokens |

E1–E3 固定 K=16、D=512、heads=8、读写层 `[4,9,14,19,24,29]`，保留原 30 层 MoT、`bridge_interval=1`、当前视觉因果保护及原 world/action flow-matching。gate 从零开始，其他投影按 R1 非零初始化。State 不跨去噪调用或 episode 缓存，不引入额外标签或教师。

四组都从同一份 `OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors` 开始，本地已记录源 SHA256：`180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d`。执行前重验权重、配置和 sidecar，保留可核实的来源 revision；不能编造缺失的预训练数据覆盖信息。

本轮“只用 clean 训练”描述追加训练协议，不自动证明 Foundation 的历史曝光与所有评测资产无交集。预训练覆盖未完全核实时明确标注，不能仅凭模型名称把 randomized 称为绝对未见环境。

原 world/action/proprio 初始参数在四组中应逐键、逐值相同；E1–E3 的共同 State 参数也应从同一初始化拷贝，不仅设置相同 seed。受控转换后严格重载。Foundation 的原数据配置不能直接当 RoboTwin 配方：必须明确替换为本轮任务集、EEF 20D↔unified 80D、相机布局及训练统计，同时保留兼容的骨干几何。

R1 的 step 201、发布 Clean2Random/Full、Wuji 权重均不进入本轮主矩阵。发布模型可作独立协议参考，但不参加匹配预算的 E0–E3 排名。从已微调 sim 权重共同继续训练是另一种 adaptation 实验，需要另立协议，不能悄悄替代 from-Foundation。

### 容量公平门槛

R1 的 `current`/`future` 不执行 `action_proj` 与 `read_action`，E3 执行；因此四种来源开关可用，不等于已经有严格的容量匹配实验。相同 `state_dict` 大小、`requires_grad` 数量或保留未使用参数，都不能证明有效容量匹配。

正式矩阵启动前必须生成 `capacity_report.json`，记录每组总参数、可训练参数、实际参与 forward/backward 的参数、各分支梯度、源 token 数、有效 attention 工作量与实测耗时。gate=0 首步的零梯度不能当作永久未参与，应检查启动后的真实计算图。

容量方案由执行者在正式训练前自主选择、验证并与训练预算一并冻结，无需额外确认：优先给 E1/E2 使用只读当前条件的等容量辅助分支，匹配 E3 新增的可学习投影/attention 预算，且不改变其允许的信息来源；新增分支必须独立测试当前信息隔离。若无法在主四组内做到合理匹配，先尝试范围内的实现修正；确需新增模型组或改变主要比较问题时，说明技术依据并请求范围变更，而不是重复请求已授权的四组预算。禁止把未使用参数或无梯度计算称为容量补偿，不得擅自增加第五组训练。

无法完全匹配计算量时，如实报告差异，不声称 compute-matched；主比较仍按 optimizer updates 和样本曝光匹配。未通过容量门槛时，只能报告工程探索，不能下“未来信息而非容量带来增益”的结论。

E2 的 future world 在原 mutual 骨干中仍可含动作信息；E3−E2 只比较额外直接读取边，不能解释为隔离了全系统动作反馈。writer 的动作 query 在 E1 中保留，但 E1 的 State 读取器本身不能接触动作或未来信息。

## 4. 数据与任务集

默认预选 10 个任务，名称已在固定 RoboTwin 版本中核对存在；数据可用性与环境专家检查尚未执行，不依据任何候选模型成绩选任务。

| 任务 | 预选覆盖面 |
|---|---|
| `adjust_bottle` | 物体抓取与姿态调整 |
| `pick_dual_bottles` | 双臂抓取 |
| `handover_block` | 双臂交接 |
| `lift_pot` | 双臂协同抬升 |
| `stack_blocks_two` | 对齐与堆叠 |
| `stack_bowls_two` | 容器堆叠 |
| `place_mouse_pad` | 目标区域放置 |
| `place_object_basket` | 容器放置 |
| `open_microwave` | 关节物体交互 |
| `press_stapler` | 接触操作 |

1. 数据默认使用 `TianxingChen/RoboTwin2.0` 的固定 revision `785feb15aa4a4f532395ad2b1d2be5f28cb561ad`，上述任务的 `aloha-agilex_clean_50`。先核对清单、许可、容量和数据契约；不下载 randomized 示范作为训练数据。
2. 每任务默认按 episode 划分 45 train / 5 offline-validation。按固定算法对 `dataset revision + task + episode id + split seed` 做稳定 SHA256 排序，取 5 条为 validation，其余训练；split seed 默认 20260912。先划分再展开窗口，不按帧随机拆分；有 session/重复轨迹信息时按组隔离并记录实际数量，不能为凑 45/5 拆开重复组。
3. 四组使用同一份显式 episode allowlist。现有 reader 的 `split` 参数不能被假定为物理数据隔离：检查其实际文件选择，并增加测试证明 train/validation 文件与 episode 无交集，重试路径也不能跨 split。
4. 只在本轮 train episodes 上拟合一份共享 RoboTwin EEF 统计，默认 min-max，保留 rot6d 等既有特殊维度规则；validation 和部署只加载该文件。不得自动回退到全数据统计、发布 Clean2Random 统计或各组分别拟合。记录参与拟合的文件清单与哈希；缺文件应报错，不隐式重算。
5. 固定 `num_frames=33`、`video_stride=4`、384×320 多视角画布、9 个视觉帧、32 个动作；核对 raw/model 映射与 mask。训练增强配置及采样 instruction 必须组间一致，validation 禁用随机增强。
6. 缺数据、环境不可用或任务替换只允许在模型比较前修订清单、说明客观原因并重新冻结协议；最终测试后不能因失败率高删除任务。此 45/5、10-task 子集是研究 pilot，不是官方 clean_50 全训练/50-task 复现。

## 5. 训练预算与随机性

以下是准备阶段的默认建议。执行者依据实测和研究目的自主确定最终方案，在正式训练前写入解析后的公共配置和 `protocol.yaml` 并锁定，向用户报告选择依据后继续执行，不等待回复：

| 项目 | 默认建议 / 约束 |
|---|---|
| 正式 train seed | 42，首轮单 seed；多训练种子留待下一 Goal |
| 公共可训练模块 | ActionDiT + proprio；E1–E3 再加 State/已验证并冻结的容量控制模块 |
| world 策略 | 参数冻结但保留输入梯度，四组相同；不能 E0 用整个 world 的 no_grad 而其他组使用不同冻结语义 |
| 精度与优化器 | bf16、AdamW、ZeRO-2、gradient checkpointing；betas/eps/weight decay/clip 显式冻结，不依赖不同源配置默认值 |
| 学习率起点 | ActionDiT 1e-5、proprio 1e-4、State 1e-4；各共享模块组间一致，world 不更新 |
| loss 与调度 | 原 video/action loss 权重组间相同；起点为常数 LR，若采用 scheduler/warmup，在正式训练前冻结公共曲线和总步数 |
| batch 起点 | per-GPU 4 × 4 卡 × accumulation 1 = global 16；这来自 R1 可运行配置，不冒称 Foundation 官方最优配方 |
| 最终预算 | 执行者结合多任务 E0/E3 实测、数据规模及可核实配方自主确定有限的正式更新数与总曝光；无需二次确认，不默认使用 200 步或任意放大 `max_steps` |

1. 准备阶段允许 E0、E3 各一个独立的 200–500 optimizer updates profiling run，默认各 200；使用训练 split，单列诊断 seed 和曝光。这些模型不进入最终结果，不用 final-eval 场景调参。
2. 测量四卡吞吐、各 rank 峰值显存、数据等待、保存/恢复、稳定性和 E0 学习曲线。在正式方案冻结前，比较有限个可行 micro-batch/worker 组合，优先选择稳定有效吞吐更高的方案；保留全部试探、OOM 和工程重跑记录。200–500 步限制针对每个 profiling run，不是正式训练上限，也不是调优总成本；调优候选与停止规则先登记，不能无限试探。
3. 训练预算记录必须填写：每组正式更新数 N、micro-batch、累积、global batch、总曝光、保存/开发评测节点、统一选 checkpoint 规则、预计训练及评测 GPU-hours、磁盘空间、容量控制成本和预计总消耗。说明 N 相对于训练窗口数的曝光轮数以及选择依据，不能只为尽快通过测试选短训练。200 步只能确认吞吐和初期学习，不证明收敛；必要时自主开展一个有事先登记步数上限与停止规则的 E0 校准 run，仅使用 train/offline-validation/开发集，单列成本并从正式矩阵中排除。正式四组仍从原始初始化重新开始。用户未指定累计时长上限，预计值不是需要等待批准的额度；预计发生明显偏差时记录原因并告知，不自动追加任务、种子或无限延长 N。
4. 默认正式四组都重新从 F0/同一 State 初值开始、使用全新优化器，运行同样 N 次更新，主结果使用第 N 步，不各自早停或挑最好 checkpoint。异常暂停后恢复必须验证优化器、scheduler、数据游标/RNG，并计入原预算；不能重新从样本零点继续却声称相同曝光序列。
5. `global_batch = micro_batch × 4 × accumulation`；相同 N 下样本曝光必须相同。优先采用共同 micro-batch/累积，某组 OOM 时不得独自改几何或有效 batch；确需不同 micro-batch，先验证累积、loss scaling 和采样等价，更新全部协议记录。
6. 固定任务采样权重和窗口顺序。建议按任务均匀采样，再在任务内抽窗口，防止长轨迹任务主导；实现后记录实际 task/episode/window id 序列或哈希与采样计数。四组应拿到相同序列，不仅相同 epoch 数。
7. 隔离新增模块初始化、sampler、worker 增强/指令、训练 world/action 噪声的 RNG。按稳定的 run seed / sample occurrence / modality 派生随机性，记录同位置输入、增强和噪声哈希；world/action 两个噪声流保持原训练独立性，但分别在组间配对。不能使用受进程影响的 Python `hash()` 或仅设置 global seed。
8. 正式启动前补充真实四卡 accumulation>1 验证（若正式配方使用），以及四组参数分组/冻结、源隔离、loss mask、严格加载、优化器恢复检查。R1 专用 `profile_training.py` 强制 State 梯度、最多 500 步且改写若干配置，不能直接改上限就作为 E0/R2 正式入口；保留 R1 命令，新增最小的公共 R2 实验入口和审计工具。

工程检查与公平性门槛通过、有限训练预算和协议已冻结后，自动从 P2 进入 P3，再完成 P4/P5，不以缺少第二次用户确认作为暂停理由。门槛未通过时继续诊断和修复，不跳过检查；只有实际外部依赖不可用或必须扩大研究范围等情况才请求用户介入。准备阶段本身不能标记为完整 pilot 完成。

### 资源利用与性能记账

- 正式配置冻结前，在真实几何、完整优化器和 checkpoint 保存/恢复条件下调优。优先提高 micro-batch 并相应调整累积，在公平性允许范围内保持选定的有效 batch；若需要调整有效 batch，应在四组正式训练前统一确定 N、曝光量与 LR 方案。正式中途不得让某一组独自改 batch、精度、分辨率或训练预算以提高利用率。
- 以稳定吞吐为目标，尽可能利用显存，同时保留动态张量、编译（如启用）、保存/恢复和评测所需峰值余量。不承诺显存占用越高越好，也不分配无用张量制造“吃满”。若采用官方推荐 batch 而仍有空闲显存，记录配方依据与性能取舍。
- 每个 run 保存 `resource_samples.jsonl` 与 `performance_summary.json`，包含 goal/protocol/run id、阶段、配置、起止时间、GPU index/UUID/型号/容量、参与进程与采样间隔。按固定周期采样，建议 1–5 秒，并记录采样缺口与监测开销；按时间线审计整个项目的同时占卡数不超过 4，而不只检查单个进程的 `WORLD_SIZE`。
- 显存与计算：逐 GPU 的利用率、显存占用及峰值；框架侧 peak allocated/reserved 单独记录，与 NVML/`nvidia-smi` 的进程/整卡显存区分。输出平均值、分位数和峰值，不仅保存最终一张截图；不能把他人任务的占用计入本项目。
- 训练：启动/预热和稳态分别记录 microstep/optimizer-update 时间、samples/s、global batch、累积、实际曝光、数据等待/预处理、forward/backward/update、保存/加载/恢复耗时。耗时拆分注明同步、重叠、采样与统计口径，不能把所有等待都称为磁盘 I/O；分析数据瓶颈时补充 CPU、RAM 和读取吞吐。
- 部署评测：记录服务装载与 warmup、生成 chunk 延迟、端到端策略请求 p50/p95、episodes/hour、模拟动作/physics steps、wall-clock 与渲染/控制等待。相同配置和负载下比较延迟，不用较高并发的吞吐冒充单请求时延改善。
- 成本：按实际使用的 GPU 数随时间积分汇总 GPU-hours，训练、profiling/调优、校准、评测、失败重跑和服务等待分别列出，给出每组和整个 Goal 的总计；共享 GPU 上的多个服务不能重复计整卡时长。记录 checkpoint/日志/视频的磁盘增量和峰值 RAM；设备支持时记录功率并估算 kWh，注明采样误差与整卡计量口径，不把估算冒充精确的模型能耗。
- 缺失监测能力应记录原因、影响与替代数据，不能填 0 或捏造利用率/FLOPs。最终 REPORT 必须包含组间性能/消耗对照和整个 Goal 的总账，明确未充分利用硬件时的瓶颈、已尝试措施及未采用更大 batch 的原因；原始采样和分析工具一并保留。

## 6. 配对评测协议

### 场景与指令冻结

- 将 offline-validation、开发 rollout、最终 pilot rollout 明确分开。默认开发集每 task/condition 5 scenes，最终集每 task/condition 20 scenes；两者环境 seed 区间不重叠，也不复用 R1 smoke 的 seed。
- 候选 seed 默认按排序后的 task 索引 t：开发从 `200000 + 10000*t`、最终从 `400000 + 10000*t` 开始，每个 task/condition 最多审查 1000 个候选。clean 与 randomized 分开成表，不要求两种条件物理状态相同；同一条件内四组共享 scene id。
- 在模型比较前独立运行上游稳定性/专家可达性筛选，保存接受/拒绝清单及原因；专家 episode 不计入策略样本量。选择规则不依赖任何策略成功率，候选耗尽时报告缺口，不放宽成功条件。
- 为每个 scene 保存任务、condition、env seed、资产实例、物体/关节初态、随机化参数、相机/灯光、原任务上限、明确的 instruction 文本、instruction 类型/来源，以及 policy seed。开发与最终 rollout 默认沿用 adapter 的 `instruction_type: unseen`，但每 scene 只选择一次并保存完整文本及哈希。上游脚本每轮重新生成/随机选指令的行为必须被固定 manifest 回放替代；只固定 env seed 不够。
- 优先验证 seed reset 后相同初态的位姿和三路观测；若采用 snapshot，必须包含并验证 controller/RNG 等必要状态。不要宣称未实现的全物理 snapshot。初态位姿/像素的容差由模型无关的重复 reset 在准备阶段标定、记录并在测试前锁定，不因某个模型不匹配而放宽。
- 最终集文件封存并记录哈希。准备阶段可做模型无关的筛选/reset 校验，但不能拿最终集策略成绩调学习率、选结构、改预算或调整指令。R1 对应的场景不进入 final 集。

### 四组完全一致的推理和控制

主比较默认四组全部 eager、关闭 DiT velocity cache，10 步同步去噪，bf16，相同 scheduler/shift、三视角预处理、normalizer、同步 executor 和 32-action chunk。这与发布 benchmark 的 compile/cache 协议不同，必须在报告中声明；发布参考成绩不与本表直接比较。

如要采用 compile，须在正式评测前对所有组验证、统一开启并冻结；近似 cache 只能作为独立推理优化实验，不允许 E0 开启而 State 关闭。E0–E3 不使用混杂的 R1 服务命令作为正式对照。

每 scene 的 policy seed 及 chunk 序号派生规则写入 manifest；使用独立 world/action RNG，组间相同序号使用相同初始噪声。相同 episode 不保证不同策略访问相同后续观测，只要求相同初态和采样协议；禁止用未来真实观测校正策略。

每个 episode 开始清空 executor buffer、prompt/cache 和所有临时状态，验证 reset ACK。保持 EEF 20D↔80D↔quaternion 16D、原 cuRobo 控制路径、物理 timestep、动作上限和任务 `check_success`；不启用额外 planner fallback，不根据模型表现扩大步数或重试。

以原 `take_action` 计策略动作，以 physics steps 计模拟时间，另计 wall-clock 与推理耗时；10 FPS 视频或 planner 插值 250 Hz 均不能直接当作 policy 控制频率。

### 重复性、错误与计数

1. 用开发集预选每 task/condition 第一个 scene，对同一 E0、同输入/噪声协议重复 2 次，记录初态、动作和结果的残余差异；共 40 个诊断 rollout，不计入主评测分母。若初态不可配对，先修复；物理残余随机性应如实记录，不靠事后筛选消除。
2. 最终默认四组 × 10 tasks × 2 conditions × 20 scenes = 1600 个策略 episodes；开发评测最多 3 个事先固定节点，每节点四组合计 400 个 episodes，最多 1200 个。容量额外对照、重复性 40 次、环境专家筛选和故障重试均需独立预算，不能隐去成本。
3. 固定模型执行顺序的轮换/交错规则，避免所有 E0 在一种机器负载下、所有 E3 在另一种负载下。记录 GPU、并发度、线程、warmup；延迟只在相同资源条件下比较。
4. 任务失败、达到上限、模型非有限动作等策略错误计失败，不能换 seed 重跑到成功。明确的基础设施故障单列 invalid，使用同一 scene 和 seed 最多重试 1 次并保留两次记录；仍失败则标记该配对单元不完整，不静默删除或替换。不能在缺失配对时宣布完整 pilot 完成。
5. 以 manifest 的预定 count 和逐 episode JSON 核算分母，禁止沿用上游硬编码 100 的 `_result.txt`。视频、成功/失败记录、配置、checkpoint 身份必须完整，包括失败样本。

## 7. 分析与解释边界

- 主要报告 clean 和 randomized 分开的、按任务等权的成功率与绝对百分点差；平均值只作为补充，不把更多 episode 的任务加权放大。
- 预先固定比较：E3−E0、E3−E1、E2−E1，以及 E3−E2；所有比较都报告，不只展示最好一项。未来信息与容量解释必须受第 3 节门槛约束；若据开发集另选主要候选，须在 final 解封前登记。
- 给出逐 task/condition 的配对结果和 95% 区间。使用保持模型间 scene 配对的、任务/scene 分层重采样或相应配对方法；重采样次数与 seed 事先固定，建议 10000 次、seed=20260912。若进行多个显著性检验，预先明确比较族并做多重比较校正。
- 明确区间只反映当前任务/场景采样下的单训练 seed pilot，不覆盖训练随机性，也不能把帧或 action 当独立 episode。不得报告“四次成功所以 100% 泛化”一类结论。
- 同时报告优化步、样本曝光、有效参数、FLOPs/耗时口径、GPU-hours、显存、保存开销和相同输入下推理延迟。主结果是 step/sample-matched，不自动等于 GPU-hour-matched。
- 判断 E0 是否仍明显欠拟合；若所有组尚处启动阶段或 E0 基础能力未建立，结论是“预算不足以判断”，不是 State 无效或已经有效。不得只延长某一组训练来追求正结果。
- E3 未超过 E1 时，不宣称未来交互有效；E2 与 E3 接近时，优先将更简单的 E2 带入后续复核。负结果是合法交付，不自动追加任务、seed、容量或超参搜索。
- 本轮不要求收益显著才能完成，但需要实验与分析完整。是否进入多训练种子复核由结果、残余方差与预算决定，另建 Goal；不自动解锁机制/真机有效性结论。

## 8. 顺序、交付与验收

| 阶段 | 交付与通过条件 | 当前状态 |
|---|---|---|
| P0 来源与数据审计 | F0/R1/环境身份、任务可用性、显式 train/validation 清单、train-only 统计 | 通过，见 REPORT 与 reports/p0_acceptance.json |
| P1 公平性工具 | 公共配置、容量方案、配对 scene/instruction/RNG、来源隔离与错误计数测试、固定协议文件 | 执行中，数据隔离与容量工具已开始 |
| P2 工程与预算 | E0/E3 四卡独立 profiling、有限调优/必要校准、持久资源监测、独立运行/中断恢复/停止释放验收，自主冻结公共预算与协议后自动继续 | 未开始 |
| P3 单 seed 训练 | P2 检查通过后直接执行，四组同初始化/样本序列/N 步/冻结策略，checkpoint、梯度与性能证据 | 未开始 |
| P4 配对 pilot | 直接接续训练，重复性检查、开发/最终集分离、1600 个主矩阵 episodes 及完整视频/失败/消耗记录 | 未开始 |
| P5 报告交付 | 所有预设比较、限制和下一步建议；本目录 REPORT、USAGE 和更新后的总索引 | 未开始 |

执行中须生成 `protocol.yaml`、数据/统计 provenance、`capacity_report.json`、开发/最终 `evaluation_manifest`、任务登记表、每个 run 的解析配置、`resource_samples.jsonl`、`performance_summary.json`、中断/恢复/停止审计记录与训练/评测证据。小型可复用配置与工具进入 Git，大体积样本清单、日志、视频、权重进入本 Goal 的产物目录，并从 REPORT 引用路径与哈希。自主选定预算的依据与冻结记录写入 GOAL，冻结后的协议修改必须带版本与原因，不能回改已知结果的成功标准。

最终 REPORT 逐项区分通过、失败、未执行、预算不足及不可外推的结论，并交付逐组性能表、全 Goal GPU-hours/存储消耗总账及资源利用分析；USAGE 只包含必要环境前提、准备、独立启动、状态/日志查看、重新接管/恢复、安全停止与 GPU 释放核验、配对评测、结果与性能审计命令，并简短说明 interrupt 与停止任务的区别。R1 回归必须继续通过；新增测试应覆盖 split/统计泄漏、参数与 RNG 配对、固定指令回放、初态不匹配、reset、计数、失败/重试、分析配对保持、资源记账不重复计数及第 2 节中断恢复验收。

只有 P0–P5 的适用项全部完成、正式预算实际执行、交付文件齐全，才能将本 Goal 标为完成。准备阶段完成不等于 R2 pilot 完成，单 seed pilot 完成也不等于整个 R2 研究阶段完成。

## 9. 明确不做

用户已正式下达本 Goal，按其范围连续完成准备、训练、评测与交付，无需第二次预算确认，但不自动包含：官方 50-task 全量成绩复现、Full 权重 adaptation、三训练种子复核、R3 bottleneck/语义监督/网格搜索、LIBERO 扩展、真机物理 rollout、跨硬件或蒸馏。需要超过 4 卡、改变主要研究问题或新增范围外实验时，停止扩展并请求新的范围确认；已有范围内的耗时估算和自主预算选择不再构成人工批准门槛。
