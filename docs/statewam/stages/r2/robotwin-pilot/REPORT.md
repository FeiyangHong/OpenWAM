# R2 RoboTwin Pilot 执行报告

Goal ID：`r2/robotwin-pilot`。开始日期：2026-09-12。状态：执行中；P2 工程验收与正式协议冻结已完成，四组正式训练和 pilot 评测尚未完成。范围见 [GOAL](GOAL.md)，已验证命令见 [USAGE](USAGE.md)。

## 版本与当前进度

R1 基点为 `98089d1fa612b7d1f0349ae68b3b8b0ef690aef7`；计划固化提交为 `368654c874cbbadc19fa18ec955e17982c7a0081`。已建立 `feat/statewam-r2-robotwin-pilot`，当前实现尚在验收。产物统一位于仓库根目录 `statewam_artifacts/robotwin/r2-robotwin-pilot/`，下文 run 路径均相对此目录。

| 阶段 | 当前事实 | 尚缺证据 |
|---|---|---|
| P0 | 数据准备、来源身份审计及加强 scene metadata 校验后的真实 reader 审计通过 | 无；P1 场景行为筛选是后续门槛 |
| P1 | 数据隔离、初始化、四组 workload/有效参数与资产审计通过；真实模型配对 WS、完整物理生命周期及 STOP/SIGKILL 恢复汇总通过；完整 100 开发 / 400 最终场景及 20 项跨进程初态独立审计通过 | 计算量非完全匹配（已在协议中如实标注） |
| P2 | 独立 CUDA 生命周期、真实 E3 四卡有序恢复/写入中断/STOP/接管、选定 batch16/global64 完整恢复、有限调优、E0/E3 各 200-step profiling、物理中断恢复汇总及三卡多服务生命周期 smoke 通过；正式协议已冻结 | GPU 0/1/2/3 当前资源不可同时确认空闲 |
| P3/P4 | 未执行 | 四组正式训练、开发/最终配对 rollout 和视频 |
| P5 | 执行报告与准备阶段 usage 已建立 | 完整比较结果、区间、性能总账与最终交付 |

当前不报告模型成功率、结构收益或收敛结论，不把 P0/P2 完成当作 pilot 完成。正式协议已冻结，但最近只读核验显示 GPU 1/2/3 有无法归属的外部 C+G 上下文，因此尚未启动正式训练；GPU 4–7 不属于本 Goal allowlist。

以上为首次准备快照；后续工程 run 见下文，不将该快照当作当前 GPU 状态。

## 数据与来源证据

数据目录：`/gaozt-test1/datasets/robotwin/r2-robotwin-pilot/clean50-785feb15-s20260912-v1/`。固定 HF revision 为 `785feb15aa4a4f532395ad2b1d2be5f28cb561ad`，variant 为 `aloha-agilex_clean_50`。数据卡标注 MIT，原文保存在 `source_manifest.json`。匿名镜像下载，没有使用或持久化 HF token。

10 个任务归档总量为 4,765,939,421 bytes；`adjust_bottle` 重验 SHA256 后链接已有归档，其余下载到本数据目录。全部归档校验固定 revision 的 LFS SHA256，检查 ZIP 成员路径并保留原始文件；不反序列化上游 pickle。500 条 episode 按 SHA256 稳定排序划分，每任务 45 train / 5 val，合计 450 / 50。精确动作轨迹与显式 session 字段参与分组；本轮不声称已检测视觉近重复或未提供的历史 session 关系。

| 任务 | Train 窗口 | Val 窗口 |
|---|---:|---:|
| adjust_bottle | 6425 | 608 |
| handover_block | 12649 | 1280 |
| lift_pot | 4993 | 406 |
| open_microwave | 22056 | 2122 |
| pick_dual_bottles | 5510 | 464 |
| place_mouse_pad | 6792 | 584 |
| place_object_basket | 11087 | 1056 |
| press_stapler | 5342 | 456 |
| stack_blocks_two | 14103 | 1389 |
| stack_bowls_two | 14053 | 1429 |
| 合计 | 103010 | 9794 |

Train 使用原 reader 的尾部 padding/mask 窗口规则；val 保留完整窗口分布。窗口数不等于均匀任务采样下各任务真实曝光，最终 N 仍待多任务 profiling 和预算校准。

- `episode_manifest.json` SHA256：`cf6612a693b22f3b71c30e6078981a93f73e917680b86350df1dd6b8272b06a5`。
- `normalization_stats.npy` SHA256：`36021ed9fa13b576b0a9019a065c5ae20a9d85f10790a95e7e24b6e2bf8ff611`。仅从 train 拟合，rot6d 维度沿用仓库 identity 规则；拟合文件清单见 `statistics_provenance.json`。
- `runs/p0-data-001/`：下载、校验、分组、统计的日志与 CPU/RAM 采样，任务成功结束。
- `runs/p0-source-audit-002/source_audit.json`：Foundation 全文件 SHA256 与 Goal 一致；R1/计划祖先关系、固定 simulator HEAD 和已记录依赖版本通过。资产源归档重验通过，解压资产的完整身份与场景行为检查留在 P1。
- `runs/p0-reader-audit-001/reader_audit.json`：通过真实 reader 检查完整文件选择与窗口索引，train/val 各 30 个实际样本的 9 帧、384×320、32×80 action、20D↔80D 往返与 val 无随机增强。发布模型配置仅用于读取兼容几何和映射，没有复用其权重或归一化统计。
- `runs/p0-reader-audit-002/reader_audit.json`：加入 scene metadata 一致性检查后复验通过；`reports/p0_acceptance.json` 汇总 P0 验收、数据身份、源码哈希与回归边界，`manifests/jobs.json` 为当时任务身份及终态快照。

Reader 新增 `episode_manifest` 时，必须加载匹配的 train-only 统计 provenance，并验证数据与指令哈希；不允许缺统计时静默禁用归一化、缺相机时填黑图、异常读取时换 episode。修复了 allowlist 筛选后按列表下标错误索引 `scene_info` 的问题，改用原 episode 编号。

## 容量方案进展

E1/E2 的候选容量控制为 `capacity_control=current_resample`：将当前帧 residual tokens 做带 mask、无新增可学习参数的自适应平均池化，变换到 action reader 的 token 数与输入维度，再使用既有 `action_proj` 和 `read_action`。E3 仍读取真实 action hidden tokens。默认 `none`，不改变已有 R1 配置和 checkpoint 结构。

小型真实 Wan/ActionDiT 测试已验证 E1/E2 与 E3 的 State 参数均参与 backward；E1/E2 辅助读取不随 future/action 值或 action padding mask 改变，E1 整体 State 对未来与动作扰动隔离，同时对当前输入敏感。尚未通过实模型 `capacity_report.json` 门槛，也不声称 compute-matched：E1 的主 world reader token 数仍少于 E2/E3。

## 中断与失败记录

`cpu_job.py` 仅用于 CPU 准备任务，不是正式 GPU 训练管理器。已通过 tmux 独立启动、原调用返回后继续采样、进程身份查询和显式停止测试；`p0-detach-check-001` 正常结束，`p0-stop-check-001` 按停止请求退出，子进程均已不存活。此为独立会话测试，不是客户端 interrupt 按钮实测。尚未实现自动 P0–P5 编排，不承诺助手中断后自动进入下一阶段。

`p0-source-audit-001` 失败原因是审计脚本把 torch 包元数据 `2.4.1` 与运行时 `2.4.1+cu121` 直接比较；已分别记录包版本与运行时版本，`p0-source-audit-002` 重验通过，未修改模拟器环境。原失败日志与消耗保留。所有已启动准备 run 均有 `job.json`、`state.json`、`resource_samples.jsonl`、`performance_summary.json`；全过程性能总账在后续阶段汇总，不能用当前 CPU 记录替代训练与评测性能证据。

扩展回归 `runs/p1-regression-001/results.xml`：269 passed、1 skipped、0 failures/errors。跳过项为 `test_wam_architecture.py` 的 CUDA ActionDiT RoPE 设备迁移检查，因为 CPU 准备入口隐藏 CUDA；必须在后续 GPU 工程验收中补跑，不能标记为已通过。新增测试覆盖 split/统计污染、缺失文件、指令/scene 修改、读取失败不替换样本、容量分支隔离/梯度、CPU 任务去重/停止/失败状态。

数据准备用时约 422 秒，2 秒周期采样 209 次，进程树峰值 RSS 约 560 MiB；源审计、两次 reader 审计、测试及失败尝试另有各自性能文件，不包含在该 422 秒中。数据目录本地逻辑文件约 11.44 GB，另有约 0.30 GB 的已存在归档链接，不把链接目标重复计为新增存储。当前无训练 GPU 消耗；整段准备阶段的 CPU 总成本未按进程退出后的完整 CPU 累计量计量，不以 RSS 或 wall-clock 冒充 CPU-hours。

## 初始化与 GPU 工程续进

2026-09-12，继续 P1/P2 工程检查，尚未启动正式训练或 profiling。

- `p1-initialize-001` 的 330 个 State 张量键名一致；抽检重载值相等，但 checkpoint 是 bf16、skeleton 是 fp32，严格 hash 因 dtype 不同而失败。`p1-initialize-003` 使用 action skeleton dtype 对齐仍失败，证明不能以 skeleton 默认 dtype 代替 checkpoint 身份。现改用初始化专用 `load_state_dict(strict=True, assign=True)`，仅在优化器创建前调用，逐张量校验 dtype、shape、value；不修改共享续训加载 API。`p1-initialize-004` 在新目录 `checkpoints/foundation-matrix-v3/` 重验，最终状态以其完整记录为准。v1/v2 失败证据保留。
- `p1-initialize-002` 因启动命令多写一个 `--`，在创建子进程前报 `FileNotFoundError`；没有创建初始化目录或占用 GPU，失败 job/state/performance 文件保留。
- GPU 预检查发现新版 NVML unavailable 常量是 ctypes 值，旧采样逻辑先后触发 `AttributeError` 和 `TypeError`；修正后又发现 v1 memory used 包含约 362 MiB 驱动保留内存，误拒绝空闲卡。现优先使用 v2 接口分别记录 used/reserved，v1 fallback 明确标注；未放宽外部进程和利用率检查。`p2-gpu-engineering-001/002/003` 仅为未创建 run 目录的启动前失败尝试，不冒称有 workload 采样。
- `p2-gpu-engineering-004`：GPU 4 UUID 绑定、单卡可见、CUDA 矩阵乘法 finite、框架显存输出、tmux 独立运行与正常释放通过。约 16.17 秒、9 次采样，分配成本约 0.00449 GPU-hours。这不是模型 profiling。
- 该 run 发现 NVML 的宿主 PID 无法通过本容器 `/proc`/NSpid 映射；CUDA 实际使用时旧摘要错误输出 `observed_process_gpu_hours=0`，该值及原 energy 汇总无效，保留原文件作为失败证据。采样器现记录 unmatched PID，归属不全时进程 GPU-hours 与项目 energy 写 null；租用时长仅作为资源分配成本，不冒充模型有效计算。当前没有解决跨 PID 命名空间的完整归属审计，不把短 CUDA 成功当作 P2 全部通过。
- `p2-gpu-stop-001`：单卡短任务实际推进 22 次矩阵运算后收到 STOP，自主退出，`stopped_cooperatively=true`、状态 stopped、退出码 0。实际进程身份已核验退出，GPU release snapshot 中该卡无进程；框架 peak allocated 42,991,616 bytes、peak reserved 58,720,256 bytes。只验证短 CUDA 生命周期，不证明训练 checkpoint 恢复或客户端 interrupt 按钮行为。
- `p2-cuda-regression-001/results.xml`：此前 CPU 回归跳过的 ActionDiT RoPE CUDA 设备迁移检查补跑通过，1 passed。完整 CPU 回归中的同一 skip 不应被重复统计为未验证。

本轮开始时全部 GPU 空闲，稍后 0–3 被其他任务使用；短工程任务分别选 4、5，项目同时最多 2 卡，未触碰其他任务。当前依然缺少完整四卡恢复、模型容量审计、正式训练预算及配对场景工具。此前 2–7 天估计不具备本轮实测依据，已在 GOAL 撤回，待 P2 吞吐与 N 冻结后重算。

`p1-regression-002` 是局部回归，167 passed、1 CPU 环境 CUDA skip；`p1-regression-003` 使用此前完整回归文件清单，305 passed、同一 CUDA skip、0 failures/errors。后者包含新增的 E0–E3 world 参数冻结但输入保留梯度检查（eager/checkpointed）、拒绝父模块 no_grad 包装、初始化 dtype 保持、NVML sentinel、GPU lease 上限/外部进程拒绝及缺失归属不填 0 的测试。CUDA skip 的独立补跑证据见 `p2-cuda-regression-001`，不能将这些小模型测试当作实模型容量验收。

`p1-initialize-004` 最终 completed，supervisor/child 均已退出。`checkpoints/foundation-matrix-v3/COMPLETE.json` 与 initialization JSON 的 SHA256 `b2f2d5424c567442a942f1fa70711b50a98ffc97f25b03aba023354d944dc1ea` 一致，四组配置和统计文件校验通过。转换保留 Foundation 的 2089 个张量，新增 330 个 State 张量；E1/E2/E3 使用同一权重文件 SHA256 `7fd0f207914ca0f3bf36d098990b1b5491b2241921f31f6143ade5234029e135`。各组所有重载张量均校验 shape/dtype/value，无 meta 张量。E0 可训练参数 1,021,367,376，E1–E3 各 1,095,479,382（State 74,112,006）；这是参数身份/冻结计数，不是有效容量结论。

该初始化脚本原在运行结束才取 `code_files` hash；期间 matrix.py 新增父模块 no_grad 拒绝检查，原 JSON 的 matrix.py hash 不能当作当次已加载源码身份。以该 run 的启动 job.json 为依据，另存 `foundation-matrix-v3/provenance_correction.json` 说明，保留原 initialization 与 COMPLETE 不变；后续脚本改为开始初始化前获取源码 hash。启动记录仍不是完整不可变源码快照，正式长任务入口须补齐源码快照及环境身份。

## 四卡恢复与模型训练入口

2026-09-12，新增 R2 `checkpoints.py`、`exposure_sampler.py` 和独立工程训练入口 `train.py`。完整状态由 Accelerate/ZeRO-2 保存到临时目录，各 rank 记录同一更新/曝光位置；同步后对文件做 SHA256 清单校验，原子发布目录并更新 `latest-complete.json`。加载先核对协议、数据、配置、初始化身份、四份 RNG/优化器分片和文件哈希，再读取本地可信 full state。失败或不完整 checkpoint 不替代旧完整版本；目前没有自动清理旧检查点。

- `p2-zero2-reference-001` 在首次四卡准备时失败：DeepSpeed 的可用显存查询强制将 CUDA_VISIBLE_DEVICES 解析成整数，不接受已在普通 CUDA probe 中通过的 UUID 字符串。没有因此修改共享安装包。启动器改用数字索引和 PCI_BUS_ID 顺序，同时记录租用 UUID；新增 CUDA Driver API 校验实际逻辑设备 UUID，且固定 PYTHON_EXEC，避免 torchrun 从 tmux 继承错误的解释器选择。四卡后续 probe 已核验实际设备为 GPU 4–7，Python 为 statewam 环境。
- `p2-zero2-reference-002` 连续四次 optimizer update；`p2-zero2-prefix-001` 保存前两次；`p2-zero2-resumed-001` 使用独立进程从完整状态继续后两次。四个 rank 的逐样本/输入哈希、loss、Python/NumPy/CPU/CUDA 随机数、最终模型张量哈希、优化器状态及 scheduler 均与连续参照逐值相同。完整 checkpoint 含四份 ZeRO-2 优化器分片、四份 RNG、模型和 scheduler。验收工具 `audit_resume_probe.py` 已执行，证据为 `reports/zero2_resume_probe_001.json`。这只是小型 synthetic 模型的四卡恢复，不是实模型续训或客户端 interrupt 实测。
- 上述失败与三个成功 probe 分配成本分别约 0.02020、0.02019、0.02021、0.01796 GPU-hours；NVML 的宿主 PID 归属问题仍存在，进程 GPU-hours 与项目 energy 仍为 null，不把分配时长冒称模型有效计算。
- `p2-loop-tests-001` 16 passed；加强断写和缺分片测试后的 `p2-loop-tests-002` 19 passed。覆盖原子发布、损坏文件、匹配 inventory 下缺 RNG/优化器分片仍被拒绝、保存异常保留旧 latest、恢复游标和配对损失反向传播。断写测试使用异常注入，尚未实测进程在大 checkpoint 写入时被杀。
- `p2-model-e0-001`：实际 Foundation E0、固定 clean train split、四卡 bf16/ZeRO-2、micro-batch 1、accumulation 1、global batch 4，共两次 optimizer update（8 次样本曝光）。全部 rank loss/梯度有限，每步有 826 个参数张量梯度记录；框架峰值 allocated 28.43–28.50 GiB。任务 completed、子进程已退出，完整状态发布到 `training/checkpoints/update_00000002/`。启动至退出约 378.20 秒、0.42022 分配 GPU-hours；保存/完整性校验约 198.69 秒，产物约 36 GiB。不能用两步耗时当作稳定吞吐或收敛证据。
- `p2-model-e3-001` 使用相同几何、batch、诊断 seed 和两步预算，初步逐 rank 记录与 E0 首步样本/噪声/loss 相同；首步 6 个 gate 非零梯度、第二步全部 330 个 State 张量非零梯度。完整 run 状态与保存结果仍需核验。尚未生成覆盖 E1/E2 的实模型 capacity_report，不声称 compute-matched。

R2 入口当前只开放 1–8 步工程检查，不能直接当作正式训练或 profiling 入口。正式协议冻结、200–500 步 profiling、实模型精确恢复、E1/E2 容量对照、scene/instruction 评测工具和源码漂移保护接入仍待完成；工程检查不缩减原 Goal 的正式训练/配对评测范围。

后续状态更新：E3、E1、E2 的两步工程任务均 completed，所有 rank 已退出、GPU lease 表为空。四组所有 rank 的两步样本/噪声完全配对，首步各项 loss 完全相同；E1–E3 第二步 330 个 State 参数张量梯度均非零。E3 整次约 383.22 秒、0.42580 分配 GPU-hours，保存约 203.06 秒；E1 约 379.73 秒、0.42193 分配 GPU-hours，保存约 198.72 秒。完整 checkpoint 的逻辑字节约 60GB（E0 59,842,266,345 bytes，E1 60,879,964,137 bytes），不同于文件系统 `du` 报告的分配空间，不混为一项。

E1/E2 已接入 `source_snapshot.py`，在运行目录复制 openwam/scripts Python 与配置源码并保存哈希，模型构造后及保存前核验漂移；恢复入口要求前后源码清单相同。E0/E3 的旧工程 run 未追溯补造源码快照。该快照未囊括第三方包/模拟器资产身份，也不等价于操作系统只读隔离；正式运行仍须记录完整环境身份。新增快照测试通过，`p2-regression-001` 扩展回归 314 passed、1 CPU CUDA skip（独立补跑证据不变）。

`p2-matrix-engineering-audit-001` 正在独立核验四组完整 checkpoint 的文件 SHA256、latest 指针、逐 rank 完成位置、样本/噪声配对和 State 梯度，输出目标 `reports/model_engineering_001.json`；文件存在且 passed 后才能作为该项最终验收。该报告不是 `capacity_report.json`，不涵盖 source token/attention 工作量、实模型恢复或收敛结论。

最终更新：`p2-matrix-engineering-audit-001` completed，`reports/model_engineering_001.json` 的 `engineering_passed=true`，SHA256 `e144a48324d9e76cb95a58bef712e1b9f051d623d37a5cc0b7126af0d0a79fe1`。四组 checkpoint 已独立逐文件重验。E0/E1/E2/E3 整次分配成本分别 0.420218/0.421926/0.427826/0.425804 GPU-hours，合计 1.695774，不含小模型 probe、失败、CPU 校验及此前工程成本。最后核验本 Goal 无存活任务，GPU leases 为空。上述工程成功不解锁正式训练：source-token/attention 成本、实模型恢复、profiling、场景协议和最终 N 仍待完成。

## 真实模型恢复前缀验证

2026-09-13，按 `manifests/model_resume_002.json` 接续真实 E3、四卡 ZeRO-2、micro-batch 1、accumulation 2、global batch 8 的恢复验收；这不是正式训练或 profiling。

- 旧协议 `r2-model-engineering-v1` 的 reference/prefix 样本和噪声配对，但恢复前已出现梯度及后续 loss 分叉，证据保存在 `reports/model_resume_001_failed.json`。未启动旧协议的 resumed 分支，不能把该失败归因于 checkpoint 加载。
- `r2-model-engineering-v2` 启用严格确定性算法及 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。本轮重新核验 reference-002 的源码快照未漂移、任务进程已退出；该四步参照 wall-clock 366.728 秒，分配成本 0.407475 GPU-hours，完整产物位于 `runs/p2-model-resume-reference-002/`。
- 本轮启动 `p2-model-resume-prefix-002`，仅占用经核验空闲的 GPU 4–7；0–3 为其他任务，未触碰。四个 rank 的四个 microsteps 与参照前缀在 position、samples、noise_hashes、metrics、gradients 字段逐值一致。prefix 正常完成，退出码 0，完整 update 2 checkpoint 已发布，supervisor/child 已退出。
- 独立 CPU run `p2-model-prefix-audit-002` 正在重验上述前缀与 checkpoint 文件身份，输出目标 `reports/model_prefix_002.json`。只有该 run 完成且报告 passed 后才启动 resumed-002；前缀一致不等于完整续训状态已经通过。

后续依次完成新进程续跑、最终模型/优化器/RNG/DeepSpeed 状态深度比较、真实写入中断与 STOP 验收；容量成本、场景回放和 200–500 步 profiling 仍是正式矩阵的前置门槛。任务实时状态以对应 `state.json` 加实际进程核验为准，不以本段历史快照推断仍在运行。

后续更新：`p2-model-prefix-audit-002` completed，`reports/model_prefix_002.json` passed；prefix 分配成本 0.396055 GPU-hours、wall-clock 356.449 秒。随后 `p2-model-resume-resumed-002` 从 update 2 完成至 update 4，next occurrence=32，四卡进程退出，分配成本 0.694819 GPU-hours、wall-clock 625.337 秒。四个 rank 的 reference 与 prefix+resumed 的全部八个 microsteps，在样本、噪声、loss、梯度记录和位置字段上逐值一致；此处梯度记录为逐参数范数/有限性，不声称保存了所有梯度张量。

完整状态审计 `p2-model-resume-audit-002` 在 DeepSpeed `fragment_address` 元数据上遇到不支持的比较类型，失败报告为 `reports/model_resume_002.json`；这是审计器覆盖不足，尚不能据此判断模型状态相同或不同。已增加该类型的 `numel/start` 精确比较及差异拒绝测试，未改变模型、优化器或训练轨迹，也未扩大容差。重验 run 为 `p2-model-resume-audit-003`，目标报告 `reports/model_resume_003.json`，结论待其完成。保存的三段训练源码快照仍保持原样。

## 场景探针首轮验证

2026-09-13，执行模型无关的 `adjust_bottle` 开发 seed 探针，使用固定 RoboTwin 版本、原专家和成功条件，没有运行策略或访问最终策略成绩。

- `p1-scene-adjust-clean-001`：seed 200000 被上游 `UnStableError` 拒绝。旧探针将其作为异常退出，保留完整失败与 0.022918 GPU-hours 成本。
- 探针现仅将专家阶段的该特定异常记为 `unstable_initial_scene` 拒绝；回放不稳定和其他异常仍报错。`p1-scene-adjust-clean-002` 对同一 seed 验证拒绝分类，completed 且 expert_eligible=false，成本 0.012414 GPU-hours。
- `p1-scene-adjust-clean-003`：按候选顺序检查 seed 200001，expert_eligible=true。固定指令及两次 reset 已保存；数值 metadata 无差异，head/left/right 三路 RGB 的 max、mean、fraction_changed 均为 0，成本 0.019714 GPU-hours。仅证明该场景的重复 reset，不等于全任务/双条件场景门槛通过。
- 以上原始证据位于对应 run 的 `probe/scene_probe.json`；有效场景另含 `instruction.json`、`reset_0/`、`reset_1/`。`p2-resume-scene-tests-002` 完成 17 项测试，涵盖元数据比较、拒绝分类、固定指令、checkpoint 和源码快照。随机条件探针 `p1-scene-adjust-random-001` 已启动，尚未记为通过。

最终更新：`p2-model-resume-audit-003` completed、退出码 0，`reports/model_resume_003.json` passed，SHA256 `c8d0100e821cb81e8dad9a6e28e479c29aad1c5a9128919cd04979e9711c3ba4`。三段工程 run 的完整 checkpoint 文件身份均通过重验；reference 与 prefix+resumed 的四 rank 全部轨迹一致，最终九份序列化状态（模型、四个优化器分片、四个 RNG 文件）的语义内容逐值/逐字节一致，包括 DeepSpeed 元数据。constant LR 无 scheduler，bf16 无活动 loss scaling；未把这些不适用项冒充额外验证。

严格恢复三段 GPU run 共 1.498349 GPU-hours；成功的 CPU 深度审计用时 625.277 秒、进程树峰值 RSS 约 112.33 GB（共享页面可能重复计入），旧审计失败另耗 223.578 秒，均保留性能原始记录。该结果只覆盖真实 E3、micro-batch 1、accumulation 2、四卡 ZeRO-2 的有序保存/新进程恢复；不覆盖强制写入中断、客户端 interrupt 按钮、其他组或其他 batch 配置。

`p1-scene-adjust-random-001` completed，seed 200000 expert_eligible=true，两次 reset 无数值 metadata 差异，三路 RGB 逐像素一致，成本 0.018490 GPU-hours。这里只完成 `adjust_bottle` 各一个有效 clean/randomized 场景的探针，不冻结全任务容差或最终 evaluation manifest，不报告策略成功率。

本轮结束核验 GPU 4–7 无本项目显存占用，0–3 仍为其他任务。下一步补真实 checkpoint 写入中断、训练 STOP 与接管验收，再完成容量成本报告、全任务场景清单和 profiling；正式 P3/P4 尚未开始。

## 故障链最终验收

`p2-model-fault-stop-001` 在恢复后的 update 3 边界完成保存并响应协作 STOP，四个 rank 的 summary 均为 stopped、position=`{optimizer_update:3,next_occurrence:24}`，进程随后全部退出。`p2-model-fault-finish-001` 从该 checkpoint 只补 update 4，正常完成并释放 GPU。

`p2-model-fault-audit-001` completed，`reports/model_fault_audit_001.json` passed，SHA256 `b11b8ff098fbb5d248b81b0f674233eeb2474d802c375e05dd457d8a60a2f5d0`。审计确认：

- update 4 写入中断的 partial 被拒绝，update 2 latest 指针和完整性未受影响；
- STOP 后的 update 3 checkpoint 与预期位置及身份一致；最终 finish update 4 与未中断 reference 的模型、四个 ZeRO-2 优化器分片、四个 RNG 文件逐值/逐字节一致；
- duplicate launch 在 live run 期间被租约拒绝，未替换原 supervisor/child；
- 四 rank 的逻辑 8 个 microsteps 实际执行 12 个，实际 48 次样本曝光中 16 次为中断后的重算；三段 GPU run 合计 1.716159 GPU-hours。

这完成了真实 E3 micro-batch 1、accumulation 2 的写入故障/协作停止/接管工程验收，但仍不等价于客户端 interrupt 按钮测试或所有正式 batch 配置的恢复覆盖。

## 真实写入中断与停止验收

2026-09-13，按 `manifests/model_fault_001.json` 启动独立工程 run `p2-model-fault-write-001`（真实 E3、4 updates、micro-batch 1、accumulation 2、save-every 2）和 CPU 观察器 `p2-model-fault-write-watch-001`。先验证项目无存活任务及租约，GPU 4–7 空闲；0–3 的其他任务未触碰。

观察器只接受本 Goal、engineering、名称含 fault、1–8 步真实训练入口的目标，核验 supervisor/child 创建时间及父子关系。计划在 update 4 的 partial checkpoint 中发现实际训练进程持有可写 `.pt` 文件、且文件跨两次观测增长时，记录目标身份后仅 SIGKILL 该写入进程。其余 rank 由 torchrun/已有 GPU supervisor 收尾；不向其他作业或 supervisor 注入故障。

触发结果写入 `reports/model_fault_write_001.json`，只证明故障是否投递，不代表恢复验收通过。后续须另行核验旧 update 2 的完整性、latest 指针未前移、不完整 update 4 被拒绝，以及进程/GPU 释放；再从 update 2 恢复并通过协作 STOP 保存退出，保留所有 partial 文件与重算消耗。故障观察器有 1800 秒上限，训练本身仍是固定四步，不自动无限重试。新增目标约束/死进程测试 2 passed。

后续更新：写入故障已投递。rank 0 持有 update 4 的 `mp_rank_00_model_states.pt` 可写 FD，跨观测从 0 增至 27,279,104 bytes；观察器记录进程创建时间并发送 SIGKILL，torchrun 报告该 rank 退出码 -9，GPU supervisor 收尾后本项目进程和 GPU 4–7 占用均释放。该工程 run 用时 352.853 秒、0.392058 GPU-hours，按预期失败，不记为训练完成。

`p2-model-fault-check-001` 完成，`reports/model_fault_check_001.json` passed：旧 update 2 完整文件哈希通过，latest 指针仍指向其原 COMPLETE 哈希；update 4.partial 保留 4,131,974,576 bytes，没有完成标记，也未发布正式 update 4，恢复验证器明确拒绝。CPU 审计耗时 64.862 秒。`p2-fault-tests-001` 的目标归属、可写文件、checkpoint 和资源相关测试 23 passed。

接着启动 `p2-model-fault-stop-001` 从旧 update 2 恢复，使用相同可执行源码及原四步总预算；独立观察器 `p2-model-fault-stop-watch-001` 待首个恢复 microstep 的持久进度后投递 STOP。重新接管测试尝试重复 launch 同一 live run，被 GPU 租约机制拒绝，原 supervisor/child 创建身份及 job 文件哈希不变，证据 `reports/model_fault_takeover_001.json`。STOP 完整保存、退出和后续剩余更新仍待验收。

## Workload 与确定性容量分支

2026-09-13，重新核验本 Goal 没有存活任务，GPU 0–7 均空闲；本轮仍只选 GPU 4–7，四卡串行执行各组两步 workload tracing，不是正式 profiling 或训练。

- `p2-workload-e0-001`（工程协议 v3）completed、全部进程退出，wall-clock 330.410 秒、分配成本 0.367122 GPU-hours。SDPA trace 首次确认该数据几何下 MoT 有 360 个 video tokens 与 32 个 action tokens；预处理的 480-token VAE attention 不与 MoT tokens 混计。E0 全部 1,021,367,376 个可训练参数参与 backward 且在两步内出现非零梯度。该时间包含诊断同步，不能当稳态吞吐。
- `p2-workload-e1-001` 在首步 backward 失败：CUDA `adaptive_avg_pool2d_backward` 不支持 `torch.use_deterministic_algorithms(True)`。四 rank 已退出并释放资源，分配成本 0.193640 GPU-hours，原日志/状态/源码快照保留。此前非严格确定性 E1/E2 工程通过不能覆盖此配置。
- 容量补偿仍为只读当前帧的无参数 pooling，但改为同一 floor/ceil 自适应区间的可分离均值：整除维度 reshape/reduce，非整除维度用固定均值矩阵；低精度输入在 FP32 中累计后转回原 dtype。不增加可学习参数或未来/action 信息。CPU double 输出/梯度与原 pooling 等价测试、真实 CUDA bf16 严格确定性反向重复测试及已有隔离/容量/workload 测试共 31 passed，证据 `p2-pooling-cuda-tests-001/results.xml`。低精度舍入路径已变化，不能冒称与旧实现逐位相同。
- 新工程协议 `r2-model-engineering-v4` 显式记录 `capacity_pooling=deterministic_separable_fp32_v2`，将重新采集四组相同协议的 workload。已启动 `p2-workload-e1-002`，固定两次 updates、micro-batch 1、accumulation 1、global batch 4。实时状态须结合其 `state.json` 和实际进程核验；尚未生成四组 `capacity_report.json`。
- 新增 `audit_workload.py` 汇总逐 rank 参数/梯度、State source、forward 与 checkpoint recomputation 的 SDPA 几何和 mask 工作量；不将 mask 允许 pairs 当作稀疏执行 FLOPs，不宣称 compute-matched。基础审计与隔离回归 `p2-workload-audit-tests-001` 为 27 passed。`profile_metrics.py` 是后续 profiling 的分段计时基础，尚未接入或执行正式 200–500 步 run。

`p2-workload-e2-001` 在模型导入前失败，原因是短暂新增的入口文件名 `profile.py` 遮蔽 Python 标准库，触发 `cProfile` 导入错误；任务没有执行训练更新，进程已释放，原失败日志保留。入口已改为 `profile_training.py`，`train.py --help` 和 profile plan/metrics 回归均通过；该失败不改变模型或数据身份。已重新启动 `p2-workload-e2-002`，使用 v4 协议，等待其终态后再采集 E3。

本轮下一步先完成 v4 四组 workload/capacity 审计，再开展有限 batch/worker 候选与 E0/E3 各 200–500 optimizer-update profiling。正式训练入口、预算、全任务场景 manifest 及策略评测仍未完成；当前工程进展不代表整个 R2 已通过。

### 四组 workload 最终结果

`p2-capacity-report-001` completed，`reports/capacity_report.json` 的 `workload_audit_passed=true`，SHA256 `b6beca8fde76862c15049bc32e3cc6265f9314828483cd08f0eb5a4f7d6f1552`。v4 有效 run 为 `p2-workload-e0-002`、`p2-workload-e1-002`、`p2-workload-e2-002`、`p2-workload-e3-001`，全部四卡两步、global batch 4，全部进程已退出。审计逐 rank 验证样本/噪声配对、zero-gate 首步 loss、参数梯度、State 六层读写及 forward/recomputation 覆盖。

| 组别 | 可训练且参与 backward 参数 | State 参数 | State 主 world tokens | State forward SDPA dense QK/AV 算术/样本 | 分配 GPU-hours |
|---|---:|---:|---:|---:|---:|
| E0 | 1,021,367,376 | 0 | 不适用 | 0 | 0.372047 |
| E1 | 1,095,479,382 | 74,112,006 | 120 | 137,035,776 | 0.375423 |
| E2 | 1,095,479,382 | 74,112,006 | 360 | 184,221,696 | 0.388576 |
| E3 | 1,095,479,382 | 74,112,006 | 360 | 184,221,696 | 0.373762 |

四个有效 run 合计 1.509809 分配 GPU-hours，不含旧 E0、失败、CUDA/CPU 测试等成本。峰值 allocated：E0 28.43–28.50 GiB，E1–E3 28.81–28.90 GiB；完整 checkpoint 保存约 166–181 秒。上述是 instrumented engineering 数据，不是稳态 profiling。language/proprio source 为 513 tokens，辅助 reader 与 action writer 为 32 tokens；实际有效 mask 有样本差异，例如 E2 的辅助当前帧分支 32 个有效 tokens、同位置 E3 action 分支 11 个有效 tokens。所有逐 rank 几何、允许 pairs、时间和显存明细均保存在报告。

容量决定：保留确定性 `current_resample` 作为 E1/E2 参数控制；实际可学习参数参与量匹配，但 E1 主视觉长度、辅助 pooling 和有效 mask 工作量并不完全相同。`compute_matched=false`，主比较按 optimizer updates/样本曝光匹配，并报告实测成本；不将 SDPA 算术当成全部 FLOPs 或实际 kernel 计算量，不据此独立证明收益只能来自未来信息。

回归证据：`p2-capacity-report-tests-001` 27 passed；`p2-workload-regression-001` 为 137 passed、1 CUDA skip（该项由 `p2-pooling-cuda-tests-001` 的真实 GPU 测试覆盖）；profile/timing 接入测试 `p2-timing-entry-tests-001` 21 passed。

### 有限 Profiling 准备

候选事先登记在 `manifests/profiling_candidates_001.json`：E3 每卡 batch/worker 分别为 4/2、8/4、16/4，每候选最多 8 updates、仅末尾保存；前 2 updates 作为预热。首次 OOM 才允许 1/2 fallback；较小 batch OOM 或不足 10 GiB 峰值余量时不继续更大 batch。选择稳定 samples/s 最优者，5% 内接近时选较小 batch；不更改模型几何、数据、精度、优化器或信息来源。不同候选有效 batch 仅用于诊断，不代表正式协议已冻结。

`profile_training.py` 为单独的 E0/E3 200–500 updates 入口，须验证计划、初始化和已通过容量报告哈希；工程 CLI 仍限定 1–8 步，尚无正式训练入口。`profile_metrics.py` 已接入可选 `--audit-timing`，通过 CUDA events/host wall 分别记录预处理、配对噪声/哈希、forward、DeepSpeed backward 和 engine step，包含关系不能相加。数据等待、微步 wall、稳态分位数、峰值显存及 checkpoint 保存另计；逐 run 的 GPU supervisor 继续持久采样。

已启动首个候选 `p2-tune-e3-b4-w2-001`，四卡、batch 4、workers 2、8 updates、save-every 8；其完成与否须以实时进程及持久状态核验。正式 E0/E3 200 步 profiling 尚未开始。

后续更新：batch 4 与 batch 8 候选均正常完成并释放资源。`select_profile.py` 从各 rank 原始步骤重算计时，核验曝光从 0 到既定总量无缺口/重复、summary 与原始计时一致，以最慢 rank 的 global samples/s 和整卡观测峰值余量选择；不将 NVML 的不可归属进程时间填成 0。

| 候选 run | micro-batch / workers | 逻辑曝光 | 排除预热及保存的 samples/s | 最小整卡峰值余量 | wall-clock / 分配 GPU-hours |
|---|---|---:|---:|---:|---|
| p2-tune-e3-b4-w2-001 | 4 / 2 | 128 | 11.0706 | 62.18 GiB | 345.706 秒 / 0.384117 |
| p2-tune-e3-b8-w4-001 | 8 / 4 | 256 | 13.3804 | 57.99 GiB | 348.671 秒 / 0.387413 |

这是各候选 6 次排除预热后的短窗口，不是正式 200 步 profiling 的稳定估计。完整任务含初始化、采样、模型保存和两次完整性读取；其他 rank 等待主 rank 发布 checkpoint 时可能出现高 GPU utilization，不能据此宣称模型计算已吃满。已按预登记上限启动最后一个 `p2-tune-e3-b16-w4-001`；其结果待核验后写入选择报告及独立 profiling plan。新增选择与 gate 回归 `p2-select-profile-tests-001` 5 passed。

最终选择：`p2-tune-e3-b16-w4-001` completed，512 次逻辑曝光，最慢 rank 14.5893 samples/s，最小整卡峰值余量 55.68 GiB，wall-clock 372.441 秒、分配成本 0.413823 GPU-hours。三档候选共 1.185353 GPU-hours，均无 OOM；batch 16 比 batch 8 高约 9%，超出 5% 近似平局范围，选择 batch 16，不继续增加未登记候选。该数值也是仓库通用配置起点，不冒称 RoboTwin 官方最优 batch；短窗口中 VAE/text 预处理与 checkpoint 固定开销仍明显，不能只用显存空余判断吞吐。

`p2-select-profile-001` completed，选择报告 `reports/profiling_selection_001.json` SHA256 `765ebae1b12f93d278e847a71cb2a71941f09f91063903113d26fda9d7cebd66`。冻结诊断计划 `manifests/profiling_plan_001.json` SHA256 `2b25ed850e8705f0f7b46d41368684d5a97dcd4c0046746b7ee11a2875ec12ff`：E0/E3 各 200 optimizer updates、四卡 per-GPU 16、accumulation 1、global 64、workers 4、seed 20260914，前 10 updates 预热，每 100 updates 保存完整状态，每组 12,800 次曝光，从 Foundation 与全新优化器重新开始。此为 profiling 冻结，不是正式训练协议或 N。

已独立启动 `p2-profile-e0-001`，session `sw-r2-p2-profile-e0-001`，只用 GPU 4–7。进程状态、日志和监控分别位于其 `state.json`、`console.log`、`resource_samples.jsonl`；退出后生成 run 级 `performance_summary.json`，逐 rank 训练计时另在 `training/timing_rank*.json`。接管先核验进程与 GPU，不重复 launch；安全停止入口为 `gpu_job.py stop --run-id p2-profile-e0-001`，其 200-step 总预算不因接管而重置。E3 profiling 尚未启动，不能将计划中的第二组写为已执行。

### E0 Profiling 完成与 E3 接续

2026-09-13，恢复 Goal 执行后核验 `p2-profile-e0-001` completed、exit_code=0，supervisor/child 与已登记进程均退出；项目其他任务没有存活进程或未结束状态，GPU 0–7 均空闲。未重新运行 E0，仅接续同一计划的 `p2-profile-e3-001`，继续只占 GPU 4–7。

`p2-profile-e0-audit-001` 的 CPU 审计已完成，报告 `reports/profiling_e0_001.json` SHA256 `b2bb459153aff7112cff09337d14a552e80585322f7396d3682c94f478f364be`。四 rank 的 200 次更新、12,800 次逻辑曝光、确定顺序/游标、有限 loss/梯度、冻结 world 无梯度及原始计时一致性检查通过；100/200 步完整 checkpoint 全文件哈希、位置和 latest-complete 指针通过。这是单组完成审计，`paired_profiling_passed=false`，不是 E0/E3 配对比较已通过。

| E0 指标 | 实测值 |
|---|---:|
| 整个 run wall-clock，含加载与保存 | 1356.786 秒（22.61 分钟） |
| 分配 GPU-hours | 1.507540 |
| 排除前 10 步预热和保存，最慢 rank 吞吐 | 14.7617 samples/s |
| 同口径 optimizer-update 平均耗时 | 4.3355 秒 |
| 包含加载与保存的端到端吞吐 | 9.4341 samples/s |
| 两次完整保存及验证耗时 | 约 161.31 / 155.64 秒 |
| 两个完整 checkpoint 的逻辑字节总数 | 119,684,535,365 bytes |

四段各 50 updates 的全 rank 平均 action training loss 为 0.040715、0.028999、0.024094、0.022335。仅说明训练早期的下降趋势，不代表 validation、收敛或策略能力。rank 0 的稳态预处理约 1.893 秒/update，forward 约 0.710 秒、backward CUDA event span 约 1.681 秒；嵌套计时不能相加为总 GPU kernel 时间，详细显存与其他 rank 数据见报告。

E3 交接：run `p2-profile-e3-001`、session `sw-r2-p2-profile-e3-001`，固定四卡 batch 16、global 64、总更新 200、保存 100/200。日志及进度在 `runs/p2-profile-e3-001/console.log`、`training/progress_rank*.json`，后台 supervisor 独立维护 `state.json` 与 `resource_samples.jsonl`。本次已核验 child 存活且开始更新，实时位置以这些文件及进程核验为准。该单任务能按固定预算独立运行，尚无跨阶段自动编排；不能把本段交接理解为完成后会自动开始正式训练。停止、查看命令见 USAGE；恢复须先核验完整 checkpoint，只执行剩余更新，不重复 launch。E0/E3 配对审计将在 E3 完成并释放后执行。

### 全任务场景准备队列

新增 `benchmarks/robotwin/r2_scene_queue.py`，按固定顺序分配 10 tasks × 2 conditions × development/final 共 40 个分片至四条 lane，每条 lane 同时仅运行一个单卡场景探针；目标合计 100 个开发与 400 个最终专家有效场景。这些是模型无关的场景准备数量，不是策略 rollout 数量或正式成功率。

队列复用 `r2_scene_manifest.py` 的每分片最多 1000 候选、每 seed 最多 2 次基础设施 attempt 限制，不自动重试失败；遇到 reset 不一致或基础设施异常停止该 lane，先诊断再接管。已完成分片复用前重验 COMPLETE/manifest 哈希、源文件、候选连续性、次数、证据路径及每场景初态/指令/RGB；中断分片按原清单恢复，队列锁拒绝重复接管。完成队列仍保留 `reset_tolerance_frozen=false`，不代替全任务容差审查、资产/环境身份冻结或策略重复性验收。

CPU 测试 `p1-scene-queue-tests-001` 16 passed；加强回归 `p1-scene-queue-tests-002` 30 passed，覆盖四 lane 无重复/无缺口、STOP 后接管、禁止自动失败重试、源码/身份变化、清单篡改、候选缺号/重复、次数超限及锁防重复。模拟器 Python 下 `r2_scene_queue.py --help` 通过。尚未运行真实场景队列；先完成 E3 profiling，再用一张已释放 GPU 验证首个真实分片，不能在四卡 profiling 期间再启动模拟器。

### 配对 Profiling 最终结果

`p2-profile-e3-001` completed、exit_code=0，全部进程退出并释放四卡。随后 `p2-profile-paired-audit-001` 完成，`reports/profiling_paired_001.json` 的 `paired_profiling_passed=true`，SHA256 `8c5ad96419ef8204c3dc35a904456216faad7e44c3bf53a53f359b80f2c251f5`。审计逐 rank 确认两组全部样本/噪声序列配对、zero-gate 首步 loss 相同、各 200 updates/12,800 曝光、100/200 完整检查点和 latest 指针、有限 loss/梯度及原始计时一致性。CPU 审计耗时 265.424 秒。

| 组别 | 稳态 samples/s（最慢 rank） | 含加载/保存 samples/s | wall-clock 秒 | 分配 GPU-hours |
|---|---:|---:|---:|---:|
| E0 | 14.7617 | 9.4341 | 1356.786 | 1.507540 |
| E3 | 14.5653 | 9.2611 | 1382.130 | 1.535700 |

两组 profiling 合计 3.043240 GPU-hours，另计调优、工程、CPU 审计等成本。E3 两次完整保存约 159.50/164.44 秒；四段各 50 updates 的平均 action training loss 为 0.040712、0.028833、0.023719、0.021841。两组均仍是早期学习诊断，微小训练 loss 差异不能当作策略泛化收益；未运行 validation rollout，也没有冻结正式 N。profiling checkpoint 不进入正式结果。

### 解压资产和环境绑定

`p1-expanded-assets-audit-001` 首次只对比源包文件，报告 `scene_assets_001.json` 如实为 failed：没有源文件缺失或变化，但发现 6 个额外 cuRobo YAML，CPU 用时 514.322 秒。经固定 RoboTwin 源码 `script/update_embodiment_config_path.py` 和 R1 安装记录核验，它们是官方将 `_tmp.yml` 中 ASSETS_PATH 替换为仓库绝对路径的派生配置；不是额外修改物理/规划参数。

审计器从已验证压缩包模板重算该生成规则，不运行安装脚本、不写资产、不直接放行额外文件。`p1-expanded-assets-audit-002` completed，`reports/scene_assets_002.json` passed，SHA256 `3a8a3d90e27143ddfc0e8f0b72fbf4c77fba57e07a64983ad8e7cd80485f3f2d`，CPU 用时 229.628 秒。objects 9,368、background_texture 11,000、embodiments 229 个实际文件全部匹配，共 20,597 个；每文件大小/SHA256、模板派生、源包与生成器身份均保留，非运行用 `__MACOSX` 条目单列。

新增 episode 账本需要进程创建时间核验，首次在模拟器 Python 导入时发现缺少 psutil。通过 `p1-simulator-ledger-dependency-001` 安装固定 `psutil==7.2.2` 后导入通过。`p1-simulator-environment-001` 的 `reports/simulator_environment_001.json` passed，SHA256 `f1e9fa27d036555a16dfbd29f7705069bf778850c248f409c5dfb2eac9dc4696`；相对 P0 仅多此包，SAPIEN/mplib 原官方依赖补丁哈希不变。包版本清单和两个补丁核验不冒称整个 site-packages 逐字节冻结。

### 首个真实分片与全队列启动

`p1-shard-adjust-clean-001` 使用单卡 GPU 4，完成 `adjust_bottle/demo_clean/development` 的固定候选顺序：5 个专家有效场景、2 个上游规则拒绝，next_seed=200007。每个有效场景的两次初态与三路 RGB 精确一致。wall-clock 374.566 秒、0.104046 GPU-hours；所有进程退出并释放 GPU。`p1-shard-adjust-audit-001` 进一步通过独立清单/文件/次数审计，报告 `reports/scene_shard_adjust_clean_001.json`，分片 manifest SHA256 `494a02e67f07f4247af29338948861f928651cf155367acc597aa6cc2f272455`。这只验收一个真实分片，不是全任务 reset 容差或真实策略恢复验收。

随后在项目无存活任务且四卡空闲时启动 `p1-scenes-lane0-001` 至 `p1-scenes-lane3-001`，分别单独租用 GPU 4、5、6、7，session 为相应 `sw-r2-<run-id>`。每条 lane 固定 10 个分片，共准备 100 个开发/400 个最终专家有效场景；每分片最多 1000 候选、每 seed 最多 2 个基础设施 attempts，不自动重试失败。首个独立诊断分片的成本保留，正式队列从自身固定清单重新准备，不能把两次相同初态当新增独立场景。

队列绑定上述资产与环境报告的路径/哈希，并在开始和结束重新核验实际解压文件、包版本与依赖补丁；源码哈希和队列/分片锁拒绝漂移与重复接管。启动前复核的 I/O 等待也计入租卡时间，不作为有效模型计算利用率。四条队列均已核验 supervisor/child 存活；各自 `scenes/progress.json`、候选日志、`state.json`、`resource_samples.jsonl` 持续落盘，结束生成 performance_summary。实时完成状态以进程和持久产物为准。

队列可独立推进本 lane 的场景准备，遇到错误停止该 lane，不启动策略评测、训练或无限重试；其他 lane 是独立的模型无关分片任务。全局 40 分片审计、跨任务容差冻结、策略重复性和真实中断恢复尚待完成。队列运行期间不得修改其 identity 中列出的场景源码或模拟器环境；后续评测代码可在不影响这些源码的模块中继续实现。队列 provenance/去重回归 `p1-queue-provenance-tests-001` 29 passed。

### 场景设备越界与文档核验

2026-09-13 补记：上述四条 lane 的“已启动/存活”是历史快照，现已全部停止。`reports/scene_gpu_spill_001.json` 记录租用 GPU 4–7 的模拟器关联 NVML PID 同时在 GPU 0 创建约 5 MiB 的 graphics 上下文，构成额外第五张物理 GPU 的越界占用；不能因显存小或未参加训练而排除。旧监控的容器/宿主 PID 归属存在盲区，不能将租卡总账当成历史完整实际占卡总账，额外时长仍须保留不确定性并单独核算。

本次仅更新文档，使用只读 `gpu_job.py status` 重验 `p1-scenes-lane0-001` 至 `p1-scenes-lane3-001`：均为 stopped，supervisor/child 均不存活，无存活登记子进程；最后一条 lane 的释放快照中 GPU 0–7 均无进程。本次 `nvidia-smi` 亦显示全部设备无运行进程。未在本次发送 STOP、启动或恢复任何计算任务。

`p1-render-gpu0-probe-001` 已 completed、exit 0，supervisor/child 均不存活，释放快照无残留进程。单个探针退出不等于整池并发隔离验收；拟采用 GPU 0 单租约共享多 lane 的新布局，`p1-scenes-pool-002` 本次核验尚无 run 目录，不记录为已启动或验收通过。旧 lane 的源码/布局身份已变化，禁止盲目复用原目录续跑；保留原产物、失败和消耗，再核验新版本。正式训练与策略评测仍未开始。

### 配对评测与推理随机性工程

新增 `r2_episode_ledger.py`：以协议、checkpoint、group/task/condition/scene、instruction、policy seed 和重复性 replicate 建立稳定 episode 键。成功和策略失败都只发布一次；视频/rollout 等关闭后的产物带哈希，终态原子写入，恢复时重验后跳过；丢失 COMPLETE 指针但完整结果仍在时可重建指针。未知 partial 必须先诊断，不能仅凭进程死亡改记 invalid；有证据的基础设施/外部中断最多允许一次原场景重试，第二次仍 invalid 则保留配对缺口。独占锁与 PID/创建时间防止重复接管，保留被中断 attempt。

`r2_policy_protocol.py` 固定 world/action 独立噪声派生规则，按共同协议、场景与 chunk 序号配对；组别、checkpoint、重试和重复性 replicate 不改变噪声，重复性两次 rollout 的账本键仍不同。reset ACK 必须确认完整身份和清空状态，动作响应核对 episode/attempt/action/chunk/噪声身份。这些是协议工具，尚未接入真实配对 WebSocket 服务；不能把纯 ACK 校验测试当作实际缓存清空验收。

E3 profiling 退出后，公共 `BaseWAMArchitecture.generate` 新增可选 `action_seed`，JointInferenceEngine 转发该条件；省略时维持原 seed 轨迹，显式传入时只改变 action 初始噪声。未修改已结束 profiling 的源快照。新推理测试实际执行公共生成函数的 CPU 替身单步路径，核验默认轨迹、双流隔离和不消耗全局 RNG；真实模型 GPU 推理与配对服务仍待验收。

测试证据：episode 账本首轮 10 passed，包括真实测试子进程 SIGKILL 后的有据分类/一次重试；该测试使用小型文件和替身视频，不是 RoboTwin episode 恢复。准备回归 `p1-evaluation-prep-tests-001`、协议回归 `p1-paired-protocol-tests-001` 和各 run 的 XML 保留。`p1-inference-noise-tests-001` 有 2 个测试替身参数重名导致失败、71 passed；修正替身参数后 `p1-inference-assets-tests-002` 93 passed（2 条第三方弃用警告），覆盖公共部署回归、独立噪声、协议、账本和资产生成规则。正式 P3/P4 尚未启动，完整 Goal 未完成。

### 单设备场景池与专家拒绝分类

`p1-scene-pool-tests-001` completed，36 passed。`p1-render-gpu0-probe-001` 在 GPU 0 完成专家筛选与两次精确 reset；25 次采样只涉及 GPU 0、关联跨设备进程数为零，23 次采样仍有容器/宿主 PID 归属盲区。耗时 48.218 秒，分配成本 0.013394 GPU-hours，不报告精确进程 GPU-hours 或模型能耗。

在真实单探针和回归通过后启动 `p1-scenes-pool-002`，四条 lane 共用物理 GPU 0 的一个租约。279 次采样只涉及 GPU 0，最多同时观察到 4 个 GPU 进程，没有关联跨设备上下文；峰值整卡显存 14,793,900,032 bytes，平均利用率 38.362%、p95 98%、峰值 100%。wall-clock 556.684 秒、分配成本 0.154634 GPU-hours。包括开始时四 lane 的资产重验等待，不把它算成有效模拟计算。

该池没有完成全清单：lane 3 在 `open_microwave/demo_randomized/230000` 专家阶段因 `target_pose cannot be None for move action` 断言退出，池记录 failed 并向其余 lane 请求 STOP；其他 lane 均有序退出，全体进程与 GPU 已释放，没有自动重试。旧 supervisor 将携带 STOP 的非零退出覆盖为 stopped，故此 run 的 `state.json` 为 stopped、exit 1，但 `scenes/progress.json` 明确 failed；此处以两者联合证据判定为失败，不视为通过。已修正后续 supervisor 的终态选择，非零退出不再被 STOP 覆盖。

固定上游 `_base_task.py:choose_grasp_pose/grasp_actor` 与 `utils/action.py:Action.__init__` 表明：无候选位姿时两个 grasp pose 都为 None，动作构造触发断言；上游 `script/eval_policy.py` 在专家阶段捕获异常后拒绝候选。新分类只识别这一精确消息、调用文件/函数和 None 局部值，并保存相关源码哈希及行号；不泛化为所有 AssertionError，不用于 reset 或策略执行。未知错误继续停机诊断。`p1-expert-classifier-tests-001` 46 passed，覆盖伪造消息、错误路径、非 None 值及 replay 阶段拒绝分类；`p1-expert-microwave-classifier-001` 已启动真实原 seed 复核。新源码版本为 preparation v2、queue v2、pool v3，旧池原始证据不覆盖、不盲目接管。

历史四条 GPU 4–7 队列的分配成本合计 0.581628 GPU-hours；其任务生命周期并集跨度 542.604 秒，对额外 GPU 0 给出保守上界 0.150723 GPU-hours。这只是这四条 lane 生命周期内的额外成本上界，不是测得的实际占用时长，也不覆盖此前独立探针/分片。首次分片的 0.104046 GPU-hours 同样只是租约口径，不能反证当时没有额外 GPU 0 上下文；全过程总账仍须补齐相关未知区间。

### R2 配对服务链路

新增 `r2_policy_service.py` 和 `r2_policy_client.py`，复用公共模型加载、三视角预处理、WAMPolicy 与同步 32-action executor；独立 R2 协议不改变公共 WS API。reset 校验 group/checkpoint/protocol/episode/attempt/固定文本，实际清空 executor、prompt 和 VACE 缓存；主配置拒绝 velocity cache，State 不跨调用持久化。每次新 chunk 注入独立 world/action seed，逐动作验证序号和身份，禁止缺相机 black-fill 与 instruction 改变后的静默 reset。客户端使用不重发的 `predict_once`，含糊断线后必须重新建立显式 episode 会话，不能重放动作请求。

服务仅允许一个连接拥有模型状态，记录发送前的响应、chunk 哈希与延迟；明确有限消息、wall-clock 和空闲预算，watchdog 在推理期间继续运行，超时 STOP 由外层 supervisor 有界升级退出。`p1-paired-service-tests-001` 29 passed；增加 NumPy proprio 序列化覆盖后 `p1-paired-service-tests-002` 36 passed（2 条第三方弃用警告）。测试包含真实本机 WebSocket、真实图像预处理/同步 executor，但模型是替身，不能据此声称真实模型或物理 episode 恢复通过。

`p1-paired-inference-001` 已在独立 GPU 1 启动 E0/E3 实模型诊断，使用 foundation-matrix-v3 的 step 0、保存的非 final 模拟器观测；每组固定 33+32 个动作请求、三次 chunk 生成，源码快照、载入身份、配对 seed、重复差异与框架显存均落盘。它不训练、不控制物理环境，不产生策略成功率；正在运行不是验收通过。后续仍需真实模拟器 episode、账本集成与中断恢复、正式配方续训、N/最终协议冻结及四组训练评测。

### 实模型 WS 完成与场景接续

上述实模型诊断现已 completed、exit 0，全体进程退出并释放 GPU 1。每组 67 条消息（2 次 reset、65 次动作请求）、3 次真实 chunk 生成；同组 reset 前后首段 32 动作逐值一致，E0 与零门控 E3 的两段观测输入下动作也逐值一致。新 chunk 使用不同的 world/action seeds 且动作不同。独立 `p1-paired-inference-audit-001` completed，核验源快照、checkpoint 哈希、reset 身份、动作数组、响应及发送前日志，无缺失或重复；`reports/paired_inference_001.json` passed，SHA256 `3b971964890d4222b5c750538db33049773336e21945bffe50052a574c414466`。不据 step-zero 相等性推断训练后效果。

| 组别 | 载入秒 | 组内总秒 | 框架 peak allocated bytes | 框架 peak reserved bytes |
|---|---:|---:|---:|---:|
| E0 | 128.186 | 134.302 | 25,121,122,304 | 25,339,887,616 |
| E3 | 121.811 | 126.396 | 25,285,078,016 | 25,459,425,280 |

整个诊断 wall-clock 268.193 秒、分配成本 0.074498 GPU-hours；CPU 独立审计约 52.693 秒。载入/预热计入总成本，E0 首次生成 2.711 秒含冷启动，不能与 E3 热态生成直接当公平延迟比较；其余逐 chunk 原始延迟保留。仍有宿主/容器 PID 归属盲区，不报告精确进程 GPU-hours 或项目能耗。

`p1-expert-microwave-classifier-001` completed、exit 0、进程已释放，wall-clock 61.702 秒、0.017140 GPU-hours。该原失败 seed 在独立运行中专家成功、双 reset 精确一致，没有触发新拒绝分类；因此真实分类分支尚未被该探针验收，且专家规划可运行间变化。旧 pool-002 的断言保留为历史失败，不改写成成功；新池用新协议一次性按候选顺序筛选，不靠失败 seed 重试获得接受。

已启动 `p1-scenes-pool-003`，GPU 0 单租约、preparation v2/queue v2/pool v3。接管核验 supervisor、池父进程及四 lane 均存活；前四个开发分片已生成 COMPLETE，后续仍执行固定 40 分片、100 开发/400 final 场景计划。它只推进场景准备，错误即共享 STOP，不自动接续训练/策略评测。保持源快照不变；旧池已完成分片未混入新版本最终清单，重复计算成本单列。

新增 `r2_rollout.py` 将真实客户端、原始 EEF 控制链、固定 instruction、原 task step limit、精确初态核验、physics step 计数和 episode 账本接成单 episode 路径；视频由真实 ffmpeg 编码并通过 ffprobe 帧数/几何及完整解码验证。`p1-rollout-loop-tests-001` 34 passed（2 条第三方弃用警告），包含实际 H264 编解码及成功/失败去重；环境和模型仍是替身，不能当作真实 SAPIEN rollout 或中断恢复验收。正式入口和完整协议门槛仍待接续。

补充验收：`p1-r2-regression-001` completed，完整 `tests/statewam` 为 228 passed、1 skipped、2 条第三方弃用警告；跳过项是 CPU supervisor 隐藏 CUDA 后的确定性池化测试。单独 `p1-cuda-pooling-regression-001` 在 GPU 1 补验相关文件，21 passed、无跳过，进程已退出并释放 GPU；wall-clock 10.175 秒、0.002826 GPU-hours。服务新增真实空闲截止和阻塞模型调用期间 watchdog 持续响应测试，包含在完整回归中。

随后 pool-003 的 `open_microwave/demo_randomized/230001/attempt_001` 实际触发 `r2-expert-no-grasp-pose-v1`：`grasp_actor` 的两个 pose 与 `Action.__init__` 的 target_pose 均为 None，已将两个源码哈希、函数与行号写入拒绝证据；此候选只运行一次、记为 expert_unsuccessful，队列按原顺序进入 230002。这补齐真实分类分支的证据，不改变先前 230000 独立复核成功的事实。

接续工作：先完成并独立审计 pool-003 的 40 分片与跨进程初态配对，再运行真实服务/模拟器 episode 及 STOP/中断去重验证；完成正式 batch16/global64 的四卡续训对照，冻结有限 N 和协议后连续执行 E0–E3 训练与配对评测。公共微调文档及发布 Clean2Random 配置均提供 `num_epochs=5` 的参考，但任务数、采样和冻结语义与本 pilot 不同；它不是本轮 N 或已验证收敛结论，正式预算仍待统一冻结。当前唯一 GPU 长任务是单设备场景池，未启动跨阶段自动编排。

### 选定 Batch 配方恢复验收

后续调度先暂停场景池，为四卡恢复验收释放设备。pool-003 已收到协作式 STOP，现核验为 stopped、exit 0，supervisor/child 的 PID 与创建时间不再对应存活进程，后代为空，八张物理 GPU 均无 compute/graphics 进程。保留原 scenes 目录、源码身份、候选进度和已用尝试次数，后续须以新 supervisor run id 和 `--resume` 接管，不重建清单。此处 supersede 上段“当前唯一 GPU 长任务”的历史描述，不改写旧运行记录。

pool-003 本段 wall-clock 935.157 秒，分配成本 0.259766 GPU-hours，466 次采样；GPU 0 平均利用率 41.588%、p95 98%、峰值 100%，峰值整卡显存 16,213,737,472 bytes。399 次采样存在归属盲区，进程 GPU-hours 与项目能耗仍为未知，不能填零。完整清单未完成，不以主动停止当作整池通过。

`manifests/batch16_recovery_plan_001.json` 登记 E3、四卡 micro-batch 16/accumulation 1/global 64、workers 4、seed 20260913、总 4 updates、prefix 2 updates、每 2 updates 保存。它是最终选定 batch 配方的工程恢复验收，不是正式训练 N。新增独立 prefix auditor 在恢复前核验四 rank 的样本/噪声/loss/梯度、源快照实际内容、完整 checkpoint 和 latest 指针；`p2-b16-prefix-tests-001` completed，24 passed，覆盖篡改快照、采样分叉、未退出任务、非零退出、缺失轨迹和 checkpoint 损坏等拒绝路径。

`p2-b16-resume-reference-001` 已在 GPU 0–3 独立启动，随后须在其完成并释放资源后执行 prefix，前缀审计通过后再执行 resumed 和完整状态审计。三段训练期间冻结 `openwam/`、`scripts/`、`configs/` 快照范围；只可推进不在该快照内且不改变场景池登记来源的独立 benchmark 工程。当前无跨阶段自动编排，不能把参照 run 已启动写成恢复通过；正式 P3/P4 仍未开始。

参照段随后 completed、exit 0，进程与 GPU 均已释放：4 updates、256 次逻辑曝光，wall-clock 632.558 秒，0.702842 GPU-hours；第 2 步完整保存耗时 172.617 秒。已按同一计划接续 `p2-b16-resume-prefix-001`（GPU 0–3），它仅执行前 2 步，尚待前缀审计和 resumed 完整状态对照。短工程 run 的整卡利用率包含模型加载、保存和分布式同步等待，不用其峰值或轮询利用率冒充有效训练吞吐。

### 真实 Episode 入口准备

新增 `benchmarks/robotwin/r2_episode_engineering.py`：一个外层两设备租约分配 GPU 0 给模拟器、另一张卡给策略服务；固定模型/统计/场景/资产/环境哈希，独立子进程，原始 action limit，加有限启动、wall-clock、空闲和消息预算，完成或失败后收尾。使用独立 runtime 和原控制链，不引入正式训练权重、不使用 final 场景、不自动重试。`physical_episode_plan_001.json` 登记 E0 step 0、adjust_bottle clean seed 200001、原上限 400 actions；计划 SHA256 `966ae99bf2dbe42b6e0f7a097c9159b065158368bdb2c1e469718e686d59fce3`。`p1-physical-plan-audit-001` CPU 输入核验完成，尚未启动真实物理 episode。

新增独立 `r2_audit_episode.py`，核验完成身份、初态、逐动作服务响应/seed、EEF 转换、chunk 哈希、physics/control 计数及真实视频解码和非空画面。入口/控制/账本测试 `p1-episode-entry-tests-001` 31 passed；增加进程生命周期与服务回归后 `p1-episode-entry-tests-002` 33 passed；独立动作审计及拒绝路径 `p1-episode-audit-tests-001` 42 passed。模型/环境在这些单元测试中仍是替身，不替代真实 rollout 或中断验收。

`p1-scene-partial-audit-001` CPU 独立核验 pool-003 已完成的 6 个分片、30 个接受场景，`reports/scene_partial_audit_001.json` passed；所有候选/初态/指令/源码身份一致。此报告仅覆盖已完成分片，完整 40 分片、跨进程 reset 和最终协议仍待完成，已停止的池不因局部审计自动恢复。

完整回归 `p1-r2-regression-002` completed：267 passed、1 skipped、2 条第三方弃用警告，33.98 秒。跳过项仍为 CPU supervisor 下的 CUDA 确定性池化测试，该模块本轮未修改，此前 GPU 补验保留。

`p2-b16-resume-prefix-001` completed、exit 0、全体进程及四卡已释放，执行 2 updates/128 次逻辑曝光；wall-clock 366.419 秒、0.407133 GPU-hours。`p2-b16-prefix-audit-001` CPU 审计 completed，四 rank 前两步的样本/增强/噪声/loss/梯度全部精确一致，源码快照实际内容、prefix 完整 checkpoint 与 latest 指针通过；报告 `reports/batch16_prefix_audit_001.json` SHA256 `03c5e91503a87a9c243515531a5d856201a2e52cdee3b5759687dd3bd39ec912`。恢复段及最终模型/优化器/RNG 对照仍待执行，不能据前缀通过宣称完整恢复通过。

在前缀 CPU 审计期间启动 `p1-physical-episode-001`，GPU 0/1 的同一两设备租约分别分配模拟器/策略服务。它按已登记计划只执行一个原 400-action 上限的真实工程 episode，不自动重试；实际启动顺序调整为 prefix 审计期间执行物理工程、其完成释放后再启动四卡 resumed，避免五卡占用。当前尚在服务加载，未取得策略终态。该 run 快照含所有 benchmark Python 文件，结束前也不得修改这些来源；场景池仍保持暂停。

### 有限正式预算候选

登记 `manifests/training_budget_candidate_001.json`，尚不允许据此启动正式训练。E0 profiling 的四段 50-update 平均 action loss 为 0.040715、0.028999、0.024094、0.022335，仍在下降；这证明早期训练有效推进，不证明收敛或策略能力。公共微调文档与发布 Clean2Random 配置均采用 5 epochs，前者示例为八卡/batch24、后者配置 batch4，不能冒称本轮四卡/冻结 world/十任务等同官方配方。

据此选择可解释且有限的共同预算候选：`ceil(103010*5/64)=8048` updates，每组 515072 次逻辑曝光，四组从相同 Foundation 重新开始，train seed 42，任务均匀有放回采样并配对序列，最终 checkpoint 为共同主结果。每 500 步及最终步保存完整状态（17 个/组），若在完成标记发布前中断，最多 500 个更新可能重算；只在最终步做一轮开发评测（400 episodes），另做重复性 40 次和最终 1600 次，不新增校准 run 或逐组早停。

估算采用已测 E0 14.761721、E3 14.565311 global samples/s，E1/E2 暂以 E3 吞吐估算而非冒充实测；每次完整保存按 175 秒、每组启动按 300 秒规划。四组串行训练合计约 42.800 小时 wall-clock、171.201 分配 GPU-hours；每组 17 份完整检查点，四组合计约 4.122 TB 逻辑存储，另需导出权重、视频与日志。此为训练部分估算，明确排除场景准备、评测、工程/审计、排队、导出与故障重算；评测吞吐仍待足够的真实多任务证据，不用一个任务外推完整 Goal 时长。最终预算/协议门槛未过，N 候选不等于正式协议已冻结。

### 首个真实物理 Episode 与收尾诊断

`p1-physical-episode-001` 已退出并释放 GPU 0/1，外层为 failed、exit 1，不改记 completed。真实循环已达到原始 400-action 上限，账本 `episodes/f53047cf8c4202c03b6326da9bd615ce97efc5dc1e8d65c626d7077d588c1fba/attempt_001/result.json` 已完整记录 failure/action_limit；400 个动作、51688 个 post-setup physics steps、原 timestep 0.004000000189989805 秒，循环及 setup/video 收尾合计 308.773 秒。视频 H264、320×240、400 帧；实际解码 contact sheet 为 `reports/physical_episode_001_contact.png`，已目视核验非空且机械臂运动可见。服务正常停止，共 401 条消息，加载 126.270 秒；整个工程 run 700.932 秒、0.389406 GPU-hours，247 次采样有 PID 归属盲区。

非零退出发生在 episode 发布后的第二次环境复核：`pkg_resources` 导入将 setuptools `_vendor` 加入 `sys.path`，进程内 `importlib.metadata.distributions()` 枚举出现额外/重复包名，字典后写值覆盖原版本。纯 CPU `p1-provenance-import-diagnostic-001` 已复现：初始枚举与冻结报告一致，单次 import 后新增 6 个包名并覆盖 6 个版本，未安装或升级任何包。证据见 `reports/provenance_import_diagnostic_001.json`。入口改为在同一模拟器解释器、固定 cwd 的干净 CPU 子进程中做前/后 provenance 核验，不修改原场景池登记的七个来源；相关 `p1-episode-entry-tests-003` 31 passed。真实修复后的整条生命周期仍待验收，不把旧失败擦除。

独立动作审计首轮 `p1-physical-recorded-actions-audit-001` 用了训练环境 NumPy 2.2.6，EEF 转换逐值检查失败；保持原转换代码和精确门槛，用执行控制的模拟器 NumPy 1.26.4 重验 `p1-physical-recorded-actions-audit-002` completed。`reports/physical_recorded_actions_002.json` 证明保存的 400 动作/服务响应/转换/chunk/初态一致，result SHA256 `79ae657ba0ffcf25c3ba0d2185b6068d37ff27e3471e3f6fb8f37b211cb1faad`；它只审计已记录数据，不替代外层成功退出或真实中断恢复。审计入口新增明确的模拟器 Python/NumPy 身份门槛，未放宽数值容差。环境子进程诊断首轮漏传 `ROBOTWIN_PATH`，已保留该失败并用明确环境重验；不是修改模拟器依赖。

在物理 run 释放后已启动 `p2-b16-resume-resumed-001`（GPU 0–3），从已通过前缀门槛的 update 2 完整状态执行剩余两步，保持原训练快照和四卡预算；完整恢复审计尚待其退出后执行。当前场景池仍暂停，不重复已完成 profiling。

### 当前接管点

resumed 随后 completed、exit 0、全体进程与四卡释放，完成 update 3/4；wall-clock 616.613 秒、0.685125 GPU-hours。三段恢复工程合计 1.795100 GPU-hours，不计入正式 N。`p2-b16-resume-audit-001` 已在 CPU 独立启动，目标为 `reports/batch16_resume_audit_001.json`；须等完整模型/优化器/RNG 比较及任务退出后再宣布恢复通过，不能仅凭末步 loss 相同推断。

`p1-provenance-subprocess-check-002` 通过包身份比较后，在背景资产哈希阶段达到 300 秒超时退出；原失败保留。为覆盖实际文件校验耗时，下一次明确诊断采用有限 900 秒，不改变物理步数、成功条件或 episode 重试额度；`p1-provenance-subprocess-check-003` 已启动 CPU-only 复核，尚待结果。精确转换环境门槛回归 `p1-episode-audit-tests-002` 32 passed；900 秒超时调整后还须复核相应测试。未来真实入口须协调服务空闲期限与资产预检耗时，不能让文件哈希消耗完服务空闲预算。

四卡恢复释放后已启动 `p1-scenes-pool-004`（GPU 0 单租约），以 unchanged preparation v2/queue v2/pool v3 和 `--resume` 接管 `runs/p1-scenes-pool-003/scenes/`。核验 supervisor、池父进程及四 lane 存活；继续固定 40 分片，不重置已用候选尝试。任务结果与采样在新 run，场景进度仍在旧 scenes 目录。当前 GPU 工作只有此单设备池，没有跨训练/评测阶段的自动编排。

接续优先项：核验两项 CPU 审计终态；完成原池全清单及跨进程配对；在保留首个已完成策略失败的前提下验证修复后的生命周期、真实 STOP/强制中断与有限重试/去重；然后补齐正式入口、统一冻结预算候选与协议，连续执行 E0–E3 训练和既定评测。首个已完成 action-limit failure 不得重跑求成功，后续去重验收应验证跳过该条结果。

900 秒文件校验超时与精确转换环境门槛调整后的 `p1-episode-entry-tests-004` completed，27 passed，0.59 秒；`git diff --check` 通过。最新核验 pool-004 的 supervisor/child/四 lane、完整恢复 CPU 审计和环境 CPU 复核均存活，尚未取得两项审计最终报告。以上为交接时观测，不替代下次实际 PID/创建时间、心跳、GPU 和产物核验。

### Batch16 完整恢复通过与正式入口准备

接续后实际核验 `p2-b16-resume-audit-001` completed、exit 0，supervisor/child 均已退出。`reports/batch16_resume_audit_001.json` passed，SHA256 `432bc7253deb360975dbcd9307256aa2aac593eea21cd29a972741fd29714515`；模型状态、四个 ZeRO-2 优化器分片及四 rank RNG 保存状态均逐值一致，身份覆盖 E3、micro16/global64、accumulation1、四卡。独立 CPU 审计 wall-clock 643.515 秒，进程树 RSS 峰值 112,017,776,640 bytes（共享页可能重复计算）。这是所选 batch 配方的短程恢复通过，不是正式 8048 步训练完成。

`p1-provenance-subprocess-check-003` 也已 completed、exit 0、进程退出，wall-clock 308.486 秒。干净同解释器子进程生成的 `environment_after.json` SHA256 为 `32d4e7694d109739dc19bf1a6276f1ddf4827811026fa0962d47a28daa64f74e`，绑定原资产及模拟器环境报告；没有改动依赖或放宽身份检查。此前 300 秒超时和首个真实 episode 后置失败继续保留。

新增 `formal_training.py` 的冻结协议门槛，固定 E0–E3、8048 updates 候选、共同 batch/优化器/推理及评测数量；绑定 P0 数据/统计、矩阵、恢复和其他必需审计，并覆盖训练及 benchmark 源码清单。未生成正式冻结协议，也不允许在缺少真实恢复或场景门槛时启动。共享训练循环的内存计时缓存改为只保存汇总所需字段，完整样本/噪声/梯度证据仍逐步落盘，不改优化计算。`p2-formal-precheck-tests-001` 23 passed；`p2-formal-entry-tests-001` 38 passed，含预算/公平性/数据身份/门槛拒绝和计时汇总等价测试，各有 2 条第三方弃用警告。

真实 episode 工程入口升级为 v2：在模型服务启动前完成 CPU 资产/环境预检，预检外层限时 1200 秒、单次校验 900 秒；物理控制完成后单独通知服务退出，再执行后置环境检查。旧 v1 计划保留审计能力但禁止重新 launch。`p1-episode-lifecycle-tests-001` 44 passed；补充预检失败不启动服务与旧计划拒绝后，`p1-r2-regression-003` 全套 309 passed、1 个 CUDA 项在 CPU run 跳过、2 条第三方弃用警告，27.75 秒。此处只证明单元/回归路径，不声明修复后的真实生命周期或故障恢复通过。

登记 `physical_episode_plan_002.json`，选择已接受开发场景 adjust_bottle clean seed 200002、E0 step 0、原 400-action 上限，无自动重试；`p1-physical-plan-audit-002` CPU 输入身份检查 completed、exit 0。它是预选的独立修复验收，不重复 seed 200001 的已完成策略失败，不计入正式策略分母。GPU 执行尚待调度。

新增 `r2_cross_process_reset.py`：完整场景清单验收后，固定选择每 task/condition 首个开发场景，在 20 个独立进程中逐一比对保存的初态与三视角像素，原环境/资产前后复核；每项 600 秒、无自动重试。`p1-cross-process-reset-tests-001` 37 passed，1.08 秒，覆盖固定选择和篡改/缺失/活进程/非零退出拒绝；尚未执行真实跨进程 20 项核验，不能据单测冻结协议。pool-004 持续准备场景，最近观察有 19 个分片完成标记，完整 40 分片收集与独立审计仍待完成。

### 开发场景验收、部署导出与生命周期接续

为两卡真实 episode 验收，pool-004 后续协作式停止，状态 stopped、exit 0，supervisor/child 与后代均已退出，实际八张设备的 compute/graphics 进程为空。保留原 scenes 目录与已用候选尝试；此次 wall-clock 1885.747 秒、0.523819 分配 GPU-hours、943 次采样。`p1-scene-partial-audit-002` completed，独立核验 20 个开发分片、100 个接受场景，报告 SHA256 `e8f46cf77aa0beb58538bceae2c6ab57906ba137b2a5e72389ddaa7858edde60`；400 个最终场景仍未全部准备完成，局部审计不是完整场景协议冻结。

完整回归 `p1-r2-regression-004` 326 passed、1 skipped、2 条第三方弃用警告，31.27 秒；GPU 2 补验 `p1-cuda-pooling-regression-002` 21 passed、无跳过，已退出释放，10.241 秒、0.002845 GPU-hours。`p2-formal-known-gates-001` 用真实 P0/容量/profiling/batch16 恢复/训练故障报告检查正式入口，已有训练门槛被接受，在缺少真实物理恢复门槛时正确拒绝；`reports/formal_known_gates_001.json` 不是完整协议通过报告。

新增 `export_checkpoint.py`，从已校验全部分片、已退出的本地完整 checkpoint 导出共同最终更新的部署权重，严格核对原始张量键/形状、bf16/有限值、配置、train-only 统计及矩阵身份；导出后逐张量比较并发布完整标记。部署权重不含优化器，不能用于冒充完整续训状态。`p3-export-tests-001` 16 passed；真实布局检查 `p3-export-layout-audit-001` completed，确认 E3 模型状态含 2419 个 bf16 张量。

`p3-engineering-export-001` completed、exit 0，已退出；从已有 E3 batch16 resumed 的 update 4 导出到 `exports/engineering-e3-b16-update4-001/`，2419 个张量逐值一致。权重 SHA256 `81178ff37185d242ac235d43bab72e06d20ba7794f5e26711ad97f060a79fb06`；`export.json` SHA256 `8a1842a7f62119c0c2a0324c683ee7cdf75618006f59c97247e1e357dcb523af`。CPU wall-clock 317.154 秒，进程树 RSS 峰值 50,418,151,424 bytes；只是工程导出验收，不是 P3 正式训练结果。GPU 2 的 `p3-export-service-probe-001` 已独立启动，以保存的非 final 观测做导出后真实模型的 67-message/3-chunk WS reset 诊断，尚待实际结果。

`p1-physical-episode-002` 已在同一 GPU 0/1 租约启动，计划 SHA256 `c9b37a667e0b7cc55848a372f88b17826aba457b665d3a073b18757a670512d7`。最近核验 supervisor、协调器与 CPU 预检子进程存活，处于资产预检；模型服务尚未启动，不视为真实 episode 或生命周期通过。单任务按有限计划独立运行，源码快照包括 `openwam/`、`scripts/`、`configs/` 及两处 benchmark 目录，运行期间冻结这些来源。与导出服务诊断同时最多租用 3 张卡；场景池保持暂停，不自动接续后续训练或场景准备。

随后 episode-002 的预检完成，模型服务及模拟器子进程启动，真实动作进度已推进。`p1-completed-failure-dedup-001` completed、exit 0：用 episode-001 的真实账本连续两次调用恢复入口，均返回已完成，attempt 数仍为 1，全部已发布产物哈希不变。`reports/completed_failure_dedup_001.json` SHA256 `4c7f738eedbb385c3274309a3033387342efd3ba83c93c2e106f44a7e501442d`。此项证明真实已完成策略失败去重，不替代物理 STOP、强制中断及同场景有限重试验收。

`p3-export-service-probe-001` 外层终态 failed、主进程 exit 0，failure 为 `Workload leader exited with live descendants`，现全部进程及 GPU 2 已释放。模型诊断内部完成 67 消息、3 chunks，导出权重动作有限、reset 后逐值一致且下一 chunk 噪声响应不同；加载 125.928 秒、模型诊断总计 132.147 秒，peak allocated/reserved 为 25,285,078,016 / 25,459,425,280 bytes。整 run 172.527 秒、0.047924 GPU-hours。内部 report 的 passed 不能覆盖外层生命周期失败。

退出采样显示前一周期存在 CPU 子进程，最后周期仅剩 supervisor 与 RSS 为 0 的工作主进程。源码检查发现 supervisor 在 `child.poll()` 前缓存 descendants，poll 之后用旧集合判定残留，存在正常退出被误判的竞态；真实残留与这一竞态需用可控测试区分，不能凭当前采样直接改写失败。episode-002 仍使用该源码快照，须待其退出后再修复和验收，不在运行中修改 supervisor。

### 完整物理生命周期通过与退出竞态修复

`p1-physical-episode-002` 随后 completed、exit 0，supervisor/child/后代均已退出，实际 compute/graphics 设备上下文释放。完成原 400 actions、33392 个 post-setup physics steps，策略 outcome 为 failure/action_limit，循环/setup/video 合计 244.038 秒；整 run 785.327 秒、0.436293 分配 GPU-hours，204 次采样有 PID 归属盲区。策略服务 401 messages、正常停止，前后环境报告一致。`reports/physical_episode_002_contact.png` 已生成并目视检查，画面非空、目标物体与控制过程可见，不据视频推断成功。

使用实际模拟器 Python/NumPy 的独立 `p1-physical-audit-002` completed、exit 0；`reports/physical_episode_002.json` passed，SHA256 `8be1adea8c493fdd9769e4ba572c363ce4d76b0526578f0b61a27820138d6701`。审计覆盖源码快照、原上限、初态、全部服务动作/分块/EEF 转换、视频解码、前后 provenance 与完整生命周期；这是修复后完整单 episode 工程通过，不是物理 STOP/强制中断恢复通过，也不是正式策略评测。

`p1-supervisor-exit-race-001` CPU 可控复现 completed：子进程正常退出后，poll 前缓存的列表仍非空，而重新读取 descendants 为空；诊断报告 SHA256 `573d1ca481d89bb4554dd1be135160f5a7f1af3f38f1579761f520bc65ccc7b3`。待 episode-002 完全退出后才修改 `gpu_job.py`：poll 得到终态时重新查询并按 PID/创建时间判断存活后代，另记录 `leader_exit_observation`；真正仍存活的后代或非零退出继续判失败，不放宽退出门槛。

`p1-supervisor-exit-tests-001` 38 passed，覆盖旧列表、真实后代和非零退出；`p1-supervisor-exit-gpu-001` 实际 CUDA 生命周期 completed、exit 0、全体进程与 GPU 2 释放，退出时 fresh descendants 为空，8.606 秒、0.002390 GPU-hours。旧导出服务 run-001 的 failed 不回写；已登记并启动修复后 `p3-export-service-probe-002`（GPU 1）复核同一工程权重及相同有限诊断，结果仍待核验。

场景池用 `p1-scenes-pool-005`（GPU 0 单租约）接管原 `runs/p1-scenes-pool-003/scenes/`，七个场景来源及候选顺序不变，保留已用尝试次数；核验 supervisor/child/四 lane 存活，处于开始时资产身份复核。与导出服务复核合计最多两卡。后续仍需全 40 分片/500 场景、20 task/condition 跨进程初态核验、真实 STOP/强制中断/有限重试、正式协议冻结与四组训练/配对评测；不能把本轮工程进展标为完整 pilot 完成。

导出服务复核 `p3-export-service-probe-002` 随后 completed、exit 0、无 failure，supervisor/child/后代已退出，GPU 1 释放；`leader_exit_observation.live_descendants=[]`。内部真实模型检查通过：67 messages、3 chunks、动作有限、reset 后首 chunk 精确重现且下一个 chunk 不同。装载 128.752 秒、模型诊断 135.092 秒，peak allocated/reserved 与首轮相同；完整 run 169.239 秒、0.047011 GPU-hours、8 次 PID 归属盲区采样。证据为该 run 的 `inference/report.json`、原始请求/响应/动作与源码快照；没有将首轮 failed 改写为通过。此时只有 pool-005 为 GPU 长任务，正式训练/评测仍未启动。

接续时先核验 pool-005 的 PID/创建时间、心跳、各 lane 进度与物理设备；不要重复已完成的 profiling、batch16 恢复或完整 episode。场景池登记的七个 benchmark 来源不得改动，其余训练/部署源码不再被已结束的工程 run 锁定。下一工程优先项是为两个预选开发场景登记真实 STOP 与强制中断，分别保留原 attempt、核验退出后仅同场景重试一次，再将这些证据与已通过的完整物理生命周期/失败去重合并验收；不得改动正式 N、成功条件或 final 分母。

### 真实中断恢复计划与启动

登记 `manifests/physical_recovery_plan_001.json`，SHA256 `ce4a86cbb3aef6bb2fb3d0ca055211370fa215b578f532ed8756cb3ef51dac1e`：E0 step 0、两个预选 adjust_bottle clean 开发场景，seed 200003 为协作式 STOP、200005 为模拟器 SIGKILL；在第 8 个已落盘动作后、最迟第 32 个动作触发。每场景最多初次尝试加一次同计划重试，共最多 4 attempts，不运行 final 场景、不改原 400-action 上限。独立协调器触发故障，但不会自动重试；核验终态/资源与 invalid 证据后由执行者接续。

工程入口 v3 按实际模拟器 PID/创建时间及 attempt 归属校验目标，再持久化故障请求和已执行证据；恢复时不再次注入已登记故障。协作式 STOP 的物理/视频/账本先正常收尾，明确记录因外部停止未执行后置资产复核，不以此声明完整 provenance；同场景重试仍须完成前后全部检查。发生 STOP 同时的网络断开记录为带 STOP 证据的 external_interrupt，不把任意策略错误归为 invalid。

新增 `r2_episode_recovery.py`，先验证首次退出方式、子进程和物理设备释放，再处理强制中断留下的 partial；没有已执行故障证据或目标仍存活时拒绝分类。恢复审计验证恰好两次尝试、同一身份/源码/初态、首个动作和噪声一致、最终完整 episode 与去重；后续物理轨迹残差逐项记录，不用残差筛选场景或扩充重试。汇总入口仍须重新核验真实原始证据后才发布 physical recovery gate。

`p1-physical-fault-tests-001` 59 passed，包括真实测试子进程 SIGKILL，但该单测不是真实模拟器故障验收。`p1-r2-regression-005` 全套 347 passed、1 CUDA 项在 CPU run 跳过、2 条第三方弃用警告，31.02 秒。`p1-physical-fault-plans-001` completed、exit 0，两份计划的来源、权重与场景身份核验通过；STOP/KILL 计划 SHA256 分别为 `2a07fdfa59c0f716d6a43cf0fb32898449e24171dd17f81866b13ad6ee809f65`、`3914f073e62ab701278178f572dba8e7dbbc4d50c3aa8a2042de58890dee511b`。

pool-005 已协作式停止，stopped/exit 0，全体进程与 GPU 释放；此前推进至 21 个分片完成标记，完整清单仍待核验。该段 wall-clock 1049.815 秒、0.291615 GPU-hours、513 次采样，其中 433 次有归属盲区。保留原场景目录及剩余候选，不重建已完成场景。

随后 `p1-physical-stop-001` 已在 GPU 0/1 独立启动，最新核验 supervisor、协调器及预检子进程存活，正在执行启动前资产复核。STOP/同场景重试、SIGKILL/同场景重试都还没有通过结论；四个工程运行期间冻结训练/benchmark 源码快照。当前场景池暂停，正式训练/评测未开始。

### STOP 首次退出与同场景接管

`p1-physical-stop-001` 已 stopped、exit 0、无 failure，全体子进程 exit 0、后代为空，GPU 0/1 释放证据已保存。协调器在第 8 个持久动作后执行登记 STOP；首次 attempt 收尾为 invalid/external_interrupt，记录 9 个策略动作、10 帧视频。外部停止明确跳过后置环境复核，不能据此声明完整 provenance。整个 run 420.779 秒、0.233766 分配 GPU-hours，199 次采样中 40 次存在进程归属盲区。

独立 `p1-physical-stop-reconcile-001` 核验已完成，`reports/physical_stop_reconcile_001.json` passed，SHA256 `d1b3cc389f51245563e672bac6e8166f20f031e79fe5c4fb9d9fef62fc7dd8a6`；首次 invalid result SHA256 `9c34cbd78254373814e178d4858a3ccb0a1ddceae7d5ffdf35973cd3e35cd88d`。这是首次故障分类与退出核验，不是恢复配对通过。

此前已启动的唯一同场景重试 `p1-physical-stop-resumed-001` 在本次接续时经 PID/创建时间核验仍存活，已完成预检并启动模型服务与模拟器。使用原计划、GPU 0/1、原 `p1-physical-stop-001/episodes/` 账本的 attempt_002，不再次注入故障；不能重新启动该 run 或增加第三次尝试。源码继续冻结，完成并核验释放后才执行配对审计。场景池保持暂停，SIGKILL 首次与重试尚未启动；没有跨阶段自动编排，正式训练/评测仍未开始。

重试随后 completed、exit 0、无 failure，全部进程与 GPU 释放；原 400-action 上限下结果为 failure/action_limit，49639 个 post-setup physics steps、400 帧视频，循环/setup/video 270.479 秒。整个重试 run 779.321 秒、0.432956 分配 GPU-hours，383 次采样中 220 次存在归属盲区；前后环境检查一致。

`p1-physical-stop-pair-audit-001` 使用模拟器 Python/NumPy 独立审计 completed、exit 0，54.613 秒；`reports/physical_stop_pair_001.json` passed，SHA256 `7859e184085ebf7c66e3407c57c9cdc9f765fd51c4134f0f162132414276f79e`。核验恰好两次尝试、相同源码快照/场景/初态/噪声、首个动作一致，前八个动作的 EEF、转换、观测/proprio 与 physics steps 均精确一致；完整重试视频、服务响应、前后来源与失败去重通过。这只证明登记 STOP 场景的恢复，不外推全轨迹或全任务确定性，不算正式策略成绩。

确认八张物理设备均无 compute/graphics 进程后，已在 GPU 0/1 启动登记的 `p1-physical-kill-001`。启动后核验 supervisor/协调器 PID 与创建时间仍存活；它只在预选 seed 200005 触发一次模拟器 SIGKILL，尚未故障核对或同场景重试。源码仍冻结，场景池暂停，不自动推进后续阶段。

### SIGKILL 核对与唯一重试

`p1-physical-kill-001` 已按登记故障退出：第 8 个持久动作、642 个 physics steps 后模拟器 exit -9，预检与策略服务均 exit 0，协调器与外层保留 failed/exit 1；supervisor/child/后代均已退出，GPU 0/1 释放。整个 run 425.167 秒、0.236204 分配 GPU-hours，213 次采样中 45 次存在归属盲区。首次视频和动作等 partial 证据保留，不冒充完整策略 episode。

`p1-physical-kill-reconcile-001` completed、exit 0，进程已退出，CPU wall-clock 28.855 秒。核验已登记故障的实际目标 PID/创建时间、执行证据、退出方式和资源释放后，将原 partial 登记为 invalid/external_interrupt；`reports/physical_kill_reconcile_001.json` passed，SHA256 `4865a39d2e0f2c37660f428dc3b982b53f5d7b64adec4004c9deac8bb7d1cd90`，invalid result SHA256 `5602c0d8f4f2f2255161b950ccca351afad08e5000e6a9541c3fce81b4cfeaaa`。分类完成不等于同场景恢复通过。

核验当前源码与首次运行快照一致、GPU 0/1 空闲后，已启动唯一重试 `p1-physical-kill-resumed-001`，仍使用 `physical_kill_plan_001.json`、原 `p1-physical-kill-001/episodes/` 账本和原场景/seed，不再次注入故障。启动后已核验 supervisor/child 的实际身份存活；GPU 4–7 此时出现其他进程，未租用或操作这些设备。当前任务可独立完成其有限 episode 与收尾，但没有后续自动编排。

下一接管点：先实际核验该重试的进程、进度与资源；完整退出并释放后，使用模拟器 Python 执行独立 SIGKILL 配对审计，再与 STOP 配对、`physical_episode_002.json`、`completed_failure_dedup_001.json` 汇总真实物理恢复门槛。此后才接续原场景池的剩余清单、20 项跨进程初态核验、正式协议冻结与四组训练评测。不得新增第三次 attempt，不重复已完成 profiling、batch16 恢复或既有完整策略失败；源码在本恢复对完成前继续冻结。

### 两种物理恢复配对通过

`p1-physical-kill-resumed-001` 随后 completed、exit 0、无 failure，全体进程与 GPU 0/1 释放，前后环境检查一致；原 400 actions 完整执行，结果 failure/action_limit，40534 个 physics steps、400 帧视频，循环/setup/video 317.660 秒。整 run 840.727 秒、0.467070 分配 GPU-hours，420 次采样中 249 次有归属盲区。STOP 和 SIGKILL 重试 contact sheet 均已生成并目视检查，分别为 `reports/physical_stop_retry_001_contact.png`、`reports/physical_kill_retry_001_contact.png`；画面非空且机械臂运动可见，不据此改写失败结果。

`p1-physical-kill-pair-audit-001` completed、exit 0、进程退出，CPU wall-clock 57.069 秒；`reports/physical_kill_pair_001.json` passed，SHA256 `c0ead504690401eebe7727c1a2aea0547a002fbbbd6672d289f7d3181dd79911`。与 STOP 对一样，源码/初态/噪声及前八个动作的 EEF、转换、观测/proprio、physics steps 均精确一致，完整重试与已完成失败去重通过。四次登记尝试合计 wall-clock 2465.994 秒、1.369996 分配 GPU-hours；另有分类、配对和汇总 CPU 审计成本，不能混同为 GPU 计算。

已启动独立 `p1-physical-recovery-bundle-001`，重新核验两组配对的原始证据及完整生命周期/真实失败去重，目标 `reports/physical_recovery_001.json`；启动本身不构成汇总通过。四个物理 run 均已退出，不再锁定当前可执行源码；历史快照和输入证据继续保留。

确认 GPU 0 空闲、pool-005 实际进程退出、原七个来源哈希不变后，`p1-scenes-pool-006` 已用单 GPU 0 租约、四 lane 接管 `runs/p1-scenes-pool-003/scenes/`，启动后 supervisor/child/四 lane 身份均已核验存活。这里只继续原 40 分片清单，不重建已完成分片、不重置候选/尝试，不自动启动后续训练/评测。场景池登记的七个来源仍不得修改；其余后续入口可在独立测试后推进，正式源码与协议尚未冻结。

汇总审计随后 completed、exit 0，进程退出，CPU wall-clock 169.368 秒；`reports/physical_recovery_001.json` passed，SHA256 `21817143eb09513dad7ac3600839abfe5a69ae97f4b8618927c64e600dae2658`。原动作上限、协作 STOP、强制中断、同场景唯一重试、完整策略失败去重、cache reset、视频解码及资源释放八项均通过；环境身份仍为 `f1e9fa27d036555a16dfbd29f7705069bf778850c248f409c5dfb2eac9dc4696`。不外推为客户端 interrupt 按钮测试、全任务重复性或正式策略效果。

接续优先项为保持 pool-006 源码身份并完成全清单，收集 500 个场景后执行固定 20 项独立进程 reset 与审计；同时补齐公共正式评测、分析和协议冻结入口。当前四组正式训练和 2040 个既定开发/重复性/最终 rollout 均未开始，不因物理恢复工程通过标记 pilot 完成。

### 正式入口、完整链审计与资源总账

本轮先核验 pool-006 supervisor/child 的 PID、创建时间和心跳，确认继续运行；场景池登记的七个源码哈希保持一致，观察到 30 个 COMPLETE 标记，但完整 40 分片及 500 场景清单尚未独立验收。只占用物理 GPU 0，未重复 profiling、batch16 恢复或任何物理故障 attempt。

新增 `evaluation_plan.py` 固定 40 次重复性、400 次开发、1600 次最终 rollout，按重复性、开发、最终顺序执行，90 个 wave / 180 个模型服务 block。候选布局为两个 simulator lane 共享 GPU 0、两个独立 policy GPU，单个三卡外层租约；按 task/condition 轮换组对与设备槽，每 wave 有完成屏障。`r2_pilot_evaluation.py` 每 block 复用模型、每 episode 新建模拟器进程，保持原账本和有限重试，无自动重试；新增 `r2_audit_pilot.py` 核验来源、服务身份/请求、逐 episode 动作、视频和生命周期。此新多 block 三卡路径尚未真实运行，不能据单元测试宣布正式部署通过。

`paired_statistics.py` 要求完整且独立审计的 2040 条结果，clean/randomized 及开发/最终分开，预设四项差值均报告；按任务/scene 保持组间配对，10000 次 percentile bootstrap、seed 20260912，记录退化区间，不做显著性检验。`freeze_protocol.py` 只有全部证据门槛通过才原子发布，拒绝覆盖；新增有明确代理局限的评测估算，使用三次完整 E0 step-zero adjust_bottle clean 工程服务计时、各场景原动作上限和固定 wave 数。共享模拟器及其他任务/模型的减速因子 1/1.5/2 是规划敏感性假设，不是实测或置信区间；超时包络另列。全场景报告尚缺，因此还未实际冻结或输出完整预算预测。

`audit_formal_training.py` 核验四 rank 完整有序 invocation 链、相同协议与源码、全部登记保存节点、完整 checkpoint 分片及最终指针。修复审计缺口：必须有 loss，四 rank 恢复边界和 checkpoint 必须相同，checkpoint 必须位于该 invocation 已记录进度内；不能接受重复 run、缺前缀或超出原预算的记录。未保存后缀在逻辑序列中丢弃重放，其已记录实际曝光单列重算成本。正式导出验证拒绝工程/profiling 权重、错误 N/batch/worker/ZeRO/协议文件及损坏文件。尚无正式训练链，不把 fixture 测试当作 8048-update 结果。

已核验 completed/exit 0 且进程退出的 CPU 测试：`p4-formal-training-audit-tests-001` 13 passed；`p4-training-chain-tests-001` 22 passed；`p4-pilot-ledger-audit-tests-001` 31 passed，包含真实 ffmpeg 视频与账本；`p2-evaluation-estimate-tests-001` 34 passed；`p2-resource-union-tests-001` 17 passed；`p4-formal-export-gate-tests-001` 26 passed。完整回归 `p1-r2-regression-007` 为 403 passed、1 CUDA 项在 CPU 环境跳过、2 条第三方警告，57.76 秒；其后增加的导出检查由上述专项覆盖，不能把回归发生时间后移。更早调度/统计、入口/生命周期测试日志继续保留。

独立 `p4-existing-physical-ledger-audit-001` 使用原模拟器 Python，对既有 episode-002 的完整物理证据运行新 pilot 逐 episode 审计，completed/exit 0、进程退出，CPU wall-clock 30.589 秒。报告 `reports/pilot_existing_physical_ledger_001.json` SHA256 `0ff5341fb2ee92974d65cdf8bc7c614692d824c76103abb740e39e9347d90575`；动作转换、服务请求、原上限、初态、视频与失败去重通过。这是新审计器读取旧工程证据，不是新策略 rollout 或正式 pilot 验收。

`resource_accounting.py` 对各 run 的物理 GPU UUID/时间区间做并集，包含启动、保存、服务等待，避免共享设备重复计时；已知 spill 按采样区间另计，归属盲区及未知尾段不填零。独立 `p2-resource-accounting-001` completed/exit 0，`reports/resource_accounting_001.json` SHA256 `340fe253a3ef92403b150e0c551d60220df4f838ac072a71e8ec9a31fd6c920a`，截点 Unix 1789266026.4423256：租约并集 19.539934 GPU-hours，加入已知 spill 采样估算为 19.548273 GPU-hours，15 次 spill 采样；pool-006 在截点仍存活。该值不是最终总账或精确模型计算/能耗，不覆盖无法归属的历史额外 GPU 上下文；租约并集峰值 4 不能推翻此前实际越界记录。CPU 审计成本独立登记。

USAGE 已整理为环境、实时状态/日志、中断/停止、证据与复核入口；逐次运行历史集中于本 REPORT 和原产物，没有另建散落文档。下一步继续真实全场景/20 项跨进程 reset、正式多服务路径验收与协议冻结，再执行四组正式训练及 2040 条 rollout；完整 Goal 保持未完成。

### 多服务组合路径工程计划

为实际验收新的正式多 block 路径，将其物理生命周期抽为 `execute_schedule`，正式入口仍强制完整冻结协议和四组最终权重；独立 `r2_pilot_smoke.py` 只接受固定 8-episode 工程计划，使用 E0/E3 step-zero、adjust_bottle 两个条件各前两个已接受开发场景，两个 wave 轮换设备槽，每服务连续处理两个新模拟器进程。保留原动作上限，无故障注入、无自动重试，不运行 final、不算正式成功率；旧工程失败的账本和结果不改变。

正式冻结增加 `pilot_lifecycle` 门槛，绑定八条逐 episode 审计、服务复用/新进程、两轮收尾、矩阵/环境/资产和完全相同的源码身份，不能仅凭公共服务曾通过而越过新组合路径验收。`p4-pilot-smoke-entry-tests-001` 30 passed、`p4-pilot-lifecycle-gate-tests-001` 53 passed，均已核验 completed/exit 0 且进程退出。已独立启动 CPU `p4-pilot-lifecycle-plan-001` 核验真实输入并生成 `manifests/pilot_lifecycle_plan_001.json`；最近 PID/创建时间核验存活，计划完成及真实 GPU 执行尚待核对。`p1-r2-regression-008` 已启动，结果待核验。

pool-006 仍存活，最近四条 lane 的持久完成数为 7/7/9/8，共 31 分片；这只是持久进度，完整清单门槛未通过。优先让原有限场景池继续；停止并释放 GPU 0 后才可调度三卡工程验收或单卡跨进程 reset，不在两个独立租约中重复使用设备。后续接管先确认计划是否完成、源码身份是否一致及各任务真实状态，不重复登记同名计划或启动同一 run。所有正式训练、配对评测仍未开始。

计划准备随后 completed/exit 0，进程退出；`manifests/pilot_lifecycle_plan_001.json` SHA256 `7a9bd62eaa7983d164861dd83fe8f0831061e9104360f4cc3b14b78dfe6cd4da`，固定 334 个可执行来源文件。四个场景为 adjust_bottle clean seeds 200001/200002、randomized seeds 200000/200001；每个分别由 E0/E3 在新工程协议身份下执行，不改变历史 episode 的终态或重试额度。独立 `p4-pilot-lifecycle-input-audit-001` 重新核对权重/配置/统计/环境/源码和场景，completed/exit 0、进程退出，CPU wall-clock 77.245 秒，报告 `reports/pilot_lifecycle_inputs_001.json` passed；仅为输入验收，尚未启动三卡 GPU 工程 run。计划源码登记后不要在未重新核验身份的情况下改动可执行文件；文档修订不影响其来源。

最终回归 `p1-r2-regression-008` completed/exit 0、进程退出，429 passed、1 CUDA 项跳过、2 条第三方警告，45.04 秒。pool-006 最新 PID/创建时间及心跳核验仍存活，持久完成分片数为 8/7/10/9，共 34/40；lane_2 即使已记录十个分片，也仍须等其后置来源校验及整池收尾。下一轮先核验该池，完成并释放后运行全清单审计；再安排固定 20 项跨进程 reset 与已登记的 8-episode 三卡工程及独立审计，全部门槛通过后冻结协议并连续进入正式矩阵。不能根据此交接把场景或新组合路径标为已通过。

### 场景池等待与后续 CPU 审计

继续执行时核验 pool-006 的相同 supervisor/child PID 与创建时间仍存活，未重启。新增整轮调度测试使用实际测试子进程，验证两个 wave 的完成屏障、GPU index/UUID 槽传递、任一 lane 失败后不启动下一轮，以及全部已归属子进程退出；`p4-pilot-wave-barrier-tests-001` completed/exit 0、进程退出，18 passed，9.36 秒。测试修改不进入登记的 334 个可执行来源，`pilot_lifecycle_plan_001.json` 的源码身份仍一致；此项仍不是真实三卡模拟器/模型验收。

lane_2 随后完成十个分片及后置资产核验并退出。独立 `p1-complete-lane2-audit-001` completed/exit 0、进程退出，CPU wall-clock 4.149 秒；`reports/complete_lane2_001.json` SHA256 `d04fa7c72080a252c6195be911b7c1454619f3cd4b69dae8ce794058f5b3eb4d`，绑定 lane manifest `c719c67437bcd8efb5b46c597a935a71062cbcf128595286bfe55068c7a039c8`。这只是 1/4 lane 的完整审计，不替代整池 40 分片/500 场景收集。

登记并启动独立 CPU `p1-full-scenes-audit-001`：最多等待 7200 秒，每 30 秒核验 pool-006 的固定 supervisor `{pid:3037547, created_at:1789261891.0}`、child `{pid:3037564, created_at:1789261893.98}` 及后代；只有实际进程退出、completed/exit 0 且租用设备释放快照完整时，才调用 `evaluation_scenes.collect` 输出 `reports/evaluation_scenes_001.json`。身份改变、失败/停止、缺失释放证据或等待超时即退出，不自动重试，不控制场景池、不申请 GPU。其脱敏完整命令持久保存在该 run 的 `job.json`；最新核验 supervisor/child 存活，日志为 `live_verified=true` 的等待记录，尚未产出全场景通过结论。这仅是有限的“等待场景池后做 CPU 审计”，不是跨训练/评测自动编排。

本轮曾观察到八张设备上额外 compute PID，核验本 Goal 的实际活进程仍只有 pool-006，未对无法归属的宿主 PID 发信号、未新增 GPU 租约；后续采样中这些额外 compute PID 消失，其余七卡显存归零。租约及统计归属盲区继续保留，不据单次检查承诺将来空闲。pool-006 当前仍占 GPU 0，持久完成数保持 8/7/10/9；剩余当前分片接受数已推进到 13/15/9（各目标 20，之后 lane_0/1 仍有余下分片）。接管时同时核验 pool-006 和 full-scenes-audit-001，不重复启动已登记的等待/收集任务。

### 评测耗时代理与持续等待

在不改动已登记可执行来源的情况下，独立 CPU `p2-evaluation-timing-proxy-001` 复核 20 个开发分片、各 task/condition 首个接受场景的原动作上限，以及三份完整物理工程服务计时。该任务 completed/exit 0，进程退出，CPU wall-clock 4.038 秒；结果为 `reports/evaluation_timing_proxy_001.json`。它不是正式协议冻结、完整场景审计或启动许可。

固定 90 个两 lane wave、40/400/1600 个 rollout，假定每次都跑到原动作上限，使用 E0 step-zero adjust_bottle 的 service wall/actions 中位数代理所有模型与任务；以开发场景原上限暂代同条件其余场景，正式冻结时仍逐一读取最终清单上限。模型启动与前后检查残余开销单列，共享模拟器/任务模型差异的减速系数 1/1.5/2 仅为敏感性假设，不是实测或置信区间：评测 wall-clock 分别 151.759 / 225.569 / 299.380 小时，对应三卡分配成本 455.276 / 676.708 / 898.139 GPU-hours；加上既有训练候选 42.8 小时 / 171.201 GPU-hours，分别为 194.559 / 268.369 / 342.180 小时与 626.477 / 847.909 / 1069.340 GPU-hours。该约 8–14 天量级不包括余下场景准备、导出、独立审计、排队和故障重算；早成功可能缩短 rollout，但不能事先假定成功率来缩减估算。已向用户告知量级及低置信度限制，不新增人工预算确认门槛、不自动追加实验。

最新实际核验 pool-006 与 full-scenes-audit-001 仍存活，后者继续核验同一进程身份等待，不重复启动。场景持久分片数推进到 9/8/10/10，共 37/40；lane_3 的十个分片完成后仍在后置校验，未据此宣称整池已完成。下一步保持原候选和剩余额度，待整池及 CPU 收集通过，再安排已登记的 reset/多服务验收。完整 Goal 保持未完成。

上述耗时代理文件 SHA256 为 `9d74e1e2cc9adbf99450ccfca90f4119e20f6d6ab8e70f2685eecd5aad276260`；固定调度的上限关键路径合计 683400 actions。开发场景原上限包括 open_microwave 1500、stack_bowls_two 900、handover_block/stack_blocks_two 800、place_object_basket 700，其余任务 400，两个条件一致；不能用单一 400-action 工程 episode 的时长直接乘整个评测数量而忽略任务上限。

lane_3 随后完成后置核验并退出。`p1-complete-lane3-audit-001` completed/exit 0、进程退出，CPU wall-clock 4.043 秒；`reports/complete_lane3_001.json` SHA256 `b99a7fd376623284501a7bc3db96da1a77d09136f9584fc1448252714e271386`，独立核验该 lane 的十个分片。最新 lane_0 最后分片 stack_blocks_two clean 接受 17/20，lane_1 当前 place_object_basket randomized 接受 9/20、另有最后一个 stack_blocks_two randomized 分片待执行。已登记的 334 个可执行来源哈希仍一致；没有 GPU 新 launch、没有重试或更改筛选门槛。整池及其有界后续审计继续运行，实时状态须以接管核验为准。

### 完整场景清单与跨进程初态检查

`p1-scenes-pool-006` 已 completed/exit 0、无 failure，supervisor、child 和后代均已退出；四条 lane、40 个分片完成，原 `runs/p1-scenes-pool-003/scenes/` 的候选顺序与尝试记录保留。退出快照显示租用 GPU 0 的 compute/graphics 进程为空。该段 wall-clock 9114.819 秒、2.531894 分配 GPU-hours、4531 次采样、峰值进程树 RSS 14,292,606,976 bytes；4332 次采样存在进程归属盲区，因此模型 GPU-time 与能耗仍为未知，不以租约成本代替。

独立 `p1-full-scenes-audit-001` 在核验场景池正常退出及释放后完成全清单收集，completed/exit 0、进程退出。`reports/evaluation_scenes_001.json` 包含 100 个开发场景和 400 个最终场景，SHA256 `57e65177e459eaec298c504defdffc373b753aa8cfba1d457a9b2c528d559886`，rows SHA256 `ccaf4f7979e720fa6d20fb421ccd5a68db890ebccaba3b48ef90c6d7c51e4d42`。该 CPU run 包含等待时间，不将其总 wall-clock 当作纯审计用时。此门槛不包含策略 rollout 或跨进程 reset 验收。

`p1-cross-process-reset-001` 已在单 GPU 0 独立运行，固定每 task/condition 首个开发场景，共 20 个新模拟器进程、每项最多 600 秒、无策略动作及自动重试。接续核验 supervisor `{pid:3170280, created_at:1789271293.96}` 与 child `{pid:3170285, created_at:1789271294.17}` 存活，已有 19 份通过的 reset 结果；后置来源核验与外层收尾尚未完成，不提前发布门槛。运行输出为 `runs/p1-cross-process-reset-001/reset/`，实际 GPU 快照仅 GPU 0 有 compute/graphics 上下文；334 个已登记可执行来源仍一致。

下一步等待该有限任务完成并核验释放，独立审计后发布 `reports/cross_process_reset_001.json`，再运行已登记的 8-episode 三卡多服务工程计划。此处没有启动正式训练或 2040 条正式 rollout，不重跑已通过的 profiling 或故障恢复实验。

跨进程 reset 随后 completed/exit 0、无 failure，全体进程退出且 GPU 0 释放；20 项均完成，前后环境一致。该 run wall-clock 1095.095 秒、0.304193 分配 GPU-hours、548 次采样、峰值进程树 RSS 4,229,648,384 bytes，392 次存在归属盲区。独立 `p1-cross-process-reset-audit-001` completed/exit 0、进程退出，CPU wall-clock 2.037 秒；`reports/cross_process_reset_001.json` passed，20 项初态一致，SHA256 `089bce979112a0ec64664953fe189fbeb33aa52007c63482f7803b396a1ec3ef`。这是模型无关的新进程初态核验，不是全轨迹重复性或中途物理快照恢复。

再次核验全部物理设备空闲及登记来源后，已启动 `p4-pilot-lifecycle-001`，同一租约使用 GPU 0/1/2，输出 `evaluation/`、新工程账本 `episodes/`，绑定原 `pilot_lifecycle_plan_001.json`。启动后 supervisor `{pid:3177811, created_at:1789273084.36}`、child `{pid:3177813, created_at:1789273084.57}` 及预检子进程经身份核验存活，正在前置资产/环境检查。计划仍为恰好 8 episodes、两轮四 block、E0/E3 step-zero、每服务两个新模拟器进程、无故障注入及自动重试。启动成功不代表生命周期验收通过；其有限运行和监测独立推进，尚无跨训练/评测自动编排。源码继续冻结，退出并释放后使用模拟器 Python 独立审计 `pilot_lifecycle_001.json`，全部门槛通过才冻结正式协议。

### 三卡工程推进与有界后续审计

后续接管已核验同一 supervisor/child 存活、心跳和 episode 动作持续推进，没有重新 launch。实际 NVML 快照显示 GPU 0 上两个 compute/graphics 模拟器上下文，GPU 1/2 各一个模型 compute 上下文，其余设备无进程；容器/宿主 PID 归属限制继续披露。共享模拟器卡出现高利用率，模型卡间歇等待，完整性能结论仍待原始采样汇总，不能据瞬时快照推断已充分利用或随意修改并发。

登记并启动 CPU `p4-pilot-lifecycle-audit-001`，固定核验上述 GPU run 的 supervisor/child 身份，最多等待 20000 秒、每 30 秒检查实际存活；正常退出且三卡释放后，以模拟器 Python 调用原 `r2_pilot_smoke.py audit`，独立审计另限 1800 秒。失败、停止、身份改变或等待超时即退出，不发送 STOP、不申请 GPU、不自动重试或启动正式训练。脱敏命令持久保存于该 run 的 `job.json`；已核验 supervisor/child 存活，正在有界等待，不把启动或等待日志视为 `pilot_lifecycle_001.json` 已通过。

新增启动响应中断回归测试分别覆盖提交前中断与已提交但 ACK 丢失，验证启动前登记保留、无 `state.json` 不能证明未提交、同 run id 拒绝重复提交；后者使用真实独立 CPU 子进程并在测试末尾核验退出。`p2-launch-interruption-tests-001` completed/exit 0、进程退出，19 passed、测试耗时 0.12 秒。它是替代 tmux/设备接口的入口测试，不是实际客户端 interrupt 按钮或真实 GPU 启动故障验收；原独立运行与恢复证据继续单独保留。仅修改测试文件，登记的 334 个可执行来源仍一致。

clean 条件已有三条完整 failure/action_limit，每条 400 actions、无重试；lane_1 两条已完成且服务退出，lane_0 第二条仍在推进。E0/E3 首条视频 contact sheet 已由 `p4-pilot-first-videos-001` 生成，completed/exit 0、进程退出，分别为 `reports/pilot_lifecycle_001_E0_clean_first_contact.png`、`reports/pilot_lifecycle_001_E3_clean_first_contact.png`，目视确认非空及机械臂运动。这些 step-zero 工程失败不是正式成功率，局部视频核验也不替代八条独立审计。接续时同时核验 GPU smoke 和 CPU 后续审计，不重复启动两者；全部结束并核验报告后再推进协议冻结。

### 三卡生命周期 smoke 完成

`p4-pilot-lifecycle-001` 已 completed、exit 0，使用物理 GPU 0/1/2，完成两轮、四个 block、8 个 episode；独立模拟器 Python 审计 `reports/pilot_lifecycle_001.json` 的 `passed=true`，8/8 records 均 `audited=true`，2 个 completed waves，固定计划 SHA256 为 `7a9bd62eaa7983d164861dd83fe8f0831061e9104360f4cc3b14b78dfe6cd4da`，审计报告 SHA256 为 `486590da725810c29750f466270cc23bebdb10cdfcdacbc0cf2821710332fbc0`。审计还核验了 334 个可执行 source files 的身份、E0/E3 step-zero 权重、服务复用及每 episode 新模拟器进程。

八条记录全部为工程阶段 `failure/action_limit`，每条恰好 400 policy actions；无故障注入、无自动重试，不能转写为 invalid、成功或正式成功率。E0/E3 首条视频 contact sheet 已确认非空且存在机械臂运动，但仅是部署检查。

GPU run 的 `performance_summary.json` 记录 wall-clock 2694.892 秒、allocated GPU-hours 2.245743、1347 个资源样本、峰值进程树 RSS 88,115,580,928 bytes。1019 个样本存在进程归属盲区，故 observed process GPU-hours 与 estimated device kWh 保持未知，不填零；GPU 0/1/2 的 release snapshot 均无 compute/graphics 进程。独立 CPU `p4-pilot-lifecycle-audit-001` 也已 completed、exit 0（wall-clock 1921.932 秒，GPU-hours 0），只做有界等待和审计，不包含 GPU 工作负载。`p2-launch-interruption-tests-001` 已 completed、exit 0，19 passed。

该门槛现在通过，但只解锁协议冻结前的最后检查，不代表 R2 pilot 完成。下一步先确认 `pilot_lifecycle_001.json`、全部输入/源码哈希和其他 gate 报告一致，再运行 `freeze_protocol.py` 原子发布 `manifests/protocol.yaml`；冻结成功后才按同一协议依次启动 E0、E1、E2、E3 正式训练。正式训练、最终 checkpoint 导出、2040 条配对 rollout、统计分析均尚未开始。

### 继续执行核验（2026-09-13 07:59 UTC）

本次接管先完成只读状态核验：`formal_training.load_protocol()` 成功加载冻结协议 `r2-pilot-v1`（8048 optimizer updates、global batch 64、world size 4），没有发现本 Goal 的正式 training/evaluation run 或存活 lease。NVML 显示 GPU 0 空闲；GPU 1/2/3 各约 14,018 MiB、74–76% 利用率，并分别存在 PID 3899245/3899249/3899252 的无法归属 `compute+graphics` 上下文；GPU 4–7 也有外部任务。最新只读阻塞证据为 `reports/gpu_allowlist_block_20260913_075905.json`。因此本次没有启动、恢复或停止任务，也没有向外部 PID 发信号；正式四卡训练继续等待物理 GPU 0/1/2/3 全部可证明空闲。

### Allowlist 重核验（2026-09-13 08:06 UTC）

本次只读重核验再次成功加载冻结协议 `r2-pilot-v1`：四组 E0–E3、8048 updates、global batch 64、world size 4、10 个任务均保持一致。`manifests/gpu_leases.json` 为空，没有本 Goal 的存活 lease、supervisor、训练或评测进程。物理 GPU 0 的显存约 0 MiB、利用率 0%，无进程；GPU 1/2/3 各约 14,017 MiB，分别有无法在容器内归属的 PID 3899245/3899249/3899252 C+G 上下文，NVML 利用率约 77%/80%/79%。GPU 4–7 也有外部 compute 进程，但不属于本 Goal allowlist。

本次未发送任何信号、未启动或恢复任务、未使用 GPU 4–7。正式四卡启动仍被阻塞，证据为 `statewam_artifacts/robotwin/r2-robotwin-pilot/reports/gpu_allowlist_block_20260913_1606.json`；只有重新核验 GPU 0/1/2/3 全部空闲且无归属盲区后，才可按冻结协议创建新的 formal training run。

随后在隐藏 CUDA 的 CPU 环境中运行 `test_r2_formal_training.py`、`test_r2_gpu_resources.py` 和 `test_r2_export_checkpoint.py`，结果为 `54 passed in 2.31s`；该回归不占用 GPU，也不替代正式训练。08:09 UTC 的再次轮询仍见 GPU 1/2/3 的同一组未归属 PID 和 GPU 4–7 的外部 compute 进程，故保持等待，不重复 launch。

08:11 UTC 接续核验：GPU 0 仍为空闲，GPU 1/2/3 仍分别为约 14,017 MiB、利用率约 74%/80%/76%，同一组外部 C+G PID 仍存在；GPU 4–7 仍在 allowlist 外运行。协议只读加载再次通过，`gpu_leases.json` 仍为空，没有 formal training/evaluation supervisor 或可接管 run。该次为连续资源等待证据，不改变冻结协议、预算或 GPU allowlist。

08:13 UTC 的 60 秒有界等待后重查仍未释放 GPU 1/2/3：三张卡各约 14,017 MiB，PID 3899245/3899249/3899252 仍被 NVML 列为 C+G；GPU 0 为 0 MiB/0%，GPU 4–7 仍有外部 compute。没有本 Goal 新进程或 lease，因此未 launch、未恢复、未停止任何任务。

08:14 UTC 接续重核验仍为相同结论：协议 `r2-pilot-v1` 可加载，`gpu_leases.json` 为空，GPU 0 空闲，GPU 1/2/3 分别约 14,017 MiB、74%/74%/77%，外部 C+G PID 3899245/3899249/3899252 仍无法在容器内归属；GPU 4–7 全部在 allowlist 外且有外部 compute。最新证据为 `reports/gpu_allowlist_block_20260913_1614.json`。在外部上下文释放并重新核验前，正式 E0–E3 不可安全启动。
