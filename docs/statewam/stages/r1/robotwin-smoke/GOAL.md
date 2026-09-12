# StateWAM R1 Profiling 与 RoboTwin Clean2Random Smoke Goal

日期：2026-09-12。
阶段：R1。Goal ID：`r1/robotwin-smoke`。
文档归属：本目录独立维护 `GOAL.md`、`REPORT.md`、`USAGE.md`；长期架构设计位于 `../../../design/`，不与执行文档混放。
状态：本 Goal 工程验收已完成；只覆盖 smoke 与 4 卡短程 profiling，不代表完整训练/成绩复现或整个研究计划完成。逐项证据见 [REPORT.md](REPORT.md)，运行命令见 [USAGE.md](USAGE.md)。

## 1. 可直接下达的 Goal

> 请按 `docs/statewam/stages/r1/robotwin-smoke/GOAL.md` 执行：在 `feat/statewam-r1-robotwin` 分支上实现 StateWAM R1 最小工程路径，完成工程 smoke 和 4 卡、200–500 个 optimizer updates 的 profiling；搭建独立的 RoboTwin 2.0 环境，使用发布的 OpenWAM Clean2Random 权重分别跑通少量 clean/randomized rollout，并完成 StateWAM 的仿真部署 smoke。本轮不进行收敛训练、完整 benchmark 评测或 R2 E0–E3 训练矩阵。数据集统一放在 `/gaozt-test1/datasets`，RoboTwin 相关运行产物统一放在 `OpenWAM/statewam_artifacts/robotwin/`。按文档管理规范，在本 Goal 目录内交付技术验收报告与简洁 usage 文档，明确区分通过、失败和未执行项目。

本文档是本轮执行范围，以用户最新约束将原研究计划的首轮 8 卡调整为 4 卡，保留 200–500 步 profiling，仿真评测仅做 smoke。原 R0–R5 研究路线不变，本轮交付工程验证与性能测量，以及发布模型和 StateWAM 的少量仿真闭环证据；不将其称为成绩复现、完成 R2 或证明 StateWAM 有控制收益。

变更记录：2026-09-12，恢复 4 卡、200–500 optimizer updates 的 profiling；按研究设计与阶段/Goal 归档文档，完整训练和全量评测仍暂缓。

执行记录：2026-09-12，用户明确下达本 Goal 后执行。完成 4 卡 200 optimizer updates profiling、优化器恢复后第 201 步，以及基线/StateWAM 各 clean、randomized 1 episode；最终回归 242 项通过，报告和 usage 在本目录交付。

## 2. 仓库与目录约定

OpenWAM 项目分支：`feat/statewam-r1-robotwin`。
分支创建点：`5be50682eb45ef5d2dace77cce88c3baa0c48446`。
创建时已有未跟踪的 `docs/statewam/` 文档，保留其内容，不清理或覆盖。

```text
/gaozt-test1/fyhong/
├── OpenWAM/
│   ├── docs/statewam/                  # 版本化的计划、执行状态与验收报告
│   ├── openwam/model/architectures/statewam/
│   ├── scripts/statewam/
│   ├── tests/statewam/
│   ├── benchmarks/robotwin/            # 适配、评测启动与结果汇总
│   └── statewam_artifacts/             # 本仓库根目录下，Git 忽略
│       ├── robotwin/
│       │   ├── checkpoints/           # 发布权重的受控转换、短训练输出
│       │   ├── manifests/             # 数据/任务/评测初态/权重来源
│       │   ├── profiles/              # 显存、吞吐、延迟
│       │   ├── runs/<run_id>/          # 配置、日志、指标、视频、运行时目录
│       │   └── reports/               # 汇总表与分析输出
│       ├── engineering/               # 跨 benchmark 的模块和加载检查
│       └── real/                      # 自有真机数据接口检查，按需建立
└── RoboTwin/                          # 独立 Git 仓库和独立 Python 环境

/gaozt-test1/datasets/
└── robotwin/                          # 下载数据集、划分与派生数据
```

- `statewam_artifacts` 位于 OpenWAM 内，所有 RoboTwin 结果必须带 `robotwin/` 命名空间；不同协议、模型与 seed 使用独立 run id。
- RoboTwin 采用同级独立 checkout，优先固定提交 `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`，不复制进 OpenWAM，也不首轮改为 submodule。
- 若该路径已有仓库或环境，先检查再复用；需要其他版本时使用独立 checkout，保留已有工作。
- 模型 checkpoint 不属于数据集：优先复用已有权重；需要下载时使用明确的模型目录，例如现有 `assets/openwam_ckpt/`。转换后的实验权重存放在上面的产物目录，不覆盖源权重。
- 所有下载的数据集及其解压、转换、教师缓存等派生数据放在 `/gaozt-test1/datasets` 下，按来源和版本分目录。先下载 smoke 所需最小数据，完整训练暂缓时无需下载完整训练集。
- 仿真资产按 RoboTwin 安装契约加载；可集中放在 datasets 下再通过配置或链接接入，记录真实路径和资产版本。
- 使用 `ROBOTWIN_PATH`、`ROBOTWIN_PYTHON` 等现有配置入口。将原生 `eval_result`、多任务日志和视频统一路由到本轮 run 目录；验证现有脚本中 runtime root 与日志参数的实际含义，必要时增加兼容的输出路径参数。
- 可复现脚本、配置模板、技术验收报告和简洁 usage 文档进入 Git；权重、数据、缓存和大体积运行结果不进入 Git。报告引用产物相对路径及校验值。
- 本 Goal 的报告、usage 和运行 manifest 统一记录 `goal_id: r1/robotwin-smoke`。每次运行使用独立 run id，并由本目录的 `REPORT.md` 索引；同阶段的后续独立目标另建 `stages/r1/<goal-id>/`，其他阶段另建对应阶段目录，不共用报告。
- 本次文档修订保留 200–500 次 optimizer update profiling 的验收要求，但不触发新的实验执行；不将 microstep、episode 或推理去噪步数计入该训练步数。

## 3. 执行范围与顺序

### A. 前置审计与最小环境

1. 阅读 `docs/statewam/design/` 中的结构方案、实现映射和仿真协议，以及[文档管理规范](../../../README.md)，检查当前源码及仓库指令。
2. 记录可用 GPU 的型号、显存、占用，已有 OpenWAM 环境、权重、数据和 RoboTwin 安装状态；选择空闲设备，不终止他人的任务。
3. 记录本轮实际使用的发布 Clean2Random checkpoint 身份、配置和文件校验值。Foundation/Wuji 和已有成功真机 run 的历史信息能从现有本地资源核实时纳入记录，不为本轮 smoke 强制下载未使用的大型权重。
4. 建立独立 RoboTwin 环境，优先按仓库已核对的版本安装：Python 3.10、SAPIEN 3.0.0b1 等依赖以 `benchmarks/robotwin/README.md` 为准。
5. 先验证资源加载、Vulkan/渲染、reset、规划器和 server/client 链路，再进行少量闭环 smoke。
6. 冻结任务清单、成功定义、episode 数、控制频率、动作 chunk、评测 seed 和输出目录。涉及数据预处理时，先划分 episode/session，再仅用 train 拟合统计。

### B. StateWAM R1 最小实现

沿用已有 world/action 骨干和本体 adapter，新增独立 StateWAM 架构与必要的共享接口扩展。

- 实现 `InteractionStateExpert` 的 reader、writer 和 gated residual；起始 `K=16`、latent dim 为 `512`。
- 对 30 层模型在 `[4, 9, 14, 19, 24, 29]` 读取和写入 State；保留原 30 层 world/action 配对及 `bridge_interval=1`。
- 支持 `off`、`current`、`future`、`future_action` 配置，满足后续 E0–E3 结构比较需要。本轮只验证其实现与接口。
- 每个去噪调用内动态读取，不跨去噪调用或 episode 缓存 State；维持当前视觉的因果保护。
- gate 初始为零，reader/writer 采用可启动的非零标准初始化；新增 latent 维度与原 proprioception `state_dim` 分开命名。
- 接通 registry、配置、训练、优化器参数组、generate 和部署生命周期。
- 实现一次性的 OpenWAM → StateWAM checkpoint 转换，严格验证原参数，只允许预定义的新 State 参数缺失；覆盖 Foundation 与 Wuji 两种源权重。部署和普通重载继续严格加载。
- 参数冻结与 `no_grad` 分开处理，确保 State 梯度可穿过固定权重的动作/世界计算。
- 首版只使用原 world/action flow-matching 目标。语义教师、Human 扩展、独立 Prior、bottleneck 训练和蒸馏留到后续。

### C. R1 工程验证

按小张量 → 单卡真实权重 → 4 卡真实几何的顺序验证：

1. shape、来源 mask、padding、all-invalid/no-action 样本及参数注册。
2. 固定输入与显式相同噪声下，原模型、State off、gate=0 的输出一致性；记录 dtype、容差和最大误差。
3. gate 首步梯度以及后续 reader/writer 的非零有限梯度；检查冻结、梯度累积和 checkpoint 重算路径。
4. 保存/重载输出一致、optimizer 参数组完整、resume 恢复优化器状态；至少使用本轮 Clean2Random 真实权重验证。Foundation/Wuji 双来源验证留待完整 R1 验收，除非权重已在本地且检查成本很低。
5. 单卡实际 batch 的预处理、forward、backward、保存和推理。真实数据可用时复核已有 Wuji adapter 的 raw 58D → model 80D 契约，不将此契约套给 RoboTwin。
6. 先用 2–10 个 optimizer updates 做训练启动 smoke，再用 4 卡、真实训练几何完成 200–500 个 optimizer updates 的 profiling。默认完成 200 步；若需确认稳态或定位稳定性问题，可延长至 500 步。分别记录启动/预热与稳态耗时、峰值显存、有效样本吞吐、数据读取比例、checkpoint 保存/重载开销和梯度证据；测量区间与实际步数入档。
7. 先 eager 且关闭近似 velocity cache 验证，再测试 compile；缓存优化独立验证，不能直接假定适用于新 State 路径。

短训练属于 R1 工程验收，用于确认模型可训练，不构成 RoboTwin 收敛训练或效果复现。

### D. 发布 Clean2Random 模型评测 smoke

使用 `OpenWAM-Alpha-Sim-RoboTwin-Clean2Random` 发布权重，分别评估 `demo_clean` 和 `demo_randomized`，不使用 Full 权重替代。

1. 选择至少一个可正常执行的任务，在 `demo_clean` 和 `demo_randomized` 下各运行 1–5 个 episodes；如果首个任务受环境或规划器问题阻塞，可换一个预先记录的任务继续定位。
2. smoke 尽量沿用公布推理设置：10 步同步去噪、compile 开启、DiT velocity cache 开启，`cosine_threshold=0.99`、`max_skips=3`；若为定位问题临时关闭优化，必须在报告中记录，并在修复后至少用目标配置完成一次闭环。
3. 保存逐 episode 结果、模型采样协议、日志、配置和必要视频，证明 reset、render、policy server、动作执行和结果落盘均能完成。
4. smoke 的验收对象是链路和协议，不以成功率为通过条件，不与仓库参考成绩作统计比较。
5. 若环境迫使使用 planner fallback 或修改协议，只能作为诊断；最终 smoke 应使用目标路径，或者在报告中将对应项标记为未通过并说明阻塞。

这一部分不要求在本地重新训练发布模型，也不运行 50 tasks × 2 conditions × 100 episodes 的完整评测。少量结果不充当成绩复现或未来从 Foundation 开始、同预算训练 E0–E3 的证据。

### E. StateWAM 仿真部署 smoke 与交付

- 将发布 Clean2Random 权重受控转换为相同 world/action 拓扑的 StateWAM checkpoint，先验证 gate=0 的部署一致性。
- 使用 State 开启的短训练 checkpoint 跑预定义少量 clean/randomized rollout，确认 observation → normalization → generate → denormalization → executor 全链路可用。
- 显式核对 RoboTwin 的 EEF/joint 动作类型和维度；保留各 benchmark 自身 adapter。
- 分别报告基线发布协议与 StateWAM 工程协议的 compile/cache 设置，不能把不同推理设置下的 smoke 成功率当公平结构比较。
- 交付可重复的分步命令、配置、模块测试结果、4 卡 profiling、Clean2Random 双条件 smoke、StateWAM smoke 证据以及未解决问题。
- 新增本目录 `REPORT.md`：记录环境、版本、实现范围、测试证据、profiling、产物索引、未通过项和明确暂缓项。
- 新增本目录 `USAGE.md`：只保留安装前提、环境变量、checkpoint 准备、测试、4 卡 profiling、模型服务、clean/randomized smoke、StateWAM smoke 和常见故障所需的简洁命令；不混入研究论证。
- 遵循 `docs/statewam/README.md` 的文档管理规范：研究设计只放 `design/`；执行材料放 `stages/<阶段>/<goal-id>/`。本轮 Goal、报告、usage 使用同一目录，不在顶层追加编号文档，不覆盖其他 Goal 的报告。后续独立 Goal 使用新的目录并登记到索引。

## 4. GPU 与 batch 约束

- 本轮进行训练启动 smoke 和 4 卡、200–500 optimizer updates 的 profiling，不做收敛训练。单卡/小张量 smoke 可用更少设备；多卡 profiling 使用 4 卡。后续若另行启动 RoboTwin 本地训练复现，也先仅使用 4 卡。
- 本轮项目同时占用的 GPU 总数不超过 4；训练、policy server 和渲染 worker 统一安排，避免额外启动评测突破资源边界。
- 优先采用对应模型、数据几何和训练协议的官方推荐 batch，明确区分 per-GPU batch 与 global batch；通用配置默认值不能自动称为特定实验的官方配方。
- 若需要从更多卡的配方适配到 4 卡，优先通过梯度累积保持目标有效 batch：`global_batch = per_gpu_batch × 4 × accumulation_steps`。
- 无明确配方时，在真实视频几何和优化器设置下逐步增大 micro-batch，尽可能提高显存利用率，同时保留保存、验证及动态峰值所需余量，以稳定吞吐作为选择依据。
- 不为填满显存擅自改变分辨率、动作 horizon、训练曝光量或实验语义；OOM 后保留记录，调整 micro-batch/累积并重新验证。
- 训练 smoke 和 profiling 均记录 optimizer updates、micro-steps、梯度累积、有效 batch、样本曝光与 GPU-hours。200–500 步指 optimizer updates，当前 trainer 的 `max_steps` 语义须从代码核实后换算。
- 在实测吞吐与渲染速度前不承诺完成天数。

## 5. 下载与认证

- 需要认证时，通过运行环境中的 `HF_TOKEN` 或受保护的凭据存储使用用户已提供的 HF 凭据。本文档不保存 token；不将其写入代码、命令示例、日志、Git 或实验 manifest。
- 后续会话若没有可用凭据，先尝试现有认证和公开访问；确实需要认证时再请求通过受保护方式提供，不从文档推测凭据。
- 优先使用 Hugging Face 官方源；遇到连接或代理问题，可尝试 `hf-mirror.com` 等可用国内镜像。先检查无需认证的下载路径，仅向经核实的必要端点发送凭据。
- 下载固定 revision 的所需文件，支持断点续传；记录 repo id、revision、来源和校验值。错误页、残缺文件或镜像不同版本不得当作有效权重/数据。
- 下载前检查目标目录容量与已有资源，复用可验证的文件；不覆盖已有数据版本。

## 6. 暂缓项目

- RoboTwin 本地完整训练复现，包括从 Foundation 重新训练 Clean2Random E0。
- 50 tasks × clean/randomized × 100 episodes 的发布成绩复现和统计对比。
- R2 E0–E3 匹配训练与多训练 seed 结构收益比较。
- R3 语义监督、bottleneck、容量/频率网格与完整机制研究。
- 真机物理 rollout、R4 效果验证、Human 数据扩展、跨硬件、蒸馏和大规模训练。
- 用户已明确下达本 Goal；授权仅限工程验证、profiling 与 smoke，以上暂缓项不随之启动。后续修改文档不自动授权新增实验。

## 7. 完成判据与状态记录

执行证据在 [REPORT.md](REPORT.md) 中维护；下表是最终验收摘要，不改变上述验收要求。早期失败和重跑记录保留在报告及原 run 目录。

| 验收项 | 当前状态 | 所需证据 |
|---|---|---|
| 独立项目分支与产物忽略规则 | 已完成 | 分支 `feat/statewam-r1-robotwin`、根 `.gitignore` |
| 环境、权重、数据与评测 manifest | 已完成 | 两套环境、数据/源码/权重哈希、配置、逐 episode 结果 |
| StateWAM 最小架构与来源开关 | 已通过 | 实现、配置、242 项针对性工程回归测试 |
| 真实权重等价、梯度、重载与续训 | 已通过 | 零 gate 等价、真实梯度、完整生成重载零误差、四 rank Adam 恢复哈希 |
| 训练启动 smoke | 已通过 | 4 卡 2 optimizer updates、权重和全状态保存 |
| 4 卡 profiling | 已通过，200 步及恢复后额外 1 步 | 稳态吞吐、峰值显存、保存/恢复开销与逐步梯度 |
| RoboTwin 环境及发布模型 smoke | 已通过 | reset/render/cuRobo 专家、server/client 与视频审计 |
| 发布 Clean2Random 双条件 smoke | 已通过 | adjust_bottle，clean/randomized 各 1 episode |
| StateWAM 仿真部署 smoke | 已通过 | step 201，六个 gate 非零，clean/randomized 各 1 episode |
| 技术验收报告 | 已交付 | 本 Goal 目录内的 [REPORT.md](REPORT.md) |
| 简洁 usage 文档 | 已交付 | 本 Goal 目录内的 [USAGE.md](USAGE.md) |

所有适用工程 smoke、4 卡 profiling 和仿真 smoke 验收项通过，且两份文档交付并更新索引后，才将本轮 Goal 标为完成。外部资源缺失或硬件限制导致无法执行时，保留已完成成果，明确具体依赖和剩余项；不得用 CPU 测试替代真实权重/4 卡检查，或将 smoke 标成完整训练/评测复现。

## 8. 依据

- [架构方案](../../../design/01_Architecture.md)
- [研究计划](../../../design/02_Research_Plan.md)
- [实现映射](../../../design/03_OpenWAM_Implementation_Map.md)
- [仿真协议](../../../design/04_Simulation_Protocol.md)
- [仓库 RoboTwin 运行接口及发布评测协议](../../../../../benchmarks/robotwin/README.md)
