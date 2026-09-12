# StateWAM v2：仿真先行的受控实验规格

日期：2026-09-12；v2 新增。\
状态：基于用户 fork 已有接口的执行计划；尚未安装环境、训练或运行仿真。

## 1. 为什么把核心消融移到仿真

真机的初始接触、物体微小姿态、手部标定、材料/摩擦、相机变化和执行时序很难严格重现。它适合回答模型是否具有实际价值，但细小架构差异容易被这些因素掩盖。

仿真适合重复 reset、固定任务与初始状态、隔离扰动、增加 rollout 数，并提供可用于机制评估的物体/接触状态。它不能保证逐位确定，也不能保证真实手部接触准确。因而采用三层证据，而非用仿真替代全部真机工作：

| 层 | 主要问题 | 推荐实验 |
|---|---|---|
| 离线与单元层 | 实现正确、状态包含什么、输入是否越界 | shape/梯度/重载、source mask、held-out probe |
| 仿真闭环层 | 架构或信息流怎样影响成功率 | E0–E3、容量/频率/bottleneck、受控扰动 |
| 真机层 | 在自有数据与真实灵巧手上是否有用 | 少量候选、抓取保持/滑脱/协调/恢复、端到端延迟 |

## 2. 环境选择与优先级

### 2.1 主环境：RoboTwin 2.0

选它是因为用户 fork 已继承 adapter、训练 reader 和评测脚本，能直接围绕 world/action 交互做闭环对照。双臂操作也与当前双侧数据契约存在研究上的联系，但标准 benchmark 的末端执行器并不等于舞肌/Sharpa 灵巧手。

建议主协议先用 Clean2Random：clean 数据训练，clean 和 randomized 分开评估；Full（clean+randomized 训练）作为拟合能力的后续对照。任务与数据量不足时使用预先定义子集，并在报告里明确子集，不能用子集分数冒充官方全套分数。

已核对 fork 的 README 给出 RoboTwin 参考 commit `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`、SAPIEN `3.0.0b1`，并提供 `single_eval.sh`、`multi_eval.sh`。实施时按该版本与其依赖进行 smoke，不混用未经核对的最新环境。

### 2.2 工程 smoke / 补充：LIBERO + LIBERO-plus

LIBERO 用于接口 smoke 或单臂外部复核；LIBERO-plus 用于 camera、noise、lighting、background 等分项扰动。若 RoboTwin 当前未安装或遇到渲染瓶颈，可以先用 LIBERO 做最小闭环，但是否将主 benchmark 切换为 LIBERO 应按环境可用性事先决定并记录。

fork 文档固定 LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`、LIBERO-plus commit `4976dc30028e805ff8094b55501d532c48fec182`，二者文档采用 MuJoCo `3.3.2`。LIBERO-plus 评价同一个在 LIBERO 上训练的 checkpoint；若在 plus 的测试扰动上调参/训练，它就不再是原来的 held-out OOD 协议。

### 2.3 自有灵巧手仿真：条件性后续

不将真实机械臂/灵巧手数字孪生作为首轮前置任务。只有研究问题明确依赖手指接触，而现有 benchmark 无法表达，才建设最小对应任务。须核对 URDF/关节耦合、驱动模式、接触和摩擦、传感延迟及任务资产；这部分本次未检查、未实现。

标准夹爪仿真阳性结果可以支持通用接口机制，不能直接支持灵巧手精细操作主张；真实灵巧手验证应在 R4 保留。

## 3. 初始化与训练协议

主矩阵：`Foundation F0 + 相同仿真数据 → E0/E1/E2/E3`。源模型、训练数据、优化步、有效 batch、图像处理和动作空间全部固定。

公开 `OpenWAM-Alpha-Sim-RoboTwin-Full`、`...-Clean2Random`、`...-Sim-LIBERO` 可用于确认 server/client 链路和 benchmark 基线。它们可能已经训练过目标 benchmark；不能把其结果和从 Foundation 开始的新模型当成相同训练预算的比较。

尤其不得从 Full 权重继续训练 clean，再把 randomized 结果称为未见分布；权重历史中的随机场景曝光不会因后续只训 clean 而消失。

如为节约启动成本，选择所有模型共同从某个已微调 sim checkpoint 继续训练，应明确报告这是 adaptation 实验，且其预训练曝光适用于所有组。不要用它替代 from-Foundation 主结论而不说明。

Wuji 初始化不需要加入首轮全部仿真网格；它作为真机初始化适用性的重要复核。在相同本体之外的仿真中，它的已有优势不能预设。

## 4. 哪些实验放在哪里

| 实验 | 首选场所 | 真机是否重复 |
|---|---|---|
| gate=0、off 路径与原模型一致 | 离线真实权重 | 少量部署 smoke |
| E0/E1/E2/E3 | 仿真 | 只保留关键 2–3 模型 |
| K=8/16/32、更新层/频率 | 仿真 | 不重复完整网格 |
| future-state bottleneck | 仿真，配合无旁路审计 | 若主张可部署压缩，再测胜出设置 |
| State→future semantic probe | 离线+仿真闭环样本 | 用少量独立交互标注检验相关性 |
| camera/noise/lighting 分项干预 | 仿真，验证集调参后锁定测试 | 选择真实相关扰动复核 |
| 接触状态/物体位姿解码 | 仿真真值用于评估 | 接触/抓持主张需现实证据 |
| 手指协调、滑脱、抓持稳定 | 真实平台为核心 | 必须 |
| Foundation 与 Wuji 来源稳健性 | 真机对应数据训练 | 最终候选在两条初始化内比较 |
| 延迟、动作执行稳定性 | 实际部署硬件 | 必须；模拟时间不能替代 |

## 5. 控制随机性的最低要求

### 5.1 三种种子分别记录

1. `train_seed`：模型初始化的新增模块、数据次序、训练噪声。
2. `env_seed`：场景生成、物体属性、初始状态和任务随机化。
3. `policy_seed`：推理阶段 world/action 采样。

正式模型差异应跨多个训练 seed 检验；仅复用一个训练模型并改变 env seed 不能说明训练稳定性。不同架构如消耗不同数量随机数，单设相同 global seed 不保证使用了相同 world/action 噪声；应在接口允许时显式构造并记录相同噪声张量，或记录独立采样协议。

### 5.2 配对环境条件

每个 task/condition 生成一个 immutable evaluation manifest，所有模型使用同一份：

- 任务和指令、物体实例、位置/朝向、关节位置/速度；
- 环境随机化参数、相机/灯光参数、版本与资产校验值；
- episode horizon、控制频率、动作 chunk 执行长度；
- env_seed、policy_seed、train_seed/run id。

如果 simulator 支持完整 snapshot，保存并恢复必要物理状态、controller 状态与 RNG；如果只支持 reset seed，则验证恢复后的观测/物体初态并如实称为配对初始条件，不假定接触求解器内部状态完全相同。任何新增 snapshot 工具都需要实现测试，现有脚本不自动提供上述全部能力。

在干预实验中，事件按固定模拟时间或预先定义的任务阶段触发，不能因为模型行为不同而给它们施加不同难度的扰动。配对后的未来轨迹自然会因策略不同而分离，这是要测量的结果。

### 5.3 仿真仍可能不确定

固定 driver、物理步长、solver 配置、render backend 与资产版本；使用少量完全相同 reset/采样输入重复 E0，测残余变异。如果存在 GPU physics/数值非确定性，用重复与区间量化，而不是宣称仿真绝对确定。

线程数、并发环境、编译和 cache 配置也要记录。不要在某一模型失败时开启另一套 planner fallback，而另一个模型仍使用原规划器。

## 6. 评价指标与信息边界

首要指标为 episode 任务成功率和 model 间的绝对差，按预定义 task×condition 单元等权；辅助报告完成阶段、任务时长、恢复率、约束违规和失败分类。

仿真中的物体位姿、接触、任务进度可以作为独立评测真值。默认不提供给模型输入；若用来训练辅助目标，必须作为特权监督单独实验，与主配置分开。评测 probe 训练也只使用 probe-train split。

不要用 world 视频看起来合理、latent loss 更低、或低噪声 teacher-forced action 更准替代闭环任务成功率。视觉上好看的预测可以与控制错误并存。

policy server 推理时环境可能暂停，因此模拟任务时间不会包含真实推理阻塞。分别报告模拟步数/模拟时间和 wall-clock latency，并在实际部署设备测 State 新增开销。

## 7. 规模、统计与停止规则

### 筛选

建议预先选 10 tasks×2 conditions×20 episodes，约 400 rollout/model；E0–E3 共 1600 次/训练种子。先单 seed pilot，只用于粗筛。task 子集按任务类型选，不按看到的模型成绩挑选。

### 复核

对 E0/E1/最佳未来状态模型补至少 3 个 train seeds；使用新的、锁定的 evaluation 初态。可以先对预先规定的 10–20 task 研究集增加到 50–100 episodes/condition/seed，再按算力做完整任务集。

示例成本：10 tasks×2 conditions×100 episodes×3 seeds×3 models = 18,000 rollout；完整 50 tasks 的同规格是 90,000 次。16 卡资源不意味着可忽略这部分成本。应先测 rollout 吞吐，再选择预登记子集或完整 benchmark；不能为了省时间把20次评测标成文档默认100次的可比成绩。

### 统计

环境初始状态在模型间配对；训练 seed 为独立训练重复。分层/配对分析尊重 task、train seed 和 scene id。置信区间不能把每帧当独立样本。评测 RNG 与训练 RNG 分开；所有原始逐 episode 结果保留。

先冻结主要比较（例如 E3/E2 中经 validation 选出的模型对 E0，并以 E1 检验容量解释）和主要 OOD。不要从大量 K/频率/扰动组合里挑最高的一项，再当成唯一预设假设。

若仿真无增益，先检查基准是否饱和、环境是否表达所需交互、训练是否充分和 State 梯度是否正常。若只有真实接触相关任务才可能体现机制，可进行预先定义的少量真实 pilot；不把仿真“必须赢”作为机械门槛，也不据无关仿真阴性直接否定灵巧手问题。

## 8. 仿真结果怎样进入真机决策

| 仿真 | 真机 | 解释与下一步 |
|---|---|---|
| 稳定改善 | 改善 | 支持接口机制与实际价值，进入扩大验证 |
| 稳定改善 | 无改善/退化 | 检查 sim/real 交互、相机、动作频率和数据适配差异；不能宣称普遍有效 |
| 无改善 | 改善 | 可能是平台/灵巧手特定收益；增加针对性对照并限定主张 |
| 无改善 | 无改善 | 关闭当前结构主张或重构，不急于扩算力 |

真机验证采用分块交错、场景模板和实际初态记录，不要求仿真式精确 reset。将预算集中在一个胜出结构和关键基线；Foundation/Wuji 初始化分别内部比较，避免以初始化差异解释为 State 增益。

## 9. 版本、资源与证据来源

- [fork 的 RoboTwin 运行接口与协议](https://github.com/FeiyangHong/OpenWAM/blob/5be50682eb45ef5d2dace77cce88c3baa0c48446/benchmarks/robotwin/README.md)
- [fork 的 LIBERO 接口](https://github.com/FeiyangHong/OpenWAM/blob/5be50682eb45ef5d2dace77cce88c3baa0c48446/benchmarks/libero/README.md)
- [fork 的 LIBERO-plus 接口](https://github.com/FeiyangHong/OpenWAM/blob/5be50682eb45ef5d2dace77cce88c3baa0c48446/benchmarks/libero-plus/README.md)
- [RoboTwin reader 配置](https://github.com/FeiyangHong/OpenWAM/blob/5be50682eb45ef5d2dace77cce88c3baa0c48446/configs/dataloader/robotwin.yaml)

上述版本与接口来自源码文档，本次未运行验证。本提案的实验分工、配对 manifest、分层分析和推进条件是研究设计建议。

8–16×A100 80GB 优先保证训练作业；policy server 与 render worker 单独调度，先验证实际节点的 Vulkan/EGL/驱动能力。必要时使用独立渲染节点，不把仿真环境依赖装进模型训练环境。首轮不下载全部 benchmark 数据，也不同时启动未测内存的多份 5B server。
