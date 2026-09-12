# StateWAM 文档索引与管理规范

本目录按研究设计、阶段和具体 Goal 管理文档。实验原始产物存放于仓库根目录的 `statewam_artifacts/`，不进入文档目录。

## 目录结构

```text
docs/statewam/
├── README.md                       # 总索引、管理规范、当前工作入口
├── design/                         # 跨阶段的研究设计
│   ├── README.md                   # 原始提案概览与背景
│   ├── 01_Architecture.md
│   ├── 02_Research_Plan.md
│   ├── 03_OpenWAM_Implementation_Map.md
│   └── 04_Simulation_Protocol.md
└── stages/
    ├── r1/
    │   └── robotwin-smoke/          # 已完成 Goal 的完整文档包
    │       ├── GOAL.md
    │       ├── REPORT.md
    │       └── USAGE.md
    └── r2/
        └── robotwin-pilot/          # 新 Goal，执行文档尚未形成
            └── GOAL.md
```

## 当前工作

| 阶段 / Goal | 状态 | 入口与交付 |
|---|---|---|
| R1 / robotwin-smoke | 工程验收完成：200 步 profiling、恢复到 201、基线/StateWAM 双条件 smoke | [GOAL](stages/r1/robotwin-smoke/GOAL.md)、[REPORT](stages/r1/robotwin-smoke/REPORT.md)、[USAGE](stages/r1/robotwin-smoke/USAGE.md) |
| R2 / robotwin-pilot | 计划已修订，待下达 Goal；下达后最多 4 卡连续完成准备、训练、评测和性能记账，无需二次预算确认 | [GOAL](stages/r2/robotwin-pilot/GOAL.md)；REPORT、USAGE 随执行交付 |

已完成 R1 工程验收，固化提交为 `98089d1fa612b7d1f0349ae68b3b8b0ef690aef7`。下一候选 Goal 是 R2 的 from-Foundation E0–E3 公平结构比较 pilot；目前只形成计划，没有启动 R2。完整 benchmark 和后续研究阶段不随文档更新自动执行。

当前提交分支：`feat/statewam-r1-robotwin`。R2 实现启动时再建立其独立分支，见对应 Goal。

## 研究设计入口

- [提案概览与背景](design/README.md)
- [结构与训练目标](design/01_Architecture.md)
- [R0–R5 研究计划](design/02_Research_Plan.md)
- [OpenWAM 实现映射](design/03_OpenWAM_Implementation_Map.md)
- [仿真实验协议](design/04_Simulation_Protocol.md)

## 管理规则

1. 顶层只保留总索引与管理规范；架构、全局研究计划、通用实现映射和协议放入 `design/`。
2. 每个执行 Goal 使用 `stages/<stage>/<goal-id>/` 独立目录，例如后续 R2 使用 `stages/r2/<goal-id>/`；同阶段的不同 Goal 也不能共用报告文件。
3. 每个 Goal 固定使用 `GOAL.md`、`REPORT.md`、`USAGE.md`。Goal 记录意图和验收，报告记录已发生的事实，usage 只给可执行的简洁步骤。附加材料确有必要时放在该 Goal 的子目录，并从报告链接。
4. Goal 在执行前创建；报告和 usage 随实现形成并在交付前验证。未运行的命令、未通过的检查和未来文件明确标注，不预先生成虚假的完成报告。
5. 在 Goal 中记录阶段、分支、日期、状态、变更原因和验收条件。范围更新在原 Goal 中维护变更记录；独立的新目标创建新目录，保留已完成目标的报告，不追加顶层全局编号文档。
6. 每次新增、完成或阻塞 Goal 时更新本索引。链接使用相对路径；移动文件后检查引用，不保留内容重复的旧路径副本。
7. 研究设计表达长期路线，当前执行以用户最新约束与对应 Goal 为准。局部执行范围缩减只记在 Goal 中；跨阶段假设或协议变更才更新设计文件并注明理由。
8. RoboTwin 的日志、checkpoint、视频、profile 和逐 episode 结果存放于 `statewam_artifacts/robotwin/`。报告记录 goal id、run id、相对路径、配置/代码版本和必要校验值，区分运行产物与文档。
9. 文档不保存凭据。示例命令通过环境变量引用认证信息。
10. Goal 的唯一标识使用 `<stage>/<goal-id>`，例如 `r1/robotwin-smoke`、`r2/robotwin-pilot`。报告、usage 和运行 manifest 均记录对应标识；同一 Goal 的重复运行使用不同 run id，不新建散落在顶层的报告。
11. 一个 Goal 的报告按运行记录追加，保留失败和重跑证据，不覆盖历史结论。跨 Goal 只通过相对链接引用前置成果，不复制整份报告或将新目标追加到旧 Goal 中。
12. 编写或修订 Goal 不等于启动执行；本次文档整理不启动新的训练、profiling 或测评。执行状态与通过结论必须由实际证据单独更新。
13. `GOAL.md` 是范围与验收条件的来源，`REPORT.md` 是执行状态与证据的来源，`USAGE.md` 是运行命令的来源。总索引仅汇总状态并链接原文，不复制详细进度；设计文档的历史状态不作为当前验收结论。
14. Goal 内的步骤 A/B/C 等属于同一目标的执行顺序，不为每个步骤另建一套散落文档。只有独立的目标才新建 `<goal-id>` 目录；完成后保留其报告和 usage，后续目标通过链接引用。

## 整理记录

2026-09-12：将原始提案文档整体移入 `design/`；将 `05_R1_RoboTwin_Goal.md` 移至 `stages/r1/robotwin-smoke/GOAL.md`，修正链接并恢复 4 卡 200–500 步 profiling。此次只整理文档，未启动实现或实验。

2026-09-12：补充 Goal 唯一标识、重复运行和跨 Goal 引用规则；保留工作区已有实现与执行记录，本次仅调整文档管理约定。

2026-09-12：用户明确下达 Goal 后，完成 `r1/robotwin-smoke` 的工程验收并交付报告和 usage；完整训练、全量评测及后续研究阶段仍未执行。

2026-09-12：按用户要求独立提交 R1；另建 `stages/r2/robotwin-pilot/GOAL.md`，补齐初始化、容量、数据/统计隔离、训练预算、配对指令/初态/RNG、统一推理配置及失败计数约束。本次只固化与规划，不启动 R2 实验。

2026-09-12：按用户要求取消 R2 的二次预算确认门槛。正式下达 Goal 后，在项目同时最多 4 卡的约束下自主调优、冻结公平训练预算并连续执行到交付；增加显存/利用率、吞吐/延迟、GPU-hours、失败重跑及存储等原始记录与汇总。本次仅修订文档，尚未启动 R2。

2026-09-12：在 R2 Goal 中补充独立任务与持久监控、对话中断边界、检查点与逐 episode 恢复、显式停止及资源释放要求，纳入 P2 验收和后续 USAGE。所有要求仍为待实现计划，本次未启动 R2。
