# R2 RoboTwin Pilot 执行报告

Goal ID：`r2/robotwin-pilot`。开始日期：2026-09-12。状态：执行中，尚未完成 pilot。范围见 [GOAL](GOAL.md)，已验证命令见 [USAGE](USAGE.md)。

## 版本与当前进度

R1 基点为 `98089d1fa612b7d1f0349ae68b3b8b0ef690aef7`；计划固化提交为 `368654c874cbbadc19fa18ec955e17982c7a0081`。已建立 `feat/statewam-r2-robotwin-pilot`，当前实现尚在验收。产物统一位于仓库根目录 `statewam_artifacts/robotwin/r2-robotwin-pilot/`，下文 run 路径均相对此目录。

| 阶段 | 当前事实 | 尚缺证据 |
|---|---|---|
| P0 | 数据准备、来源身份审计及加强 scene metadata 校验后的真实 reader 审计通过 | 无；P1 场景行为筛选是后续门槛 |
| P1 | 显式 allowlist、严格统计 provenance、当前信息容量对照已实现；相关单测通过 | 实模型容量/初始化审计、随机流、公共训练入口、固定场景与指令工具 |
| P2 | CPU 独立会话、日志采样、停止已实测 | GPU 任务管理、四卡恢复、评测恢复、200–500 步 profiling、正式预算冻结 |
| P3/P4 | 未执行 | 四组正式训练、开发/最终配对 rollout 和视频 |
| P5 | 执行报告与准备阶段 usage 已建立 | 完整比较结果、区间、性能总账与最终交付 |

当前不报告模型成功率、结构收益或收敛结论，不把 P0 完成当作 pilot 完成。首次检查 GPU 0–3 空闲，4–7 为其他任务；准备任务仅使用 CPU，不占用训练 GPU。

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
