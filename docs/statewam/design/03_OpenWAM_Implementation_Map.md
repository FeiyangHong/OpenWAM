# StateWAM v2：用户 OpenWAM fork 接口核对与实现映射

日期：2026-09-12\
主开发来源：`https://github.com/FeiyangHong/OpenWAM`\
核对 HEAD：`5be50682eb45ef5d2dace77cce88c3baa0c48446`\
上游对照：`d8dd33d8576b475f5a5cdc6fb8ca902778a199b3`\
用户 fork 研究副本：`F:/Embodied_AI/Projects/StateWAM/OpenWAM-FeiyangHong-reference`\
前次 `StateWAM-v2/` 仅是官方上游参考副本，不作为重新移植用户适配的起点。\
本文件是后续实现规格；下文“拟新增”文件目前不存在，模型代码本次未修改。

## 0. fork 的实际差异与继承规则

本次核对提交链：上游 `d8dd33d` → `3274453`（Astribot/Wuji adapter）→ `d9828b1`（预处理用法）→ `5be5068`（合并）。对上述两个端点的 diff 显示 14 个文件有变化；核心 `openwam/model/architectures/`、训练 checkpoint loader/optimizer groups、`openwam/deploy/` 以及 RoboTwin/LIBERO/LIBERO-plus 目录没有差异。此前核心架构接口分析仍适用，项目底座改为用户 fork。

当前 HEAD 的源码能力不等于历史成功 run 一定使用此提交；把两者分别记录。用户已经用自有真机数据训练 Foundation 与 Wuji 初始化并获得良好效果，这项事实来自用户，不能从仓库 README 推断具体成功率。

### 0.1 已存在的本体/数据适配，直接保留

| 已有文件 | 核对内容 | 新版处理 |
|---|---|---|
| `openwam/dataloader/wuji_real_task.py` | Wuji/Astribot 58D reader、三视角、任务文本、统计、类型与维度检查 | 保留为真机主路径 |
| `openwam/dataloader/utils/rot6d.py` | 行 rot6d→列 rot6d、58D block 重排 | 不重复转换 |
| `scripts/prepare_grasp_anything_openwam.py` | 离线转换、视频链接/复制、元数据/统计/manifest 输出 | 保留训练用途；研究 split 需补充审计 |
| `configs/dataloader/wuji_real_task.yaml` | 58D→80D 映射、三视角、quantile normalization | 按实际成功 run 继承 |
| `scripts/train_grasp_anything_wuji.sh` | 把训练环境参数转为现有 trainer 的 overrides | 作为真机启动入口；新增架构应接通该入口 |
| `tests/dataloader/test_wuji_real_task.py` | rotation/layout/unify 回环测试 | 保留并扩展，不将其当作完整真机 reader 验证 |

原始数据排列为 `[L EEF9, R EEF9, L hand20, R hand20]`，转换后 raw 排列为 `[L EEF9, L hand20, R EEF9, R hand20]`。EEF 含位置3+rot6d6；转换后使用旋转矩阵前两列。配置映射 `0–8,10–29,34–42,44–63` 到 80D 模型空间；未映射槽不应产生动作监督。raw 58D、model 80D 与新增 State latent 512D 是三个不同概念。

fork 配置的 `num_frames=33` 是 action/state window，动作 horizon 为 32 个原始采样步；`video_stride=4` 只下采样视频。不能据此推断固定物理秒数，须读取实际采样频率与时间戳。三视角为 head/left wrist/right wrist；适配文档中的样例数据量或 H20 参数不是用户当前所有数据规模或 A100 配方。

### 0.2 研究 split 的一个必要补充

已读转换器会把输入 bucket 的全部 episodes 写成 `splits.train`，并从转换后的全部 action/state 行计算归一化统计。这适用于把一个训练 bucket 完整转成训练数据；它本身不构造研究用 train/validation/test。

因此新实验要先冻结 episode/session 划分，再独立处理统计：只用训练部分拟合归一化，validation/test 复用该训练统计。不能先混合全部数据转换并计算统计，再仅修改 split 标签，就宣称严格独立测试。若历史成功 run 的 bucket 本来全是训练数据，这不构成对已有结果的否定；只需为新研究建立正确的持出数据契约。转换器仍保持原样，本次未执行数据转换。

## 1. 已核对的原生接口

| 现有位置 | 已核对行为 | 对 StateWAM 的含义 |
|---|---|---|
| `openwam/model/architectures/dual_system/joint_self_attn.py` | 组装 World/Action，forward 返回 video/action noise prediction | 保留基础骨干和外部调用契约 |
| `openwam/model/architectures/dual_system/mot_driver.py` | 每层 pre-attn→mixed-attn→post-attn；driver 不拥有参数 | State 参数必须挂到 architecture 的 nn.Module 子模块 |
| `openwam/model/architectures/utils/mask_modes.py` | isolated/video_sees_action 时，动作仍可读取第一帧 | 可以保留当前视觉，单独研究未来通路 |
| `openwam/model/architectures/tri_system/joint_self_attn.py` | 架构级 UnderstandingExpert 的 dtype/device、训练参数与生命周期范例 | 可参考模块管理，不能直接把只读理解流改名为预测状态 |
| `openwam/model/architectures/base.py` | prepare_inputs、compute_loss、checkpoint、freeze 与 generate | State 接入须同时考虑训练、重建与推理 |
| `openwam/model/architectures/registry.py` | framework/variant 映射，experimental 需要显式允许 | 新实验架构须同时接通 trainer/deploy 构建入口 |
| `openwam/train/utils/optimizer_groups.py` | 顶层可训练子模块被收集；非 world/action 模块使用基础 LR | State 可自然进入 optimizer，但应增加分组审计 |
| `openwam/train/utils/ckpt_model_loader.py` | checkpoint/live config 合并，限制不兼容架构身份 | 不能把更换架构当普通 finetune override |
| `openwam/deploy/engine.py` | 调用 architecture.generate，管理 compile/cache 等优化 | 先 eager 验证，再逐项恢复优化 |
| `openwam/dataloader/bases/` 与 benchmark adapter | 视频/action/proprio/mask 数据契约 | 复用，但须核对具体机械臂/灵巧手映射 |

### 1.1 可复核的源码链接

- [双系统架构](https://github.com/FeiyangHong/OpenWAM/blob/5be50682eb45ef5d2dace77cce88c3baa0c48446/openwam/model/architectures/dual_system/joint_self_attn.py)
- [MoT driver](https://github.com/FeiyangHong/OpenWAM/blob/5be50682eb45ef5d2dace77cce88c3baa0c48446/openwam/model/architectures/dual_system/mot_driver.py)
- [attention mask](https://github.com/FeiyangHong/OpenWAM/blob/5be50682eb45ef5d2dace77cce88c3baa0c48446/openwam/model/architectures/utils/mask_modes.py)
- [BaseWAMArchitecture](https://github.com/FeiyangHong/OpenWAM/blob/5be50682eb45ef5d2dace77cce88c3baa0c48446/openwam/model/architectures/base.py)
- [checkpoint 配置与加载](https://github.com/FeiyangHong/OpenWAM/blob/5be50682eb45ef5d2dace77cce88c3baa0c48446/openwam/train/utils/ckpt_model_loader.py)
- [Wuji 训练用法与两类初始化](https://github.com/FeiyangHong/OpenWAM/blob/5be50682eb45ef5d2dace77cce88c3baa0c48446/docs/grasp_anything_wuji_usage.md)
- [数据转换器](https://github.com/FeiyangHong/OpenWAM/blob/5be50682eb45ef5d2dace77cce88c3baa0c48446/scripts/prepare_grasp_anything_openwam.py)

## 2. 建议的代码组织

以下均为拟新增/修改计划，不是现有功能列表。

| 拟新增位置 | 职责 | 首期 |
|---|---|---|
| `openwam/model/architectures/statewam/architecture.py` | StateWAM 架构注册、生命周期、基线兼容 forward | 是 |
| `openwam/model/architectures/statewam/state_expert.py` | 当前/未来/动作条件读取器、状态→动作 writer、gate | 是 |
| `openwam/model/architectures/statewam/mot_driver.py` | 保持原 world/action 分层契约，插入 State read/write | 是 |
| `openwam/model/architectures/statewam/masks.py` | future-only bottleneck 和无旁路检查 | 第二期；首期仅定义测试 |
| `openwam/model/architectures/statewam/targets.py` | 有效未来语义目标与 optional 物理目标 | 第二期 |
| `configs/model/statewam.yaml` | 新架构参数，继承原骨干配置 | 是 |
| `scripts/statewam/initialize_checkpoint.py` | 受控的原 OpenWAM→新架构初始化 | 是 |
| `scripts/statewam/profile.py` | 真实 batch 显存/吞吐/推理测量 | 是 |
| `tests/statewam/` | 新模块与旧路径等价、梯度、mask、保存/加载/推理 | 是 |

第一版保持原双系统实现为不变对照，优先通过新架构类和新 driver 接入。若需扩展共享 lifecycle，只加入明确必要的 hook，并为原配置保持默认行为。

后续代码分支应从用户 fork 的选定提交建立，已有 `wuji_real_task` 与数据处理代码不重写。仿真继续使用 fork 自带 benchmark adapters；不能让 StateWAM 模型类导入某个特定真机 reader。

## 3. 首版配置草案

**这是规格草案，当前仓库不能直接执行以下配置。** 实现时必须增加 registry、构建入口和初始化工具。

```yaml
# Illustrative only: new keys are not implemented yet.
architecture:
  framework: statewam
  variant: interaction
  attention_mask_mode: mutual
  video_attention_mask_mode: first_frame_causal
  bridge_interval: 1
  interaction_state:
    mode: residual
    num_tokens: 16
    dim: 512
    state_bridge_layers: [4, 9, 14, 19, 24, 29]
    source: future_action  # comparison: current / future / future_action
    gate_init: 0.0
    cache_across_denoise_steps: false
    semantic_loss_weight: 0.0
```

`architecture.state_dim` 在原仓库表示 proprioception 宽度，不能改成 512 充当 State latent 维度。`interaction_state.dim` 必须独立命名。`action_dim`、proprio 宽度和统一映射继承真实 checkpoint/data contract，不从此草案推断。

真机 Wuji adapter 使用 raw 58D→model 80D；RoboTwin 的已核对 EEF reader 使用 raw 20D，joint reader 是另一种 14D 契约。共享模型接口不意味着 raw action 可以通用，仿真与真机分别保持正确的归一化与 gather/scatter。

## 4. 初始化与 checkpoint

原 `finetune_ckpt_path` 不是任意跨架构加载接口。当前 loader 检查包括 `architecture.framework` 在内的身份差异；即使绕过配置检查，严格 state_dict 加载也会发现新增 State 参数缺失。

建议单独实现一次性的初始化转换：

1. 读取原 checkpoint/config/组件 sidecar，保留来源 SHA 与文件校验值。
2. 用同一 world/action/proprio 拓扑建立新架构，再实例化 State 模块。
3. 对原模块要求参数名、shape 与 dtype 规则兼容；不接受未解释的 missing/unexpected key。
4. 只允许新增 `interaction_state.*` 与明确 gate 参数缺失并按规定初始化。
5. 确认所有 meta tensor 已实体化，State 参数包含在 optimizer 与保存清单。
6. 保存新架构自包含 checkpoint/config，附完整加载报告。
7. 从该新 checkpoint 进行普通 finetune/resume；deploy 对新模型重新使用严格加载。

不建议全局改成 `strict=False`，也不建议在每次部署都自动容忍 State 参数缺失。原 checkpoint 与新 checkpoint 应存放在不同输出目录。

上述初始化工具需分别验证 Foundation 与 Wuji 源 checkpoint，而不是只测试其中之一。两类源权重携带的数据配置或统计可能不同；新 run 必须使用目标数据契约并写到独立目录。Foundation/Wuji 身份、已训练输出与其数据曝光均写入 manifest，原始发布权重配置不原地改写。

## 5. 参数冻结与梯度

原 `freeze_modules` 除了关闭参数梯度，还递归把子模块 forward 包在 `torch.no_grad` 中。对冻结 VAE/文本教师合理，但对“固定权重、仍要让输入梯度穿过”的 world/action 网络可能不适用。

新版应区分：

- **inference-only frozen encoder**：允许 no_grad，如训练期语义教师。
- **parameter-frozen differentiable module**：权重不更新，但保留对输入的 autograd，如 State warm-up 中的固定动作处理层。

如果 State 输出经过 no_grad 包裹的动作后半层，loss 无法训练 State。测试必须验证真实接口中的梯度，而不只检查 `requires_grad`。若需要新增参数冻结机制，必须显式实现并记录，不能假定现有 freeze list 已支持。

对于 gate=0 初始化，首步 reader 内部零梯度是预期现象；gate 应先收到梯度，随后 reader/writer 获得梯度。若多步后仍全零，停止训练排查。

## 6. Driver、形状与 checkpoint 重计算

原 driver 依赖 world/action 层数与 attention head 几何匹配；Wan residual width 与 ActionDiT residual width 可以不同，但映射后共享 attention 几何必须一致。

State 的内部宽度和头数可以独立，writer 最终需映射到明确的动作 residual 或共享 attention 输出空间。首版建议插入动作 pre/post attention 之间，保持原 block suffix 合约；不要在未匹配维度的情况下直接相加。

原逐层 checkpoint 使用浅拷贝隔离 mutable hidden fields。新增 State 不得在 closure 内把 tensors 追加到共享 `extras` 或列表，避免 backward 重算污染外部状态。首版采用调用内局部变量，诊断输出以显式返回/受控采集实现；不要让日志 tensor 长期保留完整 autograd graph。

新增 query/source mask 与目标 loss mask 分开处理。原 attention 未直接消费全部 sample padding 信息，State 扩展不应假定其已自动排除 padded future；应显式处理新增目标和 State 来源有效性，并保持旧模型对照的一致性。

## 7. Loss 接口与 Human-only 分支

首版只用原 video/action loss，可保持公共 forward 返回 `(video_noise, action_noise)`。

语义目标阶段需要显式扩展训练上下文/内部 forward 输出，确保 auxiliary loss 获得有梯度的 State。`compute_loss` 内部负责噪声采样；应只采样一次并共享到各目标，避免 State loss 与 action loss 使用不同隐含条件。

推荐公共 forward 保持兼容，内部 `_forward_with_state` 返回结构化临时结果，StateWAM 的 loss 路径直接消费；不能把 `self.last_state` 当作跨 batch 的隐式训练接口。该 hook 尚未实现。

原无 noisy actions 分支可能只运行 video loop 然后返回；要训练 Human-only State，必须显式增加状态读取/目标路径，并测试全 false action mask 不产生伪动作监督。

## 8. 部署、compile 与 cache

- 沿用原 observation→normalization→generate→denormalization→executor 链。
- State 不接收独立外部 future 参数，不向机器人客户端暴露内部实现字段。
- 先用 eager + 固定种子比较 train/deploy 与保存前后结果，再测试 torch.compile。
- 初期禁用近似 DiT velocity cache。State 路径改变后，旧 cache 的跳步误差需要重新验证；不能自动继承基线 skip rate 或误差结论。
- State 在每个生成调用重置，不缓存跨 episode 的交互状态。若以后添加缓存，prompt、camera、proprio、episode reset 都必须进入有效性设计。
- 同步去噪与同步执行器分别配置，首轮保持基线执行语义。
- 延迟拆分记录图像传输/预处理、编码、去噪、State 开销、后处理、控制等待；同时报告完整时间。

## 9. 下一阶段代码验收清单

| 测试 | 所要排除的问题 |
|---|---|
| baseline/off/gate-zero 数值一致 | 新模块意外改变原输出 |
| checkpoint round-trip 和受控初始化 | 新参数遗漏、旧参数错配、meta tensor 未加载 |
| 多步梯度检查 | gate 或冻结路径阻断 State 学习 |
| current-only 来源审计 | 从动作/污染 current token 偷读未来 |
| bottleneck 全层信息路径检查 | 未来信息绕过 State |
| masked/padded/all-invalid 样本 | 无效目标、NaN 或伪动作监督 |
| 独立噪声/同步推理一致性 | 错把训练和部署 schedule 混为一谈 |
| train/eval/compile 对照 | 生命周期或编译使 State 路径失效 |
| 8 卡保存/重载与优化器分组 | 分布式参数未更新或未保存 |
| Foundation/Wuji 两种源权重初始化 | 只在一种 checkpoint 拓扑/sidecar 下有效 |
| fork 58D rotation/layout/unify 回环 | 重排、旋转约定或模型/raw 维度混淆 |
| 仿真 seed/reset/state manifest | 配对比较的初始状态实际不同 |
| 各 benchmark adapter 的端到端 rollout | 模型相同但控制频率/动作语义不同 |
| 真机短 rollout | 形状测试通过但物理动作契约错误 |

CPU/小张量测试不替代 A100 上真实权重训练，也不替代真机控制验证。本次只完成设计与源码核对，以上测试尚未执行。

## 10. 仿真接口与协议边界

fork 保留了 `benchmarks/robotwin/`、`benchmarks/libero/`、`benchmarks/libero-plus/` 的 server/client 分离评估接口。首轮不另造模拟器，也不要求将真实手臂和手全部导入仿真。根据已有环境安装情况，优先复用 RoboTwin 主比较；环境资源与驱动可用性须 smoke 验证。

仿真 State 模块与真机 State 模块共享实现；各自使用自己的 reader、动作映射和 executor。正式实验中固定并保存 simulator commit、环境配置、物体资产、reset 初态、策略采样种子和模型权重。

新增的配对 snapshot 管理、细粒度扰动和 State probe 日志是计划内容，不能仅因 benchmark 已有运行脚本就声称这些控制已经实现。详细规格见 [仿真实验执行规格](04_Simulation_Protocol.md)。
