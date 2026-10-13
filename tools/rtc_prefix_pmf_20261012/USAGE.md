> 功能已应用。后续请使用 [正式长训 usage](../../docs/wuji_rtc_prefix_pmf_usage.md)，无需再次应用补丁。以下保留准备阶段说明。

# 前缀概率采样：训练结束后手动启用

本目录是待应用功能包。当前运行目录的源码、配置和训练进程不修改；这里没有自动启动或定时切换任务。

## 配置

| 前缀长度 | 新概率 |
|---|---:|
| 0 | 10% |
| 1～7 | 合计10%，各约1.4286% |
| 8 | 15% |
| 9 | 15% |
| 10 | 35% |
| 11 | 10% |
| 12 | 5% |

这是待验证的部署区域重点采样方案，不是 GR00T 原配方，也不是部署延迟自适应功能。启用后整张概率表替代旧的“空前缀25%＋1～12步均匀”分布；不会再叠加一次25%空前缀。短有效窗口仍沿用原逻辑截短前缀，保留至少一个有效监督动作，因此实际频率可能偏离名义分布。

默认与当前 B 对照：**同步18000初始化、10 epoch、8卡、每卡batch32、累积2、有效batch512、LR 1e-5、cosine、warmup 5%、bf16、ZeRO-2、seed42**。保留 history 增强30%（非空前缀条件下，过去1～3帧完整state、不限幅）和首3步权重2；使用绝对EEF。每500 micro-step保存，保留12个checkpoint。

仅改变前缀分布，不从本轮的新权重追加训练。空前缀占比变小后，全样本中的history触发名义比例会从22.5%变为27%，条件概率仍是30%。这也是分布变化的结果。

## 现在可检查（不应用、不启动）

```bash
cd /gaozt-test1/fyhong/OpenWAM
/gaozt-test1/fyhong/envs/statewam/bin/python tools/rtc_prefix_pmf_20261012/activate.py --check
```

训练仍运行时，`training_lock_busy: true` 或 `training_pids` 非空是正常现象。

## 当前训练结束后：应用并查看配置

```bash
cd /gaozt-test1/fyhong/OpenWAM
/gaozt-test1/fyhong/envs/statewam/bin/python tools/rtc_prefix_pmf_20261012/activate.py --apply
bash scripts/train_wuji_rtc_prefix_pmf.sh seam --print-config
```

应用脚本检查训练锁、同仓库训练进程、源码和配置哈希；训练未结束或源码已被其他会话修改时拒绝应用，不覆盖冲突。它会额外保存现有 Git 状态并提交本功能，不切换当前分支、不改原暂存区。**应用不等于启动训练。**

确认初始化为同步 `checkpoint_step_18000.safetensors`，`rtc_prefix_sampling.enabled=true`，13个概率如上，`rtc_state_history.enabled=true`，`rtc_seam_loss.enabled=true`。

## 手动长训（建议 tmux）

```bash
cd /gaozt-test1/fyhong/OpenWAM
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=online \
  RTC_INIT_CKPT=/gaozt-test1/fyhong/OpenWAM/outputs/grasp_anything_base/2026-09-16_21-50-22/checkpoint_step_18000.safetensors \
  bash scripts/train_wuji_rtc_prefix_pmf.sh seam
```

输出：`outputs/wuji_rtc_prefix_pmf/seam/日期时间/`。启动配置、源码快照、初始化校验和与日志在该组 `launches/`。W&B项目和登录方式沿用当前训练，run名包含 `prefix_pmf`。脚本不停止其他GPU进程，只需自行确认显存足够。

若本轮证明接缝加权无收益，后续可以把上述组名 `seam` 改为 `baseline`：开启同一PMF但关闭接缝加权，输出到 `outputs/wuji_rtc_prefix_pmf/baseline/`。此时应与旧的不加权history实验对照，不再宣称与当前B仅差一项。

## 对照及恢复

比较对应训练进度，如新step4500与本轮B的step4500。检查接缝首3步回拉和速度、整段动作、双手、空前缀和任务成功；训练loss不能替代真机结论。

原训练入口和旧checkpoint仍可用；未指定 `training.rtc_prefix_sampling` 或设置 `enabled=false` 时保留原采样代码及RNG调用。概率配置单独保存在 `training.rtc_prefix_sampling`，不修改部署读取的 `training.rtc` 契约或网络权重结构。

同一实验恢复时必须保持PMF及其他训练配置不变；换概率需用权重初始化新实验。只含权重的safetensors不是完整优化器断点。重复长训命令会从同步18000新开实验。

## 已验证

隔离验证副本中104项CPU测试通过，含旧RTC/history/seam/trainer回归、采样频率和RNG恢复、应用时的锁/冲突保护。两组启动脚本及完整配置解析通过；新的seam配置除PMF和输出命名外，与当前B一致。准备前后对现有源码逐文件做SHA256核对，未发生变化。记录在 `outputs/rtc_prefix_pmf_prepare_20261012/`。尚未应用到正式源码，也未启动新训练；完整8卡运行及W&B联网留待手动启动时验证。
