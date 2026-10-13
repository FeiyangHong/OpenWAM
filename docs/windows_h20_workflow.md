# Windows 编辑，原 H20 仓库执行

用户明确要求：本地编辑与原服务器仓库同步，所有运行都在
`h20:/gaozt-test1/fyhong/OpenWAM` 下进行，不建立第二份远端仓库。

| 用途 | 位置 |
| --- | --- |
| Windows 代码 | `F:\Embodied_AI\Projects\OpenWAM` |
| H20 同步和运行目录 | `/gaozt-test1/fyhong/OpenWAM` |
| H20 日志、同步记录和覆盖前备份 | 原仓库内 `.h20-local/`（Git 本地忽略） |
| Python 环境 | `/gaozt-test1/zhanxch/miniconda3/envs/openwam` |
| Windows 配置、导入备份和报告 | `.local-h20/`（Git 本地忽略） |

## 常用命令

Windows PowerShell 中进入仓库根目录：

```powershell
python scripts/dev/h20.py status
python scripts/dev/h20.py sync
python scripts/dev/h20.py run -- env CUDA_VISIBLE_DEVICES= python scripts/train.py dataloader=wuji_real_task --cfg job
```

`status` 比较两端相对上次成功同步的变化；首次使用导入基线。
`sync` 上传 Git 已跟踪文件和未被忽略的新文件，更新服务器原仓库的相应文件。
它不修改服务器 Git HEAD、分支或 index，也不自动创建 commit。

服务器出现本地尚未吸收的独立文件修改、增删，或者服务器 HEAD 变化时，同步失败并提示合并，
不会静默覆盖。先读取远端变更并在本地合并，必要时有针对性地更新同步基线，不要删除基线来绕过检查。
本地删除只应用于基线中已管理且服务器未独立改变的文件；数据、模型、日志、凭据不参与源代码同步。
每次覆盖或删除前，旧文件保存在 `.h20-local/backups/<时间-ID>/files/`，变更清单在 `before.json`。

`run` 检查本地与最后同步记录一致、远端仍为该版本，然后直接在原仓库执行。
同步和通过 helper 启动的任务共用文件锁：运行期间拒绝另一次同步，避免修改正在使用的代码。
该锁不约束手工 SSH 启动的其他任务或其他人的直接编辑；这类运行需要自行协调。
整个多文件同步不是原子事务；被外部中断时保留备份，重新检查两端状态后再恢复。

## 后台运行和日志

```powershell
python scripts/dev/h20.py run --background wuji-check -- env CUDA_VISIBLE_DEVICES= bash -c 'python scripts/dev/audit_wuji_data.py --output "$OPENWAM_RUN_DIR/audit.json"'
python scripts/dev/h20.py logs <返回的run_id>
```

后台使用 tmux，SSH 断开后继续运行。每个任务在原仓库 `.h20-local/runs/<run_id>/`
保存命令、文件哈希、环境路径、日志和退出码。`OPENWAM_RUN_DIR` 指向这个目录。
`exit_code=pending` 也可能表示任务被外部终止，并不保证仍在运行。
前台命令返回真实退出码。只有明确请求训练时才启动训练，并指定 GPU 和输出位置；当前适配仅做 CPU 验证。

输出路径、数据路径和权重路径的相对解析以原仓库为准，因此原有训练脚本可以继续使用。
需要隔离一次新实验的输出时，可设置 `OUTPUT_PATH="$OPENWAM_RUN_DIR/checkpoints"`。
恢复历史训练应显式指定原 run 目录，而不是把它当作新训练的输出目录。

## 环境与凭据

本地 helper 只依赖 Python 标准库、Git 和系统 OpenSSH，无需在 Windows 安装训练环境。
Git 保持 `core.autocrlf=false`、`core.eol=lf`、`core.filemode=false`；新脚本应以 LF 保存。
服务器已有文件权限保留，新文件采用 Git 模式，新 `.sh` 默认为可执行。

执行时设置环境的 PATH、CONDA_PREFIX 和指向原仓库的 PYTHONPATH。
helper 不执行完整 conda activate；如果其他环境有额外 activate hooks，应在命令里显式启用。
Codex 在 Windows 登录，SSH 禁用 agent forwarding；不复制个人登录缓存、私钥、`.env`、`.netrc` 或 W&B 凭据。

W&B 默认 disabled。需要接入已有服务器 W&B 时，在远端运行命令中显式设置：

```bash
export WANDB_MODE=online
export NETRC=/gaozt-test1/fyhong/OpenWAM/wandb/.netrc
export WANDB_CONFIG_DIR=/gaozt-test1/fyhong/OpenWAM/wandb/config
```

## 导入与验证记录

最初导入 HEAD 为 `0fb80b3951db9d0dca3f80c612cb238bba7e1bfb`，分支
`feat/statewam-r2-robotwin-pilot`；648 个源代码/文档文件共 14,636,947 字节，全部逐文件校验一致。
Git bundle、工作区归档、staged patch 和基线保存在本地 `.local-h20/initial-snapshot/`。
原有未提交修改和本地模型配置保留；未复制服务器 Git hooks 或配置，本地暂未设置 Git remote。
`third_party/cosmos-predict2.5` 保留原有未初始化 gitlink 状态。

此前独立目录中的历史 CPU 验证日志已迁入原仓库 `.h20-local/runs/`，旧 manifest 中的原执行路径仅为历史记录。
原来的额外目录已删除。之后不再创建独立源码快照；同步 ID 仅标识当前文件内容，用于核对和日志记录。

真机数据契约、既有 43 项测试和两份数据审计详见 [Astribot/Wuji 流程](astribot_wuji_pipeline.md)。
原仓库同步测试覆盖原地运行、文件备份、冲突拒绝、路径校验、运行锁和退出码。
