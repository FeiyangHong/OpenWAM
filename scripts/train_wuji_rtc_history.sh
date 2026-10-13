#!/usr/bin/env bash
# Manual launch only. Both groups use the same initialization and training budget.
set -euo pipefail
cd "$(dirname "$0")/.."
group="${1:?Usage: bash scripts/train_wuji_rtc_history.sh clean|history [--print-config]}"
shift
mode="${1:-train}"
if [[ "$mode" != train && "$mode" != --print-config ]] || (( $# > 1 )); then
    echo 'Only --print-config is accepted after the group.' >&2; exit 2
fi
case "$group" in
    clean) enabled=false; port=29651 ;;
    history) enabled=true; port=29652 ;;
    *) echo 'Group must be clean or history.' >&2; exit 2 ;;
esac
python_bin="${RTC_PYTHON:-/gaozt-test1/fyhong/envs/statewam/bin/python}"
checkpoint="${RTC_INIT_CKPT:-$PWD/outputs/grasp_anything_base/2026-09-16_21-50-22/checkpoint_step_18000.safetensors}"
if [[ "$checkpoint" != /* || "$checkpoint" != *.safetensors || ! -f "$checkpoint" ]]; then
    echo "Missing absolute checkpoint: $checkpoint" >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#gpu_ids[@]} != 8 )); then echo 'This experiment requires exactly eight visible GPUs.' >&2; exit 2; fi
output="$PWD/outputs/wuji_rtc_history/$group"
overrides=("training.finetune_ckpt_path=$checkpoint" "training.output_path=$output"
    "training.rtc_state_history.enabled=$enabled" "project.wandb.run_name=wuji_rtc_history_${group}_8gpu_bs32_ga2")
if [[ "$mode" == --print-config ]]; then
    exec "$python_bin" scripts/train.py --config-name train_wuji_rtc_history --cfg job --resolve "${overrides[@]}"
fi
export NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
export NETRC="${NETRC:-$PWD/wandb/.netrc}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/wandb/config}"
mkdir -p "$PWD/outputs/wuji_rtc_history"
exec 9>"$PWD/outputs/wuji_rtc_history/manual_training.lock"
flock -n 9 || { echo 'Another history/clean experiment is already running.' >&2; exit 2; }
mkdir -p "$output/launches"
stamp="$(date +%Y%m%d_%H%M%S)_$$"
sha256sum "$checkpoint" > "$output/launches/${stamp}_checkpoint.sha256"
git rev-parse HEAD > "$output/launches/${stamp}_git_head.txt"
git diff --binary HEAD > "$output/launches/${stamp}_git_diff.patch"
git rev-parse refs/heads/codex/rtc-state-history > "$output/launches/${stamp}_prepared_commit.txt"
sha256sum openwam/model/architectures/utils/rtc_state_history.py \
    openwam/model/architectures/base.py openwam/dataloader/wuji_real_task.py \
    openwam/train/openwam_trainer.py configs/train_wuji_rtc_history.yaml \
    configs/train_wuji_rtc_ab.yaml scripts/train_wuji_rtc_history.sh \
    > "$output/launches/${stamp}_sources.sha256"
"$python_bin" scripts/train.py --config-name train_wuji_rtc_history --cfg job --resolve "${overrides[@]}" > "$output/launches/${stamp}_config.yaml"
printf 'Group %s; GPUs %s; effective batch 512; history %s\n' "$group" "$CUDA_VISIBLE_DEVICES" "$enabled"
"$python_bin" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-$port}" \
    scripts/train.py --config-name train_wuji_rtc_history "${overrides[@]}" \
    2>&1 | tee "$output/launches/${stamp}.log"
