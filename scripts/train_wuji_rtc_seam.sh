#!/usr/bin/env bash
# Manual launch only; eight GPUs per group. No GPU processes are stopped here.
set -euo pipefail
cd "$(dirname "$0")/.."
group="${1:?Usage: bash scripts/train_wuji_rtc_seam.sh baseline|seam [--print-config]}"
shift
mode="${1:-train}"
if [[ "$mode" != train && "$mode" != --print-config ]] || (( $# > 1 )); then
    echo 'Only --print-config is accepted after the group.' >&2; exit 2
fi
case "$group" in
    baseline) enabled=false; port=29661 ;;
    seam) enabled=true; port=29662 ;;
    *) echo 'Group must be baseline or seam.' >&2; exit 2 ;;
esac
python_bin="${RTC_PYTHON:-/gaozt-test1/fyhong/envs/statewam/bin/python}"
checkpoint="${RTC_INIT_CKPT:-$PWD/outputs/grasp_anything_base/2026-09-16_21-50-22/checkpoint_step_18000.safetensors}"
[[ "$checkpoint" == /* && "$checkpoint" == *.safetensors ]] || { echo "Initialization must be an absolute safetensors path." >&2; exit 2; }
init_name="$(basename "$checkpoint" .safetensors)"
for file in "$checkpoint" "$(dirname "$checkpoint")/config.yaml" "$(dirname "$checkpoint")/normalization_stats.npy"; do
    [[ -f "$file" ]] || { echo "Missing initialization file: $file" >&2; exit 2; }
done
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#gpu_ids[@]} != 8 )); then echo 'This experiment requires exactly eight visible GPUs.' >&2; exit 2; fi
declare -A seen=()
for gpu in "${gpu_ids[@]}"; do
    [[ -n "$gpu" && -z "${seen[$gpu]:-}" ]] || { echo 'GPU IDs must be nonempty and distinct.' >&2; exit 2; }
    seen[$gpu]=1
done
output="$PWD/outputs/wuji_rtc_seam/$group"
overrides=("training.finetune_ckpt_path=$checkpoint" "training.output_path=$output" "training.rtc_seam_loss.enabled=$enabled"
    "project.wandb.run_name=wuji_rtc_seam_${group}_${init_name}_8gpu_bs32_ga2")
if [[ "$mode" == --print-config ]]; then
    exec "$python_bin" scripts/train.py --config-name train_wuji_rtc_seam --cfg job --resolve "${overrides[@]}"
fi
export NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
export NETRC="${NETRC:-$PWD/wandb/.netrc}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/wandb/config}"
# Share the existing history lock to prevent accidental overlap of these 8-GPU jobs.
mkdir -p "$PWD/outputs/wuji_rtc_history"
exec 9>"$PWD/outputs/wuji_rtc_history/manual_training.lock"
flock -n 9 || { echo 'Another history/seam eight-GPU experiment is running.' >&2; exit 2; }
mkdir -p "$output/launches"
stamp="$(date +%Y%m%d_%H%M%S)_$$"
sha256sum "$checkpoint" "$(dirname "$checkpoint")/config.yaml" \
    "$(dirname "$checkpoint")/normalization_stats.npy" > "$output/launches/${stamp}_initialization.sha256"
git rev-parse HEAD > "$output/launches/${stamp}_git_head.txt"
git diff --binary HEAD > "$output/launches/${stamp}_git_diff.patch"
git rev-parse refs/heads/codex/rtc-seam-loss-20261012 > "$output/launches/${stamp}_prepared_commit.txt"
sources=(openwam/model/architectures/utils/{training_rtc,rtc_state_history,rtc_seam_loss}.py
    openwam/model/architectures/base.py openwam/train/openwam_trainer.py
    openwam/dataloader/wuji_real_task.py configs/train_wuji_rtc_{ab,history,seam}.yaml
    scripts/train_wuji_rtc_seam.sh)
sha256sum "${sources[@]}" > "$output/launches/${stamp}_sources.sha256"
tar -czf "$output/launches/${stamp}_sources.tar.gz" "${sources[@]}"
"$python_bin" scripts/train.py --config-name train_wuji_rtc_seam --cfg job --resolve "${overrides[@]}" > "$output/launches/${stamp}_config.yaml"
printf 'Group %s; GPUs %s; effective batch 512; seam weighting %s\n' "$group" "$CUDA_VISIBLE_DEVICES" "$enabled"
"$python_bin" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-$port}" \
    scripts/train.py --config-name train_wuji_rtc_seam "${overrides[@]}" \
    2>&1 | tee "$output/launches/${stamp}.log"
