#!/usr/bin/env bash
# A/B: initialization comparison; C/D: paired continuation without/with XYZ bias.
set -euo pipefail
cd "$(dirname "$0")/.."
group="${1:?Usage: bash scripts/train_wuji_rtc_ablation.sh A|B|C|D [--print-config]}"
shift
mode="${1:-train}"
if [[ "$mode" != train && "$mode" != --print-config ]] || (( $# > 1 )); then
    echo 'Only --print-config is accepted after the group.' >&2; exit 2
fi
python_bin="${RTC_PYTHON:-/gaozt-test1/fyhong/envs/statewam/bin/python}"
case "$group" in
    A) checkpoint="$PWD/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors"
       default_gpus=0,1,2,3; port=29641; epochs=20; lr=5e-5; aug=false; label=A_foundation ;;
    B) checkpoint="$PWD/outputs/grasp_anything_base/2026-09-16_21-50-22/checkpoint_step_18000.safetensors"
       default_gpus=4,5,6,7; port=29642; epochs=20; lr=5e-5; aug=false; label=B_sync ;;
    C|D)
       checkpoint="${RTC_INIT_CKPT:?Set RTC_INIT_CKPT to the SAME selected absolute .safetensors file for C and D}"
       epochs=5; lr=1e-5
       if [[ "$group" == C ]]; then default_gpus=0,1,2,3; port=29643; aug=false; label=C_clean
       else default_gpus=4,5,6,7; port=29644; aug=true; label=D_state_aug; fi ;;
    *) echo 'Group must be A, B, C or D.' >&2; exit 2 ;;
esac
if [[ "$checkpoint" != /* || "$checkpoint" != *.safetensors || ! -f "$checkpoint" ]]; then
    echo "Missing absolute checkpoint file: $checkpoint" >&2; exit 2
fi
output="$PWD/outputs/wuji_rtc_ablation/$label"
overrides=(
    "training.finetune_ckpt_path=$checkpoint" training.resume_ckpt_path=null
    "training.output_path=$output" "training.num_epochs=$epochs" training.max_steps=null
    "training.learning_rate=$lr" "training.rtc_state_augmentation.enabled=$aug"
    "project.wandb.run_name=wuji_rtc_$label"
)
if [[ "$mode" == --print-config ]]; then
    exec "$python_bin" scripts/train.py --config-name train_wuji_rtc_ab --cfg job --resolve "${overrides[@]}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$default_gpus}"
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#gpu_ids[@]} != 4 )); then echo 'This experiment requires exactly four visible GPUs.' >&2; exit 2; fi
export NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"
export NETRC="${NETRC:-$PWD/wandb/.netrc}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/wandb/config}"
mkdir -p "$output/launches"
launch_stamp="$(date +%Y%m%d_%H%M%S)_$$"
sha256sum "$checkpoint" > "$output/launches/${launch_stamp}_checkpoint.sha256"
"$python_bin" scripts/train.py --config-name train_wuji_rtc_ab --cfg job --resolve "${overrides[@]}" > "$output/launches/${launch_stamp}_config.yaml"
printf 'Group %s; GPUs %s; port %s; augmentation %s\n' "$group" "$CUDA_VISIBLE_DEVICES" "${MASTER_PORT:-$port}" "$aug"
"$python_bin" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-$port}" \
    scripts/train.py --config-name train_wuji_rtc_ab "${overrides[@]}" \
    2>&1 | tee "$output/launches/${launch_stamp}.log"
