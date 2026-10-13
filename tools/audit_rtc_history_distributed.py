"""Bounded 8-GPU integration audit; tiny model, real numeric samples, synthetic images.

Does not load the production checkpoint or create a W&B run.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import _build_accelerator
from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan21
from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
from openwam.model.architectures.utils.training_rtc import configure_training_rtc
from openwam.model.architectures.utils.rtc_state_history import configure_rtc_state_history
from openwam.train.openwam_trainer import OpenWAMTrainer
from openwam.train.utils.training_utils import build_cosine_scheduler
from openwam.train.utils.checkpointing import save_full_state, load_full_state, save_weights
from openwam.train.utils.seeding import per_step_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.merge(OmegaConf.load(root/'configs/train_wuji_rtc_ab.yaml'),
                          OmegaConf.load(root/'configs/train_wuji_rtc_history.yaml'))
    cfg.training.rtc_state_history.enabled = True
    torch.set_num_threads(2)
    torch.manual_seed(42)
    acc = _build_accelerator(cfg)
    assert acc.num_processes == 8
    dataset = WujiRealTaskDataset.from_config(cfg.dataloader)
    image = Image.new('RGB', (16, 16))
    def decode(*args, **kwargs):
        frames = [image.copy() for _ in range(9)]
        return (frames, None) if kwargs.get('return_missing_masks') else frames
    dataset._decode_window_video = decode
    class BoundedDataset(Dataset):
        def __len__(self): return 32 * 8 * 6
        def __getitem__(self, index): return dataset[(index * 997) % len(dataset)]
    arch = DualSystemSelfAttnArchitecture(dict(
        video_dim=96, dim=32, ffn_dim=64, num_heads=4, attn_head_dim=24,
        num_dit_layers=2, bridge_interval=1, text_dim=32, action_dim=80,
        state_dim=80, use_proprioception=True, attention_mask_mode='mutual',
        video_attention_mask_mode='first_frame_causal', mot_checkpoint_mixed_attn=True))
    dit = WanModel(dim=96, in_dim=4, ffn_dim=128, out_dim=4, text_dim=32,
        freq_dim=32, eps=1e-6, patch_size=(1,2,2), num_heads=4, num_layers=2,
        has_image_input=False, seperated_timestep=True, fuse_vae_embedding_in_latents=True)
    arch.video_backbone = Wan21(SimpleNamespace(dit=dit, vace=None, motion_controller=None,
                                               scheduler=FlowMatchScheduler()))
    arch.build_mot_driver()
    arch.set_dtype_device(torch.bfloat16, acc.device)
    arch.init_training_schedulers(1000)
    arch.set_training_runtime(use_gradient_checkpointing=True)
    configure_training_rtc(arch, cfg.training.rtc)
    configure_rtc_state_history(arch, cfg.training.rtc_state_history, dataset=dataset)
    def preprocess(**kwargs):
        n = len(kwargs['frames'])
        def zeros(*shape): return torch.zeros(*shape, dtype=arch.dtype, device=arch.device)
        return dict(input_latents=zeros(n,4,3,4,4), first_frame_latents=zeros(n,4,1,4,4),
                    context=zeros(n,4,32), context_mask=torch.ones(n,4,dtype=torch.bool,device=arch.device))
    arch.preprocess = preprocess
    opt = torch.optim.AdamW(arch.parameters(), lr=1e-5, betas=(.9,.95), weight_decay=.01)
    sched = build_cosine_scheduler(opt, total_opt_steps=2250, cfg=cfg, num_processes=8)
    loader = DataLoader(BoundedDataset(), batch_size=32, shuffle=False, collate_fn=list, num_workers=0)
    engine, opt, loader, sched = acc.prepare(arch, opt, loader, sched)
    trainer = object.__new__(OpenWAMTrainer)
    trainer.architecture = engine
    trainer.lambda_video = trainer.lambda_action = 1.
    batches = list(loader)
    assert len(batches) == 6
    assert batches[0][0]['rtc_history_states'].device.type == 'cuda'
    initial = {k: v.detach().clone() for k,v in arch.state_dict().items()}
    records = []
    def step(i):
        seed = per_step_seed(42, rank=acc.process_index, step=i)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with acc.accumulate(engine):
            losses = trainer.compute_loss(batches[i])
            loss = losses['total']
            assert torch.isfinite(loss)
            acc.backward(loss)
            if acc.sync_gradients:
                acc.clip_grad_norm_(list(engine.parameters()), 1.)
                opt.step(); sched.step(); opt.zero_grad()
        stats = arch.rtc_state_history_metrics
        names = sorted(stats)
        reduced = acc.reduce(torch.tensor([stats[k] for k in names],device=acc.device),reduction='mean')
        return dict(microstep=i+1, loss=float(loss.detach()), lr=opt.param_groups[0]['lr'],
                    stats=dict(zip(names,reduced.tolist())), scheduler_epoch=sched.scheduler.last_epoch)
    for i in range(2): records.append(step(i))
    save_weights(acc, engine, str(out), 2, final=False)
    save_full_state(acc, str(out), 2, 1, 0)
    for i in range(2,6): records.append(step(i))
    expected = {k: v.detach().clone() for k,v in arch.state_dict().items()}
    state = load_full_state(acc, str(out/'accel_state_step_2'))
    replay = [step(i) for i in range(2,6)]
    assert records[2:] == replay, (records[2:], replay)
    assert all(torch.equal(v, arch.state_dict()[k]) for k,v in expected.items())
    assert any(not torch.equal(v, initial[k]) for k,v in expected.items())
    assert sched.scheduler.last_epoch == 24
    assert all(torch.isfinite(v).all() for v in expected.values())
    report = dict(rank=acc.process_index, passed=True, records=records, replay_exact=True,
                  restore_metadata=state, parameter_count=sum(p.numel() for p in arch.parameters()),
                  peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2)
    (out/f'rank_{acc.process_index}.json').write_text(json.dumps(report, indent=2))
    acc.wait_for_everyone()
    if acc.is_main_process:
        reports=[json.loads((out/f'rank_{i}.json').read_text()) for i in range(8)]
        result=dict(passed=all(r['passed'] for r in reports), world_size=8, batch_size=32,
                    accumulation=2, dtype='bf16', zero_stage=2, optimizer_updates=3,
                    checkpoint_replay_exact=True, synthetic_image_text=True, real_numeric_data=True,
                    max_peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in reports))
        (out/'summary.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result),flush=True)
    acc.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
