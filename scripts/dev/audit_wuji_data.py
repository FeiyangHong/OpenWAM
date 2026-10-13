"""CPU audit of real Wuji conversion, geometry and normalization; never rewrites data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from openwam.dataloader.utils.rot6d import convert_wuji_58, wuji_58_rot6d_to_matrix
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.deploy.model_loader import _build_normalizer


def audit(source, target):
    info = json.loads((target / 'meta/info.json').read_text())
    source_info = json.loads((source / 'meta/info.json').read_text())
    episodes = [json.loads(line) for line in (target / 'meta/episodes.jsonl').read_text().splitlines() if line]
    parquets = list((target / 'data').rglob('*.parquet'))
    videos = list((target / 'videos').rglob('*.mp4'))
    assert len(parquets) == len(episodes) == info['total_episodes']
    assert len(videos) == len(episodes) * 3 and all(p.is_file() for p in videos)
    assert sum(e['length'] for e in episodes) == info['total_frames'] == source_info['total_frames']
    assert source_info['total_episodes'] == info['total_episodes']
    stats = np.load(target / 'meta/normalization_stats.npy', allow_pickle=True).item()['eef']
    cfg = OmegaConf.load('configs/dataloader/wuji_real_task.yaml')
    deploy_cfg = OmegaConf.create({'dataloader': OmegaConf.to_container(cfg)})
    normalizer = _build_normalizer(deploy_cfg, str(target / 'meta'))
    errors = dict(conversion=0., orthogonality=0., determinant=0., deploy_normalize=0., deploy_roundtrip_clipped=0.)
    chosen = sorted({0, len(episodes) // 2, len(episodes) - 1})
    for index in chosen:
        ep = episodes[index]['episode_index']
        rel = f'data/chunk-000/episode_{ep:06d}.parquet'
        original = pq.read_table(source / rel, columns=['action', 'observation.state'])
        converted = pq.read_table(target / rel, columns=['action', 'observation.state'])
        for column in ('action', 'observation.state'):
            raw = np.asarray(original[column].to_pylist(), dtype=np.float32)
            values = np.asarray(converted[column].to_pylist(), dtype=np.float32)
            errors['conversion'] = max(errors['conversion'], float(np.max(np.abs(convert_wuji_58(raw) - values))))
            rotations = wuji_58_rot6d_to_matrix(values)
            errors['orthogonality'] = max(errors['orthogonality'], float(np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3)))))
            errors['determinant'] = max(errors['determinant'], float(np.max(np.abs(np.linalg.det(rotations) - 1))))
            # Quantile normalization clips tails: compare the clipped domain, not the original tails.
            clipped = np.clip(values, stats['q01'], stats['q99'])
            unified = normalizer.normalize(clipped)
            recovered = normalizer.unnormalize(unified)
            errors['deploy_roundtrip_clipped'] = max(errors['deploy_roundtrip_clipped'], float(np.max(np.abs(recovered - clipped))))
            from openwam.dataloader.utils.unify_action import parse_unify_spec, unmap_from_unify
            mapped = unmap_from_unify(normalizer.normalize(values), parse_unify_spec(cfg.unify_action_map))
            expected = apply_normalization(values, stats, 'quantile')
            errors['deploy_normalize'] = max(errors['deploy_normalize'], float(np.max(np.abs(mapped - expected))))
    assert all(value < 1e-5 for value in errors.values()), errors
    return dict(source=str(source), target=str(target), episodes=len(episodes), frames=info['total_frames'],
                fps=info['fps'], parquets=len(parquets), videos=len(videos),
                video_symlinks=sum(p.is_symlink() for p in videos), sampled_episodes=[episodes[i]['episode_index'] for i in chosen],
                max_errors=errors, scope='Counts and video links for full bucket; vectors/geometry/deploy normalization for first, middle and last episode')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('/gaozt-test1/fyhong/OpenWAM/data'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for family in ('grasp_anything', 'spray_water'):
        base = args.source_root / family
        result[family] = audit(base / (family + '_eef_rot6d'), base / (family + '_eef_rot6d_col'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
