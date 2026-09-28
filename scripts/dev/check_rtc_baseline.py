"""Compare non-RTC numerics/RNG against a saved pre-RTC Git revision on CPU.

Run from the repo root: python scripts/dev/check_rtc_baseline.py --baseline-ref REF
Requires test dependencies; executes only source from the explicitly named local
Git revision. Does not checkout or alter the worktree.
"""
import argparse
import ast
import subprocess

import numpy as np
import torch

from openwam.model.architectures import base
from openwam.model.action_backbone import separate_action_dit
from tests.test_training_rtc import model_and_data


def old_method(ref, path, cls, method, namespace):
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method)
    env = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), env)
    return env[method]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ref", required=True)
    args = parser.parse_args()
    replacements = []
    for module, cls, path, names in [
        (base, base.BaseWAMArchitecture, "openwam/model/architectures/base.py", ["compute_loss", "generate"]),
        (separate_action_dit, separate_action_dit.ActionDiT,
         "openwam/model/action_backbone/separate_action_dit.py", ["prepare_state", "pre_attn_at_layer_for_compile"]),
    ]:
        for name in names:
            replacements.append((cls, name, getattr(cls, name), old_method(args.baseline_ref, path, cls.__name__, name, vars(module))))
    torch.manual_seed(12)
    model, data = model_and_data()

    def train_result():
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(91)
        result = model.compute_loss(**data)
        result["loss"].backward()
        return {**{k: v.detach().clone() for k, v in result.items()},
                **{k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None},
                "rng": torch.get_rng_state()}

    def generate_result():
        model.video_backbone.preprocess_input_for_inference = lambda **kw: {
            "latents": data["input_latents"][:1].clone(), "context": data["context"][:1],
            "first_frame_latents": data["first_frame_latents"][:1],
        }
        return model.generate([(1000, 1000), (500, 500), (0, 0)], "test", action_num_frames=4,
                              proprio=data["proprio"][:1], decode_video=False)["actions"]

    new_train, new_actions = train_result(), generate_result()
    try:
        for cls, name, new, old in replacements:
            setattr(cls, name, old)
        old_train, old_actions = train_result(), generate_result()
    finally:
        for cls, name, new, old in replacements:
            setattr(cls, name, new)
    assert new_train.keys() == old_train.keys()
    for key in new_train:
        assert torch.equal(new_train[key], old_train[key]), key
    np.testing.assert_array_equal(new_actions, old_actions)
    print(f"PASS: non-RTC loss, {len(new_train) - 4} parameter gradients, RNG and sampled actions bitwise match {args.baseline_ref}")


if __name__ == "__main__":
    main()
