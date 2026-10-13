"""Read the recorded emergency left-side target without executing reset code."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

DEFAULT_JOINT_FILE = Path('/home/astribot/wuji-gr00t-client/init_joint.md')
DEFAULT_HAND_FILE = Path('/home/astribot/wuji-gr00t-client/112_reset_body_and_hands.py')
MAX_START_ERROR_RAD = 0.2


def read_left_init(joint_file=DEFAULT_JOINT_FILE, hand_file=DEFAULT_HAND_FILE):
    lines = Path(joint_file).read_text(encoding='utf-8').splitlines()
    try:
        start = lines.index('## Desired Joint Position') + 1
    except ValueError as exc:
        raise ValueError(f'Missing Desired Joint Position in {joint_file}') from exc
    rows = {}
    for line in lines[start:]:
        if line.startswith('## '):
            break
        if line.startswith('|'):
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            if cells[0] in ('astribot_torso', 'astribot_arm_left'):
                rows[cells[0]] = [float(cell) for cell in cells[1:] if cell]
    torso = np.asarray(rows.get('astribot_torso'), dtype=np.float32)
    arm = np.asarray(rows.get('astribot_arm_left'), dtype=np.float32)
    if torso.shape != (4,) or arm.shape != (7,) or not np.isfinite(torso).all() or not np.isfinite(arm).all():
        raise ValueError('init_joint.md must contain finite torso(4) and left arm(7) desired joints')
    tree = ast.parse(Path(hand_file).read_text(encoding='utf-8'), filename=str(hand_file))
    hand = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'LEFT_HAND_INIT_POSITIONS' for target in node.targets):
            hand = np.asarray(ast.literal_eval(node.value), dtype=np.float32)
            break
    if hand is None or hand.shape != (20,) or not np.isfinite(hand).all() or np.any(np.abs(hand) > 1.57):
        raise ValueError('reset script must contain 20 finite LEFT_HAND_INIT_POSITIONS in [-1.57, 1.57]')
    return torso, arm, hand


def left_init_pose(robot, torso, arm, *, eef_frame):
    if eef_frame != 'chassis':
        raise ValueError('--hold-left-init requires --eef-frame chassis; SDK FK returns chassis pose')
    names = [robot.torso_name, robot.arm_left_name]
    current = robot.get_current_joints_position(names)
    if isinstance(current, dict):
        current = [current[name] for name in names]
    for name, measured, target in zip(names, current, (torso, arm)):
        measured = np.asarray(measured, dtype=np.float32)
        if measured.shape != target.shape or not np.isfinite(measured).all():
            raise ValueError(f'Invalid measured joints for {name}')
        if np.max(np.abs(measured - target)) > MAX_START_ERROR_RAD:
            raise RuntimeError(f'{name} is more than {MAX_START_ERROR_RAD} rad from init_joint.md; reset pose before enabling --hold-left-init')
    result = robot.get_forward_kinematics(names, [torso.tolist(), arm.tolist()])
    pose = result[robot.arm_left_name] if isinstance(result, dict) else result[1]
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(float(np.linalg.norm(pose[3:])) - 1.) > 0.05:
        raise ValueError('SDK returned an invalid left arm forward-kinematics pose')
    measured_poses = robot.get_current_cartesian_pose([robot.arm_left_name, robot.arm_right_name], frame='chassis')
    measured_pose = measured_poses[robot.arm_left_name] if isinstance(measured_poses, dict) else measured_poses[0]
    measured_pose = np.asarray(measured_pose, dtype=np.float32)
    if measured_pose.shape != (7,) or not np.isfinite(measured_pose).all():
        raise ValueError('SDK returned an invalid measured left arm pose')
    rotation_error = 2 * np.arccos(np.clip(abs(float(np.dot(pose[3:], measured_pose[3:]))), 0., 1.))
    if np.linalg.norm(pose[:3] - measured_pose[:3]) > 0.1 or rotation_error > 0.35:
        raise RuntimeError('FK init pose differs from measured chassis EEF pose; check frame and reset pose')
    return pose
