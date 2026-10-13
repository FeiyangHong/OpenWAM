import numpy as np
from openwam_robot_client.geometry import eef9_column_to_pose7, pose7_to_eef9_column
from openwam_robot_client.robot import Action58


def test_column_rot6d_round_trip():
    pose = np.array([0.2, -0.3, 1.1, 0.2, -0.3, 0.1, 0.9], dtype=np.float32)
    pose[3:] /= np.linalg.norm(pose[3:])
    reconstructed = eef9_column_to_pose7(pose7_to_eef9_column(pose))
    assert np.allclose(reconstructed[:3], pose[:3], atol=1e-6)
    assert abs(float(np.dot(reconstructed[3:], pose[3:]))) > 1.0 - 1e-6


def test_action58_layout():
    action = Action58.parse(np.arange(58, dtype=np.float32))
    assert action.left_eef.tolist() == list(range(9))
    assert action.left_hand.tolist() == list(range(9, 29))
    assert action.right_eef.tolist() == list(range(29, 38))
    assert action.right_hand.tolist() == list(range(38, 58))
