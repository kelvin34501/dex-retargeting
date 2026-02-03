"""
Retargeting Server (External Worker Protocol)

Subscribes to POEM's 3D joint output (MANO format), performs vector retargeting
using dex-retargeting, and publishes joint angles for robot hand control.

Protocol:
    1. Connect DEALER socket to master's ROUTER
    2. Send "starting" status
    3. Load retargeting config
    4. Send "model_loaded" status, wait for init config
    5. Receive retargeting config paths from master
    6. Send "ready" status
    7. Main loop: subscribe POEM pose_3d, retarget to joint angles, publish

Author: Xinyu Zhan
"""
from __future__ import annotations

import os
import sys
import json
import time
import atexit
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from collections import deque

import zmq
import numpy as np
import msgpack
import msgpack_numpy

# Import dex-retargeting
from dex_retargeting.retargeting_config import RetargetingConfig

# Import utility functions
from server_tool import log_util
from server_tool import zmq_channel_util

# MANO skeleton connectivity (parent joint for each joint index)
# Index 0 (wrist) has no parent (-1)
MANO_SKELETON = [
    -1,  # 0: wrist (root)
    0,  # 1: thumb_cmc -> wrist
    1,  # 2: thumb_mcp -> thumb_cmc
    2,  # 3: thumb_pip -> thumb_mcp
    3,  # 4: thumb_tip -> thumb_pip
    0,  # 5: index_mcp -> wrist
    5,  # 6: index_pip -> index_mcp
    6,  # 7: index_dip -> index_pip
    7,  # 8: index_tip -> index_dip
    0,  # 9: middle_mcp -> wrist
    9,  # 10: middle_pip -> middle_mcp
    10,  # 11: middle_dip -> middle_pip
    11,  # 12: middle_tip -> middle_dip
    0,  # 13: ring_mcp -> wrist
    13,  # 14: ring_pip -> ring_mcp
    14,  # 15: ring_dip -> ring_pip
    15,  # 16: ring_tip -> ring_dip
    0,  # 17: pinky_mcp -> wrist
    17,  # 18: pinky_pip -> pinky_mcp
    18,  # 19: pinky_dip -> pinky_pip
    19,  # 20: pinky_tip -> pinky_dip
]

# Z-axis offset (in meters) applied to MANO joints in local coordinates before retargeting
# This shifts the hand forward (along fingers direction) to better match robot hand kinematics
MANO_LOCAL_Z_OFFSET = np.array([0.0, 0.0, 0.02])

_logger = logging.getLogger(__name__)

# ========== Visualization utilities ==========


def setup_sapien_scene():
    """
    Setup SAPIEN scene with lighting for visualization.
    
    Returns:
        scene: sapien.Scene object
        viewer: sapien.utils.Viewer object
    """
    import sapien
    from sapien.asset import create_dome_envmap
    from sapien.utils import Viewer

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    scene = sapien.Scene()

    # Add ground plane
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.3, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)

    # Camera
    cam = scene.add_camera(name="main_cam", width=800, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.6, 0, 0.1], [0, 0, 0, -1]))

    # Viewer
    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = True  # 显示世界坐标系
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    return scene, viewer


def load_robot_to_scene(scene, urdf_path: str, hand_type: str = "right", scale: float = 1.0):
    """
    Load robot URDF into SAPIEN scene.
    
    Args:
        scene: sapien.Scene object
        urdf_path: Path to URDF file
        hand_type: "right" or "left"
        scale: Scale factor for the robot
    
    Returns:
        robot: sapien.Articulation object
    """
    import sapien
    import xml.etree.ElementTree as ET
    import tempfile

    # Pre-process URDF to fix invalid inertia values
    filepath = Path(urdf_path)
    glb_path = str(filepath).replace(".urdf", "_glb.urdf")
    urdf_to_load = glb_path if Path(glb_path).exists() else str(filepath)

    # Parse and fix inertia
    tree = ET.parse(urdf_to_load)
    root = tree.getroot()

    def fix_inertia_matrix(inertia_elem):
        """Fix inertia matrix to ensure it's physically valid (positive definite).
        
        For a valid inertia tensor, the eigenvalues must all be positive, which requires:
        - All diagonal elements > 0
        - Triangle inequalities: ixx + iyy >= izz, ixx + izz >= iyy, iyy + izz >= ixx
        - The matrix must be positive semi-definite
        
        The safest fix is to use a small spherical inertia (diagonal, no off-diagonal terms).
        """
        min_inertia = 1e-6

        # Get current values
        ixx = float(inertia_elem.get('ixx', '0'))
        ixy = float(inertia_elem.get('ixy', '0'))
        ixz = float(inertia_elem.get('ixz', '0'))
        iyy = float(inertia_elem.get('iyy', '0'))
        iyz = float(inertia_elem.get('iyz', '0'))
        izz = float(inertia_elem.get('izz', '0'))

        # Check if any diagonal is too small or negative
        needs_fix = (ixx <= min_inertia or iyy <= min_inertia or izz <= min_inertia)

        # Check triangle inequalities (physical realizability)
        if not needs_fix:
            needs_fix = (ixx + iyy < izz) or (ixx + izz < iyy) or (iyy + izz < ixx)

        # Check if off-diagonal terms could cause issues (simplified check)
        if not needs_fix:
            # For positive definiteness, we need eigenvalues > 0
            # A conservative check: off-diagonal magnitude should be small relative to diagonals
            max_off_diag = max(abs(ixy), abs(ixz), abs(iyz))
            min_diag = min(ixx, iyy, izz)
            if max_off_diag > 0.5 * min_diag:
                needs_fix = True

        if needs_fix:
            # Reset to a safe spherical inertia
            inertia_elem.set('ixx', str(min_inertia))
            inertia_elem.set('ixy', '0')
            inertia_elem.set('ixz', '0')
            inertia_elem.set('iyy', str(min_inertia))
            inertia_elem.set('iyz', '0')
            inertia_elem.set('izz', str(min_inertia))

    for link in root.findall('.//link'):
        inertial = link.find('inertial')
        if inertial is None:
            # Add minimal inertial element if missing
            inertial = ET.SubElement(link, 'inertial')
            mass = ET.SubElement(inertial, 'mass')
            mass.set('value', '0.001')
            inertia = ET.SubElement(inertial, 'inertia')
            inertia.set('ixx', '1e-6')
            inertia.set('ixy', '0')
            inertia.set('ixz', '0')
            inertia.set('iyy', '1e-6')
            inertia.set('iyz', '0')
            inertia.set('izz', '1e-6')
        else:
            # Fix zero/missing inertia values
            inertia = inertial.find('inertia')
            if inertia is not None:
                fix_inertia_matrix(inertia)
            else:
                inertia = ET.SubElement(inertial, 'inertia')
                inertia.set('ixx', '1e-6')
                inertia.set('ixy', '0')
                inertia.set('ixz', '0')
                inertia.set('iyy', '1e-6')
                inertia.set('iyz', '0')
                inertia.set('izz', '1e-6')
            # Fix zero/missing mass
            mass = inertial.find('mass')
            if mass is not None:
                val = float(mass.get('value', '0'))
                if val <= 0:
                    mass.set('value', '0.001')
            else:
                mass = ET.SubElement(inertial, 'mass')
                mass.set('value', '0.001')

    # Write fixed URDF to temp file in same directory (for relative mesh paths)
    urdf_dir = Path(urdf_to_load).parent
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', dir=urdf_dir, delete=False) as f:
        tree.write(f, encoding='unicode')
        fixed_urdf_path = f.name

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    loader.scale = scale

    try:
        robot = loader.load(fixed_urdf_path)
    finally:
        # Clean up temp file
        Path(fixed_urdf_path).unlink(missing_ok=True)

    # Position the robot
    if hand_type == "right":
        robot.set_pose(sapien.Pose([0.15, 0, 0]))
    else:
        robot.set_pose(sapien.Pose([-0.15, 0, 0]))

    return robot


def create_mano_visual_actors(scene, hand_type: str = "right") -> dict:
    """
    Create visual actors (spheres for joints, capsules for bones) for MANO hand visualization.
    
    Args:
        scene: sapien.Scene object
        hand_type: "right" or "left"
    
    Returns:
        dict with 'joints' (list of actors) and 'bones' (list of actors)
    """
    import sapien

    # Joint colors - different colors for different fingers
    joint_colors = {
        "wrist": [0.8, 0.8, 0.8, 1.0],  # gray
        "thumb": [1.0, 0.3, 0.3, 1.0],  # red
        "index": [0.3, 1.0, 0.3, 1.0],  # green
        "middle": [0.3, 0.3, 1.0, 1.0],  # blue
        "ring": [1.0, 1.0, 0.3, 1.0],  # yellow
        "pinky": [1.0, 0.3, 1.0, 1.0],  # magenta
    }

    def get_finger_name(idx: int) -> str:
        if idx == 0:
            return "wrist"
        elif 1 <= idx <= 4:
            return "thumb"
        elif 5 <= idx <= 8:
            return "index"
        elif 9 <= idx <= 12:
            return "middle"
        elif 13 <= idx <= 16:
            return "ring"
        else:
            return "pinky"

    joint_actors = []
    bone_actors = []

    # Create joint spheres
    for i in range(21):
        builder = scene.create_actor_builder()
        mat = sapien.render.RenderMaterial()
        color = joint_colors[get_finger_name(i)]
        mat.base_color = color
        mat.metallic = 0.0
        mat.roughness = 0.5
        builder.add_sphere_visual(radius=0.005, material=mat)
        actor = builder.build_kinematic(name=f"mano_{hand_type}_joint_{i}")
        joint_actors.append(actor)

    # Create bone capsules (one for each connection in skeleton)
    bone_mat = sapien.render.RenderMaterial()
    bone_mat.base_color = [0.9, 0.7, 0.5, 0.8]  # skin-like color
    bone_mat.metallic = 0.0
    bone_mat.roughness = 0.7

    for i in range(21):
        parent_idx = MANO_SKELETON[i]
        if parent_idx >= 0:  # Skip wrist (no parent)
            builder = scene.create_actor_builder()
            # Create a thin cylinder as bone placeholder
            # Actual length/orientation will be set in update_mano_visual
            builder.add_capsule_visual(radius=0.003, half_length=0.01, material=bone_mat)
            actor = builder.build_kinematic(name=f"mano_{hand_type}_bone_{i}")
            bone_actors.append(actor)
        else:
            bone_actors.append(None)

    return {"joints": joint_actors, "bones": bone_actors}


def update_mano_visual(joints_3d: np.ndarray, visual_actors: dict, offset: np.ndarray = None):
    """
    Update MANO visual actors with new joint positions.
    
    Args:
        joints_3d: (21, 3) MANO joint positions
        visual_actors: dict with 'joints' and 'bones' from create_mano_visual_actors
        offset: Optional (3,) offset to apply to all positions
    """
    import sapien

    if offset is None:
        offset = np.zeros(3)

    joint_actors = visual_actors["joints"]
    bone_actors = visual_actors["bones"]

    # Update joint positions
    for i, actor in enumerate(joint_actors):
        pos = joints_3d[i] + offset
        actor.set_pose(sapien.Pose(pos))

    # Update bone positions and orientations
    for i in range(21):
        parent_idx = MANO_SKELETON[i]
        if parent_idx >= 0 and bone_actors[i] is not None:
            # Get parent and child positions
            p1 = joints_3d[parent_idx] + offset
            p2 = joints_3d[i] + offset

            # Compute midpoint and length
            midpoint = (p1 + p2) / 2
            direction = p2 - p1
            length = np.linalg.norm(direction)

            if length > 1e-6:
                # Compute rotation to align capsule with bone direction
                # SAPIEN capsule default orientation is along X axis (half_length extends along X)
                direction_norm = direction / length
                x_axis = np.array([1, 0, 0])

                # Compute rotation axis and angle
                cross = np.cross(x_axis, direction_norm)
                dot = np.dot(x_axis, direction_norm)

                if np.linalg.norm(cross) < 1e-6:
                    if dot > 0:
                        quat = [1, 0, 0, 0]  # identity
                    else:
                        quat = [0, 0, 0, 1]  # 180 degree rotation around Z
                else:
                    cross_norm = cross / np.linalg.norm(cross)
                    angle = np.arccos(np.clip(dot, -1, 1))
                    # Quaternion from axis-angle: [w, x, y, z]
                    quat = [
                        np.cos(angle / 2),
                        cross_norm[0] * np.sin(angle / 2),
                        cross_norm[1] * np.sin(angle / 2),
                        cross_norm[2] * np.sin(angle / 2),
                    ]

                bone_actors[i].set_pose(sapien.Pose(midpoint, quat))


def get_urdf_joint_order(urdf_path: str) -> List[str]:
    """
    Parse URDF file and extract joint names in their original definition order.
    Only includes joints with DOF (revolute, continuous, prismatic).
    
    Args:
        urdf_path: Path to URDF file
    
    Returns:
        List of joint names in URDF definition order
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    urdf_joint_order = []
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        # Only include joints with DOF
        if joint_type in ['revolute', 'continuous', 'prismatic']:
            urdf_joint_order.append(joint_name)

    return urdf_joint_order


def get_pin2urdf_mapping(pin_joint_names: List[str], urdf_joint_names: List[str]) -> np.ndarray:
    """
    Get joint index mapping from pinocchio order to URDF original order.
    
    Usage: urdf_qpos = pin_qpos[idx_pin2urdf] is WRONG!
           urdf_qpos[i] = pin_qpos[idx_urdf2pin[i]], i.e., urdf_qpos = pin_qpos[idx_urdf2pin]
    
    Args:
        pin_joint_names: Joint names in pinocchio order
        urdf_joint_names: Joint names in URDF original order
    
    Returns:
        idx_urdf2pin: np.ndarray of indices such that urdf_qpos = pin_qpos[idx_urdf2pin]
    """
    # For each position in urdf order, find where that joint is in pin order
    idx_urdf2pin = np.array([pin_joint_names.index(name) for name in urdf_joint_names], dtype=int)
    return idx_urdf2pin


def get_retargeting_to_sapien_mapping(retargeting_joint_names: List[str], sapien_joint_names: List[str]) -> np.ndarray:
    """
    Get joint index mapping from retargeting output order to SAPIEN robot order.
    
    Args:
        retargeting_joint_names: Joint names in retargeting output order
        sapien_joint_names: Joint names in SAPIEN robot order
    
    Returns:
        mapping: np.ndarray of indices such that sapien_qpos = retargeting_qpos[mapping]
    """
    mapping = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)
    return mapping


# ========== Retargeting utilities ==========


def compute_wrist_transf(joints_3d: np.ndarray, hand_type: str = "right") -> np.ndarray:
    """
    Compute wrist transformation matrix (4x4) from MANO joints.
    
    Right hand frame convention:
    - X: points down (from back of hand to palm)
    - Z: points forward (from wrist to fingers)
    - Y: points left (from pinky to index)
    
    Left hand is mirrored (Y and X flipped).
    
    Args:
        joints_3d: (21, 3) MANO joint positions in world coordinates
        hand_type: "right" or "left", affects the frame orientation
    
    Returns:
        wrist_transf: (4, 4) transformation matrix T_world_wrist
    """
    assert joints_3d.shape == (21, 3), f"Expected (21, 3), got {joints_3d.shape}"

    wrist_pos = joints_3d[0]

    # Key joint indices: wrist(0), index_mcp(5), middle_mcp(9), ring_mcp(13)

    # Z-axis: from wrist to fingers (wrist -> middle_mcp)
    z = joints_3d[9] - joints_3d[0]  # middle_mcp - wrist
    z = z / (np.linalg.norm(z) + 1e-8)

    # Approximate Y direction: from ring to index
    temp_y = joints_3d[5] - joints_3d[13]  # index_mcp - ring_mcp

    # X = temp_y × Z (perpendicular to palm, from back to palm for right hand)
    x = np.cross(temp_y, z)
    x = x / (np.linalg.norm(x) + 1e-8)

    # Y = Z × X (orthogonalized Y, from pinky to index)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-8)

    # For left hand, mirror the frame
    if hand_type == "left":
        y = -y
        x = -x

    # Build rotation matrix [x, y, z]
    rotation = np.stack([x, y, z], axis=1)  # (3, 3)

    # Build 4x4 transformation matrix
    wrist_transf = np.eye(4)
    wrist_transf[:3, :3] = rotation
    wrist_transf[:3, 3] = wrist_pos

    return wrist_transf


def compute_retarget_ref_value(
    joints_3d: np.ndarray,
    origin_indices: np.ndarray,
    task_indices: np.ndarray,
) -> np.ndarray:
    """
    Compute reference value for vector retargeting from MANO joints.
    
    For vector retargeting, we compute direction vectors from origin to task joints.
    
    Args:
        joints_3d: (21, 3) MANO joint positions in world coordinates
        origin_indices: Array of origin joint indices (e.g., [0, 0, 0, 0, 0] for wrist)
        task_indices: Array of task joint indices (e.g., [4, 8, 12, 16, 20] for fingertips)
    
    Returns:
        ref_value: (N, 3) direction vectors from origin to task joints
    """
    origin_pos = joints_3d[origin_indices]  # (N, 3)
    task_pos = joints_3d[task_indices]  # (N, 3)
    ref_value = task_pos - origin_pos  # (N, 3) direction vectors
    return ref_value


def main(
    cmd_channel: str,
    sub_channel: str,
    pub_channel: str,
    identity: str = "retarget-0",
    visualize: bool = False,
):
    """
    Main retargeting server with external worker protocol.
    
    Args:
        cmd_channel: ZMQ channel for handshake with master (DEALER socket)
        sub_channel: ZMQ channel to subscribe POEM pose_3d results
        pub_channel: ZMQ channel to publish joint angles
        identity: ZMQ identity for DEALER socket
        visualize: Whether to enable SAPIEN visualization
    """
    _logger.info("Retargeting Server starting...")
    _logger.info(f"Visualization: {'enabled' if visualize else 'disabled'}")

    def parse_channel(channel: str) -> str:
        endpoint = zmq_channel_util.channel_name_to_endpoint(channel, "/dev/shm/hcc_demo")
        if zmq_channel_util.is_ipc_endpoint(endpoint):
            os.makedirs(os.path.dirname(zmq_channel_util.ipc_to_filepath(endpoint)), exist_ok=True)
        return endpoint

    ctx = zmq.Context()

    # ========== Phase 1: Handshake with master ==========

    # Command socket (DEALER) for async communication with master node
    cmd_socket = ctx.socket(zmq.DEALER)
    cmd_socket.setsockopt(zmq.IDENTITY, identity.encode())
    cmd_endpoint = parse_channel(cmd_channel)
    cmd_socket.connect(cmd_endpoint)
    _logger.info(f"Command socket (DEALER) connected to: {cmd_endpoint} with identity: {identity}")

    # Report starting status
    cmd_socket.send_string(json.dumps({
        "status": "starting",
        "msg": "Retargeting server starting...",
    }))
    _logger.info("Reported 'starting' status to master node")

    # Report model_loaded status and wait for init command
    cmd_socket.send_string(
        json.dumps({
            "status": "model_loaded",
            "msg": "Retargeting server ready, waiting for config...",
        }))
    _logger.info("Reported 'model_loaded' status, waiting for init command...")

    # Wait for init command with retargeting config from master node
    poller = zmq.Poller()
    poller.register(cmd_socket, zmq.POLLIN)
    retarget_config_right_path = None
    retarget_config_left_path = None
    urdf_base_dir = None

    while retarget_config_right_path is None:
        try:
            socks = dict(poller.poll(timeout=100))  # 100ms poll
            if cmd_socket in socks:
                cmd_msg = cmd_socket.recv_string()
                cmd_data = json.loads(cmd_msg)
                if cmd_data.get("cmd") == "init":
                    retarget_config_right_path = cmd_data.get("retarget_config_right", "")
                    retarget_config_left_path = cmd_data.get("retarget_config_left", "")
                    urdf_base_dir = cmd_data.get("urdf_base_dir", "./asset/env")
                    _logger.info(f"Received retarget_config_right: {retarget_config_right_path}")
                    _logger.info(f"Received retarget_config_left: {retarget_config_left_path}")
                    _logger.info(f"Received urdf_base_dir: {urdf_base_dir}")
                elif cmd_data.get("cmd") == "ping":
                    cmd_socket.send_string(json.dumps({
                        "status": "pong",
                        "msg": "waiting for init",
                    }))
                else:
                    _logger.warning(f"Unknown command while waiting for init: {cmd_data.get('cmd')}")
        except Exception as e:
            _logger.error(f"Error receiving init command: {e}")

    # Load retargeting configs
    _logger.info("Loading retargeting configs...")
    RetargetingConfig.set_default_urdf_dir(urdf_base_dir)

    # Right hand retargeter
    retargeter_right = None
    origin_indices_right = None
    task_indices_right = None
    idx_urdf2pin_right = None  # Mapping from URDF order to pinocchio order (always set when retargeter exists)
    if retarget_config_right_path:
        config_right = RetargetingConfig.load_from_file(retarget_config_right_path)
        retargeter_right = config_right.build()
        # Get indices from optimizer (works for both vector and DexPilot)
        # DexPilot auto-generates target_link_human_indices in optimizer init
        origin_indices_right = retargeter_right.optimizer.target_link_human_indices[0, :]
        task_indices_right = retargeter_right.optimizer.target_link_human_indices[1, :]

        # Build pin2urdf mapping for output order conversion (guaranteed to exist when retargeter exists)
        urdf_joint_names_right = get_urdf_joint_order(config_right.urdf_path)
        pin_joint_names_right = retargeter_right.joint_names
        idx_urdf2pin_right = get_pin2urdf_mapping(pin_joint_names_right, urdf_joint_names_right)
        _logger.info(
            f"Right hand retargeter loaded, type: {config_right.type}, DOF: {retargeter_right.optimizer.opt_dof}")
        _logger.info(f"Right hand URDF joint order: {urdf_joint_names_right}")
        _logger.info(f"Right hand idx_urdf2pin: {idx_urdf2pin_right.tolist()}")

    # Left hand retargeter
    retargeter_left = None
    origin_indices_left = None
    task_indices_left = None
    idx_urdf2pin_left = None  # Mapping from URDF order to pinocchio order (always set when retargeter exists)
    if retarget_config_left_path:
        config_left = RetargetingConfig.load_from_file(retarget_config_left_path)
        retargeter_left = config_left.build()
        # Get indices from optimizer (works for both vector and DexPilot)
        # DexPilot auto-generates target_link_human_indices in optimizer init
        origin_indices_left = retargeter_left.optimizer.target_link_human_indices[0, :]
        task_indices_left = retargeter_left.optimizer.target_link_human_indices[1, :]

        # Build pin2urdf mapping for output order conversion (guaranteed to exist when retargeter exists)
        urdf_joint_names_left = get_urdf_joint_order(config_left.urdf_path)
        pin_joint_names_left = retargeter_left.joint_names
        idx_urdf2pin_left = get_pin2urdf_mapping(pin_joint_names_left, urdf_joint_names_left)
        _logger.info(f"Left hand retargeter loaded, type: {config_left.type}, DOF: {retargeter_left.optimizer.opt_dof}")
        _logger.info(f"Left hand URDF joint order: {urdf_joint_names_left}")
        _logger.info(f"Left hand idx_urdf2pin: {idx_urdf2pin_left.tolist()}")
        config_left = RetargetingConfig.load_from_file(retarget_config_left_path)
        retargeter_left = config_left.build()
        # Get indices from optimizer (works for both vector and DexPilot)
        # DexPilot auto-generates target_link_human_indices in optimizer init
        origin_indices_left = retargeter_left.optimizer.target_link_human_indices[0, :]
        task_indices_left = retargeter_left.optimizer.target_link_human_indices[1, :]

        # Build pin2urdf mapping for output order conversion
        urdf_joint_names_left = get_urdf_joint_order(config_left.urdf_path)
        pin_joint_names_left = retargeter_left.joint_names
        idx_urdf2pin_left = get_pin2urdf_mapping(pin_joint_names_left, urdf_joint_names_left)
        _logger.info(f"Left hand retargeter loaded, type: {config_left.type}, DOF: {retargeter_left.optimizer.opt_dof}")
        _logger.info(f"Left hand URDF joint order: {urdf_joint_names_left}")
        _logger.info(f"Left hand idx_urdf2pin: {idx_urdf2pin_left.tolist()}")

    # ========== Phase 2.5: Setup visualization (optional) ==========

    scene = None
    viewer = None
    robot_right = None
    robot_left = None
    mano_visual_right = None
    mano_visual_left = None
    retargeting_to_sapien_right = None
    retargeting_to_sapien_left = None

    if visualize:
        try:
            _logger.info("Setting up SAPIEN visualization...")
            scene, viewer = setup_sapien_scene()

            # Load right hand robot
            if retargeter_right is not None and retarget_config_right_path:
                config_right_loaded = RetargetingConfig.load_from_file(retarget_config_right_path)
                robot_right = load_robot_to_scene(scene, config_right_loaded.urdf_path, hand_type="right")
                # Get joint mapping from URDF order to SAPIEN order
                sapien_joint_names = [j.get_name() for j in robot_right.get_active_joints()]
                urdf_joint_names_right = get_urdf_joint_order(config_right_loaded.urdf_path)
                retargeting_to_sapien_right = get_retargeting_to_sapien_mapping(urdf_joint_names_right,
                                                                                sapien_joint_names)
                _logger.info(f"Right robot loaded with {len(sapien_joint_names)} active joints")

            # Load left hand robot
            if retargeter_left is not None and retarget_config_left_path:
                config_left_loaded = RetargetingConfig.load_from_file(retarget_config_left_path)
                robot_left = load_robot_to_scene(scene, config_left_loaded.urdf_path, hand_type="left")
                # Get joint mapping from URDF order to SAPIEN order
                sapien_joint_names = [j.get_name() for j in robot_left.get_active_joints()]
                urdf_joint_names_left = get_urdf_joint_order(config_left_loaded.urdf_path)
                retargeting_to_sapien_left = get_retargeting_to_sapien_mapping(urdf_joint_names_left,
                                                                               sapien_joint_names)
                _logger.info(f"Left robot loaded with {len(sapien_joint_names)} active joints")

            # Create MANO visual actors
            if retargeter_right is not None:
                mano_visual_right = create_mano_visual_actors(scene, hand_type="right")
                _logger.info("Right MANO visual actors created")

            if retargeter_left is not None:
                mano_visual_left = create_mano_visual_actors(scene, hand_type="left")
                _logger.info("Left MANO visual actors created")

            _logger.info("SAPIEN visualization setup complete")

        except Exception as e:
            _logger.error(f"Failed to setup visualization: {e}", exc_info=True)
            visualize = False
            scene = None
            viewer = None

    # ========== Phase 3: Setup data channels ==========

    # Subscribe to POEM results (pose_3d)
    sub_socket = ctx.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
    sub_socket.setsockopt(zmq.RCVHWM, 1)
    sub_socket.setsockopt(zmq.CONFLATE, 1)
    sub_endpoint = parse_channel(sub_channel)
    sub_socket.connect(sub_endpoint)
    _logger.info(f"Subscribed to POEM channel: {sub_endpoint}")

    # Publish joint angles
    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.setsockopt(zmq.SNDHWM, 1)
    pub_socket.setsockopt(zmq.CONFLATE, 1)
    pub_endpoint = parse_channel(pub_channel)
    pub_socket.bind(pub_endpoint)
    _logger.info(f"Publishing joint angles to: {pub_endpoint}")

    # Report ready status
    cmd_socket.send_string(json.dumps({
        "status": "ready",
        "msg": "Retargeting server ready",
    }))
    _logger.info("Reported 'ready' status to master node")

    # ========== Phase 4: Main retargeting loop ==========

    def cleanup():
        _logger.info("Cleaning up...")
        cmd_socket.close()
        sub_socket.close()
        pub_socket.close()
        ctx.term()
        _logger.info("Retargeting server stopped")

    atexit.register(cleanup)

    # Stats tracking
    frame_count = 0
    retarget_times = deque(maxlen=100)

    _logger.info("Starting retargeting loop...")

    while True:
        try:
            # Receive POEM results (non-blocking)
            try:
                msg = sub_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                # No message available, but still need to render viewer to keep it responsive
                if visualize and viewer is not None:
                    try:
                        viewer.render()
                    except Exception:
                        pass
                time.sleep(0.001)
                continue

            # Decode POEM message
            data = msgpack.unpackb(msg, object_hook=msgpack_numpy.decode, raw=False)
            timestamp = data.get("sync_timestamp") or data.get("timestamp")
            pose_3d = data.get("pose_3d")

            if pose_3d is None:
                # Still render viewer even if no pose data
                if visualize and viewer is not None:
                    try:
                        viewer.render()
                    except Exception:
                        pass
                continue

            frame_count += 1
            retarget_start = time.time()

            # Retarget each hand
            result = {
                "sync_timestamp": timestamp,
                "hand_right": None,
                "hand_left": None,
                "wrist_transf_right": None,
                "wrist_transf_left": None,
            }

            # Right hand
            rh_data = pose_3d.get("rh")
            if rh_data is not None and retargeter_right is not None:
                joints_rh = rh_data.get("joints")  # (21, 3) MANO joints
                if joints_rh is not None:
                    try:
                        # Compute wrist transformation first (needed for coordinate transform)
                        wrist_transf_rh = compute_wrist_transf(joints_rh, hand_type="right")

                        # Transform joints to wrist-local coordinates
                        _inv_mat = np.linalg.inv(wrist_transf_rh)
                        joints_rh_local = (_inv_mat[:3, :3] @ joints_rh.T).T + _inv_mat[:3, 3]

                        # Shift the hand in local coordinates before retargeting
                        joints_rh_local += MANO_LOCAL_Z_OFFSET

                        # Compute reference vectors in wrist-local coordinates
                        ref_value = compute_retarget_ref_value(
                            joints_rh_local,
                            origin_indices_right,
                            task_indices_right,
                        )
                        qpos_rh = retargeter_right.retarget(ref_value)
                        # Apply joint limits (clip) before order conversion to avoid execution errors
                        qpos_rh = np.clip(qpos_rh, retargeter_right.joint_limits[:, 0],
                                          retargeter_right.joint_limits[:, 1])
                        # Convert from pinocchio order to URDF original order for output
                        qpos_rh_urdf = qpos_rh[idx_urdf2pin_right]
                        result["hand_right"] = qpos_rh_urdf.tolist()
                        result["wrist_transf_right"] = wrist_transf_rh.tolist()
                    except Exception as e:
                        _logger.warning(f"Right hand retarget failed: {e}")

            # Left hand
            lh_data = pose_3d.get("lh")
            if lh_data is not None and retargeter_left is not None:
                joints_lh = lh_data.get("joints")  # (21, 3) MANO joints
                if joints_lh is not None:
                    try:
                        # Compute wrist transformation first (needed for coordinate transform)
                        wrist_transf_lh = compute_wrist_transf(joints_lh, hand_type="left")

                        # Transform joints to wrist-local coordinates
                        _inv_mat = np.linalg.inv(wrist_transf_lh)
                        joints_lh_local = (_inv_mat[:3, :3] @ joints_lh.T).T + _inv_mat[:3, 3]

                        # Shift the hand in local coordinates before retargeting
                        joints_lh_local += MANO_LOCAL_Z_OFFSET

                        # Compute reference vectors in wrist-local coordinates
                        ref_value = compute_retarget_ref_value(
                            joints_lh_local,
                            origin_indices_left,
                            task_indices_left,
                        )
                        qpos_lh = retargeter_left.retarget(ref_value)
                        # Apply joint limits (clip) before order conversion to avoid execution errors
                        qpos_lh = np.clip(qpos_lh, retargeter_left.joint_limits[:, 0], retargeter_left.joint_limits[:,
                                                                                                                    1])
                        # Convert from pinocchio order to URDF original order for output
                        qpos_lh_urdf = qpos_lh[idx_urdf2pin_left]
                        result["hand_left"] = qpos_lh_urdf.tolist()
                        result["wrist_transf_left"] = wrist_transf_lh.tolist()
                    except Exception as e:
                        _logger.warning(f"Left hand retarget failed: {e}")

            # Publish joint angles
            pub_msg = msgpack.packb(result, default=msgpack_numpy.encode)
            pub_socket.send(pub_msg)

            # ========== Update visualization ==========
            if visualize and viewer is not None:
                try:
                    # Update right hand visualization
                    if rh_data is not None and result["hand_right"] is not None:
                        joints_rh = rh_data.get("joints")
                        if joints_rh is not None:
                            # Update MANO visual (offset to the right side)
                            if mano_visual_right is not None:
                                _inv_mat = np.linalg.inv(wrist_transf_rh)
                                joints_rh_to_viz = (_inv_mat[:3, :3] @ joints_rh.T).T + _inv_mat[:3, 3]
                                joints_rh_to_viz += MANO_LOCAL_Z_OFFSET
                                update_mano_visual(joints_rh_to_viz, mano_visual_right, offset=np.array([0.3, 0, 0]))
                            # Update robot qpos
                            if robot_right is not None and retargeting_to_sapien_right is not None:
                                qpos_rh = np.array(result["hand_right"])
                                robot_right.set_qpos(qpos_rh[retargeting_to_sapien_right])

                    # Update left hand visualization
                    if lh_data is not None and result["hand_left"] is not None:
                        joints_lh = lh_data.get("joints")
                        if joints_lh is not None:
                            # Update MANO visual (offset to the left side)
                            if mano_visual_left is not None:
                                _inv_mat = np.linalg.inv(wrist_transf_lh)
                                joints_lh_to_viz = (_inv_mat[:3, :3] @ joints_lh.T).T + _inv_mat[:3, 3]
                                joints_lh_to_viz += MANO_LOCAL_Z_OFFSET
                                update_mano_visual(joints_lh_to_viz, mano_visual_left, offset=np.array([-0.3, 0, 0]))
                            # Update robot qpos
                            if robot_left is not None and retargeting_to_sapien_left is not None:
                                qpos_lh = np.array(result["hand_left"])
                                robot_left.set_qpos(qpos_lh[retargeting_to_sapien_left])

                    # Render
                    scene.update_render()
                    viewer.render()

                except Exception as e:
                    _logger.warning(f"Visualization update failed: {e}")

            retarget_time = (time.time() - retarget_start) * 1000  # ms
            retarget_times.append(retarget_time)

            # Log periodically
            if frame_count % 30 == 0:
                avg_retarget_time = np.mean(retarget_times) if retarget_times else 0
                has_rh = result["hand_right"] is not None
                has_lh = result["hand_left"] is not None
                _logger.info(f"Frame {frame_count} | Retarget: {retarget_time:.1f}ms "
                             f"(avg: {avg_retarget_time:.1f}ms) | RH: {has_rh} | LH: {has_lh}")

        except KeyboardInterrupt:
            _logger.info("Received keyboard interrupt, stopping...")
            break
        except Exception as e:
            _logger.error(f"Error in retargeting loop: {e}", exc_info=True)
            time.sleep(0.01)

    cleanup()


if __name__ == "__main__":
    log_util.log_init()
    log_util.enable_console()

    parser = argparse.ArgumentParser(description="Retargeting Server (External Worker)")
    parser.add_argument("--server.cmd_channel",
                        type=str,
                        required=True,
                        help="ZMQ channel for handshake with master (DEALER socket)")
    parser.add_argument("--server.sub_channel",
                        type=str,
                        required=True,
                        help="ZMQ channel to subscribe POEM pose_3d results")
    parser.add_argument("--server.pub_channel", type=str, required=True, help="ZMQ channel to publish joint angles")
    parser.add_argument("--identity", type=str, default="retarget-0", help="ZMQ identity for DEALER socket")
    parser.add_argument("--visualize", action="store_true", help="Enable SAPIEN visualization for debugging")

    args = parser.parse_args()

    main(
        cmd_channel=getattr(args, "server.cmd_channel"),
        sub_channel=getattr(args, "server.sub_channel"),
        pub_channel=getattr(args, "server.pub_channel"),
        identity=args.identity,
        visualize=args.visualize,
    )
