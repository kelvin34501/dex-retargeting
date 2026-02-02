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
from typing import Tuple, Optional, Dict, Any
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

_logger = logging.getLogger(__name__)

# ========== Retargeting utilities ==========


def compute_wrist_transf(joints_3d: np.ndarray, hand_type: str = "right") -> np.ndarray:
    """
    Compute wrist transformation matrix (4x4) from MANO joints.
    
    The wrist frame is estimated using:
    - Position: MANO joint 0 (wrist)
    - Orientation: Estimated from wrist (0), index MCP (5), and middle MCP (9)
    
    Based on the method from single_hand_detector.py:estimate_frame_from_hand_points
    
    Args:
        joints_3d: (21, 3) MANO joint positions in world coordinates
        hand_type: "right" or "left", affects the frame orientation
    
    Returns:
        wrist_transf: (4, 4) transformation matrix T_world_wrist
    """
    assert joints_3d.shape == (21, 3), f"Expected (21, 3), got {joints_3d.shape}"

    # Extract key points: wrist (0), index MCP (5), middle MCP (9)
    wrist_pos = joints_3d[0]
    points = joints_3d[[0, 5, 9], :]  # wrist, index_mcp, middle_mcp

    # Compute vector from middle MCP to wrist (x-axis direction)
    x_vector = points[0] - points[2]  # wrist - middle_mcp

    # Normal fitting with SVD to find palm plane normal
    points_centered = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points_centered)
    normal = v[2, :]  # Palm plane normal (y-axis candidate)

    # Gram-Schmidt orthonormalization
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / (np.linalg.norm(x) + 1e-8)
    z = np.cross(x, normal)
    z = z / (np.linalg.norm(z) + 1e-8)

    # Ensure z-axis points from pinky to index direction
    # index_mcp (5) - middle_mcp (9) approximates this direction
    if np.sum(z * (joints_3d[5] - joints_3d[9])) < 0:
        normal = -normal
        z = -z

    # For left hand, mirror the frame
    if hand_type == "left":
        z = -z
        normal = -normal

    # Build rotation matrix [x, y, z] where y = normal
    rotation = np.stack([x, normal, z], axis=1)  # (3, 3)

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
):
    """
    Main retargeting server with external worker protocol.
    
    Args:
        cmd_channel: ZMQ channel for handshake with master (DEALER socket)
        sub_channel: ZMQ channel to subscribe POEM pose_3d results
        pub_channel: ZMQ channel to publish joint angles
        identity: ZMQ identity for DEALER socket
    """
    _logger.info("Retargeting Server starting...")

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
    if retarget_config_right_path:
        config_right = RetargetingConfig.load_from_file(retarget_config_right_path)
        retargeter_right = config_right.build()
        # Get indices from optimizer (works for both vector and DexPilot)
        # DexPilot auto-generates target_link_human_indices in optimizer init
        origin_indices_right = retargeter_right.optimizer.target_link_human_indices[0, :]
        task_indices_right = retargeter_right.optimizer.target_link_human_indices[1, :]
        _logger.info(
            f"Right hand retargeter loaded, type: {config_right.type}, DOF: {retargeter_right.optimizer.opt_dof}")

    # Left hand retargeter
    retargeter_left = None
    origin_indices_left = None
    task_indices_left = None
    if retarget_config_left_path:
        config_left = RetargetingConfig.load_from_file(retarget_config_left_path)
        retargeter_left = config_left.build()
        # Get indices from optimizer (works for both vector and DexPilot)
        # DexPilot auto-generates target_link_human_indices in optimizer init
        origin_indices_left = retargeter_left.optimizer.target_link_human_indices[0, :]
        task_indices_left = retargeter_left.optimizer.target_link_human_indices[1, :]
        _logger.info(f"Left hand retargeter loaded, type: {config_left.type}, DOF: {retargeter_left.optimizer.opt_dof}")

    # ========== Phase 2: Setup data channels ==========

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

    # ========== Phase 3: Main retargeting loop ==========

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
                time.sleep(0.001)
                continue

            # Decode POEM message
            data = msgpack.unpackb(msg, object_hook=msgpack_numpy.decode, raw=False)
            timestamp = data.get("sync_timestamp") or data.get("timestamp")
            pose_3d = data.get("pose_3d")

            if pose_3d is None:
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
                        ref_value = compute_retarget_ref_value(
                            joints_rh,
                            origin_indices_right,
                            task_indices_right,
                        )
                        qpos_rh = retargeter_right.retarget(ref_value)
                        result["hand_right"] = qpos_rh.tolist()
                        # Compute wrist transformation for IK
                        wrist_transf_rh = compute_wrist_transf(joints_rh, hand_type="right")
                        result["wrist_transf_right"] = wrist_transf_rh.tolist()
                    except Exception as e:
                        _logger.warning(f"Right hand retarget failed: {e}")

            # Left hand
            lh_data = pose_3d.get("lh")
            if lh_data is not None and retargeter_left is not None:
                joints_lh = lh_data.get("joints")  # (21, 3) MANO joints
                if joints_lh is not None:
                    try:
                        ref_value = compute_retarget_ref_value(
                            joints_lh,
                            origin_indices_left,
                            task_indices_left,
                        )
                        qpos_lh = retargeter_left.retarget(ref_value)
                        result["hand_left"] = qpos_lh.tolist()
                        # Compute wrist transformation for IK
                        wrist_transf_lh = compute_wrist_transf(joints_lh, hand_type="left")
                        result["wrist_transf_left"] = wrist_transf_lh.tolist()
                    except Exception as e:
                        _logger.warning(f"Left hand retarget failed: {e}")

            # Publish joint angles
            pub_msg = msgpack.packb(result, default=msgpack_numpy.encode)
            pub_socket.send(pub_msg)

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

    args = parser.parse_args()

    main(
        cmd_channel=getattr(args, "server.cmd_channel"),
        sub_channel=getattr(args, "server.sub_channel"),
        pub_channel=getattr(args, "server.pub_channel"),
        identity=args.identity,
    )
