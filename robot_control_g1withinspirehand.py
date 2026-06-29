"""
Example script demo'ing robot control via ZMQ.

Receives qpos data from ZMQ publisher and controls the robot in absolute position mode.
Uses a separate thread for receiving messages and a queue for buffering.
"""

import json
import time
import threading
from collections import deque
import torch as th
import numpy as np
import zmq

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
from omnigibson.utils import transform_utils as T

CONTROL_MODES = dict(
    random_test="Use random test data (default)",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    office_large="office large",
    school_chemistry="school chemistry",
    Ihlen_1_int="Ihlen 1 int",
    empty="Empty environment with no objects",
)

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return dict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = dict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    controller_names = robot.controller_order
    for controller_name in controller_names:
        controller_options = default_config[controller_name]
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options,
            name=f"{controller_name} controller",
            random_selection=random_selection,
        )

        # Add to user responses
        controller_choices[controller_name] = choice

    return controller_choices


class ZMQReceiver:
    """ZMQ message receiver running in a separate thread."""

    def __init__(self, host="tcp://127.0.0.1:6666", topic="gmr.qpos", queue_maxlen=10):
        """
        Initialize ZMQ receiver.
        
        :param host: ZMQ publisher address
        :param topic: Topic to subscribe to
        :param queue_maxlen: Maximum queue length (drops oldest when full)
        """
        self.host = host
        self.topic = topic
        self.queue = deque(maxlen=queue_maxlen)
        self.running = False
        self.thread = None

    def start(self):
        """Start the receiver thread."""
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"[ZMQ Receiver] Started, subscribing to {self.host} with topic '{self.topic}'")

    def stop(self):
        """Stop the receiver thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("[ZMQ Receiver] Stopped")

    def _receive_loop(self):
        """Main receive loop running in separate thread."""
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(self.host)
        socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        # Also subscribe to empty string as fallback
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"[ZMQ Receiver] Connected and waiting for messages...")

        while self.running:
            try:
                # Non-blocking receive with timeout
                message = socket.recv_string(flags=zmq.NOBLOCK)

                # Parse topic and data
                if " " in message:
                    recv_topic, data_str = message.split(" ", 1)
                else:
                    recv_topic = ""
                    data_str = message

                # Parse JSON data
                try:
                    data = json.loads(data_str)
                    self.queue.append(data)
                except json.JSONDecodeError as e:
                    print(f"[ZMQ Receiver] JSON decode error: {e}")

            except zmq.Again:
                # No message available, sleep briefly
                time.sleep(0.001)
            except Exception as e:
                print(f"[ZMQ Receiver] Error: {e}")
                time.sleep(0.01)

        socket.close()
        context.term()

    def get_latest(self):
        """Get the latest message from queue (non-destructive peek)."""
        if self.queue:
            return self.queue[-1]
        return None

    def pop_latest(self):
        """Get and remove the latest message from queue."""
        if self.queue:
            return self.queue.pop()
        return None


class GripperDataReceiver:
    """Separate ZMQ receiver for gripper/hand joint data."""

    def __init__(self, host="tcp://127.0.0.1:5555", queue_maxlen=10):
        """
        Initialize gripper data receiver.
        
        :param host: ZMQ publisher address for hand data
        :param queue_maxlen: Maximum queue length (drops oldest when full)
        """
        self.host = host
        self.queue = deque(maxlen=queue_maxlen)  # 使用队列缓存原始消息
        self.running = False
        self.thread = None
        self.data_lock = threading.Lock()

        # Store latest gripper data for left and right hands
        self.latest_gripper_data = {"Left": None, "Right": None}
        self.last_recv_time = {"Left": 0.0, "Right": 0.0}

        # Output gripper poses (will be updated by get_gripper_poses)
        # G1WithInspireHand has 12 fingers on right hand
        self.output_gripper_poses = {
            "gripper_right": np.zeros(12)  # Right hand has 12 finger joints
        }

    def start(self):
        """Start the gripper receiver thread."""
        self.running = True
        self.thread = threading.Thread(target=self._zmq_worker, daemon=True)
        self.thread.start()
        print(f"[Gripper Receiver] Started, subscribing to {self.host}")

    def stop(self):
        """Stop the gripper receiver thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("[Gripper Receiver] Stopped")

    def _zmq_worker(self):
        """Background thread to receive hand data from ZMQ."""
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(self.host)
        socket.setsockopt(zmq.CONFLATE, 1)  # Only keep latest message
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"[Gripper Receiver] Connected and waiting for hand data...")

        while self.running:
            try:
                events = socket.poll(timeout=10)
                if events:
                    msg = socket.recv_json(flags=zmq.NOBLOCK)

                    # 将原始消息加入队列（线程安全，deque自带锁）
                    self.queue.append(msg)

                    # 同时更新最新的手部数据用于快速访问
                    with self.data_lock:
                        for hand_side in ["Left", "Right"]:
                            if hand_side in msg and msg[hand_side] is not None:
                                self.latest_gripper_data[hand_side] = msg[hand_side]
                                self.last_recv_time[hand_side] = time.time()

            except Exception as e:
                pass

        socket.close()
        context.term()

    def peek_latest(self):
        """
        Get the latest message from queue WITHOUT removing it (non-destructive peek).
        
        :return: Latest message dict or None if queue is empty
        """
        if self.queue:
            return self.queue[-1]  # 返回队列最后一个元素（最新），但不移除
        return None

    def pop_latest(self):
        """
        Get and remove the latest message from queue (destructive).
        
        :return: Latest message dict or None if queue is empty
        """
        if self.queue:
            return self.queue.pop()  # 移除并返回最后一个元素
        return None

    def get_queue_size(self):
        """Get current queue size."""
        return len(self.queue)

    def _process_gripper_data(self, hand_side, hand_data):
        """Process gripper joint angles from hand data."""
        if "robot_joints" not in hand_data:
            return

        # Map to G1 Inspire Hand (12 fingers for right hand)
        gripper_key = "gripper_right"

        joint_angles = hand_data["robot_joints"]
        n_fingers = len(self.output_gripper_poses[gripper_key])
        alpha_g = 0.5

        # Calculate closure (0.0 = open, 1.0 = closed)
        # closure = np.clip((np.abs(rad_array) - self.MIN_RAD) / (self.MAX_RAD - self.MIN_RAD), 0.0, 1.0)
        rad_array = np.array(joint_angles)
        closure = np.clip(np.abs(rad_array) / 1.2, 0.0, 1.0)

        if n_fingers == 12:
            mapping = [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5]
            for a_i, d_i in enumerate(mapping):
                if d_i < len(closure):
                    c_val = closure[d_i]
                    raw_val = (c_val * 2.0 - 1.0) if hand_side == "Right" else (1.0 - c_val * 2.0)
                    self.output_gripper_poses[gripper_key][a_i] = (
                            alpha_g * raw_val + (1 - alpha_g) * self.output_gripper_poses[gripper_key][a_i]
                    )
        else:
            # Generic mapping for other finger counts
            min_len = min(n_fingers, len(closure))
            for i in range(min_len):
                c_val = closure[i]
                raw_val = (c_val * 2.0 - 1.0) if hand_side == "Right" else (1.0 - c_val * 2.0)
                self.output_gripper_poses[gripper_key][i] = (
                        alpha_g * raw_val +
                        (1 - alpha_g) * self.output_gripper_poses[gripper_key][i]
                )

    def get_gripper_poses(self):
        """
        Get the latest gripper poses (thread-safe).
        This method processes the latest queued message if available.
        
        :return: Dictionary containing gripper pose arrays
        """
        # 尝试从队列中获取最新消息并处理
        latest_msg = self.peek_latest()
        if latest_msg is not None:
            with self.data_lock:
                for hand_side in ["Left", "Right"]:
                    if hand_side in latest_msg and latest_msg[hand_side] is not None:
                        # 检查是否需要更新（避免重复处理）
                        if self.last_recv_time[hand_side] > 0:
                            self._process_gripper_data(hand_side, latest_msg[hand_side])

        with self.data_lock:
            return self.output_gripper_poses.copy()


def qpos_to_action_tensor(qpos_data, robot, gripper_receiver=None):
    """
    Convert qpos array to flattened action tensor for the robot.
    
    :param qpos_data: Dictionary containing 'qpos' key with array/list
    :param robot: Robot instance
    :param gripper_receiver: Optional GripperDataReceiver instance for hand data
    :return: Flattened 1D tensor for env.step({robot.name: action})
    """
    try:
        # Fix 1: Use np.concatenate instead of list.extend (which returns None)
        import numpy as np
        raw_qpos = qpos_data['qpos']

        # Convert to numpy array if needed
        if isinstance(raw_qpos, th.Tensor):
            raw_qpos = raw_qpos.cpu().numpy()
        elif isinstance(raw_qpos, list):
            raw_qpos = np.array(raw_qpos)

        # Pad with 12 zeros
        padding_zeros = np.zeros(12)
        qpos_np = np.concatenate([raw_qpos, padding_zeros])

        # Convert to torch tensor
        qpos = th.tensor(qpos_np, dtype=th.float32)

        # G1WithInspireHand joint structure (48 dimensions after padding):
        # [0:3]: Reserved/unused
        # [3:7]: Reserved/unused  
        # [7:13]: Left leg joints (6 DOF)
        # [13:19]: Right leg joints (6 DOF)
        # [19:22]: Trunk joints (3 DOF: waist_yaw, waist_roll, waist_pitch)
        # [22:29]: Left arm joints (7 DOF)
        # [29:36]: Right arm joints (7 DOF)
        # [36:48]: Right gripper fingers (12 DOF - Inspire Hand)

        # Note: Left gripper is currently empty (no fingers configured)

        # Define joint segments based on the padded qpos structure
        # Note: Indices correspond to the qpos array after padding with 12 zeros
        joint_segments = {
            # 'base': qpos[:3], # Base is often handled separately or fixed
            'trunk': qpos[19:22],
            'arm_left': qpos[22:29],
            # 'gripper_left': qpos[...], # Add if needed and indices are known
            "leg_left": qpos[7:13],
            'arm_right': qpos[29:36],
            # 'gripper_right' will be filled from gripper_receiver if available
            "leg_right": qpos[13:19],
        }

        # Get gripper data from separate receiver if available
        if gripper_receiver is not None:
            gripper_poses = gripper_receiver.get_gripper_poses()
            if 'gripper_right' in gripper_poses:
                joint_segments['gripper_right'] = th.tensor(
                    gripper_poses['gripper_right'],
                    dtype=th.float32
                )

        # Map joint segments to controllers
        action_list = []
        for ctrl_name in robot.controller_order:
            controller = robot.controllers[ctrl_name]
            expected_dim = getattr(controller, 'command_dim', None) or \
                           getattr(controller, 'dof', None) or 1

            if ctrl_name in joint_segments:
                segment = joint_segments[ctrl_name]
                # Adjust dimension if needed
                if len(segment) != expected_dim:
                    if len(segment) > expected_dim:
                        segment = segment[:expected_dim]
                    else:
                        segment = th.cat([segment, th.zeros(expected_dim - len(segment))])
                action_list.append(segment.flatten())
            else:
                # Use current position as fallback
                if hasattr(controller, 'get_current_joint_positions'):
                    action_list.append(controller.get_current_joint_positions().flatten())
                else:
                    action_list.append(th.zeros(expected_dim))

        action_tensor = th.cat(action_list)

        # Verify dimensions
        if len(action_tensor) != robot.action_dim:
            print(f"[Action] ERROR: Action dim {len(action_tensor)} != expected {robot.action_dim}")
            return None

        return action_tensor

    except Exception as e:
        print(f"[Action] Error converting qpos: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    """
    Robot control demo with ZMQ remote control.
    Receives qpos data from ZMQ publisher and controls the robot.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose scene to load
    scene_model = "empty"
    if not quickstart:
        scene_model = choose_from_options(options=SCENES, name="scene", random_selection=random_selection)

    # Choose robot to create
    robot_name = "G1WithInspireHand"
    if not quickstart:
        robot_name = choose_from_options(
            options=list(sorted(REGISTERED_ROBOTS.keys())), name="robot", random_selection=random_selection
        )

    scene_cfg = dict()
    if scene_model == "empty":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model

    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    # Disable normalization for absolute position control
    robot0_cfg["action_normalize"] = False
    # Fix the base link to prevent movement
    # robot0_cfg["fixed_base"] = True
    # Move robot up by 1 meter (z-axis)
    # robot0_cfg["position"] = [0.0, 0.0, 1.0]
    # robot0_cfg["self_collisions"] = False  #好像不起作用

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)

    # Choose robot controller to use
    robot = env.robots[0]

    # Check actual robot structure
    print(f"\n{'=' * 70}")
    print(f"[DEBUG] Robot Analysis")
    print(f"{'=' * 70}")
    print(f"Robot name: {robot.name}")
    print(f"Robot n_dof: {robot.n_dof}")
    print(f"Robot action_dim: {robot.action_dim}")
    print(f"Controller order: {robot.controller_order}")
    print(f"\nController details:")

    total_action_dim = 0
    for ctrl_name in robot.controller_order:
        ctrl = robot.controllers[ctrl_name]
        if hasattr(ctrl, 'command_dim'):
            dim = ctrl.command_dim
        elif hasattr(ctrl, 'dof'):
            dim = ctrl.dof
        else:
            current_pos = ctrl.get_current_joint_positions()
            dim = len(current_pos) if hasattr(current_pos, '__len__') else 1

        total_action_dim += dim
        print(f"  {ctrl_name:20s}: {type(ctrl).__name__:35s} - {dim:2d} DOF")

    print(f"\nTotal calculated action_dim: {total_action_dim}")
    print(f"Expected action_dim: {robot.action_dim}")
    print(f"Match: {'✓' if total_action_dim == robot.action_dim else '✗'}")
    print(f"{'=' * 70}\n")

    # Determine controller configuration based on robot type
    print(f"robot.name: {robot.name}")

    # G1WithInspireHand configuration with JointController for arms and grippers
    controller_choices = {
        # "base": "HolonomicBaseJointController",
        "trunk": "JointController",
        "arm_left": "JointController",  # 7 DOF (direct joint control)
        "arm_right": "JointController",  # 7 DOF
        "gripper_right": "MultiFingerGripperController",  # 12 DOF (inspire hand fingers)
        "leg_left": "JointController",  # 6 DOF (synchronized)
        "leg_right": "JointController",  # 6 DOF
    }
    print("[Config] Using G1WithInspireHand controller configuration")
    print("[Config] Arms use JointController - send joint angles directly!")
    print("[Config] Gripper uses MultiFingerGripperController - send finger positions!")

    if not quickstart:
        controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    # Prepare controller config
    controller_config = {}
    for component, name in controller_choices.items():
        print(f"[Config] {component}: {name}")
        # Only JointController supports use_delta_commands parameter
        if name == "JointController":
            controller_config[component] = {"name": name, "use_delta_commands": False}
        elif name == "MultiFingerGripperController":
            controller_config[component] = {"name": name, "mode": "independent"}
        else:
            controller_config[component] = {"name": name}

    # Update the control mode of the robot
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_file()

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.46949, -3.97358, 2.21529]),
        orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    # Reset environment and robot
    env.reset()
    robot.reset()

    # Initialize ZMQ receiver
    zmq_receiver = ZMQReceiver(host="tcp://192.168.10.78:6666", topic="gmr.qpos", queue_maxlen=2)
    zmq_receiver.start()

    # Initialize Gripper Data Receiver
    gripper_receiver = GripperDataReceiver(host="tcp://127.0.0.1:7777", queue_maxlen=2)
    gripper_receiver.start()

    print("\n[ZMQ Control Mode] Waiting for messages from publisher...")
    print("[ZMQ Control Mode] Press ESC to quit\n")

    # Main control loop
    max_steps = -1 if not short_exec else 100
    step = 0
    last_message_time = time.time()
    timeout = 5.0  # Timeout if no message received for 5 seconds

    try:
        while step != max_steps:
            # Get latest message from queue
            message = zmq_receiver.get_latest()
            # message = "sdfsdfsdfsdf"

            if message is not None:
                # Convert qpos to action
                # action_tensor = qpos_to_action_tensor(message, robot, gripper_receiver=None)
                action_tensor = qpos_to_action_tensor(message, robot, gripper_receiver=gripper_receiver)

                if action_tensor is not None:
                    # Send action
                    action = {robot.name: action_tensor}
                    env.step(action=action)
                    last_message_time = time.time()

                    # # Print status occasionally
                    # if step % 60 == 0:
                    #     frame_idx = message.get('frame_idx', 'N/A')
                    #     print(f"[Step {step}] Received frame {frame_idx}, controlling robot...")
                else:
                    print("[Warning] Failed to convert qpos to action, maintaining current position")
                    # Maintain current position
                    action_list = []
                    for controller_name in robot.controller_order:
                        ctrl = robot.controllers[controller_name]
                        if hasattr(ctrl, 'get_current_joint_positions'):
                            action_list.append(ctrl.get_current_joint_positions().flatten())
                        else:
                            action_list.append(th.zeros(ctrl.command_dim))
                    action_tensor = th.cat(action_list)
                    action = {robot.name: action_tensor}
                    env.step(action=action)
            else:
                # No message available, maintain current position
                action_list = []
                for controller_name in robot.controller_order:
                    ctrl = robot.controllers[controller_name]
                    if hasattr(ctrl, 'get_current_joint_positions'):
                        action_list.append(ctrl.get_current_joint_positions().flatten())
                    else:
                        action_list.append(th.zeros(ctrl.command_dim))
                action_tensor = th.cat(action_list)
                action = {robot.name: action_tensor}
                env.step(action=action)

                # Check timeout
                if time.time() - last_message_time > timeout:
                    print(f"[Warning] No message received for {timeout} seconds")

            step += 1

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user")
    finally:
        print("\nShutting down...")
        zmq_receiver.stop()
        gripper_receiver.stop()
        og.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Control robot via ZMQ messages.")

    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Whether the example should be loaded with default settings for a quick start.",
    )
    args = parser.parse_args()
    main(quickstart=args.quickstart)
