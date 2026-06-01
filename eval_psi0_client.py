#!/usr/bin/env python3
"""
本地 OmniGibson 仿真评估脚本 - 使用 json_numpy
终极修复版：使用 OmniGibson 原生 dof_idx 进行绝对位置路由
"""

import sys
import time
import argparse
import numpy as np
import cv2
import requests
import json_numpy
from loguru import logger
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController

json_numpy.patch()

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.GUI = True


def to_numpy(x):
    if x is None: return None
    if hasattr(x, 'cpu'): return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray): return x
    if hasattr(x, '__array__'): return np.asarray(x)
    try:
        return np.array(x)
    except Exception:
        return None


def get_rgb_from_obs(obs):
    for key, value in obs.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    if 'rgb' in sub_value:
                        return to_numpy(sub_value['rgb'])
                    for deep_key, deep_value in sub_value.items():
                        if 'rgb' in deep_key.lower() and hasattr(deep_value, 'shape'):
                            return to_numpy(deep_value)
    return None


class RemotePSI0Client:
    def __init__(self, server_ip: str, server_port: int, timeout: float = 5.0):
        self.base_url = f"http://{server_ip}:{server_port}"
        self.act_url = f"{self.base_url}/act"
        self.health_url = f"{self.base_url}/health"
        self.timeout = timeout
        self.session = requests.Session()

        logger.info(f"Connecting to PSI0 server at {self.base_url}...")
        try:
            resp = self.session.get(self.health_url, timeout=5.0)
            if resp.status_code == 200:
                logger.success("✅ Connected to PSI0 server")
            else:
                raise ConnectionError(f"Health check failed: {resp.status_code}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to PSI0 server: {e}")
            raise

    def predict_action(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray:
        if state.ndim == 1: state = state.reshape(1, -1)
        payload = {
            "image": {"observation.images.egocentric": image},
            "state": {"states": state},
            "instruction": instruction,
            "history": {}, "condition": {}, "gt_action": None, "dataset_name": None, "timestamp": None
        }
        try:
            resp = self.session.post(self.act_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, str):
                logger.error(f"Server returned error string: {result}")
                return None
            if "action" in result:
                return np.array(result["action"], dtype=np.float32)
            else:
                return None
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None


def create_g1_robot_config():
    ctrl_config = {}
    for comp in ["base", "camera", "head", "torso", "arm_left", "arm_right", "gripper_left", "gripper_right"]:
        if comp == "base":
            ctrl_config[comp] = {"name": "JointController", "use_delta_commands": True}
        elif "arm" in comp:
            # 手臂：使用绝对位置控制 (use_delta_commands=False)
            ctrl_config[comp] = {
                "name": "JointController",
                "use_delta_commands": False,
                "motor_type": "position"
            }
        elif "gripper" in comp:
            ctrl_config[comp] = {
                "name": "MultiFingerGripperController",
                "mode": "independent",
                "command_input_limits": None,
                "command_output_limits": None,
            }

        else:
            ctrl_config[comp] = {"name": "JointController", "use_delta_commands": True}

    return {
        "type": "G1WithDex3Hand", "action_type": "continuous", "action_normalize": False,
        "obs_modalities": ["rgb"], "grasping_mode": "physical",
        "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_width": 640, "image_height": 480}}},
        "controller_config": ctrl_config
    }


def create_apple_scene():
    return {
        "scene": {"type": "Scene", "scene_model": "empty"},
        "objects": [
            {"type": "DatasetObject", "name": "table", "category": "breakfast_table", "model": "lcsizg",
             "position": [0.6, 0.0, 0.035], "fixed_base": True},
            {"type": "DatasetObject", "name": "bottle_of_barbecue_sauce", "category": "bottle_of_barbecue_sauce",
             "model": "ikbsox",
             "position": [0.4, -0.05, 0.2]},
            # {"type": "DatasetObject", "name": "apple_1", "category": "apple", "model": "agveuv", "density": 50.0,
            #  "position": [0.4, -0.05, 0.2]},
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", type=str, required=True)
    parser.add_argument("--server-port", type=int, default=8014)
    parser.add_argument("--task", type=str, default="Pick up the apple")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    client = RemotePSI0Client(server_ip=args.server_ip, server_port=args.server_port, timeout=args.timeout)

    logger.info("Starting OmniGibson environment...")
    env = og.Environment(configs={**create_apple_scene(), "robots": [create_g1_robot_config()]})
    robot = env.robots[0]
    obs, _ = env.reset()
    logger.success("✅ Environment ready")

    kb = KeyboardRobotController(robot=robot)
    reset_flag, quit_flag = [False], [False]

    kb.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.R, description="Reset",
                                  callback_fn=lambda: reset_flag.__setitem__(0, True))
    kb.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.Q, description="Quit",
                                  callback_fn=lambda: quit_flag.__setitem__(0, True))

    target_dt = 1.0 / args.fps
    step_count, action_idx = 0, 0
    action_buffer = None

    while step_count < args.max_steps and not quit_flag[0]:
        loop_start = time.time()
        _ = kb.get_teleop_action()

        if reset_flag[0]:
            obs, _ = env.reset()
            action_buffer, action_idx, step_count, reset_flag[0] = None, 0, 0, False
            continue

        rgb_img = get_rgb_from_obs(obs)
        if rgb_img is None:
            obs, *_ = env.step(np.zeros(robot.action_dim))
            step_count += 1
            continue

        if rgb_img.dtype in [np.float32, np.float64]:
            rgb_img = (rgb_img * 255).astype(np.uint8) if rgb_img.max() <= 1.0 else rgb_img.astype(np.uint8)
        elif rgb_img.dtype != np.uint8:
            rgb_img = rgb_img.astype(np.uint8)
        if len(rgb_img.shape) == 2:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
        elif rgb_img.shape[2] == 4:
            rgb_img = rgb_img[:, :, :3]

        # 提取 State
        state_t = robot.get_joint_positions()
        if hasattr(state_t, 'cpu'): state_t = state_t.cpu().numpy()

        state_36 = np.zeros(36, dtype=np.float32)
        upper_body_len = min(len(state_t), 36)
        state_36[:upper_body_len] = state_t[:upper_body_len]

        if action_buffer is not None and action_idx < len(action_buffer):
            action = action_buffer[action_idx]
            action_idx += 1
            if action_idx >= len(action_buffer): action_buffer, action_idx = None, 0
        else:
            action_seq = client.predict_action(rgb_img, state_36.reshape(1, -1), args.task)
            if action_seq is None:
                action = state_36.copy()
            else:
                if len(action_seq.shape) == 2:
                    action_buffer = action_seq
                    action = action_buffer[0]
                    action_idx = 1
                else:
                    action = action_seq

        env_action = np.zeros(robot.action_dim, dtype=np.float32)

        # ==================== 【终极修复：依赖原生 dof_idx】 ====================
        # 在 OmniGibson 中，每个 controller 都有 dof_idx 属性，
        # 它代表这个控制器负责的关节在 robot.get_joint_positions() 中的绝对下标。
        for comp_name, act_indices in robot.controller_action_idx.items():
            if comp_name == "base" or "head" in comp_name or "camera" in comp_name:
                continue

            controller = robot.controllers[comp_name]
            dof_indices = getattr(controller, 'dof_idx', [])

            for act_i, dof_i in zip(act_indices, dof_indices):
                # 只有当底层物理索引在我们录制的 36 维度以内时才赋值
                if dof_i < 36:
                    env_action[int(act_i)] = action[int(dof_i)]
        # =========================================================================

        # 增强诊断日志：开局验证映射关系
        if step_count == 0:
            logger.info(f"--- 诊断：底层物理映射表 ---")
            for comp_name, act_indices in robot.controller_action_idx.items():
                if comp_name == "base": continue
                controller = robot.controllers[comp_name]
                dof_indices = getattr(controller, 'dof_idx', [])
                logger.info(f" [{comp_name}] EnvAction索引 {act_indices}  <---  模型State/Action维度 {dof_indices}")
            logger.info(f"--- 诊断：首帧模型 Action 值 ---")
            logger.info(f" State (36维前31): {np.round(state_36[:31], 3)}")
            logger.info(f" Action(36维前31): {np.round(action[:31], 3)}")
            logger.info(f"---------------------------")

        if step_count <= 3:
            for comp_name in ["gripper_left", "gripper_right"]:
                if comp_name in robot.controller_action_idx:
                    act_indices = robot.controller_action_idx[comp_name]
                    controller = robot.controllers[comp_name]
                    dof_indices = getattr(controller, 'dof_idx', [])
                    gripper_act = env_action[act_indices] if len(act_indices) > 0 else []
                    gripper_model = action[dof_indices] if len(dof_indices) > 0 else []
                    logger.info(f" Step{step_count} [{comp_name}] model输出: {np.round(gripper_model, 3)} → env_action: {np.round(gripper_act, 3)}")

        if step_count % 50 == 0:
            left_arm_idx = robot.controller_action_idx.get("arm_left", [])
            right_arm_idx = robot.controller_action_idx.get("arm_right", [])

            if len(left_arm_idx) > 0:
                l_start = int(left_arm_idx[0])
                logger.info(
                    f"Step {step_count}: Left Arm 最终送入仿真的指令  -> {np.round(env_action[l_start:l_start + 6], 3)}")
            if len(right_arm_idx) > 0:
                r_start = int(right_arm_idx[0])
                logger.info(
                    f"Step {step_count}: Right Arm 最终送入仿真的指令 -> {np.round(env_action[r_start:r_start + 6], 3)}")

        obs, *_ = env.step(env_action)
        step_count += 1

        elapsed = time.time() - loop_start
        if elapsed < target_dt: time.sleep(target_dt - elapsed)

    env.close()
    logger.success(f"✅ Evaluation complete! Total steps: {step_count}")


if __name__ == "__main__":
    main()