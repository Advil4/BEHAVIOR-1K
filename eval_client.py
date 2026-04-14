import sys
import time

import cv2
import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from loguru import logger
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from transformers import AutoTokenizer

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.GUI = True

CKPT_PATH = "/home/ubuntu/Project/BEHAVIOR-1K/smolvla_omni_formal/checkpoints/020000/pretrained_model"
VLM_PATH = "/home/ubuntu/Project/BEHAVIOR-1K/SmolVLM2-500M-Video-Instruct"
DEVICE = torch.device("cuda")
TASK_TEXT = "Put the apple on the plate"


def load_policy():
    logger.info(f"正在加载模型: {CKPT_PATH}")
    policy = SmolVLAPolicy.from_pretrained(CKPT_PATH)
    policy.to(DEVICE)
    policy.eval()

    logger.info(f"正在加载分词器: {VLM_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(VLM_PATH)
    tokens = tokenizer(
        TASK_TEXT, return_tensors="pt", padding="max_length", truncation=True, max_length=48
    ).to(DEVICE)
    return policy, tokens


def get_rgb_from_obs(obs):
    for key, value in obs.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    if 'rgb' in sub_value: return to_numpy(sub_value['rgb'])
                    for deep_key, deep_value in sub_value.items():
                        if 'rgb' in deep_key.lower() and hasattr(deep_value, 'shape'):
                            return to_numpy(deep_value)
    return None


def to_numpy(x):
    """安全剥离 PyTorch Tensor，确保返回干净的 NumPy 数组"""
    if x is None: return None
    if hasattr(x, 'detach'): return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray): return x
    if hasattr(x, '__array__'): return np.asarray(x)
    try:
        return np.array(x)
    except Exception:
        return None


def main():
    ctrl_config = {}
    for comp in ["base", "camera", "head", "torso", "arm_left", "arm_right", "gripper_left", "gripper_right"]:
        if comp == "base":
            ctrl_config[comp] = {"name": "HolonomicBaseJointController"}
        elif "arm" in comp or "gripper" in comp:
            ctrl_config[comp] = {
                "name": "JointController",
                "use_delta_commands": True,
                "command_output_limits": None
            }
        else:
            ctrl_config[comp] = {
                "name": "JointController",
                "use_delta_commands": True
            }

    robot_cfg = {
        "type": "R1ProWithSvhHand",
        "action_type": "continuous",
        "obs_modalities": ["rgb", "proprio"],
        "grasping_mode": "assisted",
        "sensor_config": {
            "VisionSensor": {"sensor_kwargs": {"image_width": 640, "image_height": 480}}
        },
        "controller_config": ctrl_config
    }

    cfg = {"scene": {"type": "Scene", "scene_model": "empty"},
           "objects": [
               {"type": "DatasetObject", "name": "table", "category": "breakfast_table", "model": "lcsizg",
                "position": [0.6, 0.0, 1.2], "fixed_base": True},
               {"type": "DatasetObject", "name": "apple_1", "category": "apple", "model": "agveuv", "density": 50.0,
                "position": [0.6, -0.1, 1.3]},
               {"type": "DatasetObject", "name": "plate_1", "category": "plate", "model": "aewthq",
                "position": [0.6, 0.05, 1.4]},
           ],
           "robots": [robot_cfg]}

    logger.info("正在启动 OmniGibson 仿真器...")
    env = og.Environment(configs=cfg)
    robot = env.robots[0]

    obs, _ = env.reset()

    policy, tokens = load_policy()
    logger.success("✅ 模型加载完毕，准备开始闭环控制！")

    kb_generator = KeyboardRobotController(robot=robot)
    reset_flag = [False]

    def trigger_reset():
        reset_flag[0] = True
        logger.info("🔄 检测到 'R' 键，准备重置环境和模型缓存...")

    kb_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset Environment and Policy",
        callback_fn=trigger_reset,
    )
    logger.info("👉 提示：在仿真界面按下 'R' 键可以随时安全重置环境。")

    TARGET_FPS = 30.0
    target_dt = 1.0 / TARGET_FPS

    while True:
        loop_start = time.time()
        _ = kb_generator.get_teleop_action()

        # 处理重置请求
        if reset_flag[0]:
            obs, _ = env.reset()

            if hasattr(policy, 'reset'):
                policy.reset()  # 清空模型缓存

            reset_flag[0] = False
            logger.success("✅ 环境已重置，重新开始预测！")
            continue

        rgb_img = get_rgb_from_obs(obs)
        if rgb_img is None or not isinstance(rgb_img, np.ndarray):
            obs, _, _, _, _ = env.step(np.zeros(robot.action_dim))
            continue

        if rgb_img.dtype in [np.float32, np.float64]:
            rgb_img = (rgb_img * 255).astype(np.uint8) if rgb_img.max() <= 1.0 else rgb_img.astype(np.uint8)
        elif rgb_img.dtype != np.uint8:
            rgb_img = rgb_img.astype(np.uint8)

        if len(rgb_img.shape) == 2:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
        elif rgb_img.shape[2] == 4:
            rgb_img = rgb_img[:, :, :3]

        img_resized = cv2.resize(rgb_img, (640, 480), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        current_joints_raw = robot.get_joint_positions()
        current_joints = to_numpy(current_joints_raw)

        state_tensor = torch.from_numpy(current_joints).float()
        if state_tensor.shape[0] < 64:
            state_tensor = torch.cat([state_tensor, torch.zeros(64 - state_tensor.shape[0])])
        elif state_tensor.shape[0] > 64:
            state_tensor = state_tensor[:64]
        state_tensor = state_tensor.unsqueeze(0).to(DEVICE)

        batch = {
            "observation.image": img_tensor,
            "observation.state": state_tensor,
            "observation.language.tokens": tokens["input_ids"],
            "observation.language.attention_mask": tokens["attention_mask"].bool(),
        }

        with torch.no_grad():
            actions = policy.select_action(batch)
            target_joints = actions[0].cpu().numpy()

        env_action = np.zeros(robot.action_dim)

        safe_target_joints = np.zeros(max(64, len(target_joints)))
        safe_target_joints[:len(target_joints)] = target_joints

        safe_current_joints = np.zeros(max(64, len(current_joints)))
        safe_current_joints[:len(current_joints)] = current_joints

        for comp in ["arm_right", "gripper_right"]:
            if comp in robot.controllers:
                action_idx = robot.controller_action_idx.get(comp, [])

                try:
                    dof_idx = robot.controllers[comp].dof_idx
                except AttributeError:
                    continue

                if len(action_idx) == 0 or len(dof_idx) == 0: continue

                delta = safe_target_joints[dof_idx] - safe_current_joints[dof_idx]

                limit = 0.1 if "arm" in comp else 0.3
                safe_delta = np.clip(delta, -limit, limit)

                env_action[action_idx] = safe_delta

        obs, reward, done, truncated, info = env.step(env_action)

        elapsed = time.time() - loop_start
        if elapsed < target_dt: time.sleep(target_dt - elapsed)


if __name__ == "__main__":
    main()
