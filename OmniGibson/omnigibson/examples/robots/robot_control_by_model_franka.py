import os
import time
import json
import math
import glob
import torch as th
import numpy as np
import argparse
from collections import deque

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
import omnigibson.utils.transform_utils as T

# HuggingFace Tokenizer (用于解决 language tokens 报错)
from transformers import AutoTokenizer

# LeRobot 导入
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

try:
    from safetensors.torch import safe_open
except ImportError:
    safe_open = None

# 优化 OmniGibson 性能
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


SCENES = dict(
    Pomaria_2_int_data_collection_8="Pomaria_2_int_data_collection_8",
    Rs_int="Realistic interactive home environment (default)",
    office_large="office large",
    school_chemistry="school chemistry",
    Ihlen_1_int="Ihlen 1 int",
    empty="Empty environment with no objects",
)

DEFAULT_SCENE_MODEL = "Pomaria_2_int_data_collection_8"
DEFAULT_ROBOT_NAME = "franka"
DEFAULT_END_EFFECTOR = "mounted"
DEFAULT_TASK_DESCRIPTION = "Robot teleoperation task"
TARGET_TABLE_NAME = "breakfast_table_mwsyqb"
APPLE_NAME = "apple_yyuiva"
BANANA_NAME = "banana_znakxm"
COFFEE_CUP_NAME = "coffee_cup_nhzrei"
MICROPHONE_NAME = "microphone_eyleyk"
STRAWBERRY_NAME = "strawberry_table_extra"
BOTTLE_NAME = "bottle_table_extra"
PEN_NAME = "pen_table_extra"
PLATE_NAME = "plate_table_extra"
SPONGE_NAME = "sponge_table_extra"
SOAP_NAME = "soap_table_extra"
LEMON_NAME = "lemon_table_extra"
CANDY_NAME = "candy_table_extra"
LAPTOP_NAME = "laptop_izydvb"
DESK_PHONE_NAME = "desk_phone_gntohf"
EXTRA_TABLE_OBJECT_SPECS = (
    {"name": STRAWBERRY_NAME, "category": "strawberry", "model": "ntxofi", "mass": 0.03},
    {"name": BOTTLE_NAME, "category": "salt_bottle", "model": "wdpcmk", "mass": 0.06},
    {"name": PEN_NAME, "category": "pen", "model": "gzrbrx", "mass": 0.02},
    {"name": PLATE_NAME, "category": "plate", "model": "vitdwc", "mass": 0.08},
    {"name": SPONGE_NAME, "category": "sponge", "model": "klwueh", "mass": 0.03},
    {"name": SOAP_NAME, "category": "bar_soap", "model": "lyigsj", "mass": 0.05},
    {"name": LEMON_NAME, "category": "lemon", "model": "bbvonm", "mass": 0.04},
    {"name": CANDY_NAME, "category": "hard_candy", "model": "vwegtt", "mass": 0.02},
)
TABLE_OBJECT_NAMES = (
    APPLE_NAME,
    BANANA_NAME,
    COFFEE_CUP_NAME,
    MICROPHONE_NAME,
    STRAWBERRY_NAME,
    BOTTLE_NAME,
    PEN_NAME,
    PLATE_NAME,
    SPONGE_NAME,
    SOAP_NAME,
    LEMON_NAME,
    CANDY_NAME,
)
HIDDEN_TABLE_OBJECT_NAMES = (LAPTOP_NAME, DESK_PHONE_NAME)
HIDDEN_OBJECT_BASE_POSITION = th.tensor([0.0, 0.0, -10.0], dtype=th.float32)
TABLE_RANDOMIZE_EDGE_MARGIN = 0.06
TABLE_RANDOMIZE_MIN_OBJECT_DISTANCE = 0.09
TABLE_RANDOMIZE_Z_OFFSET = 0.015
ARM_TELEOP_SPEED_SCALE = 0.2
RESET_EEF_LIFT_HEIGHT = 0.2
RESET_EEF_LIFT_STEPS = 10
RESET_EEF_SETTLE_STEPS = 10
CAMERA_IMAGE_SIZE = 512
DEFAULT_TARGET_HZ = 30
DEFAULT_IMAGE_HISTORY_FRAMES = 1
DEFAULT_IMAGE_HISTORY_INTERVAL = 0.5
FRANKA_TABLE_FRONT_YAW = math.pi
FRANKA_TABLE_FRONT_POSITION = th.tensor([-0.30, 0.234, 0.0], dtype=th.float32)
FRANKA_TABLE_FRONT_ORIENTATION = T.euler2quat(th.tensor([0.0, 0.0, FRANKA_TABLE_FRONT_YAW], dtype=th.float32))
VIEWER_CAMERA_POSITION = th.tensor([1.35, 0.15, 1.45], dtype=th.float32)
VIEWER_CAMERA_ORIENTATION = T.euler2quat(th.tensor([1.10, 0.0, math.pi / 2.0], dtype=th.float32))


class LeRobotInferencer:
    """LeRobot 推理器：加载模型，并手动复现 LeRobot 的 state/action 归一化处理。"""
    
    def __init__(self, model_dir, robot, device="cuda", task_description=DEFAULT_TASK_DESCRIPTION):
        self.robot = robot
        self.device = device
        self.task_description = task_description
        self.config = self._load_model_config(model_dir)
        self.input_features = self.config.get("input_features", {})
        self.output_features = self.config.get("output_features", {})
        self.expected_image_keys = [
            key for key, feature in self.input_features.items()
            if key.startswith("observation.image") or feature.get("type") == "VISUAL"
        ]
        self.state_dim = self.input_features.get("observation.state", {}).get("shape", [len(robot.get_joint_positions())])[0]
        self.action_dim = self.output_features.get("action", {}).get("shape", [robot.action_dim])[0]
        self.normalizer_stats = self._load_normalizer_stats(model_dir)
        
        print(f"🔄 正在加载模型权重自: {model_dir}")
        self.policy = SmolVLAPolicy.from_pretrained(model_dir, local_files_only=True)
        self.policy.to(self.device)
        self.policy.eval()
        
        print(f"🔄 正在加载 Tokenizer...")
        # 加载分词器，解决 language.tokens KeyError 的问题
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        
        print("✅ 模型和 Tokenizer 加载成功！")
        print(f"📦 模型配置: state_dim={self.state_dim}, action_dim={self.action_dim}")
        print(f"📷 模型图像输入: {self.expected_image_keys}")
        print(f"📝 当前任务描述: {self.task_description}")
        
        # 检测机器人有哪些摄像头
        self.camera_names = self._detect_cameras()

    def _load_model_config(self, model_dir):
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"⚠️ 未找到模型 config.json，将使用 Franka 当前机器人维度: {config_path}")
            return {
                "input_features": {"observation.state": {"type": "STATE", "shape": [len(self.robot.get_joint_positions())]}},
                "output_features": {"action": {"type": "ACTION", "shape": [self.robot.action_dim]}},
            }
        with open(config_path, "r") as f:
            return json.load(f)

    def _load_normalizer_stats(self, model_dir):
        if safe_open is None:
            print("⚠️ safetensors 未安装，state/action 将不做 mean/std 归一化或反归一化。")
            return {}

        stats_path = os.path.join(model_dir, "policy_preprocessor_step_5_normalizer_processor.safetensors")
        if not os.path.exists(stats_path):
            candidates = sorted(glob.glob(os.path.join(model_dir, "*normalizer*.safetensors")))
            stats_path = candidates[0] if candidates else stats_path
        if not os.path.exists(stats_path):
            print(f"⚠️ 未找到 normalizer stats: {stats_path}")
            return {}

        stats = {}
        with safe_open(stats_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                feature_name, stat_name = key.rsplit(".", 1)
                stats.setdefault(feature_name, {})[stat_name] = f.get_tensor(key).to(self.device)
        return stats

    def _fit_vector(self, value, target_dim):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(value) < target_dim:
            value = np.pad(value, (0, target_dim - len(value)), mode="constant")
        elif len(value) > target_dim:
            value = value[:target_dim]
        return value

    def _mean_std_normalize(self, value, feature_key):
        feature_stats = self.normalizer_stats.get(feature_key)
        if not feature_stats or "mean" not in feature_stats or "std" not in feature_stats:
            return value
        mean = feature_stats["mean"].to(dtype=value.dtype, device=value.device)
        std = feature_stats["std"].to(dtype=value.dtype, device=value.device)
        return (value - mean) / (std + 1e-8)

    def _mean_std_unnormalize(self, value, feature_key):
        feature_stats = self.normalizer_stats.get(feature_key)
        if not feature_stats or "mean" not in feature_stats or "std" not in feature_stats:
            return value
        mean = feature_stats["mean"].to(dtype=value.dtype, device=value.device)
        std = feature_stats["std"].to(dtype=value.dtype, device=value.device)
        return value * std + mean
    
    def _detect_cameras(self):
        """检测机器人有哪些摄像头"""
        cameras = {}
        if hasattr(self.robot, 'sensors'):
            for sensor_name, sensor in self.robot.sensors.items():
                from omnigibson.sensors import VisionSensor
                if isinstance(sensor, VisionSensor):
                    sensor_lower = sensor_name.lower()
                    if any(keyword in sensor_lower for keyword in ['eyes', 'head', 'zed']):
                        cameras['head'] = sensor_name
                    elif any(keyword in sensor_lower for keyword in ['eef', 'wrist', 'hand', 'gripper', 'realsense']):
                        if 'left' in sensor_lower:
                            cameras['left_wrist'] = sensor_name
                        elif 'right' in sensor_lower:
                            cameras['right_wrist'] = sensor_name
                        else:
                            cameras['wrist'] = sensor_name
        return cameras

    def _find_rgb_image(self, obs_dict, sensor_name):
        if sensor_name in obs_dict and isinstance(obs_dict[sensor_name], dict) and 'rgb' in obs_dict[sensor_name]:
            return obs_dict[sensor_name]['rgb']

        robot_prefix = sensor_name.split(':')[0] if ':' in sensor_name else sensor_name
        if robot_prefix in obs_dict and isinstance(obs_dict[robot_prefix], dict):
            if sensor_name in obs_dict[robot_prefix] and isinstance(obs_dict[robot_prefix][sensor_name], dict):
                if 'rgb' in obs_dict[robot_prefix][sensor_name]:
                    return obs_dict[robot_prefix][sensor_name]['rgb']

        def find_rgb_in_dict(d, target_key, depth=0):
            if depth > 3:
                return None
            if isinstance(d, dict):
                if target_key in d and isinstance(d[target_key], dict) and 'rgb' in d[target_key]:
                    return d[target_key]['rgb']
                for _, v in d.items():
                    result = find_rgb_in_dict(v, target_key, depth + 1)
                    if result is not None:
                        return result
            return None

        return find_rgb_in_dict(obs_dict, sensor_name)

    def extract_image_tensors(self, obs_dict):
        """Extract current camera images as CHW float tensors in [0, 1]."""
        image_tensors = {}
        for cam_name, sensor_name in self.camera_names.items():
            image_key = f"observation.image.{cam_name}"
            if self.expected_image_keys and image_key not in self.expected_image_keys:
                continue

            rgb_img = self._find_rgb_image(obs_dict, sensor_name)
            if rgb_img is None:
                print(f"⚠️ 推理错误: 找不到摄像头 {cam_name} 的图像！")
                continue

            if hasattr(rgb_img, 'detach'):
                rgb_img = rgb_img.detach().cpu().numpy()
            if rgb_img.shape[-1] == 4:
                rgb_img = rgb_img[:, :, :3]

            img_chw = np.transpose(rgb_img, (2, 0, 1)).astype(np.float32) / 255.0
            image_tensors[image_key] = th.from_numpy(img_chw)

        return image_tensors

    def reset(self):
        if hasattr(self, "policy"):
            self.policy.reset()
        
    def predict_action(self, obs_dict, state, image_sequences=None):
        """
        将当前环境状态传入模型，获取预测动作，并处理维度对齐
        """
        # ==================== 1. 处理 State：先对原始 state_dim 做归一化，policy 内部再 pad 到 max_state_dim ====================
        state_np = self._fit_vector(state, self.state_dim)
        state_tensor = th.from_numpy(state_np).to(self.device)
        state_tensor = self._mean_std_normalize(state_tensor, "observation.state")
            
        batch = {
            "observation.state": state_tensor.unsqueeze(0)
        }
        
        # ==================== 2. 处理摄像头图像 ====================
        if image_sequences is None:
            image_sequences = {
                image_key: [image_tensor]
                for image_key, image_tensor in self.extract_image_tensors(obs_dict).items()
            }

        for image_key, frames in image_sequences.items():
            if self.expected_image_keys and image_key not in self.expected_image_keys:
                continue
            if not frames:
                continue

            frames = [frame.to(dtype=th.float32) for frame in frames]
            if len(frames) == 1:
                img_tensor = frames[-1].unsqueeze(0).to(self.device)
            else:
                img_tensor = th.stack(frames, dim=0).unsqueeze(0).to(self.device)
            batch[image_key] = img_tensor

        # ==================== 3. 处理文本指令 (解决 KeyError) ====================        
        # 使用 tokenizer 把文本转成 ID 和 Attention Mask
        tokenized = self.tokenizer(self.task_description, return_tensors="pt")
        batch["observation.language.tokens"] = tokenized["input_ids"].to(self.device)
        batch["observation.language.attention_mask"] = tokenized["attention_mask"].to(self.device).bool()
        # 兼容性保留
        batch["task"] = [self.task_description]
        
        # ==================== 4. 模型前向推理 ====================
        with th.no_grad():
            time_begin = time.time()
            action_tensor = self.policy.select_action(batch)
            print("模型纯推理时间:", time.time() - time_begin)
            
        if action_tensor.dim() > 1:
            action_tensor = action_tensor.squeeze(0)

        # select_action 返回的是归一化后的 action，需要反归一化回仿真动作空间。
        action_tensor = self._mean_std_unnormalize(action_tensor, "action")
        action_np = action_tensor.detach().cpu().numpy().astype(np.float32)

        robot_action_dim = self.robot.action_dim
        if len(action_np) >= robot_action_dim:
            action_np = action_np[:robot_action_dim]
        else:
            action_np = np.pad(action_np, (0, robot_action_dim - len(action_np)), mode='constant')
            
        return action_np


class ImageHistoryBuffer:
    """Keeps recent camera frames and samples them by fixed control-step intervals."""

    def __init__(self, inferencer, history_frames=DEFAULT_IMAGE_HISTORY_FRAMES, interval_sec=DEFAULT_IMAGE_HISTORY_INTERVAL, target_hz=DEFAULT_TARGET_HZ):
        self.inferencer = inferencer
        self.history_frames = max(1, int(history_frames))
        self.interval_sec = max(0.0, float(interval_sec))
        self.target_hz = float(target_hz)
        self.interval_steps = max(1, int(round(self.interval_sec * self.target_hz)))
        self.maxlen = max(1, (self.history_frames - 1) * self.interval_steps + 1)
        self.records = deque(maxlen=self.maxlen)

    def clear(self):
        self.records.clear()

    def append(self, obs_dict, step_idx):
        self.records.append(
            {
                "step": int(step_idx),
                "images": self.inferencer.extract_image_tensors(obs_dict),
            }
        )

    def sample(self):
        if not self.records:
            return None

        current_step = self.records[-1]["step"]
        sampled = {}
        for history_idx in range(self.history_frames):
            target_step = current_step - (self.history_frames - 1 - history_idx) * self.interval_steps
            record = self._record_at_or_before(target_step)
            for image_key, image_tensor in record["images"].items():
                sampled.setdefault(image_key, []).append(image_tensor)

        return sampled

    def _record_at_or_before(self, target_step):
        fallback = self.records[0]
        for record in reversed(self.records):
            if record["step"] <= target_step:
                return record
        return fallback


def choose_controllers(robot, random_selection=False):
    """交互式选择每个关节的控制方式"""
    controller_choices = dict()
    default_config = robot._default_controller_config
    controller_names = robot.controller_order
    
    for controller_name in controller_names:
        controller_options = default_config[controller_name]
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options,
            name=f"{controller_name} controller",
            random_selection=random_selection,
        )
        controller_choices[controller_name] = choice

    return controller_choices


def get_robot_state(robot):
    """Use Franka joint positions as the low-dimensional policy state."""
    return robot.get_joint_positions().to(dtype=th.float32)


def fix_franka_in_front_of_table(robot):
    robot.set_position_orientation(
        position=FRANKA_TABLE_FRONT_POSITION,
        orientation=FRANKA_TABLE_FRONT_ORIENTATION,
    )
    print(
        f"[init] Fixed {robot.name} in front of {TARGET_TABLE_NAME}: "
        f"pos={FRANKA_TABLE_FRONT_POSITION.tolist()}, yaw={FRANKA_TABLE_FRONT_YAW:.3f} rad"
    )


def open_robot_gripper(robot):
    if not hasattr(robot, "gripper_control_idx"):
        return

    joint_positions = robot.get_joint_positions()
    control_limits = robot.control_limits.get("position", robot.control_limits.get("joint_position"))
    if control_limits is None:
        print("[init-warn] Could not open gripper; robot position control limits were not found.")
        return

    lower_limits, upper_limits = control_limits
    open_limits = lower_limits if getattr(robot, "_grasping_direction", "lower") == "upper" else upper_limits
    for arm, gripper_idx in robot.gripper_control_idx.items():
        open_qpos = open_limits[gripper_idx]
        joint_positions[gripper_idx] = open_qpos.to(dtype=joint_positions.dtype, device=joint_positions.device)
        print(
            f"[init] Opened gripper for arm {arm}: "
            f"qpos={open_qpos.tolist()}, grasping_direction={getattr(robot, '_grasping_direction', 'lower')}"
        )

    robot.set_joint_positions(joint_positions, drive=False)
    robot.set_joint_velocities(th.zeros(robot.n_dof, dtype=joint_positions.dtype, device=joint_positions.device))


def set_gripper_open_action(action, robot):
    action = np.asarray(action, dtype=np.float32)
    for controller_name, action_indices in robot.controller_action_idx.items():
        if not controller_name.startswith("gripper"):
            continue

        controller = robot.controllers[controller_name]
        action_indices_np = action_indices.cpu().numpy() if hasattr(action_indices, "cpu") else np.asarray(action_indices)
        if getattr(controller, "_mode", None) == "binary":
            action[action_indices_np] = 1.0
        elif getattr(controller, "_motor_type", None) == "position" and hasattr(controller, "dof_idx"):
            lower_limits, upper_limits = robot.control_limits["position"]
            open_limits = lower_limits if getattr(robot, "_grasping_direction", "lower") == "upper" else upper_limits
            action[action_indices_np] = _to_numpy(open_limits[controller.dof_idx])
        else:
            action[action_indices_np] = 1.0
    return action


def reset_keyboard_gripper_to_open(action_generator):
    if not hasattr(action_generator, "binary_grippers"):
        return

    for binary_gripper in action_generator.binary_grippers:
        action_generator.gripper_direction[binary_gripper] = 1.0
        action_generator.persistent_gripper_action[binary_gripper] = 1.0
    action_generator.toggling_gripper = False


def step_with_open_gripper(env, robot, num_steps=10):
    obs = None
    open_action = set_gripper_open_action(np.zeros(robot.action_dim, dtype=np.float32), robot)
    for _ in range(num_steps):
        obs, _, _, _, _ = env.step(action=open_action)
    return obs


def lift_end_effector_after_reset(env, robot):
    action = set_gripper_open_action(np.zeros(robot.action_dim, dtype=np.float32), robot)

    arm_indices = robot.controller_action_idx.get("arm_0")
    if arm_indices is None:
        print("[init-warn] Could not lift EEF after reset; arm_0 action indices were not found.")
        return step_with_open_gripper(env, robot, num_steps=RESET_EEF_SETTLE_STEPS)

    arm_indices_np = arm_indices.cpu().numpy() if hasattr(arm_indices, "cpu") else np.asarray(arm_indices)
    if len(arm_indices_np) < 3:
        print("[init-warn] Could not lift EEF after reset; arm_0 command has fewer than 3 dimensions.")
        return step_with_open_gripper(env, robot, num_steps=RESET_EEF_SETTLE_STEPS)

    # IK input is normalized to [-1, 1], which maps to roughly [-0.2m, 0.2m].
    action[arm_indices_np[2]] = min(1.0, RESET_EEF_LIFT_HEIGHT / 0.2 / RESET_EEF_LIFT_STEPS)
    for _ in range(RESET_EEF_LIFT_STEPS):
        obs, _, _, _, _ = env.step(action=action)

    obs = step_with_open_gripper(env, robot, num_steps=RESET_EEF_SETTLE_STEPS)
    print(f"[init] Lifted EEF upward by about {RESET_EEF_LIFT_HEIGHT:.2f}m for a wider global view.")
    return obs


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _sample_table_xy(table, obj, rng, occupied_xy):
    table_low, table_high = (_to_numpy(corner) for corner in table.aabb)
    obj_extent = _to_numpy(obj.aabb_extent)
    clearance = np.maximum(obj_extent[:2] * 0.5 + TABLE_RANDOMIZE_EDGE_MARGIN, TABLE_RANDOMIZE_EDGE_MARGIN)

    low_xy = table_low[:2] + clearance
    high_xy = table_high[:2] - clearance
    if np.any(low_xy > high_xy):
        table_center_xy = (table_low[:2] + table_high[:2]) * 0.5
        table_half_extent_xy = np.maximum((table_high[:2] - table_low[:2]) * 0.25, TABLE_RANDOMIZE_EDGE_MARGIN)
        low_xy = table_center_xy - table_half_extent_xy
        high_xy = table_center_xy + table_half_extent_xy

    min_distance = max(TABLE_RANDOMIZE_MIN_OBJECT_DISTANCE, float(np.max(obj_extent[:2])))
    for _ in range(100):
        xy = rng.uniform(low=low_xy, high=high_xy)
        if all(np.linalg.norm(xy - prev_xy) >= min_distance for prev_xy in occupied_xy):
            return xy

    return rng.uniform(low=low_xy, high=high_xy)


def _zero_object_velocity(obj, dtype=th.float32, device=None):
    if hasattr(obj, "set_linear_velocity"):
        obj.set_linear_velocity(th.zeros(3, dtype=dtype, device=device))
    if hasattr(obj, "set_angular_velocity"):
        obj.set_angular_velocity(th.zeros(3, dtype=dtype, device=device))


def hide_unwanted_table_objects(env):
    hidden_log = []
    for idx, obj_name in enumerate(HIDDEN_TABLE_OBJECT_NAMES):
        obj = env.scene.object_registry("name", obj_name)
        if obj is None:
            print(f"[init-warn] Could not hide {obj_name}; object was not found.")
            continue

        old_pos, old_orn = obj.get_position_orientation()
        hidden_pos = HIDDEN_OBJECT_BASE_POSITION.to(dtype=old_pos.dtype, device=old_pos.device).clone()
        hidden_pos[1] += 0.5 * idx

        obj.set_position_orientation(position=hidden_pos, orientation=old_orn)
        obj.visible = False
        if hasattr(obj, "disable_collisions"):
            obj.disable_collisions()
        if hasattr(obj, "disable_gravity"):
            obj.disable_gravity()
        _zero_object_velocity(obj, dtype=old_pos.dtype, device=old_pos.device)
        hidden_log.append(obj_name)

    if hidden_log:
        print(f"[init] Hidden table objects: {', '.join(hidden_log)}")


def set_object_mass(obj, mass):
    dynamic_links = [link for link in obj.links.values() if hasattr(link, "mass")]
    if not dynamic_links:
        return

    link_mass = float(mass) / len(dynamic_links)
    for link in dynamic_links:
        link.mass = link_mass


def ensure_extra_table_objects(env):
    added_log = []
    table_objects = []
    for spec in EXTRA_TABLE_OBJECT_SPECS:
        obj = env.scene.object_registry("name", spec["name"])
        if obj is None:
            obj = DatasetObject(
                name=spec["name"],
                category=spec["category"],
                model=spec["model"],
                fixed_base=False,
            )
            env.scene.add_object(obj)
            added_log.append(f"{spec['name']} ({spec['category']}/{spec['model']})")
        table_objects.append((obj, spec))

    if added_log:
        was_playing = og.sim.is_playing()
        if not was_playing:
            og.sim.play()
        og.sim.step()
        if not was_playing:
            og.sim.stop()

    for idx, (obj, spec) in enumerate(table_objects):
        if "mass" in spec:
            set_object_mass(obj, spec["mass"])
        hidden_pos = HIDDEN_OBJECT_BASE_POSITION.clone()
        hidden_pos[1] += 0.5 * (idx + len(HIDDEN_TABLE_OBJECT_NAMES))
        obj.set_position_orientation(position=hidden_pos, orientation=th.tensor([0.0, 0.0, 0.0, 1.0]))
        obj.visible = True
        _zero_object_velocity(obj)

    if added_log:
        print(f"[init] Added extra table objects: {', '.join(added_log)}")


def randomize_table_object_positions(env, rng):
    table = env.scene.object_registry("name", TARGET_TABLE_NAME)
    if table is None:
        print(f"[init-warn] Could not randomize table objects; table {TARGET_TABLE_NAME} was not found.")
        return

    _, table_high = table.aabb
    table_top_z = float(_to_numpy(table_high)[2])
    placed = []
    placement_log = []

    for obj_name in TABLE_OBJECT_NAMES:
        obj = env.scene.object_registry("name", obj_name)
        if obj is None:
            print(f"[init-warn] Could not randomize {obj_name}; object was not found.")
            continue

        old_pos, old_orn = obj.get_position_orientation()
        obj_low, _ = obj.aabb
        old_pos_np = _to_numpy(old_pos)
        bottom_offset = float(old_pos_np[2] - _to_numpy(obj_low)[2])

        xy = _sample_table_xy(table=table, obj=obj, rng=rng, occupied_xy=placed)
        new_pos = th.tensor(
            [xy[0], xy[1], table_top_z + bottom_offset + TABLE_RANDOMIZE_Z_OFFSET],
            dtype=old_pos.dtype,
            device=old_pos.device,
        )

        euler = T.quat2euler(old_orn)
        euler[2] = float(rng.uniform(-math.pi, math.pi))
        new_orn = T.euler2quat(euler)

        obj.set_position_orientation(position=new_pos, orientation=new_orn)
        obj.visible = True
        _zero_object_velocity(obj, dtype=old_pos.dtype, device=old_pos.device)

        placed.append(xy)
        placement_log.append(f"{obj_name}: pos={[round(v, 3) for v in new_pos.tolist()]}, yaw={float(euler[2]):.3f}")

    if placement_log:
        print("[init] Randomized table object positions:")
        for line in placement_log:
            print(f"       {line}")


def scale_arm_teleop_action(action, action_generator):
    arm_info = action_generator.controller_info.get("arm_0")
    if arm_info is None:
        return action

    start_idx = arm_info["start_idx"]
    end_idx = start_idx + arm_info["command_dim"]
    action[start_idx:end_idx] *= ARM_TELEOP_SPEED_SCALE
    return action


def main(
    model_path,
    quickstart=False,
    random_selection=False,
    task_description=DEFAULT_TASK_DESCRIPTION,
    image_history_frames=DEFAULT_IMAGE_HISTORY_FRAMES,
    image_history_interval=DEFAULT_IMAGE_HISTORY_INTERVAL,
):
    """闭环推理主函数"""
    og.log.info(f"Starting Inference with model: {model_path}")

    scene_model = DEFAULT_SCENE_MODEL
    if not quickstart:
        scene_model = choose_from_options(options=SCENES, name="scene", random_selection=random_selection)

    scene_cfg = dict()
    if scene_model == "empty":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model
        scene_file = os.path.abspath(
            os.path.join(
                "datasets",
                "behavior-1k-assets",
                "scenes",
                scene_model,
                "json",
                f"{scene_model}_best.json",
            )
        )
        if os.path.exists(scene_file):
            scene_cfg["scene_file"] = scene_file

    robot_name = DEFAULT_ROBOT_NAME
    if not quickstart:
        robot_name = choose_from_options(
            options=list(sorted(REGISTERED_ROBOTS)), name="robot", random_selection=random_selection
        )

    robot0_cfg = dict()
    robot0_cfg["type"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb"]
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = False
    robot0_cfg["fixed_base"] = True
    robot0_cfg["end_effector"] = DEFAULT_END_EFFECTOR
    robot0_cfg["position"] = FRANKA_TABLE_FRONT_POSITION.tolist()
    robot0_cfg["orientation"] = FRANKA_TABLE_FRONT_ORIENTATION.tolist()
    robot0_cfg["sensor_config"] = {
        "VisionSensor": {
            "sensor_kwargs": {
                "image_width": CAMERA_IMAGE_SIZE,
                "image_height": CAMERA_IMAGE_SIZE
            }
        }
    }

    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])
    env = og.Environment(configs=cfg)
    ensure_extra_table_objects(env)

    robot = env.robots[0]
    
    controller_choices = {
        "arm_0": "InverseKinematicsController",
        "gripper_0": "MultiFingerGripperController",
    }
    if not quickstart:
        controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)
    env.scene.update_initial_file()

    og.sim.viewer_camera.set_position_orientation(
        position=VIEWER_CAMERA_POSITION,
        orientation=VIEWER_CAMERA_ORIENTATION,
    )

    placement_rng = np.random.default_rng()

    env.reset()
    robot.reset()
    fix_franka_in_front_of_table(robot)
    open_robot_gripper(robot)
    hide_unwanted_table_objects(env)
    randomize_table_object_positions(env, placement_rng)

    # 初始化推理器
    inferencer = LeRobotInferencer(model_dir=model_path, robot=robot, task_description=task_description)
    image_history = ImageHistoryBuffer(
        inferencer=inferencer,
        history_frames=image_history_frames,
        interval_sec=image_history_interval,
        target_hz=DEFAULT_TARGET_HZ,
    )
    
    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)
    reset_keyboard_gripper_to_open(action_generator)

    flags = {"reset": False, "inferencing": False}

    def request_reset():
        flags["reset"] = True

    def toggle_control_mode():
        flags["inferencing"] = not flags["inferencing"]
        if flags["inferencing"]:
            inferencer.reset()
            print("\n▶️ [S] 已切换到模型控制")
        else:
            print("\n⌨️ [S] 已切换到键盘控制")

    # Register custom bindings
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot and pause inference",
        callback_fn=request_reset,
    )
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.S,
        description="Toggle keyboard/model control",
        callback_fn=toggle_control_mode,
    )
    action_generator.print_keyboard_teleop_info()

    print("\n" + "="*80)
    print(f"🚀 开始模型自动控制 (Inference)!")
    print(f"🤖 机器人: {robot_name} (真实动作维度: {len(robot.get_joint_positions())})")
    print(f"🕹️ 控制器配置: {controller_choices}")
    print(f"📝 当前任务描述: {task_description}")
    print(f"📷 摄像头分辨率: {CAMERA_IMAGE_SIZE}x{CAMERA_IMAGE_SIZE}")
    print(
        f"🕰️ 图像历史: {image_history.history_frames} 帧, "
        f"间隔 {image_history.interval_sec:.3f}s ({image_history.interval_steps} control steps)"
    )
    print(f"🍎🍌 每次重置会随机摆放桌面物体: {', '.join(TABLE_OBJECT_NAMES)}")
    print(f"🧹 已从桌面移除: {', '.join(HIDDEN_TABLE_OBJECT_NAMES)}")
    print(f"🧭 Franka 固定位置: {FRANKA_TABLE_FRONT_POSITION.tolist()}, yaw={FRANKA_TABLE_FRONT_YAW:.3f}")
    print("  [S] - 切换键盘控制 / 模型控制")
    print("  [R] - 复位并回到键盘控制")
    print("  [CTRL+C] - 退出程序")
    print("  当前状态: 键盘控制模式，按 [S] 可切换到模型控制")
    print("="*80 + "\n")

    obs = lift_end_effector_after_reset(env, robot)
    image_history.clear()
    image_history.append(obs, step_idx=0)
    
    step = 0
    target_hz = DEFAULT_TARGET_HZ
    frame_time = 1.0 / target_hz

    try:
        while True:
            start_time = time.time()

            if flags["reset"]:
                env.reset()
                robot.reset()
                fix_franka_in_front_of_table(robot)
                open_robot_gripper(robot)
                hide_unwanted_table_objects(env)
                randomize_table_object_positions(env, placement_rng)
                reset_keyboard_gripper_to_open(action_generator)
                inferencer.reset()
                flags["inferencing"] = False
                flags["reset"] = False
                obs = lift_end_effector_after_reset(env, robot)
                step = 0
                image_history.clear()
                image_history.append(obs, step_idx=step)
                print("\n🔄 [R] 已复位，当前为键盘控制模式；按 [S] 可切换到模型控制")
                continue

            if flags["inferencing"]:
                state = get_robot_state(robot)

                print("*" * 100)
                print(state)
                print(f"状态维度: {len(state)}")
                image_sequences = image_history.sample()
                predicted_action = inferencer.predict_action(
                    obs_dict=obs,
                    state=state,
                    image_sequences=image_sequences,
                )

                print(f"动作维度: {len(predicted_action)}")
                print(predicted_action)
                action = predicted_action
            else:
                action = action_generator.get_teleop_action()
                action = scale_arm_teleop_action(action, action_generator)
            
            time_begin = time.time()
            obs, _, _, _, _ = env.step(action=action)
            step += 1
            image_history.append(obs, step_idx=step)
            print(f"仿真器执行单步花费时间: {time.time() - time_begin}")
            
            if step % 10 == 0:
                print(f"⚡ 运行中... Step: {step}", end='\r')

            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
                
    except KeyboardInterrupt:
        print("\n⏹️ 检测到终止信号，正在退出仿真...")
    except Exception as e:
        print("\n❌ 运行过程中发生崩溃！具体报错原因如下：")
        import traceback
        traceback.print_exc()
    finally:
        og.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using trained LeRobot model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/ubuntu/Desktop/open-code/BEHAVIOR-1K/output/omnigibson_robot_merged_5_512/checkpoints/010000",
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Whether to skip interactive selections and use defaults.",
    )
    parser.add_argument(
        "--random_selection",
        action="store_true",
        help="Whether to randomly select options instead of asking user.",
    )
    parser.add_argument(
        "--task-description",
        "--task",
        default=DEFAULT_TASK_DESCRIPTION,
        help=f"Task instruction passed to the policy. Default: {DEFAULT_TASK_DESCRIPTION!r}.",
    )
    parser.add_argument(
        "--image-history-frames",
        type=int,
        default=DEFAULT_IMAGE_HISTORY_FRAMES,
        help=(
            "Number of image frames sent to the policy, including the current frame. "
            f"Use 1 for the old single-frame inference mode. Default: {DEFAULT_IMAGE_HISTORY_FRAMES}."
        ),
    )
    parser.add_argument(
        "--image-history-interval",
        type=float,
        default=DEFAULT_IMAGE_HISTORY_INTERVAL,
        help=(
            "Seconds between sampled image history frames. "
            f"Default: {DEFAULT_IMAGE_HISTORY_INTERVAL}."
        ),
    )
    args = parser.parse_args()
    
    if os.path.exists(os.path.join(args.model_path, "pretrained_model")):
        model_path = os.path.join(args.model_path, "pretrained_model")
    else:
        model_path = args.model_path
        
    main(
        model_path=model_path,
        quickstart=args.quickstart,
        random_selection=args.random_selection,
        task_description=args.task_description,
        image_history_frames=args.image_history_frames,
        image_history_interval=args.image_history_interval,
    )
