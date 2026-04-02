import os
import sys
import time

import cv2
import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy
import zmq
from loguru import logger
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
from scipy.spatial.transform import Rotation as R
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    level="INFO",
    colorize=True
)

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.GUI = True


def to_numpy(x):
    """安全剥离 PyTorch Tensor"""
    if hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
    return np.array(x)


def get_matching_component(hand_side, component_dict):
    target_side = hand_side.lower()
    for key in component_dict.keys():
        if target_side in key:
            return key
    if len(component_dict) == 1:
        return list(component_dict.keys())[0]
    return None


class VisionRobotController:
    def __init__(self, robot, server_ip="127.0.0.1"):
        self.robot = robot
        self.arm_indices = {}
        self.gripper_indices = {}

        for key, idx_list in robot.controller_action_idx.items():
            if key.startswith("arm"):
                self.arm_indices[key] = idx_list
            elif key.startswith("gripper"):
                self.gripper_indices[key] = idx_list

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(f"tcp://{server_ip}:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.output_arm_deltas = {key: np.zeros(6) for key in self.arm_indices.keys()}
        self.output_gripper_poses = {key: np.zeros(len(idx_list)) for key, idx_list in self.gripper_indices.items()}

        # 信箱缓存机制
        self.latest_hand_data = {"Left": None, "Right": None}
        self.last_recv_time = {"Left": 0.0, "Right": 0.0}

        self.smoothed_cam_pos = {"Left": None, "Right": None}
        self.smoothed_cam_rot = {"Left": None, "Right": None}
        self.prev_cam_pos = {"Left": None, "Right": None}
        self.prev_cam_rot = {"Left": None, "Right": None}

        # 滤波参数
        self.alpha_x = 0.4
        self.alpha_y = 0.4
        self.alpha_z = 0.4
        self.alpha_rot = 0.5
        self.alpha_g = 0.5

        # 右手系矩阵
        self.T_c2r = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

    def update_and_get_vision_parts(self):
        for key in self.output_arm_deltas.keys():
            self.output_arm_deltas[key] = np.zeros(6)

        while True:
            try:
                msg = self.socket.recv_json(flags=zmq.NOBLOCK)
                for hand_side in ["Left", "Right"]:
                    if hand_side in msg and msg[hand_side] is not None:
                        self.latest_hand_data[hand_side] = msg[hand_side]
                        self.last_recv_time[hand_side] = time.time()
            except zmq.Again:
                break

        curr_time = time.time()

        for hand_side in ["Left", "Right"]:
            arm_key = get_matching_component(hand_side, self.arm_indices)
            gripper_key = get_matching_component(hand_side, self.gripper_indices)

            # 丢失信号保护
            if curr_time - self.last_recv_time[hand_side] > 0.5:
                self.latest_hand_data[hand_side] = None
                self.smoothed_cam_pos[hand_side] = None
                continue

            hand_data = self.latest_hand_data[hand_side]
            if hand_data is None:
                continue

            # 开始处理数据
            if "wrist_pose" in hand_data and arm_key is not None:
                T_curr = np.array(hand_data["wrist_pose"])
                raw_pos = T_curr[:3, 3]
                raw_rot = R.from_matrix(T_curr[:3, :3])

                # 1. 抓取初始点
                if self.smoothed_cam_pos[hand_side] is None:
                    self.smoothed_cam_pos[hand_side] = raw_pos.copy()
                    self.smoothed_cam_rot[hand_side] = raw_rot
                    self.prev_cam_pos[hand_side] = raw_pos.copy()
                    self.prev_cam_rot[hand_side] = raw_rot
                    logger.success(f"🔗[{hand_side}] 信号捕获！相对增量已激活。")
                    continue

                # 2. EMA 滤波
                curr_smoothed_pos = self.smoothed_cam_pos[hand_side].copy()
                curr_smoothed_pos[0] = self.alpha_x * raw_pos[0] + (1 - self.alpha_x) * curr_smoothed_pos[0]
                curr_smoothed_pos[1] = self.alpha_y * raw_pos[1] + (1 - self.alpha_y) * curr_smoothed_pos[1]
                curr_smoothed_pos[2] = self.alpha_z * raw_pos[2] + (1 - self.alpha_z) * curr_smoothed_pos[2]
                self.smoothed_cam_pos[hand_side] = curr_smoothed_pos

                q_raw = raw_rot.as_quat()
                q_prev = self.smoothed_cam_rot[hand_side].as_quat()
                if np.dot(q_raw, q_prev) < 0: q_raw = -q_raw
                q_new = self.alpha_rot * q_raw + (1 - self.alpha_rot) * q_prev
                q_new /= np.linalg.norm(q_new)
                curr_smoothed_rot = R.from_quat(q_new)
                self.smoothed_cam_rot[hand_side] = curr_smoothed_rot

                # 3. 提取增量
                delta_cam_pos = curr_smoothed_pos - self.prev_cam_pos[hand_side]
                local_delta_cam_rot = self.prev_cam_rot[hand_side].inv() * curr_smoothed_rot

                # 过滤极小的高频颤动
                if np.linalg.norm(delta_cam_pos) < 0.0005:
                    delta_cam_pos = np.zeros(3)

                # 4. 坐标映射与动作放大
                SCALE = 3.0
                delta_robot_pos = (self.T_c2r @ delta_cam_pos) * SCALE

                # 旋转映射
                cam_local_rotvec = local_delta_cam_rot.as_rotvec()
                if hand_side == "Right":
                    robot_local_rotvec = np.array([cam_local_rotvec[1], -cam_local_rotvec[2], -cam_local_rotvec[0]])
                else:
                    robot_local_rotvec = np.array([-cam_local_rotvec[1], cam_local_rotvec[2], -cam_local_rotvec[0]])

                arm_name = arm_key.replace("arm_", "") if "arm_" in arm_key else self.robot.default_arm
                try:
                    _, curr_quat = self.robot.get_relative_eef_pose(arm_name)
                    curr_rot_mat = R.from_quat(to_numpy(curr_quat)).as_matrix()
                except:
                    curr_rot_mat = np.eye(3)

                delta_robot_rotvec = curr_rot_mat @ robot_local_rotvec

                # 5. 单帧增量安全钳制
                pos_norm = np.linalg.norm(delta_robot_pos)
                if pos_norm > 0.1:
                    delta_robot_pos = (delta_robot_pos / pos_norm) * 0.1

                rot_norm = np.linalg.norm(delta_robot_rotvec)
                if rot_norm > 0.1:
                    delta_robot_rotvec = (delta_robot_rotvec / rot_norm) * 0.1

                self.output_arm_deltas[arm_key][:3] = delta_robot_pos
                self.output_arm_deltas[arm_key][3:] = delta_robot_rotvec

                self.prev_cam_pos[hand_side] = curr_smoothed_pos.copy()
                self.prev_cam_rot[hand_side] = curr_smoothed_rot

            # 灵巧手兼容抓取
            if "robot_joints" in hand_data and gripper_key in self.gripper_indices:
                joint_angles = hand_data["robot_joints"]
                n_fingers = len(self.gripper_indices[gripper_key])
                alpha_g = self.alpha_g
                rad_array = np.array(joint_angles)

                MIN_RAD = 0.1
                MAX_RAD = 0.6
                closure = np.clip((np.abs(rad_array) - MIN_RAD) / (MAX_RAD - MIN_RAD), 0.0, 1.0)
                # closure = np.clip(np.abs(rad_array) / 1.2, 0.0, 1.0)

                if n_fingers == 12:  # Inspire 灵巧手
                    mapping = [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5]
                    for a_i, d_i in enumerate(mapping):
                        if d_i < len(closure):
                            c_val = closure[d_i]
                            raw_val = (c_val * 2.0 - 1.0) if hand_side == "Right" else (1.0 - c_val * 2.0)
                            self.output_gripper_poses[gripper_key][a_i] = (
                                    alpha_g * raw_val + (1 - alpha_g) * self.output_gripper_poses[gripper_key][a_i]
                            )

                elif n_fingers == 20:  # SVH 灵巧手
                    min_len = min(n_fingers, len(closure))
                    for i in range(min_len):
                        c_val = closure[i]
                        if hand_side == "Right":
                            raw_val = c_val * 2.0 - 1.0
                        else:
                            if i in [1, 2, 3, 4, 8, 9, 13]:
                                raw_val = 1.0 - c_val * 2.0
                            else:
                                raw_val = c_val * 2.0 - 1.0
                        self.output_gripper_poses[gripper_key][i] = (
                                alpha_g * raw_val + (1 - alpha_g) * self.output_gripper_poses[gripper_key][i]
                        )

                elif n_fingers == 2 and len(joint_angles) >= 11:  # Pinch
                    pinch = np.linalg.norm(np.array(joint_angles[4:7]) - np.array(joint_angles[8:11]))
                    c_val = np.clip(1.0 - (pinch / 0.08), 0.0, 1.0)
                    raw_val = (c_val * 2.0 - 1.0) if hand_side == "Right" else (1.0 - c_val * 2.0)
                    self.output_gripper_poses[gripper_key][:] = (
                            alpha_g * raw_val + (1 - alpha_g) * self.output_gripper_poses[gripper_key][:]
                    )
                else:  # 通用直接映射
                    min_len = min(n_fingers, len(closure))
                    for i in range(min_len):
                        c_val = closure[i]
                        raw_val = (c_val * 2.0 - 1.0) if hand_side == "Right" else (1.0 - c_val * 2.0)
                        self.output_gripper_poses[gripper_key][i] = (
                                alpha_g * raw_val + (1 - alpha_g) * self.output_gripper_poses[gripper_key][i]
                        )

        return self.output_arm_deltas, self.output_gripper_poses


# LeRobot 录制器核心逻辑
class LeRobotRecorder:
    def __init__(self, robot, fps=30):
        self.is_recording = False
        self.fps = fps
        self.dataset = None
        self.discard_current_episode = False
        self.current_episode_id = None

        timestamp = int(time.time())
        self.repo_id = f"omnigibson_dex_{timestamp}"
        self.local_dir = f"./lerobot_data/{self.repo_id}"
        self.task_description = "Put the apple on the plate"

        self.action_dim = robot.action_dim
        self.state_dim = robot.n_dof
        self.saved_episode_count = 0
        self.episode_buffer = []

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            # 开启录制
            self.episode_buffer = []
            if self.discard_current_episode:
                self.discard_current_episode = False
                self.current_episode_id = None

            logger.info("🔴 开始录制！数据正在暂存至【内存】中...")
            logger.info("💡 提示：若操作失误，按 'B' 键丢弃本次录制")
        else:
            # 结束保存
            if self.discard_current_episode:
                self.current_episode_id = None
                self.discard_current_episode = False
                logger.warning("🗑️ 已丢弃当前录制片段！(未向硬盘写入任何脏数据)")
                logger.info(f"📊 数据集已保存 episode 数：{self.saved_episode_count}")
            else:
                logger.info("⏹️ 停止录制！正在将有效数据打包写入硬盘...")
                if len(self.episode_buffer) == 0:
                    logger.warning("当前缓存没有数据，跳过保存。")
                    return

                if self.dataset is None:
                    import shutil
                    from pathlib import Path
                    dataset_path = Path(self.local_dir)

                    if dataset_path.exists():
                        logger.warning("检测到已存在的数据集，清理中...")
                        try:
                            shutil.rmtree(dataset_path)
                            logger.success("已清理旧数据集")
                        except Exception as e:
                            logger.error(f"清理失败：{e}")
                            self.is_recording = False
                            return

                    dataset_path.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        self.dataset = LeRobotDataset.create(
                            repo_id=self.repo_id,
                            fps=self.fps,
                            root=self.local_dir,
                            features={
                                "observation.image": {
                                    "dtype": "image",
                                    "shape": (480, 640, 3),
                                    "names": ["height", "width", "channel"],
                                },
                                "observation.state": {
                                    "dtype": "float32",
                                    "shape": (self.state_dim,),
                                    "names": ["joint_positions"],
                                },
                                "action": {
                                    "dtype": "float32",
                                    "shape": (self.action_dim,),
                                    "names": ["action_vector"],
                                },
                            }
                        )
                        logger.success("数据集初始化成功！")
                        logger.info("📷 图像分辨率：640x480 (VGA - LeRobot 推荐)")
                    except Exception as e:
                        logger.error(f"创建数据集失败：{e}")
                        self.is_recording = False
                        return

                logger.info(f"⏳ 正在写入 {len(self.episode_buffer)} 帧数据...")
                try:
                    for frame_data in self.episode_buffer:
                        try:
                            self.dataset.add_frame(frame_data, task=self.task_description)
                        except TypeError:
                            self.dataset.add_frame(frame_data)

                    self.dataset.save_episode()
                    self.saved_episode_count += 1
                    self.current_episode_id = None

                    logger.success("💾 Episode 保存完毕！")
                    logger.info(
                        f"📊 数据统计: 本次写入 {len(self.episode_buffer)} 帧 | 总 Episode 数: {self.saved_episode_count}")
                except Exception as e:
                    logger.error(f"保存写入失败：{e}")

            self.episode_buffer = []

    def discard_episode(self):
        """手动取消当前录制的 episode"""
        if self.is_recording:
            self.discard_current_episode = True
            self.is_recording = False
            logger.warning("🚫 已标记取消当前录制，内存缓存已清空。")

    def step(self, obs_img, robot_state, action):
        if self.is_recording:
            if obs_img is None:
                logger.warning("输入图像为 None，跳过此帧")
                return

            try:
                import torch
                if hasattr(obs_img, 'cpu'):
                    obs_img = obs_img.cpu().numpy()
                elif hasattr(obs_img, 'get_array'):
                    obs_img = np.array(obs_img)
            except Exception:
                pass

            if not isinstance(obs_img, np.ndarray):
                logger.warning(f"图像类型错误 {type(obs_img)}，跳过此帧")
                return

            if len(obs_img.shape) < 2:
                logger.warning(f"图像维度不正确 {obs_img.shape}，跳过此帧")
                return

            if len(obs_img.shape) == 2:
                obs_img = cv2.cvtColor(obs_img, cv2.COLOR_GRAY2RGB)
            elif obs_img.shape[2] == 4:
                obs_img = obs_img[:, :, :3]

            if obs_img.dtype in [np.float32, np.float64]:
                if obs_img.max() <= 1.0:
                    obs_img = (obs_img * 255).astype(np.uint8)
                else:
                    obs_img = obs_img.astype(np.uint8)
            elif obs_img.dtype != np.uint8:
                obs_img = obs_img.astype(np.uint8)

            try:
                img_resized = cv2.resize(obs_img, (640, 480), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                logger.error(f"图像缩放失败：{e} | 原始 shape: {obs_img.shape}, dtype: {obs_img.dtype}")
                return

            if img_resized.shape != (480, 640, 3):
                logger.warning(f"最终图像尺寸不正确 {img_resized.shape}")
                return

            frame_dict = {
                "observation.image": img_resized,
                "observation.state": np.array(robot_state, dtype=np.float32),
                "action": np.array(action, dtype=np.float32)
            }
            self.episode_buffer.append(frame_dict)


def main():
    # robot_name = choose_from_options(options=list(sorted(REGISTERED_ROBOTS)), name="robot")
    # scene_model = choose_from_options(options=["empty", "Rs_int"], name="scene")

    robot_cfg = {
        "type": "R1ProWithSvhHand",
        "action_type": "continuous",
        "action_normalize": False,
        "obs_modalities": ["rgb", "proprio"],
        "grasping_mode": "assisted",
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_width": 640,
                    "image_height": 480,
                }
            }
        }
    }

    cfg = {"scene": {"type": "Scene", "scene_model": "empty"},
           "objects": [{
               "type": "DatasetObject",
               "name": "table",
               "category": "breakfast_table",
               "model": "lcsizg",
               "position": [0.6, 0.0, 1.2],
               "fixed_base": True,
           },
               {
                   "type": "DatasetObject",
                   "name": "apple_1",
                   "category": "apple",
                   "model": "agveuv",
                   "density": 50.0,
                   "position": [0.6, -0.1, 1.3],
               },
               {
                   "type": "DatasetObject",
                   "name": "plate_1",
                   "category": "plate",
                   "model": "aewthq",
                   "position": [0.6, 0.05, 1.4],
               },
           ],
           "robots": [robot_cfg]}

    env = og.Environment(configs=cfg)
    robot = env.robots[0]

    controller_config = {}
    for component in robot.controller_order:
        if component == "base":
            controller_config[component] = {"name": "HolonomicBaseJointController"}
        elif component.startswith("arm"):
            controller_config[component] = {
                "name": "InverseKinematicsController",
                "mode": "pose_delta_ori",
            }
        elif component.startswith("gripper"):
            controller_config[component] = {
                "name": "MultiFingerGripperController",
                "mode": "independent",
                "command_input_limits": None,  # 确保不限死输入
                "isaac_kp": 100.0,  # 比例增益，增加马达响应力（默认值通常只有几十）
                "isaac_kd": 10.0,  # 阻尼增益，防止抖动（Kp 太大容易抖）
            }
        else:
            controller_config[component] = {"name": "JointController", "use_delta_commands": True}

    robot.reload_controllers(controller_config=controller_config)
    env.reset()

    kb_generator = KeyboardRobotController(robot=robot)
    vs_generator = VisionRobotController(robot=robot)

    kb_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset all objects",
        callback_fn=lambda: env.reset(),
    )

    recorder = LeRobotRecorder(robot=robot, fps=30)

    kb_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.T,
        description="Toggle LeRobot Data Recording",
        callback_fn=lambda: recorder.toggle_recording(),
    )

    kb_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.B,
        description="Discard current recording and reset",
        callback_fn=lambda: recorder.discard_episode(),
    )

    base_idx = robot.controller_action_idx.get("base", [])
    arm_indices = {k: v for k, v in robot.controller_action_idx.items() if k.startswith("arm")}
    gripper_indices = {k: v for k, v in robot.controller_action_idx.items() if k.startswith("gripper")}
    other_indices = []
    for comp in ["trunk", "camera", "legs", "head"]:
        other_indices.extend(robot.controller_action_idx.get(comp, []))

    logger.success("【数采模式启动】图像 API 及按键绑定已就绪！")
    logger.info("👉 按 'T' 键 开始/保存 Episode，按 'B' 键 取消当前录制！")
    logger.info(f"📁 录制数据将存放至：{recorder.local_dir}")

    TARGET_FPS = 30.0
    target_dt = 1.0 / TARGET_FPS

    while True:
        loop_start = time.time()
        kb_action = kb_generator.get_teleop_action()
        arm_poses, gripper_poses = vs_generator.update_and_get_vision_parts()

        combined_action = np.zeros(robot.action_dim)
        if len(base_idx) > 0: combined_action[base_idx] = kb_action[base_idx]
        for idx in other_indices: combined_action[idx] = kb_action[idx]
        for arm_key, arm_idx in arm_indices.items():
            if len(arm_idx) > 0 and arm_key in arm_poses: combined_action[arm_idx] = arm_poses[arm_key]
        for gripper_key, gripper_idx in gripper_indices.items():
            if len(gripper_idx) > 0 and gripper_key in gripper_poses: combined_action[gripper_idx] = gripper_poses[
                gripper_key]

        obs, _, _, _, _ = env.step(combined_action)

        rgb_img = None

        for key, value in obs.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        if 'rgb' in sub_value:
                            rgb_img = sub_value['rgb']
                            break
                        for deep_key, deep_value in sub_value.items():
                            if 'rgb' in deep_key.lower() and hasattr(deep_value, 'shape'):
                                rgb_img = deep_value
                                break

        robot_state = robot.get_joint_positions()
        recorder.step(obs_img=rgb_img, robot_state=robot_state, action=combined_action)

        elapsed = time.time() - loop_start
        if elapsed < target_dt: time.sleep(target_dt - elapsed)


if __name__ == "__main__":
    main()
