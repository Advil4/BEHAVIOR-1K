import os
import sys
import threading
import time

import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy
import torch
import zmq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from loguru import logger
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from scipy.spatial.transform import Rotation as R

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>", level="DEBUG")

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.GUI = True


def to_numpy(x):
    if hasattr(x, 'cpu'): return x.detach().cpu().numpy()
    return np.array(x)


def get_matching_component(hand_side, component_dict):
    target_side = hand_side.lower()
    for key in component_dict.keys():
        if target_side in key:
            return key
    if len(component_dict) == 1:
        only_key = list(component_dict.keys())[0]
        if "left" not in only_key.lower() and "right" not in only_key.lower():
            return only_key
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
        self.socket.connect(f"tcp://{server_ip}:7777")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.output_arm_deltas = {key: np.zeros(6) for key in self.arm_indices.keys()}
        self.output_gripper_poses = {key: np.zeros(len(idx_list)) for key, idx_list in self.gripper_indices.items()}

        self.latest_hand_data = {"Left": None, "Right": None}
        self.last_recv_time = {"Left": 0.0, "Right": 0.0}
        self.data_lock = threading.Lock()

        # 用于计算增量的历史状态
        self.smoothed_cam_pos = {"Left": None, "Right": None}
        self.smoothed_cam_rot = {"Left": None, "Right": None}
        self.prev_cam_pos = {"Left": None, "Right": None}
        self.prev_cam_rot = {"Left": None, "Right": None}

        self.alpha_pos = 0.4
        self.alpha_rot = 0.5

        # 相机坐标系到机器人基坐标系的旋转映射矩阵
        self.T_c2r = np.array([[0, 0, -1],
                               [1, 0, 0],
                               [0, -1, 0]
                               ])

        self._running = True
        self.zmq_thread = threading.Thread(target=self._zmq_worker, daemon=True)
        self.zmq_thread.start()
        logger.info("📡 后台 ZMQ 网络线程已启动")

    def _zmq_worker(self):
        while self._running:
            try:
                events = self.socket.poll(timeout=10)
                if events:
                    msg = self.socket.recv_json(flags=zmq.NOBLOCK)
                    with self.data_lock:
                        for hand_side in ["Left", "Right"]:
                            if hand_side in msg and msg[hand_side] is not None:
                                self.latest_hand_data[hand_side] = msg[hand_side]
                                self.last_recv_time[hand_side] = time.time()
            except Exception as e:
                pass

    def update_and_get_vision_parts(self):
        # 每帧清零增量，如果没有收到新手势，机器人绝对静止
        for key in self.output_arm_deltas.keys():
            self.output_arm_deltas[key] = np.zeros(6)

        with self.data_lock:
            local_hand_data = self.latest_hand_data.copy()
            local_recv_time = self.last_recv_time.copy()

        curr_time = time.time()
        for hand_side in ["Left", "Right"]:
            arm_key = get_matching_component(hand_side, self.arm_indices)
            gripper_key = get_matching_component(hand_side, self.gripper_indices)

            if curr_time - local_recv_time[hand_side] > 0.5:
                with self.data_lock:
                    self.latest_hand_data[hand_side] = None
                self.smoothed_cam_pos[hand_side] = None
                self.prev_cam_pos[hand_side] = None
                continue

            hand_data = local_hand_data[hand_side]
            if hand_data is None:
                continue

            if "wrist_pose" in hand_data and arm_key is not None:
                T_curr = np.array(hand_data["wrist_pose"])
                raw_pos = T_curr[:3, 3]
                raw_rot = R.from_matrix(T_curr[:3, :3])

                if self.prev_cam_pos[hand_side] is None:
                    self.smoothed_cam_pos[hand_side] = raw_pos.copy()
                    self.smoothed_cam_rot[hand_side] = raw_rot
                    self.prev_cam_pos[hand_side] = raw_pos.copy()
                    self.prev_cam_rot[hand_side] = raw_rot
                    logger.success(f"🔗[{hand_side}] 信号捕获！增量追踪已激活。")
                    continue

                curr_smoothed_pos = (self.alpha_pos * raw_pos) + (
                        (1 - self.alpha_pos) * self.smoothed_cam_pos[hand_side])
                self.smoothed_cam_pos[hand_side] = curr_smoothed_pos

                q_raw = raw_rot.as_quat()
                q_prev = self.smoothed_cam_rot[hand_side].as_quat()
                if np.dot(q_raw, q_prev) < 0: q_raw = -q_raw
                q_new = self.alpha_rot * q_raw + (1 - self.alpha_rot) * q_prev
                q_new /= np.linalg.norm(q_new)
                curr_smoothed_rot = R.from_quat(q_new)
                self.smoothed_cam_rot[hand_side] = curr_smoothed_rot

                delta_cam_pos = curr_smoothed_pos - self.prev_cam_pos[hand_side]
                delta_cam_rot = curr_smoothed_rot * self.prev_cam_rot[hand_side].inv()

                if np.linalg.norm(delta_cam_pos) < 0.0005:
                    delta_cam_pos = np.zeros(3)

                SCALE = 3.0
                delta_robot_pos = self.T_c2r @ delta_cam_pos * SCALE

                ROT_SCALE = 2.0
                delta_robot_rot_mat = self.T_c2r @ delta_cam_rot.as_matrix() @ self.T_c2r.T
                delta_robot_rotvec = R.from_matrix(delta_robot_rot_mat).as_rotvec()

                delta_robot_rotvec = delta_robot_rotvec * ROT_SCALE

                pos_norm = np.linalg.norm(delta_robot_pos)
                if pos_norm > 0.1:
                    delta_robot_pos = (delta_robot_pos / pos_norm) * 0.1

                rot_norm = np.linalg.norm(delta_robot_rotvec)
                if rot_norm > 0.3:
                    delta_robot_rotvec = (delta_robot_rotvec / rot_norm) * 0.3

                self.output_arm_deltas[arm_key][:3] = delta_robot_pos
                self.output_arm_deltas[arm_key][3:] = delta_robot_rotvec

                self.prev_cam_pos[hand_side] = curr_smoothed_pos.copy()
                self.prev_cam_rot[hand_side] = curr_smoothed_rot

            if "robot_joints" in hand_data and gripper_key is not None and gripper_key in self.gripper_indices:
                joint_angles = np.array(hand_data["robot_joints"])
                n_fingers = len(self.gripper_indices[gripper_key])
                alpha_g = 0.9

                if n_fingers == 7:
                    if len(joint_angles) >= 2:
                        index_raw = np.mean([joint_angles[0], joint_angles[1]])
                        index_finger_bend = np.clip(index_raw / 1.57, 0.0, 1.0)
                    else:
                        index_finger_bend = 0.0

                    thumb_linkage_ratio = 1.0

                    for i in range(n_fingers):
                        if i < len(joint_angles):
                            target_rad = joint_angles[i]

                            if i >= 4:
                                if i == 4:
                                    # thumb_0: 基节关节，控制拇指向外展/内收
                                    # 正值 = 向中指方向靠拢，负值 = 向食指方向靠拢
                                    thumb_max_angle = 0.8  # 从 -0.8 改为 +0.8（正方向）
                                    linkage_ratio = 0.9  # 联动比例
                                elif i == 5:
                                    # thumb_1: 中节关节，主要弯曲（保持负值）
                                    thumb_max_angle = -1.047
                                    linkage_ratio = 1.0
                                else:
                                    # thumb_2: 远节关节，跟随弯曲（保持负值）
                                    thumb_max_angle = -0.8
                                    linkage_ratio = 0.9

                                thumb_target = index_finger_bend * thumb_max_angle * linkage_ratio

                                self.output_gripper_poses[gripper_key][i] = (
                                        alpha_g * thumb_target + (1 - alpha_g) * self.output_gripper_poses[gripper_key][
                                    i]
                                )
                            else:
                                self.output_gripper_poses[gripper_key][i] = (
                                        alpha_g * target_rad + (1 - alpha_g) * self.output_gripper_poses[gripper_key][i]
                                )
                elif n_fingers == 12:
                    mapping = [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5]
                    for a_i, d_i in enumerate(mapping):
                        if d_i < len(joint_angles):
                            target_rad = joint_angles[d_i]
                            self.output_gripper_poses[gripper_key][a_i] = (
                                    alpha_g * target_rad + (1 - alpha_g) * self.output_gripper_poses[gripper_key][a_i]
                            )
                else:
                    min_len = min(n_fingers, len(joint_angles))
                    for i in range(min_len):
                        target_rad = joint_angles[i]
                        self.output_gripper_poses[gripper_key][i] = (
                                alpha_g * target_rad + (1 - alpha_g) * self.output_gripper_poses[gripper_key][i]
                        )

        return self.output_arm_deltas, self.output_gripper_poses


class LeRobotRecorderV3:
    def __init__(self, robot, fps=30):
        self.robot = robot
        self.is_recording = False
        self.fps = fps
        self.dataset = None

        timestamp = int(time.time())
        self.repo_id = f"omnigibson_dex_{timestamp}"
        self.local_dir = os.path.abspath(f"./lerobot_data/{self.repo_id}")
        self.task_description = "Put the apple on the plate"

        # =========[ 修改点 1：匹配 Psi-Zero (Ψ0) 要求 ] =========
        # 论文要求: State 补齐至 36 维，Action 补齐至 36 维
        self.state_dim = 36
        self.action_dim = 36
        # ==========================================================

        self.episode_buffer = []
        self.saved_episode_count = 0

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.episode_buffer = []
            logger.info("🔴 开始录制！数据正在暂存至内存...")
        else:
            self.is_recording = False
            if not self.episode_buffer:
                logger.warning("本次录制无有效帧，跳过。")
                return

            logger.info(f"⏹️ 录制结束，正在将 {len(self.episode_buffer)} 帧写入 V3.0 数据集...")
            if self.dataset is None:
                self.dataset = LeRobotDataset.create(
                    repo_id=self.repo_id,
                    fps=self.fps,
                    root=self.local_dir,
                    features={
                        "observation.image": {"dtype": "video", "shape": (3, 480, 640), "names": ["c", "h", "w"]},
                        "observation.state": {"dtype": "float32", "shape": (self.state_dim,), "names": ["dim"]},
                        "action": {"dtype": "float32", "shape": (self.action_dim,), "names": ["dim"]}
                    }
                )

            try:
                for frame in self.episode_buffer:
                    self.dataset.add_frame(frame)
                self.dataset.save_episode()
                self.saved_episode_count += 1
                logger.success(f"💾 第 {self.saved_episode_count} 集已保存！")
            except Exception as e:
                logger.error(f"写入磁盘失败: {e}")

            self.episode_buffer = []

    def discard_current(self):
        if self.is_recording or self.episode_buffer:
            self.is_recording = False
            self.episode_buffer = []
            logger.warning("🗑️ 录制已丢弃，内存已清空。")

    def finalize_and_exit(self):
        if self.dataset is not None:
            logger.warning("正在执行最终 finalize()... 请稍候...")
            self.dataset.finalize()
            logger.success(f"🎉 数据集封存完成！路径: {self.local_dir}")
        else:
            logger.info("没有生成任何数据，直接退出。")

    def step(self, img_hwc, state, action):
        if not self.is_recording: return
        if hasattr(img_hwc, 'detach'):
            img_hwc = img_hwc.detach().cpu().numpy()
        if img_hwc.shape[-1] == 4:
            img_hwc = img_hwc[:, :, :3]
        img_chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.uint8)

        self.episode_buffer.append({
            "observation.image": torch.from_numpy(img_chw).clone(),
            "observation.state": torch.from_numpy(state.astype(np.float32)).clone(),
            "action": torch.from_numpy(action.astype(np.float32)).clone(),
            "task": self.task_description
        })


def main():
    robot_name = "G1WithDex3Hand"
    robot_cfg = {
        "type": robot_name,
        "action_type": "continuous",
        "action_normalize": False,
        "obs_modalities": ["rgb"],
        "grasping_mode": "physical",
        # 提示：Psi-Zero 预训练的输入图片尺寸通常会被缩放到 360x240 或 320x240
        # 你可以保持 640x480 的收集分辨率以便于观看，在送入模型前再做 resize 处理。
        "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_width": 640, "image_height": 480}}}
    }

    cfg = {"scene": {"type": "Scene", "scene_model": "empty"},
           "objects": [
               # {"type": "DatasetObject", "name": "carpet", "category": "carpet", "model": "ctclvd",
               #  "position": [0.6, 0.0, 0.035], "fixed_base": True},
               {"type": "DatasetObject", "name": "table", "category": "breakfast_table", "model": "lcsizg",
                "position": [0.6, 0.0, 0.035], "fixed_base": True},
               # {"type": "DatasetObject", "name": "apple_1", "category": "apple", "model": "agveuv",
               # #  "position": [0.4, -0.05, 0.2]},
               # {"type": "DatasetObject", "name": "bottle_of_alfredo_sauce", "category": "bottle_of_alfredo_sauce",
               #  "model": "xwzqjr",
               #  "position": [0.4, -0.05, 0.2]},
               # {"type": "DatasetObject", "name": "bottle_of_apple_cider", "category": "bottle_of_apple_cider",
               #  "model": "frekrp",
               #  "position": [0.4, -0.05, 0.1]},
               {"type": "DatasetObject", "name": "bottle_of_barbecue_sauce", "category": "bottle_of_barbecue_sauce",
                "model": "ikbsox",
                "position": [0.4, -0.05, 0.2]},
               # {"type": "DatasetObject", "name": "plate_1", "category": "plate", "model": "aewthq",
               #  "position": [0.4, 0.055, 0.2]},
           ],
           "robots": [robot_cfg]}

    env = og.Environment(configs=cfg)
    robot = env.robots[0]

    controller_config = {}
    for comp in robot.controller_order:
        if comp == "base":
            controller_config[comp] = {"name": "JointController", "use_delta_commands": True}
        elif comp.startswith("arm"):
            controller_config[comp] = {"name": "InverseKinematicsController", "mode": "pose_delta_ori"}
        elif comp.startswith("gripper"):
            controller_config[comp] = {
                "name": "MultiFingerGripperController",
                "mode": "independent",
                "command_input_limits": None,
                "command_output_limits": None
            }
        else:
            controller_config[comp] = {"name": "JointController", "use_delta_commands": True}

    robot.reload_controllers(controller_config=controller_config)
    env.scene.update_initial_file()

    obs, _ = env.reset()
    kb = KeyboardRobotController(robot=robot)
    vs = VisionRobotController(robot=robot, server_ip="127.0.0.1")
    recorder = LeRobotRecorderV3(robot=robot, fps=30)

    flags = {"reset": False, "quit": False}

    def set_reset():
        flags["reset"] = True

    def set_quit():
        flags["quit"] = True

    kb.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.R, description="Reset Env & Discard",
                                  callback_fn=set_reset)
    kb.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.T, description="Toggle Record",
                                  callback_fn=lambda: recorder.toggle_recording())
    kb.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.B, description="Discard Current",
                                  callback_fn=lambda: recorder.discard_current())
    kb.register_custom_keymapping(key=lazy.carb.input.KeyboardInput.Q, description="Safe Finalize & Exit",
                                  callback_fn=set_quit)

    logger.success("【G1 灵巧手视觉遥操作 + 数据录制 V3 (兼容 Psi-Zero)】已启动！")

    TARGET_FPS = 30.0
    target_dt = 1.0 / TARGET_FPS

    while True:
        loop_start = time.time()
        _ = kb.get_teleop_action()

        if flags["quit"]:
            vs._running = False
            recorder.finalize_and_exit()
            env.close()
            sys.exit(0)

        if flags["reset"]:
            obs, _ = env.reset()
            recorder.discard_current()
            flags["reset"] = False
            logger.info("🔄 环境复位，缓冲区已清空。")
            continue

        state_t = robot.get_joint_positions()

        rgb_img = None
        for k, v in obs.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, dict) and 'rgb' in sv: rgb_img = sv['rgb']; break

        kb_act = kb.get_teleop_action()
        vs_arm, vs_grip = vs.update_and_get_vision_parts()

        combined_act = np.zeros(robot.action_dim)

        for arm_k, arm_idx in vs.arm_indices.items():
            if arm_k in vs_arm: combined_act[arm_idx] = vs_arm[arm_k]

        for grp_k, grp_idx in vs.gripper_indices.items():
            if grp_k in vs_grip: combined_act[grp_idx] = vs_grip[grp_k]

        combined_act[:len(kb_act)] = np.where(combined_act[:len(kb_act)] == 0, kb_act, combined_act[:len(kb_act)])
        next_obs, _, _, _, _ = env.step(combined_act)
        state_next = robot.get_joint_positions()

        # 处理 State (期望 36维)
        s36 = np.zeros(36, dtype=np.float32)
        upper_body_len = min(len(state_t), 36)
        s36[:upper_body_len] = state_t[:upper_body_len]

        # 处理 Action (期望 36维)
        a36 = np.zeros(36, dtype=np.float32)
        next_upper_body_len = min(len(state_next), 36)
        a36[:next_upper_body_len] = state_next[:next_upper_body_len]

        recorder.step(img_hwc=rgb_img, state=s36, action=a36)

        obs = next_obs
        elapsed = time.time() - loop_start
        if elapsed < target_dt: time.sleep(target_dt - elapsed)


if __name__ == "__main__":
    main()
