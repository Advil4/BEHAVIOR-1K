import os
import time
import cv2
import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy
import zmq
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
from scipy.spatial.transform import Rotation as R

# 尝试导入 LeRobot 数据集管理器
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    LEROBOT_AVAILABLE = True
except ImportError:
    print("⚠️ 未检测到 lerobot，请执行：pip install lerobot")
    LEROBOT_AVAILABLE = False

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.GUI = True


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

        # 调试信息：如果网络发生断流，打印日志
        curr_time = time.time()

        for hand_side in ["Left", "Right"]:
            arm_key = get_matching_component(hand_side, self.arm_indices)
            gripper_key = get_matching_component(hand_side, self.gripper_indices)

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
                    print(f"🔗 [{hand_side}] 信号捕获！相对增量已激活。")
                    continue

                # 2. EMA 滤波
                curr_smoothed_pos = self.smoothed_cam_pos[hand_side].copy()
                alpha_x = self.alpha_x
                alpha_y = self.alpha_y
                alpha_z = self.alpha_z

                curr_smoothed_pos[0] = alpha_x * raw_pos[0] + (1 - alpha_x) * curr_smoothed_pos[0]
                curr_smoothed_pos[1] = alpha_y * raw_pos[1] + (1 - alpha_y) * curr_smoothed_pos[1]
                curr_smoothed_pos[2] = alpha_z * raw_pos[2] + (1 - alpha_z) * curr_smoothed_pos[2]
                self.smoothed_cam_pos[hand_side] = curr_smoothed_pos

                alpha_rot = self.alpha_rot
                q_raw = raw_rot.as_quat()
                q_prev = self.smoothed_cam_rot[hand_side].as_quat()
                if np.dot(q_raw, q_prev) < 0: q_raw = -q_raw
                q_new = alpha_rot * q_raw + (1 - alpha_rot) * q_prev
                q_new /= np.linalg.norm(q_new)
                curr_smoothed_rot = R.from_quat(q_new)
                self.smoothed_cam_rot[hand_side] = curr_smoothed_rot

                # 3. 提取增量
                delta_cam_pos = curr_smoothed_pos - self.prev_cam_pos[hand_side]

                # 注意这里的顺序：prev.inv() * curr，代表“以自身为基准转了多少度”
                local_delta_cam_rot = self.prev_cam_rot[hand_side].inv() * curr_smoothed_rot

                # 过滤极小的高频颤动
                if np.linalg.norm(delta_cam_pos) < 0.0005:
                    delta_cam_pos = np.zeros(3)

                # 4. 坐标映射与动作放大
                SCALE = 3.0
                delta_robot_pos = (self.T_c2r @ delta_cam_pos) * SCALE

                # 旋转映射：将相机的局部旋转轴映射给机器人的局部旋转轴
                cam_local_rotvec = local_delta_cam_rot.as_rotvec()

                # 手腕坐标翻转
                if hand_side == "Right":
                    robot_local_rotvec = np.array([cam_local_rotvec[1], -cam_local_rotvec[2], -cam_local_rotvec[0]])
                else:
                    robot_local_rotvec = np.array([-cam_local_rotvec[1], cam_local_rotvec[2], -cam_local_rotvec[0]])

                # 获取机器人的当前真实姿态
                arm_name = arm_key.replace("arm_", "") if "arm_" in arm_key else self.robot.default_arm
                try:
                    _, curr_quat = self.robot.get_relative_eef_pose(arm_name)
                    curr_rot_mat = R.from_quat(
                        curr_quat.detach().cpu().numpy() if hasattr(curr_quat, 'cpu') else np.array(
                            curr_quat)).as_matrix()
                except:
                    curr_rot_mat = np.eye(3)

                # 用当前姿态矩阵乘以局部旋转向量，将其转换为 IK 接收的全局旋转增量
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

                # 更新下一帧基准
                self.prev_cam_pos[hand_side] = curr_smoothed_pos.copy()
                self.prev_cam_rot[hand_side] = curr_smoothed_rot

            # 灵巧手兼容抓取
            if "robot_joints" in hand_data and gripper_key in self.gripper_indices:
                joint_angles = hand_data["robot_joints"]
                n_fingers = len(self.gripper_indices[gripper_key])
                alpha_g = self.alpha_g

                # 统一转换为 0(纯张开) ~ 1(纯握拳) 的闭合度
                rad_array = np.array(joint_angles)
                closure = np.clip(np.abs(rad_array) / 1.2, 0.0, 1.0)

                if n_fingers == 12:  # Inspire 灵巧手模式
                    mapping = [8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5]
                    for a_i, d_i in enumerate(mapping):
                        if d_i < len(closure):
                            c_val = closure[d_i]
                            raw_val = (c_val * 2.0 - 1.0) if hand_side == "Right" else (1.0 - c_val * 2.0)
                            self.output_gripper_poses[gripper_key][a_i] = (
                                    alpha_g * raw_val + (1 - alpha_g) * self.output_gripper_poses[gripper_key][a_i]
                            )

                elif n_fingers == 20:  # SVH 灵巧手模式
                    min_len = min(n_fingers, len(closure))
                    for i in range(min_len):
                        c_val = closure[i]

                        if hand_side == "Right":
                            raw_val = c_val * 2.0 - 1.0
                        else:
                            if i in [1, 2, 3, 4, 8, 9, 13]:
                                raw_val = 1.0 - c_val * 2.0  # 反转
                            else:
                                raw_val = c_val * 2.0 - 1.0  # 正向

                        self.output_gripper_poses[gripper_key][i] = (
                                alpha_g * raw_val + (1 - alpha_g) * self.output_gripper_poses[gripper_key][i]
                        )

                elif n_fingers == 2 and len(joint_angles) >= 11:  # Pinch 捏合夹爪
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


# =======================================================================
# 【新增】：LeRobot 录制器核心逻辑
# =======================================================================
class LeRobotRecorder:
    def __init__(self, robot, fps=30):
        self.is_recording = False
        self.fps = fps
        self.dataset = None
        
        # 使用时间戳生成唯一目录名，避免冲突
        import time
        timestamp = int(time.time())
        self.repo_id = f"omnigibson_dex_{timestamp}"
        self.local_dir = f"./lerobot_data/{self.repo_id}"
        
        # 添加任务描述（LeRobot 要求）
        self.task_description = "Omnigibson dexterous manipulation teleoperation"

        # 提取机器人状态空间的维度
        self.action_dim = robot.action_dim
        self.state_dim = robot.n_dof  # 关节总数

    def toggle_recording(self):
        if not LEROBOT_AVAILABLE:
            print("⚠️ 请先安装 lerobot: pip install lerobot")
            return

        self.is_recording = not self.is_recording
        if self.is_recording:
            print("\n=========================================")
            print("🔴 开始录制，数据正在写入缓存中...")
            print("=========================================")
            if self.dataset is None:
                import shutil
                from pathlib import Path
                
                dataset_path = Path(self.local_dir)
                
                # 如果目录已存在（无论是否为空），直接删除
                if dataset_path.exists():
                    print(f"\n⚠️ 检测到已存在的目录，清理中...")
                    try:
                        shutil.rmtree(dataset_path)
                        print(f"✅ 已清理旧目录")
                    except Exception as e:
                        print(f"❌ 清理失败：{e}")
                        self.is_recording = False
                        return
                
                # 确保父目录存在
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 现在创建数据集（此时目录肯定不存在）
                try:
                    self.dataset = LeRobotDataset.create(
                        repo_id=self.repo_id,
                        fps=self.fps,
                        root=self.local_dir,
                        features={
                            "observation.image": {
                                "dtype": "image",
                                "shape": (480, 640, 3),  # LeRobot 推荐的标准尺寸
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
                    print(f"✅ 数据集创建成功！路径：{self.local_dir}")
                    print(f"📷 图像分辨率：640x480 (VGA - LeRobot 推荐)")

                except Exception as e:
                    print(f"❌ 创建数据集失败：{e}")
                    self.is_recording = False
        else:
            print("\n=========================================")
            print("⏹️ 停止录制！正在保存 Episode 到磁盘...")
            if self.dataset is not None:
                try:
                    self.dataset.save_episode()
                    print(f"💾 Episode 保存完毕！路径：{self.local_dir}")
                    print(f"📊 数据统计:")
                    print(f"   - 总帧数：{len(self.dataset)}")
                    print(f"   - Episode 数：{self.dataset.num_episodes}")
                except Exception as e:
                    print(f"❌ 保存失败：{e}")
                    print("请检查磁盘空间或权限设置")
            print("=========================================")

    def step(self, obs_img, robot_state, action):
        if self.is_recording and self.dataset is not None:

            # 1. 检查输入是否为 None
            if obs_img is None:
                print("⚠️ 警告：输入图像为 None，跳过此帧")
                return
            
            # 2. 转换为 numpy 数组（如果还不是）
            try:
                import torch
                if hasattr(obs_img, 'cpu'):
                    # PyTorch Tensor → Numpy
                    obs_img = obs_img.cpu().numpy()
                elif hasattr(obs_img, 'get_array'):
                    # PIL Image → Numpy
                    obs_img = np.array(obs_img)
            except Exception:
                pass
            
            # 3. 确保是 numpy 数组
            if not isinstance(obs_img, np.ndarray):
                print(f"⚠️ 警告：图像类型错误 {type(obs_img)}，跳过此帧")
                return
            
            # 4. 检查图像维度
            if len(obs_img.shape) < 2:
                print(f"⚠️ 警告：图像维度不正确 {obs_img.shape}，跳过此帧")
                return
            
            # 5. 处理灰度图（添加通道维度）
            if len(obs_img.shape) == 2:
                obs_img = cv2.cvtColor(obs_img, cv2.COLOR_GRAY2RGB)
            
            # 6. 处理 RGBA 图像（去掉 Alpha 通道）
            elif obs_img.shape[2] == 4:
                obs_img = obs_img[:, :, :3]
            
            # 7. 确保数据类型正确
            if obs_img.dtype == np.float32 or obs_img.dtype == np.float64:
                # 浮点型 [0,1] → uint8 [0,255]
                if obs_img.max() <= 1.0:
                    obs_img = (obs_img * 255).astype(np.uint8)
                else:
                    obs_img = obs_img.astype(np.uint8)
            elif obs_img.dtype != np.uint8:
                # 其他类型强制转换为 uint8
                obs_img = obs_img.astype(np.uint8)
            
            # 8. 缩放到标准尺寸 (W=640, H=480)
            try:
                img_resized = cv2.resize(obs_img, (640, 480), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                print(f"❌ 图像缩放失败：{e}")
                print(f"   原始 shape: {obs_img.shape}, dtype: {obs_img.dtype}")
                return
            
            # 9. 最终验证
            if img_resized.shape != (480, 640, 3):
                print(f"⚠️ 警告：最终图像尺寸不正确 {img_resized.shape}")
                return
            
            # 10. 添加到数据集（添加 task 参数）
            frame_dict = {
                "observation.image": img_resized,
                "observation.state": np.array(robot_state, dtype=np.float32),
                "action": np.array(action, dtype=np.float32)
            }
            
            # 尝试带 task 参数调用 add_frame
            try:
                self.dataset.add_frame(frame_dict, task=self.task_description)
            except TypeError:
                # 如果当前版本不支持 task 参数，则不带 task 调用
                self.dataset.add_frame(frame_dict)

def main():
    # robot_name = choose_from_options(options=list(sorted(REGISTERED_ROBOTS)), name="robot")
    # scene_model = choose_from_options(options=["empty", "Rs_int"], name="scene")

    robot_cfg = {
        "type": "R1ProWithSvhHand",
        "action_type": "continuous",
        "action_normalize": False,
        "finger_static_friction": 1000.0,
        "finger_dynamic_friction": 1000.0,
        "obs_modalities": ["rgb", "proprio"],  # 强制开启 RGB 渲染
    }
    
    cfg = {"scene": {"type": "Scene",
                     "scene_model": "empty"},
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
                   "position": [0.5, 0.1, 1.3],
               },
               {
                   "type": "DatasetObject",
                   "name": "bottle_1",
                   "category": "wine_bottle",
                   "model": "hlzfxw",
                   "fixed_base": False,
                   "visual_only": False,
                   "position": [0.5, -0.1, 1.4],
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
            controller_config[component] = {"name": "MultiFingerGripperController", "mode": "independent"}
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

    # 初始化数据录制器
    recorder = LeRobotRecorder(robot=robot, fps=30)

    # 注册键盘按键 'T' 为录制开关
    kb_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.T,
        description="Toggle LeRobot Data Recording",
        callback_fn=lambda: recorder.toggle_recording(),
    )

    base_idx = robot.controller_action_idx.get("base", [])
    arm_indices = {k: v for k, v in robot.controller_action_idx.items() if k.startswith("arm")}
    gripper_indices = {k: v for k, v in robot.controller_action_idx.items() if k.startswith("gripper")}
    other_indices = []
    for comp in ["trunk", "camera", "legs", "head"]:
        other_indices.extend(robot.controller_action_idx.get(comp, []))

    print(f"\n🟢 【数采模式启动】已修复图像 API！")
    print(f"👉 按下键盘 'T' 键开始录制动作，再按一次 'T' 保存 Episode！")

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

        # 步进环境并获取观测字典
        obs, _, _, _, _ = env.step(combined_action)

        rgb_img = None

        # 从多层嵌套字典中提取 RGB 图像
        for key, value in obs.items():
            if isinstance(value, dict):
                # 第一层嵌套：robot_zqkfvs
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        # 第二层嵌套：zed_link:Camera:0
                        if 'rgb' in sub_value:
                            rgb_img = sub_value['rgb']
                            break
                        # 或者可能是其他 key 名
                        for deep_key, deep_value in sub_value.items():
                            if 'rgb' in deep_key.lower() and hasattr(deep_value, 'shape'):
                                rgb_img = deep_value
                                break
        
        # 提取机器人的关节状态
        robot_state = robot.get_joint_positions()

        # 灌入录制器
        recorder.step(obs_img=rgb_img, robot_state=robot_state, action=combined_action)

        elapsed = time.time() - loop_start
        if elapsed < target_dt: time.sleep(target_dt - elapsed)


if __name__ == "__main__":
    main()
