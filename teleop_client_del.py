import time

import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy
import zmq
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options
from scipy.spatial.transform import Rotation as R

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


def main():
    robot_name = choose_from_options(options=list(sorted(REGISTERED_ROBOTS)), name="robot")
    scene_model = choose_from_options(options=["empty", "Rs_int"], name="scene")
    robot_cfg = {"type": robot_name,
                 "action_type": "continuous",
                 "action_normalize": False,
                 "finger_static_friction": 1000.0,
                 "finger_dynamic_friction": 1000.0,

                 "grasping_mode": "assisted"
                 }

    cfg = {"scene": {"type": "Scene" if scene_model == "empty" else "InteractiveTraversableScene",
                     "scene_model": scene_model},
           "objects": [{
               "type": "DatasetObject",
               "name": "table",
               "category": "breakfast_table",
               "model": "lcsizg",
               "position": [0.6, 0.0, 1.2],
               "fixed_base": True,
           },
               # {
               #     "type": "PrimitiveObject",
               #     "name": "my_cube",
               #     "primitive_type": "Cube",
               #     "rgba": [1.0, 0.0, 0.0, 1.0],
               #     "scale": [0.05, 0.05, 0.05],
               #     "position": [0.5, -0.1, 1.3],
               # },
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
                   "fixed_base": False,  # 是否固定基座（False=可移动，True=固定）
                   "visual_only": False,
                   "position": [0.5, -0.1, 1.4],
               },
               # {
               #     "type": "DatasetObject",  # 物体类型：从数据集加载
               #     "name": "my_tissues_0",  # 唯一名称：类别_随机后缀_实例号
               #     "category": "box_of_tissues",  # 语义类别（用于 BDDL 任务和查询）
               #     "model": "ntbrtz",  # 3D 模型标识符（对应 USD文件）
               #     "scale": [1.0, 1.0, 1.0],  # XYZ 缩放比例（可选，默认 1.0）
               #     "fixed_base": False,  # 是否固定基座（False=可移动，True=固定）
               #     "visual_only": False,  # 是否仅视觉（False=有碰撞，True=无碰撞）
               #     "position": [2.80, 0.69, 0.54],  # 初始位置 [x, y, z] (单位：米)
               #     "orientation": [0, 0, 0, 1],  # 初始方向四元数 [x, y, z, w]
               #     "in_rooms": ["kitchen_0"],  # 所在房间（可选，用于语义定位）
               # }
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
                "mode": "pose_delta_ori"
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

    base_idx = robot.controller_action_idx.get("base", [])
    arm_indices = {k: v for k, v in robot.controller_action_idx.items() if k.startswith("arm")}
    gripper_indices = {k: v for k, v in robot.controller_action_idx.items() if k.startswith("gripper")}
    other_indices = []
    for comp in ["trunk", "camera", "legs", "head"]:
        other_indices.extend(robot.controller_action_idx.get(comp, []))

    print(f"\n🟢 服务端启动")

    target_dt = 1.0 / 30.0
    while True:
        loop_start = time.time()
        kb_action = kb_generator.get_teleop_action()

        arm_deltas, gripper_poses = vs_generator.update_and_get_vision_parts()

        combined_action = np.zeros(robot.action_dim)
        if len(base_idx) > 0: combined_action[base_idx] = kb_action[base_idx]
        for idx in other_indices: combined_action[idx] = kb_action[idx]

        for arm_key, arm_idx in arm_indices.items():
            if len(arm_idx) > 0 and arm_key in arm_deltas:
                combined_action[arm_idx] = arm_deltas[arm_key]

        for gripper_key, gripper_idx in gripper_indices.items():
            if len(gripper_idx) > 0 and gripper_key in gripper_poses:
                combined_action[gripper_idx] = gripper_poses[gripper_key]

        env.step(combined_action)
        elapsed = time.time() - loop_start
        if elapsed < target_dt: time.sleep(target_dt - elapsed)


if __name__ == "__main__":
    main()
