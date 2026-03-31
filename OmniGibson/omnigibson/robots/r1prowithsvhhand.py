from functools import cached_property

import numpy as np
import torch as th

from omnigibson.robots.r1 import R1
from scipy.spatial.transform import Rotation as R


class DummyAGMarker:
    """完美兼容 OmniGibson 底层辅助抓取 (Assisted Grasp) 的动态雷达点"""

    def __init__(self, robot, link_name, local_offset):
        self.robot = robot
        self.link_name = link_name
        self.local_offset = local_offset

    @property
    def position(self):
        """物理引擎每帧都会呼叫这个属性，来获取指尖此时此刻的绝对世界坐标"""
        import torch

        # 获取机械臂对应手指的连杆对象
        link = self.robot.links.get(self.link_name)
        if link is None:
            return torch.zeros(3)

        # 实时获取当前物理姿态 (OmniGibson 默认返回 Torch Tensor)
        pos, quat = link.get_position_orientation()

        # 安全剥离 Tensor，进行数学矩阵计算
        is_tensor = hasattr(pos, "cpu")
        pos_np = pos.detach().cpu().numpy() if is_tensor else np.array(pos)
        quat_np = quat.detach().cpu().numpy() if hasattr(quat, "cpu") else np.array(quat)

        # 将局部偏移坐标(local_offset)转换到世界坐标系
        rot_mat = R.from_quat(quat_np).as_matrix()
        world_pos = pos_np + rot_mat @ np.array(self.local_offset)

        # 伪装回 Tensor 发送给引擎
        if is_tensor:
            return torch.tensor(world_pos, dtype=torch.float32, device=pos.device)
        return world_pos


class R1ProWithSvhHand(R1):
    """
    R1 Pro Robot
    """

    @property
    def tucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]

        # 🦵 弯曲膝盖/躯干姿态 (向前弯曲 + 轻微下蹲)
        # torso_joint1: 向前弯曲 0.25 弧度 (~14 度)
        # torso_joint2: 保持 0 (无侧倾)
        # torso_joint3: 向下弯曲 -0.15 弧度 (~8.5 度)
        # torso_joint4: 保持 0 (无旋转)
        pos[self.trunk_control_idx] = th.tensor([0.3, 0.0, -0.45, 0.0])

        for arm in self.arm_names:
            pos[self.gripper_control_idx[arm]] = th.tensor([0.0, 0.0, 0.0, 0.0, 0.0,
                                                            0.0, 0.0, 0.0, 0.0, 0.0,
                                                            0.0, 0.0, 0.0, 0.0, 0.0,
                                                            0.0, 0.0, 0.0, 0.0, 0.0])  # open gripper
        return pos

    @property
    def untucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]

        # 🦵 弯曲膝盖/躯干姿态 (展开姿态，稍微弯曲)
        # torso_joint1: 向前弯曲 0.15 弧度 (~8.5 度)
        # torso_joint2: 保持 0 (无侧倾)
        # torso_joint3: 向下弯曲 -0.08 弧度 (~4.5 度)
        # torso_joint4: 保持 0 (无旋转)
        # pos[self.trunk_control_idx] = th.tensor([0.1, 0, -0.45, 0])

        for arm in self.arm_names:
            pos[self.gripper_control_idx[arm]] = th.tensor([0.0, 0.0, 0.0, 0.0, 0.0,
                                                            0.0, 0.0, 0.0, 0.0, 0.0,
                                                            0.0, 0.0, 0.0, 0.0, 0.0,
                                                            0.0, 0.0, 0.0, 0.0, 0.0])  # open gripper
        pos[self.arm_control_idx["left"]] = th.tensor([-1.57, 0.0, -3.14, 0.0, 0.0, 0.0, 0.0])
        pos[self.arm_control_idx["right"]] = th.tensor([-1.57, 0.0, 3.14, 0.0, 0.0, 0.0, 0.0])
        return pos

    @cached_property
    def floor_touching_base_link_names(self):
        return ["wheel_motor_link1", "wheel_motor_link2", "wheel_motor_link3"]

    @cached_property
    def arm_link_names(self):
        return {arm: [f"{arm}_arm_link{i}" for i in range(1, 8)] for arm in self.arm_names}

    @cached_property
    def arm_joint_names(self):
        return {arm: [f"{arm}_arm_joint{i}" for i in range(1, 8)] for arm in self.arm_names}

    @cached_property
    def finger_link_names(self):
        link_suffixes = [
            "e1",
            "base_link",
            "z",
            "a",
            "b",
            "c",
            "virtual_l",
            "l",
            "p",
            "t",
            "e2",
            "virtual_i",
            "i",
            "m",
            "q",
            "virtual_j",
            "j",
            "n",
            "r",
            "virtual_k",
            "k",
            "o",
            "s",
            "base_link_01",
            "thtip",  # 拇指末端
            "fftip",  # 食指末端
            "lftip",  # 小指末端
            "rftip",  # 无名指末端
            "mftip"  # 中指末端
        ]

        return {arm: [f"{arm}_hand_{suffix}" for suffix in link_suffixes] for arm in self.arm_names}

    @cached_property
    def finger_joint_names(self):
        joint_suffixes = [
            "Thumb_Opposition",  # 0 拇指对掌
            "Thumb_Flexion",  # 1 拇指屈曲
            "j3",  # 2 拇指关节3
            "j4",  # 3 拇指关节4
            "index_spread",  # 4 食指展开(特殊，镜像关节但没目标)
            "Index_Finger_Proximal",  # 5 食指近端
            "Index_Finger_Distal",  # 6 食指远端
            "j14",  # 7 食指关节14
            "j5",  # 8 不确定
            "Finger_Spread",  # 9 手指展开
            "Pinky",  # 10 小指
            "j13",  # 11 小指关节13
            "j17",  # 12 小指关节17
            "ring_spread",  # 13 无名指展开
            "Ring_Finger",  # 14 无名指
            "j12",  # 15 无名指关节12
            "j16",  # 16 无名指关节16
            "Middle_Finger_Proximal",  # 17 中指近端
            "Middle_Finger_Distal",  # 18 中指远端
            "j15"  # 19 中指关节15
        ]

        return {arm: [f"{arm}_hand_{suffix}" for suffix in joint_suffixes] for arm in self.arm_names}

    @property
    def arm_workspace_range(self):
        return {arm: th.deg2rad(th.tensor([-45, 45], dtype=th.float32)) for arm in self.arm_names}

    @property
    def disabled_collision_pairs(self):
        # 1. 保留你原有的、手动配置的其他部位的忽略碰撞对
        collision_pairs = [
            ["left_arm_link1", "torso_link4"],
            ["left_arm_link2", "torso_link4"],
            ["right_arm_link1", "torso_link4"],
            ["right_arm_link2", "torso_link4"],
            ["left_arm_link5", "left_arm_link7"],
            ["right_arm_link5", "right_arm_link7"],
            ["base_link", "wheel_motor_link1"],
            ["base_link", "wheel_motor_link2"],
            ["base_link", "wheel_motor_link3"],
            ["torso_link2", "torso_link4"],
        ]

        import itertools  # 请把这行加在你的 Python 文件最顶部的 import 区域

        # 2. 动态生成手部内部的忽略碰撞对 (极大减少工作量)
        for arm in self.arm_names:
            # 获取当前手的所有 link 名称 (比如 29 个)
            hand_links = self.finger_link_names[arm]

            # 使用 itertools.combinations 自动生成这 29 个 link 的所有两两不重复组合 (C_29^2 = 406对)
            # hand_pairs 会生成类似: [('left_hand_e1', 'left_hand_base_link'), ('left_hand_e1', 'left_hand_z'), ...]
            hand_pairs = list(itertools.combinations(hand_links, 2))

            # 将 tuple 转换为 list，以符合 OmniGibson 的格式要求
            hand_pairs_list = [[link1, link2] for link1, link2 in hand_pairs]

            # 追加到总的过滤列表中
            collision_pairs.extend(hand_pairs_list)

        # (可选) 如果你想让左手和右手之间也不发生碰撞，可以取消下面代码的注释：
        # left_hand_links = self.finger_link_names["left"]
        # right_hand_links = self.finger_link_names["right"]
        # lr_pairs = [[l_link, r_link] for l_link in left_hand_links for r_link in right_hand_links]
        # collision_pairs.extend(lr_pairs)

        return collision_pairs

    @property
    def assisted_grasp_start_points(self):
        """
        射线发射起点：手掌心 (base_link)。
        """
        return {
            arm: [
                DummyAGMarker(self, f"{arm}_hand_base_link", [0.0, 0.0, 0.0]),
                DummyAGMarker(self, f"{arm}_hand_base_link", [0.0, 0.0, 0.0]),
                DummyAGMarker(self, f"{arm}_hand_base_link", [0.0, 0.0, 0.0]),
                DummyAGMarker(self, f"{arm}_hand_base_link", [0.0, 0.0, 0.0]),
            ] for arm in self.arm_names
        }

    @property
    def assisted_grasp_end_points(self):
        """
        射线接收终点：各个指尖 (tips)。
        """
        return {
            arm: [
                DummyAGMarker(self, f"{arm}_hand_fftip", [0.0, 0.0, 0.0]),  # 食指尖
                DummyAGMarker(self, f"{arm}_hand_mftip", [0.0, 0.0, 0.0]),  # 中指尖
                DummyAGMarker(self, f"{arm}_hand_rftip", [0.0, 0.0, 0.0]),  # 无名指尖
                DummyAGMarker(self, f"{arm}_hand_lftip", [0.0, 0.0, 0.0]),  # 小指尖
            ] for arm in self.arm_names
        }

    # 【新增代码段】在这里强行注入质量
    def _post_load(self):
        # 先调用父类的 _post_load，确保所有 Link 都已经加载到了内存中
        super()._post_load()

        # 遍历左右手
        for arm in self.arm_names:
            # 获取这只手所有的 link 名字
            for link_name in self.finger_link_names[arm]:
                # 安全检查：确保这个 link 确实在机器人的部件字典中
                if link_name in self.links:
                    # 强行设置质量为 0.02 (千克)
                    self.links[link_name].mass = 0.02
