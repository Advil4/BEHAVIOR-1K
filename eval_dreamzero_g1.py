#!/usr/bin/env python3
"""
DreamZero G1 仿真评估脚本

在 OmniGibson 仿真环境中运行 DreamZero 策略，通过 WebSocket 连接远程
DreamZero 推理服务器。

用法:
    # 先在远程服务器启动推理服务：
    python scripts/inference/serve_dreamzero_g1.py --port 8000

    # 然后在有 OmniGibson 的机器上运行评估：
    python eval_dreamzero_g1.py \
        --server-host <SERVER_IP> \
        --server-port 8000 \
        --instruction "Pick up the bottle" \
        --episodes 5

快捷键:
    R  重置环境
    Q  退出
"""

import os
import sys
import time
import uuid
import argparse
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardRobotController
from loguru import logger

# DreamZero WebSocket 客户端
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval_utils"))
from policy_client import WebsocketClientPolicy

logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>", level="INFO")

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.GUI = True


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "__array__"):
        return np.asarray(x)
    try:
        return np.array(x)
    except Exception:
        return None


def get_rgb_from_obs(obs):
    """从 OmniGibson 观测字典中提取第一帧可用的 RGB 图像。"""
    for key, value in obs.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict) and "rgb" in sub_value:
                    return to_numpy(sub_value["rgb"])
    return None


# ---------------------------------------------------------------------------
# 机器人配置（与 teleop_client_g1.py 保持一致）
# ---------------------------------------------------------------------------

def create_g1_robot_config():
    """创建 G1 机器人配置，使用 JointController 实现绝对关节位置控制。

    注意：机械臂使用 JointController（绝对位置），因为 DreamZero 模型预测的是
    绝对关节位置（而非增量）。遥操作客户端在数据采集时使用了
    InverseKinematicsController 配合 pose_delta_ori，但记录的 "action" 是
    下一状态的关节位置（绝对值），因此模型学会了从观测预测绝对关节位置。
    """
    ctrl_config = {}
    for comp in [
        "base", "camera", "head", "torso", "trunk",
        "arm_left", "arm_right", "gripper_left", "gripper_right",
    ]:
        if comp == "base":
            ctrl_config[comp] = {"name": "JointController", "use_delta_commands": False, "motor_type": "position"}
        elif comp in ("trunk", "torso", "head", "camera"):
            ctrl_config[comp] = {"name": "JointController", "use_delta_commands": False, "motor_type": "position"}
        elif "arm" in comp:
            # 绝对位置控制: 模型输出即为期望的关节角度
            ctrl_config[comp] = {"name": "JointController", "use_delta_commands": False, "motor_type": "position"}
        elif "gripper" in comp:
            ctrl_config[comp] = {
                "name": "MultiFingerGripperController",
                "mode": "independent",
                "command_input_limits": None,
                "command_output_limits": None,
            }

    return {
        "type": "G1WithDex3Hand",
        "action_type": "continuous",
        "action_normalize": False,
        "obs_modalities": ["rgb"],
        "grasping_mode": "physical",
        "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_width": 640, "image_height": 480}}},
        "controller_config": ctrl_config,
    }


# ---------------------------------------------------------------------------
# 场景配置
# ---------------------------------------------------------------------------

def create_apple_scene():
    return {
        "scene": {"type": "Scene", "scene_model": "empty"},
        "objects": [
            {
                "type": "DatasetObject",
                "name": "table",
                "category": "breakfast_table",
                "model": "lcsizg",
                "position": [0.6, 0.0, 0.035],
                "fixed_base": True,
            },
            {
                "type": "DatasetObject",
                "name": "bottle_of_barbecue_sauce",
                "category": "bottle_of_barbecue_sauce",
                "model": "ikbsox",
                "position": [0.4, -0.05, 0.2],
            },
        ],
    }


# ---------------------------------------------------------------------------
# 评估器
# ---------------------------------------------------------------------------

class DreamZeroG1Evaluator:
    """在 G1 仿真环境中评估 DreamZero 策略。

    通过 WebSocket 连接远程 DreamZero 推理服务器。

    关键：必须与训练数据格式一致：
      - 状态：robot.get_joint_positions() 的前 36 维（完整 env_action 空间）
      - 动作：模型输出 36 维，直接映射到 env_action（无需切片）
      - 旧版错误：状态被切片为 8 维（7 关节 + 1 夹爪），现已改为完整 36 维
    """

    def __init__(
        self,
        server_host: str,
        server_port: int,
        instruction: str = "Pick up the bottle",
        max_steps_per_episode: int = 200,
        replan_steps: int = 8,
        gripper_gain: float = 1.0,
        gripper_scale: float = 1.0,
        gripper_delay_steps: int = 0,
    ):
        self.instruction = instruction
        self.max_steps_per_episode = max_steps_per_episode
        self.replan_steps = replan_steps  # 每隔 N 步重新查询模型（≤ action_horizon）
        self.gripper_gain = gripper_gain  # ≥1.0 放大闭合增量；1.0 = 直通
        self.gripper_scale = gripper_scale  # 直接缩放夹爪绝对值：>1 = 手指更弯, <1 = 手指更直
        self.gripper_delay_steps = gripper_delay_steps  # 前 N 步夹爪保持张开（0 = 不延迟）

        # WebSocket 客户端
        self.client = WebsocketClientPolicy(server_host, server_port)
        self.session_id = str(uuid.uuid4())

        # 动作块：模型输出 (24, 36) = action_horizon × 36 维关节
        self.action_horizon = 24  # 与 checkpoint 配置一致
        self.actions_from_chunk = 0
        self.action_chunk = None

        self.robot = None

        logger.info(f"✓ 已连接 DreamZero 服务器 {server_host}:{server_port}")
        logger.info(f"✓ 会话 ID: {self.session_id}")
        logger.info(f"✓ 重规划间隔: 每 {self.replan_steps} 步 (action_horizon={self.action_horizon})")
        logger.info(f"✓ 夹爪增益: {self.gripper_gain}x")
        logger.info(f"✓ 夹爪缩放: {self.gripper_scale}x")
        if self.gripper_delay_steps > 0:
            logger.info(f"✓ 夹爪延迟: 前 {self.gripper_delay_steps} 步保持张开")
        logger.info(f"✓ 指令: {instruction}")

    # ---- 状态提取 ------------------------------------------------

    def _extract_state(self, robot) -> np.ndarray:
        """提取与训练数据格式匹配的完整 36 维关节位置。

        模型在全部 36 维关节位置上训练（不拆分）。
        直接返回 (36,) float64 数组。
        """
        jp = robot.get_joint_positions()
        if hasattr(jp, "cpu"):
            jp = jp.cpu().numpy()

        state_36 = np.zeros(36, dtype=np.float64)
        n_copy = min(36, len(jp))
        state_36[:n_copy] = jp[:n_copy]
        return state_36

    def _extract_robot_joint_map(self, robot) -> dict:
        """构建映射: {控制器名称: list of (act_i, dof_i) 对}。

        包含躯干控制器（物理关节 0-2），因为模型预测所有物理关节 0-7。
        使用 zip(controller_action_idx[comp], controller.dof_idx) 与
        eval_psi0_client.py 类似，将模型输出维度正确映射到 env_action 索引。
        """
        mapping = {}
        for comp_name in ["trunk", "arm_left", "arm_right", "gripper_left", "gripper_right"]:
            if comp_name not in robot.controller_action_idx:
                continue
            act_indices = robot.controller_action_idx[comp_name]
            controller = robot.controllers[comp_name]
            dof_indices = getattr(controller, 'dof_idx', [])
            # 存储为 (act_i, dof_i) 对列表
            mapping[comp_name] = list(zip(act_indices, dof_indices))
        return mapping

    # ---- 观测构建 --------------------------------------------

    def _build_request(self, rgb_img: np.ndarray) -> dict:
        """构建发送到推理服务器的观测字典。

        G1 模型需要：
          - observation/exterior_image_1_left: (H, W, 3) uint8 — 单视角
          - observation/joint_position: (36,) float64 — 完整 36 维关节位置
          - prompt: str
          - session_id: str
        """
        joint_36 = self._extract_state(self.robot)

        # 确保 RGB 图像格式正确
        if rgb_img is not None:
            if rgb_img.dtype in (np.float32, np.float64):
                rgb_img = (rgb_img * 255).astype(np.uint8) if rgb_img.max() <= 1.0 else rgb_img.astype(np.uint8)
            elif rgb_img.dtype != np.uint8:
                rgb_img = rgb_img.astype(np.uint8)
            rgb_img = np.ascontiguousarray(rgb_img[..., :3]) if rgb_img.shape[-1] == 4 else np.ascontiguousarray(rgb_img)
        else:
            rgb_img = np.zeros((480, 640, 3), dtype=np.uint8)

        return {
            "observation/exterior_image_1_left": rgb_img,
            "observation/joint_position": joint_36,
            "prompt": self.instruction,
            "session_id": self.session_id,
        }

    # ---- 动作推理 ------------------------------------------------

    def infer_action(self, rgb_img: np.ndarray) -> np.ndarray:
        """从模型获取一个 36 维动作。

        每隔 replan_steps（默认 8）步重新查询模型，而不是等完整 24 步的
        动作块耗尽。更频繁的重规划让模型更快获得视觉反馈，减少开环预测
        漂移导致的夹爪振荡。

        返回:
            ndarray of shape (36,): 完整 36 维关节位置
        """
        if self.action_chunk is None or self.actions_from_chunk >= self.replan_steps:
            # 从服务器请求新的动作块
            self.actions_from_chunk = 0
            request_data = self._build_request(rgb_img)
            result = self.client.infer(request_data)

            if isinstance(result, dict) and "actions" in result:
                chunk = result["actions"]
            elif isinstance(result, np.ndarray):
                chunk = result
            else:
                raise TypeError(f"Unexpected server response type: {type(result)}")

            # 形状: (action_horizon, 36)
            assert chunk.ndim == 2, f"Expected 2D action chunk, got {chunk.shape}"
            assert chunk.shape[1] == 36, f"Expected 36 action dims, got {chunk.shape[1]}"

            self.action_chunk = chunk
            logger.info(f"✓ 新动作块: shape={chunk.shape}, "
                        f"min={chunk.min():.4f}, max={chunk.max():.4f}, "
                        f"mean={chunk.mean():.4f}")
            # 打印动作块第一个动作用于调试
            logger.info(f"  chunk[0] = {np.round(chunk[0], 4)}")

            # --- 诊断：全 24 步的右臂轨迹（检查模型是否预测了"拿起"） ---
            # 右臂 DOF：[4, 6, 8, 10, 12, 14, 16]
            #   DOF 4  = 肩膀俯仰（抬起前伸）
            #   DOF 10 = 肘部弯曲（向前伸展）
            #   DOF 12 = 前臂（抬升/放下）
            arm_r_dof_names = {4: "肩俯仰", 6: "肩横滚", 8: "上臂转", 10: "肘弯", 12: "前臂", 14: "腕摆", 16: "腕转"}
            logger.info(f"  右臂 24 步轨迹（检查是否预测抬起）:")
            for dof_i in [4, 6, 8, 10, 12, 14, 16]:
                if dof_i < chunk.shape[1]:
                    traj = chunk[:, dof_i]
                    delta_total = traj[-1] - traj[0]
                    logger.info(
                        f"    DOF {dof_i:2d} ({arm_r_dof_names.get(dof_i, '?'):4s}): "
                        f"起点={traj[0]:.4f} → 终点={traj[-1]:.4f} "
                        f"(Δ={delta_total:+.4f}) "
                        f"min={traj.min():.4f} max={traj.max():.4f}"
                    )
            # 右夹爪 24 步轨迹
            logger.info(f"  右夹爪 24 步轨迹:")
            gripper_r_dofs_key = [20, 21, 22, 26, 27, 28, 30]
            for dof_i in gripper_r_dofs_key:
                if dof_i < chunk.shape[1]:
                    traj = chunk[:, dof_i]
                    logger.info(
                        f"    DOF {dof_i:2d}: 起点={traj[0]:.4f} 终点={traj[-1]:.4f} "
                        f"min={traj.min():.4f} max={traj.max():.4f}"
                    )

        action = self.action_chunk[self.actions_from_chunk]
        self.actions_from_chunk += 1
        return action

    # ---- 动作 → 机器人映射 ------------------------------------------

    def _apply_action(self, robot, action_36: np.ndarray, step_count: int = 0) -> np.ndarray:
        """将模型 36 维动作（get_joint_positions 顺序）路由到 env_action
        （controller_action_idx 顺序）。

        参考 eval_psi0_client.py：仅映射实际活动的 arm/gripper 控制器，
        通过 zip(controller_action_idx[comp], controller.dof_idx) 将模型输出
        维度正确路由到仿真器 action 索引。跳过 trunk（关节 0-2 固定不动）。

        关键：模型被训练预测 state_next = get_joint_positions()[:36]，
        使用的是机器人本体的 DOF 顺序。但 env.step() 期望的动作顺序是
        controller_action_idx 顺序。对于 G1，左右臂/手关节是交错的：
          - arm_left dof=[3,5,7,9,11,13,15]   → action[3:10]
          - arm_right dof=[4,6,8,10,12,14,16] → action[10:17]
          - gripper_left dof=[17,23,18,24,19,25,29]  → action[17:24]
          - gripper_right dof=[20,26,21,27,22,28,30] → action[24:31]

        注意：动作反归一化（q99 逆运算）在服务器端由 sim_policy.unapply()
        处理——客户端不需要反归一化。
        """
        env_action = np.zeros(robot.action_dim, dtype=np.float32)

        # 活动控制器：参考 eval_psi0_client.py，仅映射手臂和灵巧手（跳过 trunk）
        active_controllers = ["arm_left", "arm_right", "gripper_left", "gripper_right"]

        # --- 构建映射，仅覆盖活动控制器 ---
        dof_to_act: dict[int, int] = {}
        comp_info: dict[str, dict] = {}  # 诊断用

        for comp_name in active_controllers:
            if comp_name not in robot.controller_action_idx:
                continue
            act_indices = robot.controller_action_idx[comp_name]
            controller = robot.controllers[comp_name]
            dof_indices = getattr(controller, 'dof_idx', [])

            # 回退：如果控制器没有 dof_idx（如 MultiFingerGripperController），
            # 则假设直接 1:1 映射（act_i == dof_i）
            if len(dof_indices) == 0:
                dof_indices = list(act_indices)

            comp_info[comp_name] = {
                "act_indices": act_indices,
                "dof_indices": dof_indices,
                "n_act": len(act_indices),
                "n_dof": len(dof_indices),
                "fallback": len(getattr(controller, 'dof_idx', [])) == 0,
            }
            for act_i, dof_i in zip(act_indices, dof_indices):
                dof_i_int = int(dof_i)
                act_i_int = int(act_i)
                dof_to_act[dof_i_int] = act_i_int

        # 存储供后续调试日志使用（目标 vs 实际对比）
        self._debug_dof_to_act = dof_to_act

        # ---- 第一步详细诊断 ----
        if step_count == 0:
            logger.info("=" * 72)
            logger.info("DOF → Action Index 映射（参考 eval_psi0_client.py 格式）")
            logger.info(f"  get_joint_positions() 长度: {len(robot.get_joint_positions())}")
            logger.info(f"  robot.action_dim: {robot.action_dim}")
            logger.info(f"  模型输出维度: {action_36.shape}")
            logger.info("-" * 72)

            # --- 存储每个控制器的实际 dof_idx 供后续调试日志使用 ---
            self._debug_controller_dofs: dict[str, list[int]] = {}

            for comp_name in active_controllers:
                info = comp_info[comp_name]
                act = info["act_indices"]
                dof = info["dof_indices"]
                self._debug_controller_dofs[comp_name] = [int(d) for d in dof]
                fallback_tag = " [回退: act_i=dof_i]" if info.get("fallback") else ""
                if len(dof) > 0:
                    dof_str = f"dof={dof}{fallback_tag}" if len(dof) <= 12 else f"dof=[{dof[0]}..{dof[-1]}] len={len(dof)}{fallback_tag}"
                else:
                    dof_str = "⚠️  dof_idx=[] — 动作被丢弃!"
                logger.info(f"  [{comp_name:15s}] EnvAction {list(act)}  <---  模型维度 {dof_str}")

            # --- 显示被丢弃的模型输出维度 ---
            max_dof = min(len(action_36), 36)
            dropped = [i for i in range(max_dof) if i not in dof_to_act]
            if dropped:
                logger.warning(
                    f"⚠️  模型输出维度被丢弃（不在任何活动控制器的 dof_idx 中）: "
                    f"{dropped} (共 {len(dropped)} 维，含 trunk [0,1,2] + padding [31-35])"
                )
            else:
                logger.info("✓ 所有模型输出维度 (0-35) 均已映射到 action 槽位")

            # --- 按控制器打印完整初始状态 ---
            logger.info("-" * 72)
            logger.info("初始关节位置（按控制器排列）:")
            jp_init = robot.get_joint_positions()
            if hasattr(jp_init, 'cpu'):
                jp_init = jp_init.cpu().numpy()
            for comp_name in active_controllers:
                dofs = self._debug_controller_dofs.get(comp_name, [])
                dofs_36 = [d for d in dofs if d < 36]
                if dofs_36:
                    vals = [float(jp_init[d]) if d < len(jp_init) else 0.0 for d in dofs_36]
                    logger.info(f"  {comp_name:22s} dof={dofs_36}")
                    logger.info(f"  {'':22s} val={np.round(vals, 5)}")

            # --- 显示模型预测的初始状态变化量（第 0 步） ---
            logger.info("-" * 72)
            logger.info("模型 ACTION[0] vs 初始状态 (增量 = 目标 - 当前):")
            for comp_name in active_controllers:
                dofs = self._debug_controller_dofs.get(comp_name, [])
                dofs_36 = [d for d in dofs if d < 36 and d in dof_to_act]
                if dofs_36:
                    cur_vals = [float(jp_init[d]) if d < len(jp_init) else 0.0 for d in dofs_36]
                    tgt_vals = [float(action_36[d]) for d in dofs_36]
                    deltas = [t - c for t, c in zip(tgt_vals, cur_vals)]
                    max_delta = max(abs(d) for d in deltas)
                    logger.info(f"  {comp_name:22s} cur={np.round(cur_vals, 4)}")
                    logger.info(f"  {'':22s} tgt={np.round(tgt_vals, 4)}")
                    logger.info(f"  {'':22s} Δ  ={np.round(deltas, 4)}  max|Δ|={max_delta:.4f}")
            logger.info("=" * 72)

        # --- 先获取当前关节位置作为未映射关节的回退值 ---
        current_jp = robot.get_joint_positions()
        if hasattr(current_jp, 'cpu'):
            current_jp = current_jp.cpu().numpy()
        # 使用当前关节位置填充所有控制器的 action 槽位（作为安全回退）
        for dof_i in range(min(len(current_jp), robot.action_dim)):
            if dof_i in dof_to_act:
                env_action[dof_to_act[dof_i]] = current_jp[dof_i]

        # --- 将模型预测覆盖到相应 action 槽位（参考 psi0: env_action[act_i] = model[dof_i]）---
        # 构建夹爪 DOF 集合用于增益放大
        gripper_dofs: set[int] = set()
        for comp_name in active_controllers:
            if comp_name.startswith("gripper"):
                controller = robot.controllers[comp_name]
                dof_indices = getattr(controller, 'dof_idx', [])
                if len(dof_indices) == 0:
                    dof_indices = list(robot.controller_action_idx[comp_name])
                for d in dof_indices:
                    gripper_dofs.add(int(d))

        for dof_i in range(min(len(action_36), 36)):
            if dof_i not in dof_to_act:
                continue
            target = float(action_36[dof_i])

            # 夹爪动作放大：
            #   gripper_scale 直接缩放绝对值（用于测试模型值是否太小）
            #   gripper_gain 放大增量（用于克服 PD 接触力不足）
            if dof_i in gripper_dofs:
                # 夹爪延迟：手臂到位前，强制保持张开（目标 = 当前值，即不闭合）
                if self.gripper_delay_steps > 0 and step_count < self.gripper_delay_steps:
                    if step_count == 0:
                        logger.info(f"  ⏸️  夹爪延迟生效：前 {self.gripper_delay_steps} 步保持张开，此后放开模型控制")
                    env_action[dof_to_act[dof_i]] = 0.0  # 保持打开
                    continue

                if self.gripper_scale != 1.0:
                    target = target * self.gripper_scale
                if self.gripper_gain > 1.0:
                    current = float(current_jp[dof_i]) if dof_i < len(current_jp) else target
                    delta = target - current
                    if abs(delta) > 0.001:
                        target = current + delta * self.gripper_gain
                        if step_count == 0:
                            logger.info(
                                f"  夹爪 DOF {dof_i}: 模型={float(action_36[dof_i]):.4f} "
                                f"缩放→{float(action_36[dof_i]) * self.gripper_scale:.4f} "
                                f"增量放大→{target:.4f}"
                            )

            env_action[dof_to_act[dof_i]] = target

        return env_action

    # ---- 回合循环 ----------------------------------------------------

    def run_episode(self, env, robot, episode_idx: int, save_video: bool = True) -> dict:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"回合 {episode_idx + 1}")
        logger.info(f"{'=' * 60}")

        obs, _ = env.reset()
        self.robot = robot
        self.action_chunk = None
        self.actions_from_chunk = 0
        self.session_id = str(uuid.uuid4())
        try:
            self.client.reset({"session_id": self.session_id})
        except Exception:
            pass  # reset 是尽力而为的

        # 构建关节映射（保留供参考，不再用于动作映射）
        _joint_map = self._extract_robot_joint_map(robot)
        logger.info(f"--- 诊断：第 {episode_idx + 1} 回合初始化 ---")
        logger.info(f"  robot.action_dim: {robot.action_dim}")
        logger.info(f"  模型输出 36 维, 通过 dof_idx→action_idx 映射到 env_action")
        logger.info(f"------------------------------------")

        frames = [] if save_video else None
        episode_success = False
        step_count = 0

        for step in range(self.max_steps_per_episode):
            step_start = time.time()

            rgb_img = get_rgb_from_obs(obs)
            if rgb_img is None:
                obs, *_ = env.step(np.zeros(robot.action_dim, dtype=np.float32))
                step_count += 1
                continue

            # 从 DreamZero 模型获取动作（36 维，直接在 env_action 空间）
            action_36 = self.infer_action(rgb_img)

            # 模型输出即为 env_action（都是 36 维关节位置）
            env_action = self._apply_action(robot, action_36, step_count=step_count)

            # 调试：打印前几步的 env_action
            if step_count < 5:
                logger.info(f"Step {step_count}: env_action[:16] = {np.round(env_action[:16], 4)}")
                logger.info(f"Step {step_count}: env_action[16:] = {np.round(env_action[16:], 4)}")

            # 步进仿真
            next_obs, reward, terminated, truncated, info = env.step(env_action)

            # --- 调试：目标 vs 实际（前 5 步） ---
            if step_count < 5:
                jp_after = robot.get_joint_positions()
                if hasattr(jp_after, 'cpu'):
                    jp_after = jp_after.cpu().numpy()

                dof_to_act = getattr(self, '_debug_dof_to_act', {})
                ctrl_dofs = getattr(self, '_debug_controller_dofs', {})

                # --- 按控制器分组的目标 vs 实际（仅活动控制器，跳过 trunk） ---
                for comp_name in ["arm_left", "arm_right", "gripper_left", "gripper_right"]:
                    dofs_all = ctrl_dofs.get(comp_name, [])
                    # 仅检查模型 36 维输出范围内的 DOF
                    dofs = [d for d in dofs_all if d < 36 and d in dof_to_act]
                    if not dofs:
                        continue

                    actual = [float(jp_after[d]) if d < len(jp_after) else 0.0 for d in dofs]
                    target = [float(env_action[dof_to_act[d]]) for d in dofs]
                    delta = [t - a for t, a in zip(target, actual)]
                    max_err = max(abs(d) for d in delta)
                    logger.info(f"Step {step_count}: {comp_name:22s} actual={np.round(actual, 4)}")
                    logger.info(f"Step {step_count}: {'':22s} target={np.round(target, 4)}  max_err={max_err:.4f}")

            # 记录视频帧
            if save_video:
                frame = get_rgb_from_obs(next_obs)
                if frame is not None:
                    # 去除 alpha 通道（mediapy 期望 RGB，非 RGBA）
                    if frame.shape[-1] == 4:
                        frame = frame[..., :3]
                    frame_uint8 = (frame * 255).astype(np.uint8) if frame.dtype in (np.float32, np.float64) and frame.max() <= 1.0 else frame.astype(np.uint8)
                    frames.append(np.ascontiguousarray(frame_uint8))

            # 检查成功
            if info.get("success", False):
                episode_success = True
                logger.success("✅ 任务完成!")
                break

            obs = next_obs
            step_count += 1

            if (step + 1) % 10 == 0:
                elapsed = time.time() - step_start
                logger.info(f"  Step {step + 1}/{self.max_steps_per_episode} | {elapsed:.3f}s")

        # 保存视频
        if save_video and frames:
            self._save_video(frames, episode_idx, episode_success)

        result = {"episode": episode_idx, "steps": step_count, "success": episode_success, "instruction": self.instruction}
        logger.info(f"Episode {episode_idx + 1}: {step_count} steps, success={episode_success}")
        return result

    def _save_video(self, frames, episode_idx, success):
        if not frames:
            logger.warning(f"回合 {episode_idx + 1}: 0 帧已收集, 跳过视频保存")
            return

        logger.info(f"正在保存视频: {len(frames)} 帧, "
                    f"frame[0] shape={frames[0].shape}, dtype={frames[0].dtype}")

        try:
            import mediapy

            output_dir = Path("eval_videos")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status = "success" if success else "fail"
            path = output_dir / f"ep{episode_idx:03d}_{status}_{timestamp}.mp4"
            mediapy.write_video(str(path), frames, fps=10)
            logger.info(f"💾 视频已保存: {path}")
        except Exception as e:
            import traceback
            logger.warning(f"视频保存失败: {e}")
            logger.warning(traceback.format_exc())
            # 回退：尝试 cv2
            try:
                self._save_video_cv2(frames, episode_idx, success)
            except Exception as e2:
                logger.warning(f"cv2 回退也失败: {e2}")

    def _save_video_cv2(self, frames, episode_idx, success):
        output_dir = Path("eval_videos")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "success" if success else "fail"
        path = output_dir / f"ep{episode_idx:03d}_{status}_{timestamp}_cv2.mp4"

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(path), fourcc, 10, (w, h))
        for frame in frames:
            # 确保 cv2 的 uint8 BGR
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()
        logger.info(f"💾 视频已保存 (cv2 回退): {path}")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DreamZero G1 仿真评估")
    parser.add_argument("--server-host", type=str, required=True, help="DreamZero 推理服务器地址")
    parser.add_argument("--server-port", type=int, default=8000, help="DreamZero 推理服务器端口")
    parser.add_argument("--instruction", type=str, default="Pick up the bottle", help="任务指令")
    parser.add_argument("--episodes", type=int, default=5, help="评估回合数")
    parser.add_argument("--max-steps", type=int, default=200, help="每回合最大步数")
    parser.add_argument("--replan-steps", type=int, default=8, help="每 N 步重新查询模型 (默认 8, 最大 24)")
    parser.add_argument("--gripper-gain", type=float, default=1.0, help="放大夹爪增量: >1 = 更强闭合 (建议 2.0-3.0)")
    parser.add_argument("--gripper-scale", type=float, default=1.0, help="直接缩放夹爪绝对值: >1 = 手指更弯 (建议 1.5-3.0)")
    parser.add_argument("--gripper-delay-steps", type=int, default=0, help="前 N 步夹爪保持张开，等手臂到位后再闭合 (建议 20-40)")
    parser.add_argument("--headless", action="store_true", help="无 GUI 运行")
    parser.add_argument("--no-video", action="store_true", help="禁用视频录制")
    args = parser.parse_args()

    if args.headless:
        gm.GUI = False
        logger.info("正在以无头模式运行")

    # 创建 G1 环境
    cfg = {**create_apple_scene(), "robots": [create_g1_robot_config()]}
    logger.info("正在创建 OmniGibson 环境...")
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    logger.success("✓ 环境就绪")
    logger.info(f"  Robot action_dim: {robot.action_dim}")
    logger.info(f"  Controller order: {robot.controller_order}")
    logger.info(f"  Controller action_idx: {robot.controller_action_idx}")

    # 键盘控制 (R=重置, Q=退出)
    kb = KeyboardRobotController(robot=robot)
    reset_flag = [False]
    quit_flag = [False]
    kb.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R, description="重置", callback_fn=lambda: reset_flag.__setitem__(0, True)
    )
    kb.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.Q, description="退出", callback_fn=lambda: quit_flag.__setitem__(0, True)
    )

    # 评估器
    evaluator = DreamZeroG1Evaluator(
        server_host=args.server_host,
        server_port=args.server_port,
        instruction=args.instruction,
        max_steps_per_episode=args.max_steps,
        replan_steps=args.replan_steps,
        gripper_gain=args.gripper_gain,
        gripper_scale=args.gripper_scale,
        gripper_delay_steps=args.gripper_delay_steps,
    )

    # 运行回合
    results = []
    for ep_idx in range(args.episodes):
        if quit_flag[0]:
            break
        if reset_flag[0]:
            reset_flag[0] = False
            continue

        result = evaluator.run_episode(env, robot, episode_idx=ep_idx, save_video=not args.no_video)
        results.append(result)

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("评估汇总")
    logger.info("=" * 60)
    total = len(results)
    success = sum(1 for r in results if r["success"])
    avg_steps = np.mean([r["steps"] for r in results]) if results else 0
    logger.info(f"回合数: {total}")
    logger.info(f"成功率: {success}/{total} ({success / max(total, 1) * 100:.1f}%)")
    logger.info(f"平均步数: {avg_steps:.1f}")
    logger.info(f"指令: {args.instruction}")

    env.close()
    logger.success("✓ 完成")


if __name__ == "__main__":
    main()
