from functools import cached_property

import torch as th

from omnigibson.robots.articulated_trunk_robot import ArticulatedTrunkRobot
from omnigibson.robots.mobile_manipulation_robot import MobileManipulationRobot
from omnigibson.utils.python_utils import classproperty


class G1WithDex3Hand(ArticulatedTrunkRobot, MobileManipulationRobot):
    """
    G1 Robot
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        fixed_base=False,  # Add fixed_base parameter
        visual_only=False,
        self_collisions=True,
        link_physics_materials=None,
        load_config=None,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        disable_grasp_handling=False,
        finger_static_friction=None,
        finger_dynamic_friction=None,
        # Unique to MobileManipulationRobot
        default_reset_mode="untuck",
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not. If True, creates a FixedJoint 
                connecting the robot's root link to the world, preventing any base movement.
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            link_physics_materials (None or dict): If specified, dictionary mapping link name to kwargs used to generate
                a specific physical material for that link's collision meshes, where the kwargs are arguments directly
                passed into the isaacsim.core.api.materials.physics_material.PhysicsMaterial constructor, e.g.: "static_friction",
                "dynamic_friction", and "restitution"
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                we will automatically set the control frequency to be at the render frequency by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self._default_joint_pos will be used instead.
                Note that _default_joint_pos are hardcoded & precomputed, and thus should not be modified by the user.
                Set this value instead if you want to initialize the robot with a different rese joint position.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is ["rgb", "proprio"].
                Valid options are "all", or a list containing any subset of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            include_sensor_names (None or list of str): If specified, substring(s) to check for in all raw sensor prim
                paths found on the robot. A sensor must include one of the specified substrings in order to be included
                in this robot's set of sensors
            exclude_sensor_names (None or list of str): If specified, substring(s) to check against in all raw sensor
                prim paths found on the robot. A sensor must not include any of the specified substrings in order to
                be included in this robot's set of sensors
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            sensor_config (None or dict): nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. This will override any default values specified by this class.
            grasping_mode (str): One of {"physical", "assisted", "sticky"}.
                If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
                If "assisted", will magnetize any object touching and within the gripper's fingers.
                If "sticky", will magnetize any object touching the gripper's fingers.
            disable_grasp_handling (bool): If True, will disable all grasp handling for this object. This means that
                sticky and assisted grasp modes will not work unless the connection/release methodsare manually called.
            finger_static_friction (None or float): If specified, specific static friction to use for robot's fingers
            finger_dynamic_friction (None or float): If specified, specific dynamic friction to use for robot's fingers.
                Note: If specified, this will override any ways that are found within @link_physics_materials for any
                robot finger gripper links
            default_reset_mode (str): Default reset mode for the robot. Should be one of: {"tuck", "untuck"}
                If reset_joint_pos is not None, this will be ignored (since _default_joint_pos won't be used during initialization).
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,  # Pass fixed_base to parent class
            visual_only=visual_only,
            self_collisions=self_collisions,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            include_sensor_names=include_sensor_names,
            exclude_sensor_names=exclude_sensor_names,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            disable_grasp_handling=disable_grasp_handling,
            finger_static_friction=finger_static_friction,
            finger_dynamic_friction=finger_dynamic_friction,
            default_reset_mode=default_reset_mode,
            **kwargs,
        )

    # # Name of the actual root link that we are interested in. Note that this is different from self.root_link_name,
    # # which is "base_footprint_x", corresponding to the first of the 6 1DoF joints to control the base.
    # @property
    # def base_footprint_link_name(self):
    #     return "base_link"

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("G1 does not support discrete actions!")

    @property
    def _raw_controller_order(self):
        # 暂时注释掉腿部控制器，先确保基本功能正常
        controllers = [
                    #    "base", 
                        "trunk", 
                        "arm_left", 
                        "arm_right", 
                        "gripper_left",
                        "gripper_right",
                        #    "leg_left",  # 暂时禁用
                        #    "leg_right"  # 暂时禁用
                        ]
        
        return controllers

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        # We use joint controllers for base as default
        # controllers["base"] = "HolonomicBaseJointController"
        controllers["trunk"] = "JointController"
        # arm_left, gripper_left, arm_right, gripper_right 由父类根据 arm_names 自动生成
        # 暂时不添加腿部控制器
        # controllers["leg_left"] = "JointController"
        # controllers["leg_right"] = "JointController"
        
        # controllers["trunk"] = "JointController"
        # controllers["arm_left"] = "JointController"
        # controllers["arm_right"] = "JointController"
        # controllers["gripper_left"] = "MultiFingerGripperController"
        # controllers["gripper_right"] = "MultiFingerGripperController"
        # controllers["leg_left"] = "JointController"
        # controllers["leg_right"] = "JointController"
        
        return controllers

    @property
    def tucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        # pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        # for arm in self.arm_names:
        #     pos[self.gripper_control_idx[arm]] = th.tensor([0.05, 0.05])  # open gripper
        return pos

    @property
    def untucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        # pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        # for arm in self.arm_names:
        #     pos[self.gripper_control_idx[arm]] = th.tensor([0.05, 0.05])  # open gripper
        #     pos[self.arm_control_idx[arm]] = th.tensor([0.0, 1.906, -0.991, 1.571, 0.915, -1.571])
        return pos

    # @cached_property
    # def floor_touching_base_link_names(self):
    #     return ["wheel_link1", "wheel_link2", "wheel_link3"]

    @cached_property
    def trunk_link_names(self):
        return ["pelvis", "torso_link", "waist_yaw_link", "waist_roll_link"]

    @cached_property
    def trunk_joint_names(self):
        return ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

    @classproperty
    def n_arms(cls):
        return 2

    @classproperty
    def arm_names(cls):
        return ["left", "right"]

    @classproperty
    def leg_names(cls):
        return ["left", "right"]

    @cached_property
    def arm_link_names(self):
        return {
                "left": ["left_shoulder_pitch_link", 
                         "left_shoulder_roll_link", 
                         "left_shoulder_yaw_link", 
                         "left_elbow_link", 
                         "left_wrist_roll_link", 
                         "left_wrist_pitch_link", 
                         "left_wrist_yaw_link",
                         ],
                "right": ["right_shoulder_pitch_link", 
                          "right_shoulder_roll_link", 
                          "right_shoulder_yaw_link", 
                          "right_elbow_link", 
                          "right_wrist_roll_link", 
                          "right_wrist_pitch_link", 
                          "right_wrist_yaw_link",
                          ],
                }

    @cached_property
    def arm_joint_names(self):
        return {
                "left": ["left_shoulder_pitch_joint",
                         "left_shoulder_roll_joint",
                         "left_shoulder_yaw_joint",
                         "left_elbow_joint",
                         "left_wrist_roll_joint",
                         "left_wrist_pitch_joint",
                         "left_wrist_yaw_joint"],
                "right": ["right_shoulder_pitch_joint",
                          "right_shoulder_roll_joint",
                          "right_shoulder_yaw_joint",
                          "right_elbow_joint",
                          "right_wrist_roll_joint",
                          "right_wrist_pitch_joint",
                          "right_wrist_yaw_joint"],
                }

    # @cached_property
    # def leg_link_names(self):
    #     return {
    #             "left": ["left_hip_pitch_link",
    #                      "left_hip_roll_link",
    #                      "left_hip_yaw_link",
    #                      "left_knee_link",
    #                      "left_ankle_pitch_link",
    #                      "left_ankle_roll_link"],
    #             "right": ["right_hip_pitch_link",
    #                       "right_hip_roll_link",
    #                       "right_hip_yaw_link",
    #                       "right_knee_link",
    #                       "right_ankle_pitch_link",
    #                       "right_ankle_roll_link"]
    #             }


    # @cached_property
    # def leg_joint_names(self):
    #     return {
    #             "left": ["left_hip_pitch_joint", 
    #                      "left_hip_roll_joint", 
    #                      "left_hip_yaw_joint", 
    #                      "left_knee_joint", 
    #                      "left_ankle_pitch_joint", 
    #                      "left_ankle_roll_joint"],
    #             "right": ["right_hip_pitch_joint", 
    #                       "right_hip_roll_joint", 
    #                       "right_hip_yaw_joint", 
    #                       "right_knee_joint", 
    #                       "right_ankle_pitch_joint", 
    #                       "right_ankle_roll_joint"]
    #             }

    @cached_property
    def eef_link_names(self):
        return {"left": "left_hand_eef_link", 
                "right": "right_hand_eef_link"}

    @cached_property
    def finger_link_names(self):
        return {
                "left": [   "left_hand_palm_link", "left_hand_index_0_link", 
                            "left_hand_index_1_link", "left_hand_middle_0_link",
                            "left_hand_middle_1_link", "left_hand_thumb_0_link",
                            "left_hand_thumb_1_link", "left_hand_thumb_2_link"],
                "right": [  "right_hand_palm_link", "right_hand_index_0_link", 
                            "right_hand_index_1_link", "right_hand_middle_0_link",
                            "right_hand_middle_1_link", "right_hand_thumb_0_link",
                            "right_hand_thumb_1_link", "right_hand_thumb_2_link"],
                }

    @cached_property
    def finger_joint_names(self):
        return {
                "left": [   "left_hand_index_0_joint", "left_hand_index_1_joint",
                            "left_hand_middle_0_joint", "left_hand_middle_1_joint",
                            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint","left_hand_thumb_2_joint",
                        ],
                "right": [  
                            "right_hand_index_0_joint", "right_hand_index_1_joint",
                            "right_hand_middle_0_joint", "right_hand_middle_1_joint",
                            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint","right_hand_thumb_2_joint",
                        ],
                }

    @property
    def arm_workspace_range(self):
        return {"left": th.deg2rad(th.tensor([-45, 45], dtype=th.float32)), 
                "right": th.deg2rad(th.tensor([-45, 45], dtype=th.float32))}

    # @cached_property
    # def leg_control_idx(self):
    #     """
    #     Returns:
    #         dict: Dictionary mapping leg appendage name to indices in low-level control
    #             vector corresponding to leg joints.
    #     """
    #     return {
    #         leg: th.tensor([list(self.joints.keys()).index(name) for name in self.leg_joint_names[leg]])
    #         for leg in self.leg_names
    #     }

    @property
    def _default_controller_config(self):
        """
        Generate default controller configuration.
        Ensures that all components mentioned in _default_controllers have a configuration template here.
        """
        # Get default config from parent classes (ManipulationRobot etc.)
        # This automatically includes arm_left, arm_right, gripper_left, gripper_right based on arm_names
        cfg = super()._default_controller_config
        
        # Get control frequency
        freq = self._control_freq
        
        # # Add configuration for Trunk (if not added by parent)
        # if "trunk" not in cfg:
        #     cfg["trunk"] = {
        #         "JointController": {
        #             "name": "JointController",
        #             "control_freq": freq,
        #             "control_limits": self.control_limits,
        #             "dof_idx": self.trunk_control_idx,
        #             "command_output_limits": None,
        #             "motor_type": "position",
        #             "use_delta_commands": False,
        #         }
        #     }
            
        # # Add configuration for Legs
        # for leg_side in self.leg_names:
        #     leg_name = f"leg_{leg_side}"
        #     if leg_name not in cfg:
        #         cfg[leg_name] = {
        #             "JointController": {
        #                 "name": "JointController",
        #                 "control_freq": freq,
        #                 "control_limits": self.control_limits,
        #                 "dof_idx": self.leg_control_idx[leg_side],
        #                 "command_output_limits": None,
        #                 "motor_type": "position",
        #                 "use_delta_commands": False,
        #             }
        #         }
                
        return cfg

    @property
    def disabled_collision_pairs(self):
        import itertools

        all_links =[]
        
        # 1. 收集躯干部位
        all_links.extend(self.trunk_link_names)
        
        # 2. 收集左右手臂、末端执行器、手指
        for arm in self.arm_names:
            all_links.extend(self.arm_link_names[arm])
            # all_links.append(self.eef_link_names[arm])
            all_links.extend(self.finger_link_names[arm])
            
        # 3. 收集腿部（提前为你写好，这样你以后取消腿部注释时也能自动生效）
        if hasattr(self, 'leg_names') and hasattr(self, 'leg_link_names'):
            for leg in self.leg_names:
                all_links.extend(self.leg_link_names[leg])
                
        # 去重（防止有重复的部件名字）
        all_links = list(set(all_links))
        
        # 4. 使用 itertools 生成所有可能的一对一组合
        # 比如 [A, B, C] 会生成 (A,B), (A,C), (B,C)
        all_pairs = list(itertools.combinations(all_links, 2))

        # 遍历左右手
        for arm in self.arm_names:
            # 获取这只手所有的 link 名字
            for link_name in self.finger_link_names[arm]:
                # 安全检查：确保这个 link 确实在机器人的部件字典中
                if link_name in self.links:
                    # 强行设置质量为 0.02 (千克)
                    self.links[link_name].mass = 0.02
        
        # 转换为 OmniGibson 需要的 list of lists 格式
        return[list(pair) for pair in all_pairs]


