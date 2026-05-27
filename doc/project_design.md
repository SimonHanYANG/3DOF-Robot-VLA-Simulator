# UR5 取放仿真平台实施方案

## 1. 项目概述
本方案旨在构建一个**仿真—数据生成—模型推理—可视化**一体化平台，用于：
- 生成高质量取放轨迹数据，训练视觉语言动作（VLA）模型
- 对 OpenVLA、ACT、π0、π0.5 等主流 VLA 模型进行标准化测试与可视化

**机械臂**：UR5e 六自由度机械臂 + 简化平行二指夹爪（从 MuJoCo Menagerie 获取臂模型，自定义夹爪）。
**任务**：将桌面上的物体（立方体/球体/圆柱体，后续扩展杯/瓶/碗）从随机初始位置**抓取**到用户指定的目标位置放下，全程避开桌面边界。
**动作空间**：7D 笛卡尔增量 + 夹爪（dx, dy, dz, droll, dpitch, dyaw, gripper_open），通过逆运动学转换为关节目标。
**UI**：提供完整的 3D 交互界面，支持设置任务、加载模型并实时播放推理结果。

---

## 2. 系统总体架构
```
┌───────────────────────────────┐
│         前端 UI (Web)         │
│  React + Three.js / R3F       │
│  - 3D 场景渲染与交互          │
│  - 任务设置（物体/目标位置）   │
│  - 推理过程动画播放           │
└──────────┬────────────────────┘
           │ WebSocket (JSON)
┌──────────▼────────────────────┐
│       后端服务 (Python)        │
│  FastAPI + WebSocket           │
│  - 仿真引擎接口                │
│  - 模型推理管理                │
│  - 数据记录与任务调度          │
└──────────┬────────────────────┘
           │
┌──────────▼────────────────────┐
│      仿真核心 (MuJoCo)         │
│  - UR5e 6-DOF 机械臂           │
│  - 平行二指夹爪                │
│  - 物体 (立方体/球体/圆柱体)   │
│  - IK 求解器 (DLS Jacobian)   │
│  - 碰撞检测与物理推进          │
│  - 相机传感器 (RGB 渲染)       │
└───────────────────────────────┘
```

- **通信协议**：前端与后端通过 WebSocket 双向传输 JSON 消息（场景状态、动作指令、推理结果）。
- **仿真引擎**：MuJoCo（高性能、原生支持碰撞与驱动，Python 绑定完善）。
- **前端渲染**：Three.js + react-three-fiber，完全在浏览器中运行。
- **IK 求解**：基于 MuJoCo Jacobian 的阻尼最小二乘法（DLS）逆运动学。

---

## 3. 环境建模与物理仿真

### 3.1 UR5e 机械臂定义
- **模型来源**：MuJoCo Menagerie (`google-deepmind/mujoco_menagerie`) 中的 `universal_robots_ur5e/`
- **关节构型**：6 个旋转关节（shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3）
- **基座位置**：固定于世界坐标系，使机械臂工作空间覆盖桌面
- **驱动方式**：位置控制（position actuator），kp 增益参考 Menagerie 默认值

### 3.2 平行二指夹爪
- **结构**：安装于 UR5e 腕部法兰（wrist_3_link），两个对称滑动手指
- **关节**：2 个 slide 关节，通过 `mimic` 元素由单一执行器控制
- **行程**：0 ~ 0.04m（闭合~张开）
- **摩擦**：手指表面高摩擦（μ=1.0），保证稳定抓取

### 3.3 桌面与物体
- **桌面**：0.8m × 0.8m × 0.02m 矩形平面，z=0
- **初始物体**（几何体）：
  | 类型 | 几何参数 | 质量 | 颜色 |
  |------|----------|------|------|
  | 立方体 | 边长 0.04m | 0.1kg | 红 |
  | 球体 | 半径 0.025m | 0.08kg | 绿 |
  | 圆柱体 | 底半径 0.015m, 高 0.06m | 0.12kg | 蓝 |

- **扩展物体**（S18，mesh 模型）：杯、瓶、碗

### 3.4 相机传感器
| 名称 | 类型 | 位置 | 朝向 | 分辨率 |
|------|------|------|------|--------|
| `top_cam` | fixed | (0, 0, 0.8) | 垂直向下 (fovy=60°) | 224×224 |
| `wrist_cam` | fixed (parented to gripper) | 夹爪前方 | 沿夹爪接近方向 | 224×224 |

---

## 4. 逆运动学

### 4.1 DLS-Jacobian IK 求解器
- 使用 MuJoCo 的 `mj_jac` 计算 6×N Jacobian
- 阻尼最小二乘法：`dq = J^T (J J^T + λ²I)^{-1} e`
- 误差 `e` 为 6D 笛卡尔误差（3 位置 + 3 角度轴旋转）
- 迭代 20-50 次收敛，精度 < 1mm 位置 / < 1° 姿态

### 4.2 笛卡尔增量转换
- 输入：当前关节角 `(θ₁...θ₆)` + 增量 `(dx, dy, dz, droll, dpitch, dyaw)`
- 计算当前末端位姿 → 叠加增量 → IK 求解新关节角
- 用于将 VLA 模型的 7D 动作转换为关节目标

---

## 5. 任务定义与数据生成

### 5.1 任务描述模板
每一条任务数据包含：
- **指令文本**：`"pick the red cube and place it at (0.20, 0.15)"`
- **初始状态**：随机物体位姿（在桌面工作空间内，不重叠）
- **目标位置**：用户指定的桌面坐标 (x, y)
- **物体类型**：cube / sphere / cylinder（后续扩展 cup / bottle / bowl）

### 5.2 取放轨迹生成
采用**程序化规划**生成无碰撞取放路径：
1. **抓取规划**：根据物体类型计算顶部抓取位姿（pre_grasp → grasp）
2. **轨迹生成**：home → approach → descend → grasp → lift → transport → descend → release → home
3. 每个阶段通过笛卡尔空间线性插值 + IK 转换为关节轨迹
4. 在 MuJoCo 中重播验证，记录图像和状态

### 5.3 数据集结构（HDF5）
```
dataset.h5
├── metadata/
├── trajectory_000000/
│   ├── "instruction"          : str
│   ├── "object_type"          : str
│   ├── "object_init_pose"     : (7,) float32   # pos + quat
│   ├── "target_pose"          : (3,) float32
│   ├── "actions"              : (T, 7) float32  # 7D 动作
│   ├── "ee_poses"             : (T, 7) float32
│   ├── "joint_positions"      : (T, 6) float32
│   ├── "gripper_states"       : (T,) float32
│   ├── "top_images"           : (T, 224, 224, 3) uint8
│   └── "wrist_images"         : (T, 224, 224, 3) uint8
├── trajectory_000001/
├── ...
```

**数据集规模**：几何体 5k 条 + 日常物体 5k 条 = 10k+ 条轨迹。

---

## 6. VLA 模型集成与推理

### 6.1 统一动作接口
- **输入**：`image`（224×224×3 RGB，顶视图或双视图拼接）+ `instruction`（字符串）+ `state`（关节角/末端位姿，可选）
- **输出**：7D 动作 `(dx, dy, dz, droll, dpitch, dyaw, gripper_open)`

### 6.2 各模型适配
| 模型 | 输入 | 输出 | 适配要点 |
|------|------|------|----------|
| **OpenVLA** | 图像+文本 | 离散 action tokens | 6D 连续 bin + 1D 二值化 |
| **ACT** | 图像+关节状态 | 动作块 (chunk_len, 7) | CVAE 架构，chunk 预测 |
| **π0/π0.5** | 多模态 | 连续 7D 动作 | 扩散模型，双视图输入 |

### 6.3 推理闭环
```
重置环境 → 获取初始观测（图像+状态）
loop:
    观测 → 模型推理 → 7D 动作
    IK 转换 → 执行动作 → 仿真步进
    获取新观测
    检查终止条件（成功抓取放置 或 超时）
```

---

## 7. 数据流与消息协议

### 7.1 WebSocket 消息类型
| 消息名 | 方向 | 内容 |
|--------|------|------|
| `set_object` | 前端→后端 | `{object_type, position}` 设置物体 |
| `set_target` | 前端→后端 | `{x, y}` 设置目标位置 |
| `load_model` | 前端→后端 | `{model_name, weight_path}` |
| `start_inference` | 前端→后端 | `{}` |
| `stop_inference` | 前端→后端 | `{}` |
| `state_update` | 后端→前端 | `{ee_pose, joints, gripper, obj_poses, images, ...}` |
| `inference_action` | 后端→前端 | `{action_7d, model_name, timestamp}` |
| `grasp_result` | 后端→前端 | `{success, object_lifted}` |
| `done` | 后端→前端 | `{success, final_error}` |

---

## 8. 开发路线图

| 阶段 | 内容 | 产出物 / 里程碑 |
|------|------|-----------------|
| 1 | 仿真核心：UR5e+夹爪+场景+IK+控制器 | 可运行仿真，Python 控制末端移动 |
| 2 | 数据生成：抓取规划+取放轨迹+数据录制 | 生成 100 条验证轨迹 |
| 3 | 后端服务：FastAPI+WebSocket | 可通过 WS 控制环境 |
| 4 | 前端：Three.js UR5 渲染+交互面板 | 浏览器显示场景并控制 |
| 5 | 模型训练：ACT/OpenVLA/π0 训练管线 | 各模型可训练并产出 checkpoint |
| 6 | 推理闭环：模型适配器+推理引擎 | 完整推理流程 |
| 7 | 系统集成与扩展：日常物体+端到端测试 | 全功能平台 |

---

## 9. 技术栈与依赖

- **仿真物理**：MuJoCo ≥ 3.1
- **后端**：Python 3.10+, FastAPI, websockets, numpy, opencv-python, Pillow
- **IK 求解**：scipy（旋转矩阵/四元数转换）
- **模型推理**：PyTorch (OpenVLA/ACT), JAX (π0.5), HuggingFace transformers
- **前端**：Node.js, React, @react-three/fiber, @react-three/drei, three.js
- **通信**：WebSocket
- **数据存储**：HDF5 (h5py)

---

## 10. 关键注意事项
- **IK 收敛性**：目标不可达时需优雅降级，返回最近可行解
- **夹爪稳定性**：抓取后物体可能滑动，需足够摩擦力和适当闭合速度
- **碰撞过滤**：手臂自碰撞需禁用（contype=0），仅夹爪手指与物体碰撞
- **动作裁剪**：7D 动作执行前强制裁剪到安全范围
- **可扩展性**：架构支持后续添加更多物体类型和更复杂的任务
