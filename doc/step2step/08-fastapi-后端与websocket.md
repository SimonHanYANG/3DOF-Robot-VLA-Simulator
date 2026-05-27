# S08 — FastAPI 后端与 WebSocket

**状态**：🔴 未开始
**阶段**：三、后端服务搭建
**依赖**：S04-S06（仿真控制接口、轨迹生成可用）

## 目标
搭建 FastAPI + WebSocket 后端服务，定义 7D 动作空间的消息协议，实现前端与仿真环境的实时双向通信。

## 前置条件
- S04 完成：SimController 可用
- S06 完成：轨迹生成可用（用于测试）

## 实现细节

### 1. 项目结构

```
backend/
├── __init__.py
├── main.py                  # FastAPI 入口
├── config.py                # 配置（场景路径、端口等）
├── ws_handler.py            # WebSocket 连接管理与消息路由
├── schemas/
│   ├── __init__.py
│   ├── messages.py          # Pydantic 消息模型
│   └── enums.py             # 枚举定义
└── services/
    ├── __init__.py
    ├── sim_service.py       # 仿真服务（S09 实现）
    └── inference_engine.py  # 推理引擎（S09 实现）
```

### 2. WebSocket 消息协议

#### 前端 → 后端
| 消息类型 | 字段 | 说明 |
|----------|------|------|
| `set_object` | `{object_type, position: [x,y,z]}` | 设置物体类型和位置 |
| `set_target` | `{position: [x,y]}` | 设置目标放置位置 |
| `set_ee_pose` | `{pos: [x,y,z], quat: [w,x,y,z]}` | 直接设置末端位姿（拖拽用） |
| `load_model` | `{model_name, checkpoint_path}` | 加载推理模型 |
| `start_inference` | `{speed: float}` | 开始推理 |
| `stop_inference` | `{}` | 停止推理 |

#### 后端 → 前端
| 消息类型 | 字段 | 说明 |
|----------|------|------|
| `state_update` | `{ee_pose, joints, gripper, obj_poses, target_pose, top_image, wrist_image}` | 场景状态（25Hz） |
| `inference_action` | `{action: [7], model_name, step}` | 推理动作 |
| `grasp_result` | `{success, object_lifted}` | 抓取结果 |
| `done` | `{success, final_error}` | 任务完成 |

### 3. Pydantic 消息模型

```python
# schemas/messages.py
from pydantic import BaseModel

class SetObjectMsg(BaseModel):
    type: str = "set_object"
    object_type: str              # "cube" / "sphere" / "cylinder"
    position: list[float]         # [x, y, z]

class SetTargetMsg(BaseModel):
    type: str = "set_target"
    position: list[float]         # [x, y]

class StateUpdateMsg(BaseModel):
    type: str = "state_update"
    ee_pose: list[float]          # [x, y, z, qw, qx, qy, qz] (7D)
    joints: list[float]           # [j1, j2, j3, j4, j5, j6] (6D)
    gripper: float                # 0-1
    obj_poses: dict               # {name: {pos: [...], quat: [...]}}
    target_pose: list[float]      # [x, y, z]
    top_image: str                # base64 编码
    wrist_image: str              # base64 编码

class InferenceActionMsg(BaseModel):
    type: str = "inference_action"
    action: list[float]           # [dx,dy,dz,droll,dpitch,dyaw,gripper] (7D)
    model_name: str
    step: int
```

### 4. FastAPI 主入口

```python
# main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="UR5 VLA Simulator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # 消息路由 → ws_handler
```

### 5. WebSocket 连接管理

```python
# ws_handler.py
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    async def broadcast(self, message: dict):
        for conn in self.active_connections:
            await conn.send_json(message)

async def handle_message(msg: dict, sim_service, inference_engine):
    """消息路由"""
    msg_type = msg.get("type")
    if msg_type == "set_object":
        await sim_service.set_object(msg)
    elif msg_type == "start_inference":
        await inference_engine.start(msg)
    # ...
```

### 6. 配置

```python
# config.py
SCENE_PATH = "sim/assets/scene.xml"
WS_PORT = 8765
STATE_BROADCAST_HZ = 25
RENDER_SIZE = (224, 224)
```

## 关键文件
| 文件 | 说明 |
|------|------|
| `backend/main.py` | FastAPI 入口 |
| `backend/config.py` | 配置 |
| `backend/ws_handler.py` | WebSocket 消息路由 |
| `backend/schemas/messages.py` | Pydantic 消息模型 |
| `backend/schemas/enums.py` | 枚举定义 |

## 验证标准
- [ ] `uvicorn backend.main:app` 成功启动
- [ ] WebSocket 客户端可连接到 `ws://localhost:8765/ws`
- [ ] 发送 `set_object` 消息后仿真环境正确重置
- [ ] 后端以 25Hz 广播 `state_update` 消息
- [ ] 图像 base64 编码/解码正确，尺寸 224×224
- [ ] 多客户端连接时消息正确广播

## 注意事项
- base64 编码的 224×224 RGB 图像约 95KB/帧，25Hz ≈ 2.4MB/s，需考虑带宽
- WebSocket 连接断开时需清理资源（停止推理循环、释放 GPU 显存）
- 使用 FastAPI 的 lifespan 事件管理 SimController 生命周期
- CORS 配置在开发环境允许所有来源，生产环境需限制
