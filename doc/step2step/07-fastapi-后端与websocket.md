# S07 — FastAPI 后端与 WebSocket 通信

**状态**：🔴 未开始
**阶段**：三、后端服务
**依赖**：S03（仿真控制器可用，可先不依赖 S04-S06 的数据集）

## 目标
搭建 FastAPI 后端框架，实现 WebSocket 通信端点，定义并实现前后端 JSON 消息协议。

## 前置条件
- S03 完成：`SimController` 可独立运行（本步骤可与 S04-S06 并行开发）

## 实现细节

### 1. FastAPI 项目结构

```
backend/
├── __init__.py
├── main.py                   # FastAPI 应用入口
├── config.py                 # 配置管理（端口、CORS 等）
├── ws_handler.py             # WebSocket 连接管理与消息路由
├── schemas/                  # 消息模型定义
│   ├── __init__.py
│   ├── messages.py           # 所有 WS 消息的 Pydantic 模型
│   └── enums.py              # 枚举类型（物体类型、模型名称等）
└── services/                 # 业务逻辑层（S08 实现）
    ├── __init__.py
    └── sim_service.py        # 仿真服务封装（S08）
```

### 2. WebSocket 消息协议（8 种消息）

使用 JSON 文本帧，每条消息包含 `type` 字段标识消息类型：

#### 2.1 前端 → 后端（控制消息）

| type | 字段 | 说明 |
|------|------|------|
| `set_ee_pose` | `{x, y, z}` | 拖拽末端目标位姿 |
| `set_task` | `{object_type, init_pose: {x,y}, target_pose: {x,y}}` | 设置任务参数 |
| `set_object` | `{object_type, color, pose: {x,y}}` | 放置物体 |
| `set_target` | `{pose: {x,y}}` | 设置目标位置标记 |
| `load_model` | `{model_name, checkpoint_path}` | 加载模型权重 |
| `start_inference` | `{}` | 开始闭环推理 |
| `stop_inference` | `{}` | 停止推理 |
| `set_speed` | `{speed: 0.5/1.0/2.0}` | 设置播放速度 |

#### 2.2 后端 → 前端（状态消息）

| type | 字段 | 说明 |
|------|------|------|
| `state_update` | `{ee_pose, obj_pose, joints, images, ...}` | 周期性状态更新 |
| `inference_action` | `{action: {dx,dy,dz}, model_name, timestamp}` | 模型输出动作 |
| `collision_event` | `{body_A, body_B, position}` | 碰撞事件 |
| `done` | `{success: bool, final_error: float}` | 推理结束 |
| `error` | `{code, message}` | 错误通知 |

### 3. Pydantic 消息模型

```python
# schemas/messages.py
from pydantic import BaseModel
from typing import Optional, Literal

class SetEEPoseMsg(BaseModel):
    type: Literal["set_ee_pose"] = "set_ee_pose"
    x: float
    y: float
    z: float

class StateUpdateMsg(BaseModel):
    type: Literal["state_update"] = "state_update"
    ee_pose: tuple[float, float, float]
    obj_pose: tuple[float, float, float]
    target_pose: tuple[float, float]
    top_image: str          # base64 编码的 JPEG
    angle_image: str        # base64 编码的 JPEG
    simulation_time: float
    collision: bool
# ... 其他消息类型
```

### 4. WebSocket 连接管理

```python
# ws_handler.py
class ConnectionManager:
    """管理所有活跃的 WebSocket 连接"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """向所有连接的客户端广播消息"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass  # 断开的连接在 disconnect 中处理

    async def send_personal(self, message: dict, websocket: WebSocket):
        """向特定客户端发送消息"""
        await websocket.send_json(message)
```

### 5. WebSocket 端点

```python
# main.py
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # 根据 type 字段路由到对应处理函数
            await route_message(data, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### 6. CORS 配置
- 允许前端开发服务器（`http://localhost:5173`）跨域访问
- 生产环境改为具体域名

## 关键文件
| 文件 | 说明 |
|------|------|
| `backend/main.py` | FastAPI 应用 + WebSocket 端点 |
| `backend/ws_handler.py` | 连接管理器 + 消息路由 |
| `backend/schemas/messages.py` | 消息 Pydantic 模型 |
| `backend/config.py` | 配置（端口、CORS） |

## 验证标准
- [ ] `uvicorn backend.main:app --reload` 正常启动
- [ ] 使用 `wscat -c ws://localhost:8000/ws` 可建立 WebSocket 连接
- [ ] 发送 `{"type": "set_ee_pose", "x": 0.1, "y": 0, "z": 0.05}` 收到响应（即使 S08 未完成，至少消息路由不报错）
- [ ] 多个客户端同时连接时各自独立收发消息
- [ ] 客户端断开后连接管理器正确清理
- [ ] 发送非法 JSON 或未知消息类型时返回 error 消息而非崩溃
- [ ] FastAPI 自动生成的 `/docs` Swagger 页面可访问

## 注意事项
- WebSocket 的 `receive_json()` 默认无超时，需考虑长时间无消息时的处理（心跳机制可后续添加）
- 图像数据以 base64 编码传输，对于 224×224 图像约 50KB/帧，25Hz 下带宽约 1.25MB/s，在局域网内可接受
- 生产环境建议在 FastAPI 前加 nginx 做反向代理和 WS 升级
- WebSocket 消息处理需异步，避免阻塞事件循环——仿真步进和图像渲染用 `run_in_executor` 放到线程池执行
