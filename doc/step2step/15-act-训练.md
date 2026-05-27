# S15 — ACT 训练

**状态**：🔴 未开始
**阶段**：五、VLA 模型训练
**依赖**：S13（ACT 格式数据集就绪）+ S12（训练基础设施就绪）

## 目标
搭建 ACT（Action Chunking Transformer）模型，在自定义推动数据集上从头训练或基于预训练编码器微调，保存模型 checkpoint。

## 前置条件
- S13 完成：ACT 格式 HDF5 数据集已生成（images + joint_states + action_chunks）
- S12 完成：GPU 环境、PyTorch 已安装

## 实现细节

### 1. ACT 模型架构概述

ACT 的核心设计：
- **视觉编码器**：ResNet-18/50（或 ViT）将图像编码为特征向量
- **关节状态编码**：MLP 将当前末端位置编码为特征向量
- **Transformer 编码器**：融合图像特征和状态特征
- **Transformer 解码器**：自回归或一次性预测未来动作序列块（action chunk）
- **输出**：`(chunk_len, action_dim)` 的增量动作序列

架构变体：
- **ACT**（原始）：CNN 编码器 + Transformer encoder-decoder
- **ACT++**（改进）：更强的视觉编码器（如 DINOv2）+ 更好的位置编码

### 2. 模型实现

```python
# models/training/act_model.py
import torch
import torch.nn as nn

class ACTModel(nn.Module):
    """Action Chunking Transformer"""

    def __init__(self,
                 action_dim=3,          # (dx, dy, dz)
                 chunk_len=100,         # 预测 100 步动作
                 hidden_dim=256,
                 n_encoder_layers=6,
                 n_decoder_layers=6,
                 n_heads=8,
                 state_dim=3,           # 末端位置 (x, y, z)
                 ):
        super().__init__()
        # 视觉编码器
        self.vision_encoder = self._build_vision_encoder(hidden_dim)

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 位置编码（可学习的）
        self.pos_encoding = nn.Parameter(torch.randn(1, chunk_len, hidden_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers)

        # 输出头
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def _build_vision_encoder(self, hidden_dim):
        """使用 ResNet-18 作为视觉骨干"""
        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 替换第一层以适配 224×224 输入
        # 移除最后的 fc 层，用 1×1 卷积投影到 hidden_dim
        modules = list(resnet.children())[:-2]  # 去掉 avgpool 和 fc
        backbone = nn.Sequential(*modules)
        # 输出: (B, 512, 7, 7) → 投影到 hidden_dim
        self.vision_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        return backbone

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """图像编码：ResNet-18 特征提取 + 投影 + 全局平均池化"""
        feat = self.vision_encoder(image)              # (B, 512, 7, 7)
        feat = self.vision_proj(feat)                  # (B, hidden_dim, 7, 7)
        feat = feat.flatten(2).mean(dim=-1)            # (B, hidden_dim) 全局平均池化
        return feat

    def forward(self, images, joint_states):
        """
        images: (B, 3, 224, 224) 或 (B, 2, 3, 224, 224) 双视图
        joint_states: (B, 3)
        返回: (B, chunk_len, 3) 动作序列
        """
        B = images.shape[0]
        # 视觉编码
        if images.dim() == 5:  # 双视图
            img1, img2 = images[:, 0], images[:, 1]
            feat1 = self._encode_image(img1)
            feat2 = self._encode_image(img2)
            vis_feat = torch.cat([feat1, feat2], dim=-1)
        else:
            vis_feat = self._encode_image(images)

        # 状态编码
        state_feat = self.state_encoder(joint_states)

        # 融合特征作为 encoder 输入
        encoder_input = vis_feat.unsqueeze(1) + state_feat.unsqueeze(1)
        memory = self.encoder(encoder_input)

        # decoder 输入（可学习的 query + 位置编码）
        queries = self.pos_encoding.repeat(B, 1, 1)
        decoder_output = self.decoder(queries, memory)

        # 预测动作序列
        actions = self.action_head(decoder_output)  # (B, chunk_len, 3)
        return actions
```

### 3. 损失函数

```python
def act_loss(pred_actions, true_actions, mask=None):
    """
    pred_actions: (B, chunk_len, 3)
    true_actions: (B, chunk_len, 3) — 来自 S13 数据集
    mask: (B, chunk_len) — 标记填充的无效步（轨迹末尾）
    """
    # L1 损失 + L2 损失的组合
    l1_loss = F.l1_loss(pred_actions, true_actions, reduction='none')
    l2_loss = F.mse_loss(pred_actions, true_actions, reduction='none')

    # L1 为主，L2 为辅（L1 更鲁棒）
    total_loss = l1_loss + 0.1 * l2_loss

    if mask is not None:
        # 只计算有效步的损失
        mask = mask.unsqueeze(-1)  # (B, chunk_len, 1)
        total_loss = (total_loss * mask).sum() / mask.sum()
    else:
        total_loss = total_loss.mean()

    return total_loss
```

### 4. 训练循环

```python
def train_act(config, train_loader, val_loader):
    model = ACTModel(
        action_dim=3,
        chunk_len=config.chunk_len,
        hidden_dim=config.hidden_dim,
    ).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs * len(train_loader)
    )

    logger = ExperimentLogger(config, project="3dof-arm-vla-act")

    for epoch in range(config.num_epochs):
        model.train()
        for batch in train_loader:
            images = batch["images"].to(config.device)
            states = batch["joint_states"].to(config.device)
            actions = batch["action_chunks"].to(config.device)

            pred = model(images, states)
            loss = act_loss(pred, actions, mask=batch.get("mask"))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            logger.log({"train/loss": loss.item()})

        # 验证
        val_loss = evaluate_act(model, val_loader, config.device)
        logger.log({"val/loss": val_loss})

        # 保存 checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, ...)
```

### 5. 动作分块策略

训练时预测整个 chunk，推理时每步可以：
- **方案 A**：每步重新推理，取 chunk 的第一个动作执行
- **方案 B**：每隔 K 步推理一次，顺序执行 chunk 中的动作
- **方案 C**：取多个 chunk 的重叠部分做加权平均（temporal ensemble）

推荐初始使用方案 A（简单可靠），后续可尝试方案 C 提升平滑性。

### 6. 评估指标

```python
def evaluate_act(model, val_loader, device):
    """在验证集上计算平均预测误差"""
    model.eval()
    total_loss = 0
    total_steps = 0
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch["images"].to(device),
                        batch["joint_states"].to(device))
            loss = act_loss(pred, batch["action_chunks"].to(device))
            total_loss += loss.item()
            total_steps += 1
    return total_loss / total_steps
```

额外指标（在仿真闭环中评估，见 S18）：
- 成功率（物体到达目标）
- 平均终点误差
- 平均完成步数

## 关键文件
| 文件 | 说明 |
|------|------|
| `models/training/act_model.py` | ACT 模型定义 |
| `models/training/act_train.py` | ACT 训练脚本 |
| `models/training/act_dataloader.py` | ACT 数据加载器 |
| `models/training/configs/act_config.yaml` | Hydra 训练配置 |

## 验证标准
- [ ] ACT 模型在随机输入下可完成前向传播，输出 shape 正确：(B, chunk_len, 3)
- [ ] 训练 loss 在 50 epoch 内下降 ≥ 50%
- [ ] 验证 loss 在 10 epoch 后不再明显下降（趋于收敛）
- [ ] 单步推理耗时 < 20ms（满足 50Hz 闭环需求）
- [ ] checkpoint 可正常保存和加载并继续训练
- [ ] 在仿真中单步推理（S17 接口）输出的动作使物体向目标方向移动
- [ ] WandB 上可看到 loss 曲线、学习率调度

## 注意事项
- ACT 训练收敛通常需要较多数据（≥ 50k 步），10k 轨迹 × 45 步 ≈ 450k 步足够
- chunk_len 设太大（>200）会导致训练不稳定，设太小（<50）预测能力不足
- 如果视觉编码器用 ResNet-18，参数量约 11M（含 Transformer），单 GPU 训练可行
- 若双视图图像输入，可用两个独立视觉编码器，或拼接为 224×448 的宽图用一个编码器
- ACT 对学习率敏感——建议从 1e-4 开始，若不收敛可降至 5e-5
- 若物体类型多样（球/方块/锥体），可考虑在输入中加入物体类型嵌入（one-hot → MLP）
