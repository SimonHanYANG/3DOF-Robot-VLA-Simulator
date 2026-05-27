# S13 — ACT 训练（7D 动作空间）

**状态**：🔴 未开始
**阶段**：五、VLA 模型训练
**依赖**：S12（ACT 格式数据集就绪）

## 目标
实现 ACT（Action Chunking with Transformers）模型训练脚本，适配 7D 动作空间的取放任务。训练可运行，loss 下降。

## 前置条件
- S12 完成：ACT 格式数据集已转换
- GPU 环境可用

## 实现细节

### 1. ACT 模型架构

ACT 使用 CVAE（条件变分自编码器）+ Transformer 架构：
- **编码器**：图像（CNN）+ 状态（MLP）→ 编码向量
- **解码器**：Transformer decoder → 预测动作块 (chunk_len, 7)
- **VAE**：训练时编码真实动作块为隐变量，推理时采样

输入变化（相比旧 3DOF 版本）：
- 图像：224×224×3（不变）
- 状态：3D → 7D（末端位姿 pos+quat）或 7D（6 关节 + 1 夹爪）
- 输出：(chunk_len, 3) → (chunk_len, 7)

### 2. 模型定义

```python
class ACTModel(nn.Module):
    def __init__(self, action_dim=7, state_dim=7, chunk_len=50,
                 hidden_dim=256, n_heads=8, n_layers=4):
        super().__init__()
        self.action_dim = action_dim  # 7
        self.chunk_len = chunk_len    # 50

        # 图像编码器（ResNet18 backbone）
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, hidden_dim)

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # VAE 编码器（训练时使用）
        self.vae_encoder = nn.Sequential(
            nn.Linear(chunk_len * action_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),  # μ and log σ
        )

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 动作预测头
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, image, state, actions=None):
        # image: (B, 3, 224, 224)
        # state: (B, 7)
        # actions: (B, chunk_len, 7) - 训练时的 ground truth

        img_feat = self.backbone(image)           # (B, hidden_dim)
        state_feat = self.state_encoder(state)     # (B, hidden_dim)

        if self.training and actions is not None:
            # VAE 编码
            flat_actions = actions.reshape(B, -1)
            vae_out = self.vae_encoder(flat_actions)
            mu, logvar = vae_out.chunk(2, dim=-1)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = torch.randn(B, hidden_dim).to(image.device)

        # 条件向量
        cond = img_feat + state_feat + z  # (B, hidden_dim)

        # Transformer 解码
        queries = torch.zeros(B, self.chunk_len, hidden_dim).to(image.device)
        queries = queries + cond.unsqueeze(1)
        decoded = self.decoder(queries, cond.unsqueeze(0))

        # 预测动作
        pred_actions = self.action_head(decoded)  # (B, chunk_len, 7)
        return pred_actions
```

### 3. 训练循环

```python
def train(config):
    model = ACTModel(
        action_dim=7,
        state_dim=7,
        chunk_len=config.chunk_len,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    dataset = ACTDataset(config.data_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            image = batch["image"].to(config.device)
            state = batch["state"].to(config.device)
            actions = batch["actions"].to(config.device)

            pred = model(image, state, actions)
            loss = F.mse_loss(pred, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch {epoch}: loss={avg_loss:.6f}")
```

### 4. 训练配置

```yaml
# models/training/configs/act_config.yaml
model:
  action_dim: 7
  state_dim: 7
  chunk_len: 50
  hidden_dim: 256
  n_heads: 8
  n_layers: 4

training:
  epochs: 50
  batch_size: 32
  lr: 1.0e-4
  weight_decay: 1.0e-4
  lr_scheduler: cosine
  warmup_epochs: 5

data:
  train_path: "data/processed/act/train"
  val_path: "data/processed/act/val"
  num_workers: 4

logging:
  wandb_project: "ur5-act"
  save_every: 5
  eval_every: 1
```

## 关键文件
| 文件 | 说明 |
|------|------|
| `models/training/act_train.py` | ACT 训练脚本 |
| `models/training/act_model.py` | ACT 模型定义 |
| `models/training/configs/act_config.yaml` | 训练配置 |

## 验证标准
- [ ] 模型加载 7D 数据集，输入输出维度正确
- [ ] 训练 3 个 epoch，loss 从初始值下降
- [ ] 无 OOM 错误（batch_size=32 在 GPU 上可行）
- [ ] WandB 显示 loss 曲线
- [ ] checkpoint 可保存和重新加载
- [ ] 加载 checkpoint 后单步推理输出 shape 为 (chunk_len, 7)

## 注意事项
- ResNet18 backbone 使用 ImageNet 预训练权重，视觉编码器初始冻结 5 epoch
- chunk_len=50 对取放任务足够（典型轨迹 50-100 步）
- VAE 的 KL 散度损失权重需调优（初始 0.001-0.01）
- 状态维度 7 包含 (x, y, z, qw, qx, qy, qz)，需归一化
- 动作维度 7 各维度量级不同（位置 cm 级 vs 旋转 rad 级 vs 夹爪 0-1），考虑加权损失
