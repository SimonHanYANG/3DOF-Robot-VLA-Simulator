# S14 — OpenVLA 微调训练

**状态**：🔴 未开始
**阶段**：五、VLA 模型训练
**依赖**：S12（OpenVLA 格式数据集就绪）

## 目标
加载 OpenVLA 预训练权重，在自采集取放数据集上进行 LoRA 微调，适配 7D 动作空间。保存可用的模型 checkpoint。

## 前置条件
- S12 完成：OpenVLA 格式数据集已转换
- GPU 环境、transformers/peft/accelerate 已安装

## 实现细节

### 1. 7D 动作离散化

OpenVLA 使用离散 action token。7D 动作的编码策略：

| 维度 | 含义 | 范围 | 编码方式 |
|------|------|------|----------|
| dx | 位置 X | ±0.02m | 256 bins |
| dy | 位置 Y | ±0.02m | 256 bins |
| dz | 位置 Z | ±0.02m | 256 bins |
| droll | 旋转 Roll | ±0.1rad | 256 bins |
| dpitch | 旋转 Pitch | ±0.1rad | 256 bins |
| dyaw | 旋转 Yaw | ±0.1rad | 256 bins |
| gripper | 夹爪 | 0/1 | 2 bins（二值） |

总计：6 × 256 + 2 = 1538 个 action tokens

### 2. 动作编解码

```python
def encode_action_7d(action: np.ndarray, num_bins=256) -> np.ndarray:
    """7D 连续动作 → 离散 bin IDs"""
    bins = np.zeros(7, dtype=np.int64)
    # 前 6 维：连续值 → bin
    ranges = [(-0.02, 0.02)] * 3 + [(-0.1, 0.1)] * 3
    for i in range(6):
        lo, hi = ranges[i]
        normalized = (action[i] - lo) / (hi - lo)
        bins[i] = np.clip(int(normalized * num_bins), 0, num_bins - 1)
    # 第 7 维：gripper 二值化
    bins[6] = 1 if action[6] > 0.5 else 0
    return bins

def decode_action_7d(bins: np.ndarray, num_bins=256) -> np.ndarray:
    """离散 bin IDs → 7D 连续动作"""
    action = np.zeros(7)
    ranges = [(-0.02, 0.02)] * 3 + [(-0.1, 0.1)] * 3
    for i in range(6):
        lo, hi = ranges[i]
        action[i] = lo + (bins[i] / num_bins) * (hi - lo)
    action[6] = 1.0 if bins[6] > 0.5 else 0.0
    return action
```

### 3. LoRA 微调配置

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)
```

### 4. 数据集类

```python
class OpenVLAPickPlaceDataset(torch.utils.data.Dataset):
    """OpenVLA 微调用取放数据集"""

    def __init__(self, data_path, processor, num_bins=256):
        self.dataset = datasets.load_from_disk(data_path)
        self.processor = processor
        self.num_bins = num_bins

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]           # PIL Image
        instruction = sample["instruction"]
        action_bins = sample["action_bins"]  # (7,) 离散 bin IDs

        inputs = self.processor(
            images=image,
            text=instruction,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "labels": torch.tensor(action_bins),
        }
```

### 5. 训练配置

```python
training_args = TrainingArguments(
    output_dir="models/checkpoints/openvla-pickplace",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 等效 batch_size=16
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    bf16=True,
    report_to="wandb",
    run_name="openvla-pickplace-lora",
)
```

### 6. 评估指标

```python
def evaluate_action_accuracy(pred_bins, true_bins):
    """各维度 token 准确率"""
    return (pred_bins == true_bins).float().mean(dim=0)  # (7,) 每维度准确率

def evaluate_continuous_error(pred_actions, true_actions):
    """连续动作 MAE"""
    return torch.abs(pred_actions - true_actions).mean(dim=0)  # (7,) 每维度 MAE
```

## 关键文件
| 文件 | 说明 |
|------|------|
| `models/training/openvla_train.py` | OpenVLA 微调训练脚本 |
| `models/training/openvla_model.py` | 模型加载和适配 |
| `models/training/configs/openvla_config.yaml` | 训练配置 |
| `models/training/utils/openvla_utils.py` | 动作编解码工具 |

## 验证标准
- [ ] OpenVLA 预训练模型成功加载到 GPU
- [ ] 自定义数据集正确加载，7D action_bins 格式正确
- [ ] LoRA 微调后训练 loss 下降
- [ ] 验证集动作 token 准确率 > 30%（6D 连续 + 1D 二值）
- [ ] 无 OOM
- [ ] checkpoint 可保存和重新加载
- [ ] WandB 显示 loss 曲线

## 注意事项
- OpenVLA 7B 模型 bf16 约需 14GB 显存，加上优化器约需 24GB
- gripper 维度只有 2 个 bin，训练初期准确率可能很高（多数样本 gripper=开）
- 视觉编码器的预处理需与 OpenVLA 官方完全一致
- 首次训练建议跑 100 步观察 loss 是否下降
