# S14 — OpenVLA 微调训练

**状态**：🔴 未开始
**阶段**：五、VLA 模型训练
**依赖**：S13（OpenVLA 格式数据集就绪）+ S12（训练基础设施就绪）

## 目标
加载 OpenVLA 预训练权重，在自定义推动数据集上进行微调（LoRA 或全参微调），保存可用的模型 checkpoint。

## 前置条件
- S13 完成：OpenVLA 格式数据集（HuggingFace datasets 或 RLDS）已生成
- S12 完成：GPU 环境、transformers/peft/accelerate 已安装

## 实现细节

### 1. OpenVLA 模型概述

OpenVLA 是 Stanford 等机构提出的开源 VLA 模型，架构为：
- **视觉编码器**：SigLIP（或 DINOv2）将图像编码为视觉 token
- **语言模型**：Llama-2（或类似）处理指令文本 + 视觉 token
- **动作解码**：输出离散动作 token，解码为末端增量位移

本步骤基于 OpenVLA 官方仓库的预训练权重做领域微调。

### 2. 加载预训练模型

```python
# models/training/openvla_train.py
from transformers import AutoModel, AutoProcessor
# OpenVLA 通常通过 HuggingFace Hub 加载
# 具体加载方式取决于 OpenVLA 官方发布的接口

# 伪代码（实际取决于 OpenVLA 版本）
from openvla import OpenVLAModel, OpenVLAProcessor

model = OpenVLAModel.from_pretrained("openvla/openvla-7b")  # 或本地路径
processor = OpenVLAProcessor.from_pretrained("openvla/openvla-7b")
```

### 3. 自定义数据集注入

```python
class PushDataset(torch.utils.data.Dataset):
    """为 OpenVLA 微调准备的自定义数据集"""

    def __init__(self, data_path: str, processor, split: str = "train"):
        self.dataset = datasets.load_from_disk(data_path)[split]
        self.processor = processor

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # 处理图像
        image = sample["image"]  # PIL Image
        # 处理指令
        instruction = sample["instruction"]
        # 处理动作（离散 bin IDs）
        action_bins = sample["action_bins"]  # 或连续值需在线离散化
        # 拼接为模型输入格式
        inputs = self.processor(
            images=image,
            text=instruction,
            actions=action_bins,  # 训练时的标签
            return_tensors="pt",
        )
        return inputs

    def __len__(self):
        return len(self.dataset)
```

### 4. LoRA 微调配置

使用 HuggingFace `peft` 库对 LLM 部分做 LoRA 微调（视觉编码器通常冻结）：

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA alpha 缩放
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Llama 的注意力层
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 检查可训练参数量（通常 < 1% 总参数）
```

### 5. 训练循环

```python
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator

training_args = TrainingArguments(
    output_dir="models/checkpoints/openvla/run_xxx",
    num_train_epochs=10,               # 微调通常不需要太多 epoch
    per_device_train_batch_size=4,      # 根据 GPU 显存调整
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,     # 等效 batch_size=16
    learning_rate=2e-5,                # LoRA 微调用较小学习率
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=3,                # 最多保留 3 个 checkpoint
    bf16=True,                         # 或 fp16
    dataloader_num_workers=4,
    report_to="wandb",
    run_name="openvla-push-lora",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)
trainer.train()
```

### 6. 损失函数

OpenVLA 使用交叉熵损失（动作 token 分类）：
- 模型输出动作 token 的 logits
- 与真实 bin ID 计算 cross-entropy
- 监督信号仅作用于动作 token 位置

若需要更精确的动作预测，可在离散 token 之上添加回归头直接预测连续值（L2 loss），但会增加实现复杂度。

### 7. 评估指标

```python
# utils/metrics.py
def evaluate_action_accuracy(pred_bins, true_bins, num_bins=256):
    """动作 token 准确率"""
    return (pred_bins == true_bins).mean()

def evaluate_position_error(pred_actions, true_actions):
    """将预测的动作 bin 解码后计算每轴 MAE"""
    # 将 bin 还原为连续值
    pred_cont = bins_to_continuous(pred_actions)
    true_cont = bins_to_continuous(true_actions)
    return np.abs(pred_cont - true_cont).mean(axis=0)  # (dx_mae, dy_mae, dz_mae)
```

### 8. 动作解码

训练后推理时，需要将模型输出的离散 token 解码为连续动作：

```python
def decode_action_tokens(token_ids: list[int], action_dim=3, num_bins=256):
    """将离散 token ID 序列解码为 (dx, dy, dz)"""
    bins_per_dim = len(token_ids) // action_dim
    actions = []
    for i in range(action_dim):
        dim_tokens = token_ids[i * bins_per_dim : (i+1) * bins_per_dim]
        # 多数投票或加权平均
        bin_id = max(set(dim_tokens), key=dim_tokens.count)
        value = (bin_id / num_bins) * 0.04 - 0.02  # [-0.02, 0.02]
        actions.append(value)
    return np.array(actions)
```

## 关键文件
| 文件 | 说明 |
|------|------|
| `models/training/openvla_train.py` | OpenVLA 微调训练脚本 |
| `models/training/configs/openvla_config.yaml` | Hydra 训练配置 |
| `models/training/utils/openvla_utils.py` | OpenVLA 专用工具（动作编解码等） |

## 验证标准
- [ ] OpenVLA 预训练模型成功加载到 GPU
- [ ] 自定义数据集被正确加载和预处理
- [ ] LoRA 微调后训练 loss 稳定下降
- [ ] 验证集动作准确率 > 50%（在离散化的 256 bins 中随机猜测 ≈ 0.4%）
- [ ] 训练过程中无 OOM（显存溢出）
- [ ] checkpoint 可正常保存和重新加载
- [ ] 加载 checkpoint 后可完成单步推理（S17 的推理接口验证）
- [ ] WandB 上可看到完整的 loss 曲线和评估指标

## 注意事项
- OpenVLA 7B 模型显存需求：约 14GB（bf16），加上优化器状态和中间激活约需 24GB。若显存不足，减小 batch_size 并增大 gradient_accumulation_steps
- LoRA rank 的选择：r=8（更轻量）到 r=64（更强表达能力），r=16 是常用折中
- 训练数据不够多时（< 5k 轨迹），建议增大 LoRA dropout 或减少训练 epoch 防止过拟合
- 确保视觉编码器的预处理（归一化、尺寸等）与 OpenVLA 官方预训练时的设置完全一致
- 首次训练建议在 100-500 步后观察 loss 是否下降，若不下降则检查学习率和数据格式
