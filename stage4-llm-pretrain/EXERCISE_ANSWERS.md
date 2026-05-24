# Stage 4 · 练习题答案

> LLaMA 架构改进 + 预训练实践

---

## 练习 1：RMSNorm vs LayerNorm 训练对比

**难度**: ⭐ | **目标**: 对比两种归一化的训练速度和收敛

### 答案

修改 `llama_block.py`，在 LLaMABlock 中增加一个 `--norm_type` 参数：
- `"rmsnorm"`: 使用 `RMSNorm`（默认）
- `"layernorm"`: 替换为 `nn.LayerNorm`

训练 tiny 模型 1 epoch，对比：

| 归一化 | Time/epoch (CPU) | Train Loss | 内存 |
|--------|------------------|------------|------|
| RMSNorm | ~420s | ~3.80 | 低 |
| LayerNorm | ~435s | ~3.78 | 略高 |

**结论**：RMSNorm 比 LayerNorm 快约 3-5%，收敛效果几乎一致。

**代码关键修改**：
```python
# 在 LLaMABlock.__init__ 中
if norm_type == "rmsnorm":
    self.attention_norm = RMSNorm(dim)
    self.ffn_norm = RMSNorm(dim)
else:
    self.attention_norm = nn.LayerNorm(dim)
    self.ffn_norm = nn.LayerNorm(dim)
```

---

## 练习 2：纯 NumPy / 纯 PyTorch Tensor 手写 SwiGLU 前向传播

**难度**: ⭐⭐ | **目标**: 深入理解 SwiGLU 的计算过程

### 答案

```python
import torch
import torch.nn.functional as F

def swiglu_forward(x, gate_weight, up_weight, down_weight):
    """
    SwiGLU 前向传播，逐步计算。

    Args:
        x: (B, T, d) 输入
        gate_weight: (d, hidden) 门控投影权重
        up_weight: (d, hidden) 上投投影权重
        down_weight: (hidden, d) 下投投影权重

    Returns:
        (B, T, d) 输出
    """
    B, T, d = x.shape
    hidden = gate_weight.shape[1]  # 8d/3

    # Step 1: gate = SiLU(x @ W_gate)
    #           = (x @ W_gate) * sigmoid(x @ W_gate)
    gate_logits = x @ gate_weight          # (B, T, hidden)
    gate = gate_logits * torch.sigmoid(gate_logits)  # (B, T, hidden)

    # Step 2: up = x @ W_up
    up = x @ up_weight                     # (B, T, hidden)

    # Step 3: hidden = gate ⊙ up  (逐元素乘)
    hidden_state = gate * up               # (B, T, hidden)

    # Step 4: output = hidden @ W_down
    output = hidden_state @ down_weight    # (B, T, d)

    return output

# ── 验证 ──
def test_swiglu_manual():
    d, hidden = 16, 40  # 40 ≈ 8*16/3 向上取整
    B, T = 2, 4

    # 随机权重
    gate_w = torch.randn(d, hidden)
    up_w = torch.randn(d, hidden)
    down_w = torch.randn(hidden, d)

    x = torch.randn(B, T, d)

    # 手动实现
    y_manual = swiglu_forward(x, gate_w, up_w, down_w)

    # 参考实现 (nn.Linear)
    from swiglu import SwiGLU
    swiglu = SwiGLU(d, hidden_dim=hidden)
    swiglu.gate_proj.weight.data = gate_w.T
    swiglu.up_proj.weight.data = up_w.T
    swiglu.down_proj.weight.data = down_w.T

    y_ref = swiglu(x)

    print(f"手动输出: {y_manual[0, 0, :4]}")
    print(f"参考输出: {y_ref[0, 0, :4]}")
    print(f"最大差值: {(y_manual - y_ref).abs().max():.10f}")
    # 预期: 最大差值 < 1e-6

test_swiglu_manual()
```

**关键理解**：
- gate 和 up 输入相同（都是 x），但经过不同的线性变换
- gate 经过 SiLU → 输出范围 (-0.278, +∞)，负值会被部分抑制
- up 不经过任何激活 → 直接参与门控乘法
- gate ⊙ up 的语义：gate 控制 up 的哪个部分通过

---

## 练习 3：RoPE vs NoPE vs Learnable PE 长度外推对比

**难度**: ⭐⭐ | **目标**: 验证 RoPE 的长度外推能力

### 答案

训练阶段：使用 seq_len=256，三种位置编码方式各训练一个 tiny 模型

测试阶段：使用 seq_len=512 的序列做评估，观察 PPL 变化

**实验设计**：

```python
import torch
from model import LLaMA

def compare_extrapolation():
    """对比三种 PE 方式的长度外推能力"""
    results = {}

    # 三种配置
    # 1. RoPE: 默认（LLaMA）
    # 2. NoPE: 完全不用位置编码
    # 3. Learnable PE: 添加 nn.Embedding(max_seq_len, dim)

    for pe_type in ["rope", "nope", "learnable"]:
        # 训练: seq_len=256
        # ... 训练 1 epoch ...

        # 测试: 用 seq_len=512 评估
        val_loss_256 = evaluate(model, val_loader_256)  # 训练长度
        val_loss_512 = evaluate(model, val_loader_512)  # 外推长度

        ppl_256 = math.exp(val_loss_256)
        ppl_512 = math.exp(val_loss_512)
        degradation = ppl_512 - ppl_256  # PPL 增加量

        results[pe_type] = {
            "ppl@256": ppl_256,
            "ppl@512": ppl_512,
            "degradation": degradation
        }
    return results
```

**预期结果**：

| PE 类型 | PPL@256 | PPL@512 | PPL 增量 | 外推能力 |
|---------|---------|---------|----------|----------|
| RoPE | 25.0 | 27.5 | +2.5 | **优** |
| Learnable PE | 25.0 | 50+ | +25+ | 差（仅限训练长度） |
| NoPE | 28.0 | 28.5 | +0.5 | 良（无位置信息） |

**原因分析**：
- **RoPE**：只依赖相对位置，qₙ·kₘ 由 (n-m) 决定。256 步内见过的相对距离，在 512 步中依然有效 → 温和劣化
- **Learnable PE**：位置 m 的 embedding 是独立参数。m>255 的 embedding 从未被训练过 → 随机初始化 → 严重劣化
- **NoPE**：完全不使用位置信息，所以长度变化不影响 → 但模型本身质量差（缺少位置信息做注意力）

**可视化**：

```
PPL 随序列长度的变化:

PPU
 ^
 |                                    Learnable PE 断裂
 |                                    /
 |                                  /
 |  50+ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    /
 |                            /
 |  30 ─ ─ ─ ─ ─ ─ ─ ─ ─ RoPE (温和上升)
 |  28 ─ NoPE (几乎不变)
 |
 +───┼───────┼───────┼──────────> Seq Len
    128     256     384     512
```

---

## 练习 4：GQA 分组比例与 KV Cache 计算

**难度**: ⭐ | **目标**: 量化 GQA 对推理内存的节省

### 答案

**公式**：

每层 KV Cache (per token) = 2 × n_kv_head × d_k × sizeof(dtype)

**计算器**：

```python
def kv_cache_size(n_layer, n_kv_head, d_k, seq_len, dtype_bytes=2):
    """计算 KV Cache 总大小。

    Args:
        n_layer: 层数
        n_kv_head: KV 头数
        d_k: 每头维度
        seq_len: 序列长度
        dtype_bytes: 2 for FP16/BF16, 4 for FP32

    Returns:
        (per_token_bytes, total_bytes_for_seq_len)
    """
    # 每层、每 token: 2 (K+V) × n_kv_head × d_k
    per_layer_per_token = 2 * n_kv_head * d_k * dtype_bytes

    # 全部层、全部序列
    total = n_layer * per_layer_per_token * seq_len
    return per_layer_per_token, total

# ── LLaMA 3 8B 为例 ──
n_layer = 32
d_model = 4096
d_k = 128   # d_model / n_head = 4096 / 32

for n_kv_head, name in [(32, "MHA"), (8, "GQA (LLaMA3)"), (1, "MQA")]:
    per_token, total = kv_cache_size(n_layer, n_kv_head, d_k, seq_len=8192)
    print(f"{name}: n_kv_head={n_kv_head:2d}, "
          f"per_token={per_token}B, total={total/1024**3:.2f} GB")
```

**输出**：

```
MHA:            n_kv_head=32, per_token=16384B, total=4.00 GB
GQA (LLaMA3):   n_kv_head= 8, per_token= 4096B, total=1.00 GB  ← 节省 75%
MQA:            n_kv_head= 1, per_token= 512B,  total=0.12 GB
```

**结论**：
- LLaMA 3 8B 用 GQA (n_kv_head=8) 把 KV Cache 从 4.0GB 降到 1.0GB
- 如果降到 MQA (n_kv_head=1)，再降到 0.12GB，但质量问题更大
- GQA 是工程上最优的折中：4x 内存节省 + 几乎没有质量损失

---

## 练习 5：梯度累积 + AMP 正确性验证

**难度**: ⭐⭐ | **目标**: 验证梯度累积 + AMP 的数值正确性

### 答案

核心问题：梯度累积把 loss 除以 grad_accum_steps，AMP 的 GradScaler 也会缩放 loss。两者的交互需要验证。

```python
import torch

def verify_grad_accum_amp():
    """
    验证逻辑:
      正常训练（无 grad accum）→ 保存梯度 G_ref
      训练（grad_accum=4）    → 保存梯度 G_accum
      → 两个梯度应几乎相等
    """
    from model import LLaMA, create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    # 两个相同的模型
    model_ref = create_model("tiny", vocab_size=1000, max_seq_len=256).to(device)
    model_acc = create_model("tiny", vocab_size=1000, max_seq_len=256).to(device)
    model_acc.load_state_dict(model_ref.state_dict())  # 精确复制

    # 数据
    B = 4  # small batch
    T = 256
    x = torch.randint(0, 999, (B, T), device=device)
    y = torch.randint(0, 999, (B, T), device=device)

    # ── 参考：正常训练 ──
    logits_ref, loss_ref = model_ref(x, targets=y)
    loss_ref.backward()
    ref_grads = {n: p.grad.clone() for n, p in model_ref.named_parameters() if p.grad is not None}

    # ── 实验：梯度累积 (模拟 grad_accum=4，即只有 1/4 的 batch) ──
    model_acc.zero_grad()
    grad_accum = 4
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    # 每个 micro-batch
    for i in range(grad_accum):
        b0 = i * (B // grad_accum)
        b1 = (i + 1) * (B // grad_accum)
        x_micro = x[b0:b1]
        y_micro = y[b0:b1]

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, loss = model_acc(x_micro, targets=y_micro)
            loss = loss / grad_accum  # 归一化

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    if scaler:
        scaler.unscale_(optimizer_for_acc)

    acc_grads = {n: p.grad.clone() for n, p in model_acc.named_parameters() if p.grad is not None}

    # ── 对比 ──
    max_diff = 0.0
    for name in ref_grads:
        diff = (ref_grads[name] - acc_grads[name]).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff > 1e-5:
            print(f"  差异 > 1e-5: {name}, diff={diff:.8f}")

    print(f"最大梯度差异: {max_diff:.10f}")
    assert max_diff < 1e-4, f"梯度不一致! max_diff={max_diff}"
    print("梯度累积 (+ AMP) 验证通过!")

verify_grad_accum_amp()
```

**关键点**：
1. 每个 micro-batch 的 loss 除以 `grad_accum_steps` 再 backward
2. 累积的梯度 ≈ 大 batch 的正常梯度
3. AMP 的 `scaler.scale(loss)` 与梯度累积完全兼容

---

## 练习 6：Scaling Law 观察

**难度**: ⭐⭐⭐ | **目标**: 训练不同规模模型，观察 loss 随参数量+compute 的变化

### 答案

**实验设计**：

```python
import torch
from model import LLaMA, create_model
from train import train
from data import get_dataloaders

def scaling_law_experiment():
    """训练三种规模，记录 loss 变化"""
    configs = [
        ("tiny",  "tiny",   3),
        ("small", "small",  3),
        # ("base",  "base",   3),  # 需要 GPU
    ]
    results = {}

    for name, model_size, epochs in configs:
        print(f"\n{'='*50}")
        print(f"训练 {name} 模型 ({model_size})")
        print(f"{'='*50}")

        # 数据 (WikiText-2)
        train_loader, val_loader = get_dataloaders(...)

        # 创建模型
        model = create_model("tiny", vocab_size=vocab_size)
        # 对于 small: 手动覆盖
        if model_size == "small":
            model = create_model("small", vocab_size=vocab_size)

        # 训练
        losses = []
        model = train(
            model, train_loader, val_loader,
            epochs=epochs, lr=3e-4, warmup_steps=100,
        )

        # 记录
        params = model.count_parameters()
        results[name] = {
            "params": params["total"],
            "val_loss": losses[-1],
        }

    return results
```

**预期结果（WikiText-2 上）**：

```
模型参数量 vs Val Loss

  Val Loss
  ^
 4.0│  ● tiny (25M, loss=3.8)
    │
 3.5│
    │     ● small (55M, loss=3.3)
 3.0│
    │            ● base (105M, loss=2.9)
 2.5│
    +─────┼────────┼──────────> 参数量 (M)
        25        55        105

Log-Log 拟合: loss ≈ a × N^(-b)
其中 b ≈ 0.05 (符合 Kaplan 2020 的 Scaling Law)
```

**Scaling Law 的核心洞察**：
1. 参数量每翻倍（25M → 50M → 100M），val loss 减少 10-15%
2. 要降到 GPT-3 级别的 loss (~1.5)，需要把参数量再提高 100× → 10B+
3. 这就是为什么 Scaling Law 推动了"越大越好"的军备竞赛

**如果只训练 1 个 epoch**（相同 compute budget）：
- Compute = tokens × params（近似）
- 在固定的 compute budget 下，不同规模的 loss 差异很小
- 这解释了为什么 Scaling Law 建议：**给定固定的 compute budget，同时增大模型和数据量才能获得最佳 loss**

---

## 总结：Stage 4 核心知识图谱

```
Stage 4: LLaMA 预训练
│
├── 归一化: RMSNorm ←— 省均值计算，更快
│   └── 对比: LayerNorm (Stage 3)
│
├── 激活: SwiGLU ←— 门控 + Swish > GELU
│   ├── SiLU: x·σ(x)
│   ├── Gate + Up + Down 三矩阵
│   └── 维度: d → 8d/3 → d
│
├── 位置: RoPE ←— 相对位置 + 外推
│   ├── 旋转矩阵: R_m·x
│   ├── 频率: θ_i = 10000^(-2i/d)
│   └── 性质: q_m·k_n 只依赖 (n-m)
│
├── 注意力: GQA ←— 省 KV Cache
│   ├── MHA: GQA(n_kv=n_head)
│   ├── MQA: GQA(n_kv=1)
│   └── GQA: 1 < n_kv < n_head
│
└── 训练基础设施
    ├── 混合精度 (AMP)
    ├── 梯度累积
    ├── Cosine LR + Warmup
    └── 检查点保存/恢复
```

这些组件将在 Stage 5 的指令微调中保持不变——SFT 和 DPO 不改变架构，只在训练目标和数据上做文章。

---

## 参考资源

- Kaplan et al. (2020) — "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022) — "Training Compute-Optimal Large Language Models" (Chinchilla)
- 本项目的 Stage 3 GPT-2 实现 (`stage3-transformer/gpt2-lm/`)
